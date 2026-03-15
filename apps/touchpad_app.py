import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from screeninfo import get_monitors
from tsai.models.InceptionTime import InceptionTime

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from OneEuroFilter import OneEuroFilter
from models.tcn_lstm import TCNLSTMStreaming
from utils.cd_ratio import CDRatioEstimator
from utils.imu_processor import IMUProcessor
from utils.window import Window

try:
    from common.stream_window import RingStreamWindowBase
    from common.ui_style import APP_STYLE_SHEET
except ImportError:
    from apps.common.stream_window import RingStreamWindowBase
    from apps.common.ui_style import APP_STYLE_SHEET

try:
    from pynput import mouse

    HAS_PC_CONTROL = True
    PC_CONTROL_ERROR = ""
except Exception as exc:
    mouse = None
    HAS_PC_CONTROL = False
    PC_CONTROL_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET


@dataclass(frozen=True)
class TouchpadModelConfig:
    gesture_ckpt: Path
    motion_ckpt: Path
    gesture_window_size: int = 20
    gesture_calculate_interval: int = 2
    gesture_history_size: int = 5
    threshold_down: int = 1
    threshold_up: int = 2
    click_interval_seconds: float = 0.1
    idle_gyro_threshold: float = 0.03


@dataclass
class PendingImuReading:
    address: str
    data: np.ndarray


@dataclass
class RingTouchpadRuntime:
    gesture_detector: "GestureDetector"
    motion_model: TCNLSTMStreaming
    imu_processor: IMUProcessor
    cd_ratio_estimator: CDRatioEstimator
    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    imu_data: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=1))
    last_gesture: int = 3
    last_down_time: float = 0.0
    packet_count: int = 0


class GestureDetector:
    def __init__(self, model: InceptionTime, config: TouchpadModelConfig, device: torch.device):
        self.device = device
        self.model = model
        self.config = config
        self.data_buffer: deque[np.ndarray] = deque()
        self.imu_window: Window[np.ndarray] = Window(config.gesture_window_size)
        self.cur_time_state = 0
        self.history_queue = deque(maxlen=config.gesture_history_size)

    def detect(self, data: np.ndarray) -> int:
        outputs = self.model(torch.from_numpy(data.T).float().view(1, 6, 20).to(self.device))
        pred = int(torch.argmax(outputs, dim=1).detach().cpu().item())
        self.history_queue.append(pred)

        count_0 = self.history_queue.count(0)
        count_1 = self.history_queue.count(1)

        if count_0 >= self.config.threshold_down:
            self.history_queue.clear()
            return 4
        if count_1 >= self.config.threshold_up:
            self.history_queue.clear()
            return 3
        return 0

    def calculate(self, data: np.ndarray) -> int:
        self.data_buffer.append(data)
        while self.data_buffer:
            self.imu_window.push(self.data_buffer.popleft())
        data_window = self.imu_window.map(lambda item: item).to_numpy().reshape(-1, 6)
        if data_window.shape[0] != self.config.gesture_window_size:
            return 0
        if self.cur_time_state < self.config.gesture_calculate_interval:
            self.cur_time_state += 1
            return 0
        self.cur_time_state = 0
        return self.detect(data_window)


class RingTouchpadEngine:
    def __init__(self, config: TouchpadModelConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gesture_state_dict = self._load_gesture_state_dict(config.gesture_ckpt)
        self.motion_state_dict = self._load_motion_state_dict(config.motion_ckpt)
        self.ring_states: dict[str, RingTouchpadRuntime] = {}
        self.mouse_controller = mouse.Controller() if HAS_PC_CONTROL else None
        self.last_status = "等待 IMU 数据"

        monitors = []
        if HAS_PC_CONTROL:
            try:
                monitors = get_monitors()
            except Exception:
                monitors = []
        if monitors:
            self.left = min(m.x for m in monitors)
            self.top = min(m.y for m in monitors)
            self.right = max(m.x + m.width for m in monitors)
            self.bottom = max(m.y + m.height for m in monitors)
        else:
            self.left = -10**9
            self.top = -10**9
            self.right = 10**9
            self.bottom = 10**9

    def clear_states(self, addresses: list[str] | None = None) -> None:
        if addresses is None:
            self.ring_states.clear()
            self.last_status = "状态已清空"
            return
        for address in addresses:
            self.ring_states.pop(address, None)

    def snapshot(self) -> dict[str, object]:
        return {
            "ring_count": len(self.ring_states),
            "last_status": self.last_status,
            "active_rings": [addr for addr, state in self.ring_states.items() if state.last_gesture == 4],
            "packet_count": sum(state.packet_count for state in self.ring_states.values()),
        }

    def process_imu(self, address: str, data: np.ndarray) -> None:
        state = self._state_for(address)
        state.packet_count += 1

        linear_data = data.copy()
        if state.imu_processor.pre_timestamp == 0:
            state.imu_processor.init_quaternion_and_timestamp(data[:3], timestamp=1)
        else:
            state.imu_processor.update_attitude_by_data(data[3:6], data[:3])
        linear_data[:3] = state.imu_processor.calculate_linear_acceleration(data[:3])

        state.imu_data.append(linear_data)

        current_gesture = state.gesture_detector.calculate(linear_data)
        if current_gesture != 0:
            self._handle_gesture_transition(state, address, current_gesture)

        if len(state.imu_data) < state.imu_data.maxlen:
            return

        merged_data = torch.tensor(np.array(state.imu_data), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = state.motion_model(merged_data.unsqueeze(0))
        outputs = output.detach().cpu().numpy()
        if outputs.shape != (1, 1, 2):
            raise RuntimeError(f"unexpected motion model output shape: {outputs.shape}")

        result = outputs[0]
        if np.mean(np.abs(data[3:6])) < self.config.idle_gyro_threshold:
            result = np.array([[0.0, 0.0]], dtype=np.float32)

        if state.last_gesture == 4:
            self._move_mouse(state, result)

    def _handle_gesture_transition(self, state: RingTouchpadRuntime, address: str, current_gesture: int) -> None:
        now = time.time()
        if current_gesture == 4:
            state.last_down_time = now
            if state.last_gesture != 4:
                self.last_status = f"{address} Down"
        elif current_gesture == 3:
            if state.last_gesture != 3:
                self.last_status = f"{address} Up"
            if state.last_gesture == 4 and now - state.last_down_time < self.config.click_interval_seconds:
                if self.mouse_controller is not None:
                    self.mouse_controller.click(mouse.Button.left)
                self.last_status = f"{address} Click"
        state.last_gesture = current_gesture

    def _move_mouse(self, state: RingTouchpadRuntime, outputs: np.ndarray) -> None:
        if self.mouse_controller is None:
            return

        for output in outputs:
            vx = float(output[0]) / 10.0 * 9.0
            vy = float(output[1]) / 10.0 * 9.0
            velocity_norm = float(np.linalg.norm([vx, vy]))
            cd_ratio = state.cd_ratio_estimator.update(velocity_norm)
            vx = float(state.filter_x(vx))
            vy = float(state.filter_y(vy))
            vx *= cd_ratio
            vy *= cd_ratio

            cx, cy = self.mouse_controller.position
            if cx + vx < self.left:
                vx = self.left - cx
            elif cx + vx > self.right:
                vx = self.right - cx
            if cy + vy < self.top:
                vy = self.top - cy
            elif cy + vy > self.bottom:
                vy = self.bottom - cy

            self.mouse_controller.move(vx, vy)
            self.last_status = f"移动鼠标 ({vx:.2f}, {vy:.2f})"

    def _state_for(self, address: str) -> RingTouchpadRuntime:
        state = self.ring_states.get(address)
        if state is not None:
            return state

        gesture_model = InceptionTime(6, 4, nf=16).to(self.device)
        gesture_model.load_state_dict(self.gesture_state_dict)
        gesture_model.eval()

        motion_model = TCNLSTMStreaming(6, 128, 2, [64, 128], 3, 0.2).to(self.device)
        motion_model.load_state_dict(self.motion_state_dict)
        motion_model.eval()

        state = RingTouchpadRuntime(
            gesture_detector=GestureDetector(gesture_model, self.config, self.device),
            motion_model=motion_model,
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            cd_ratio_estimator=CDRatioEstimator(min_cd=0.3, max_cd=2.0, c=0.2),
            filter_x=OneEuroFilter(freq=200, mincutoff=0.0004, beta=0.08),
            filter_y=OneEuroFilter(freq=200, mincutoff=0.0004, beta=0.08),
        )
        self.ring_states[address] = state
        return state

    def _load_gesture_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"gesture checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"invalid gesture checkpoint: {checkpoint_path}")
        return checkpoint["model"]

    def _load_motion_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"motion checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"invalid motion checkpoint: {checkpoint_path}")
        return checkpoint


class RingTouchpadWorker:
    def __init__(self, config: TouchpadModelConfig):
        self.engine = RingTouchpadEngine(config)
        self.pending_readings: deque[PendingImuReading] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def enqueue(self, address: str, data: np.ndarray) -> None:
        with self._condition:
            self.pending_readings.append(PendingImuReading(address=address, data=data))
            self._condition.notify()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            pending_count = len(self.pending_readings)
        snapshot = self.engine.snapshot()
        snapshot["pending_count"] = pending_count
        return snapshot

    def clear_states(self, addresses: list[str] | None = None) -> None:
        with self._condition:
            self.engine.clear_states(addresses)
            if addresses is None:
                self.pending_readings.clear()
            else:
                blocked = set(addresses)
                self.pending_readings = deque(
                    [item for item in self.pending_readings if item.address not in blocked],
                    maxlen=4096,
                )

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and not self.pending_readings:
                    self._condition.wait(timeout=0.2)
                if not self._running:
                    return
                reading = self.pending_readings.popleft()
            self.engine.process_imu(reading.address, reading.data)


class TouchpadAppWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int, config: TouchpadModelConfig):
        super().__init__(host=host, port=port)
        self.setWindowTitle("触控板应用")
        self.resize(860, 560)

        self.worker_error = ""
        self.touchpad_worker: RingTouchpadWorker | None = None
        if not HAS_PC_CONTROL:
            self.worker_error = f"桌面控制不可用：{PC_CONTROL_ERROR}"
        else:
            try:
                self.touchpad_worker = RingTouchpadWorker(config)
            except Exception as exc:
                self.worker_error = str(exc)

        self.status_text = QLabel("数据流状态：未连接")
        self.status_text.setObjectName("statusTag")
        self.runtime_text = QLabel(self.runtime_hint_text())
        self.runtime_text.setWordWrap(True)
        self.metrics_text = QLabel("包数: 0\n队列: 0\n活动戒指: 无")

        self._build_ui()
        self.set_stream_status_label(self.status_text)
        self._bind_signals()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_metrics)
        self.refresh_timer.start(200)

        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        panel = QFrame()
        panel.setProperty("class", "panel")
        panel_layout = QVBoxLayout(panel)

        title = QLabel("戒指触控板")
        title.setProperty("class", "title")
        desc = QLabel(
            "将 `ring_app_touchpad` 移植为独立流式应用。内部仍使用同一套手势模型、位移模型和点击判定逻辑。"
        )
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_text)
        panel_layout.addWidget(self.runtime_text)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择要作为触控板的戒指（不勾选表示订阅全部）",
        )

        row = QHBoxLayout()
        self.btn_apply = QPushButton("应用订阅")
        self.btn_apply.setProperty("class", "btn-primary")
        row.addWidget(self.btn_apply)
        row.addStretch()

        panel_layout.addLayout(row)
        panel_layout.addWidget(self.metrics_text)

        root.addWidget(panel)

    def _bind_signals(self) -> None:
        self.btn_apply.clicked.connect(self.apply_subscription)
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def handle_stream_message(self, message: dict) -> None:
        if self.touchpad_worker is None:
            return
        if message.get("type") != "imu":
            return

        address = str(message.get("address", ""))
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not address or not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        data = np.asarray(
            [
                float(acc_raw[0]),
                float(acc_raw[1]),
                float(acc_raw[2]),
                float(gyr_raw[0]),
                float(gyr_raw[1]),
                float(gyr_raw[2]),
            ],
            dtype=np.float32,
        )
        self.touchpad_worker.enqueue(address, data)

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        if self.touchpad_worker is None:
            return
        reset_addresses = removed_addresses + changed_connection_addresses
        if reset_addresses:
            self.touchpad_worker.clear_states(reset_addresses)

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])

    def runtime_hint_text(self) -> str:
        if self.worker_error:
            return f"初始化失败：{self.worker_error}"
        return "逻辑保持与旧版一致：Down 手势进入移动态，Up 手势退出；在短时间内 Down 后 Up 会触发左键点击。"

    def refresh_metrics(self) -> None:
        if self.touchpad_worker is None:
            self.runtime_text.setText(self.runtime_hint_text())
            return

        snapshot = self.touchpad_worker.snapshot()
        active_rings = snapshot["active_rings"]
        active_text = "、".join(active_rings) if active_rings else "无"
        self.runtime_text.setText(
            "\n".join(
                [
                    self.runtime_hint_text(),
                    f"最近状态：{snapshot['last_status']}",
                ]
            )
        )
        self.metrics_text.setText(
            "\n".join(
                [
                    f"包数: {int(snapshot['packet_count'])}",
                    f"队列: {int(snapshot['pending_count'])}",
                    f"活动戒指: {active_text}",
                    f"已初始化戒指: {int(snapshot['ring_count'])}",
                ]
            )
        )

    def closeEvent(self, event):
        self.refresh_timer.stop()
        self.client.stop()
        self.client.wait(1500)
        if self.touchpad_worker is not None:
            self.touchpad_worker.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    parser.add_argument("--gesture-ckpt", default=str(APP_ROOT / "checkpoint" / "tap-94.pth"))
    parser.add_argument("--motion-ckpt", default=str(APP_ROOT / "checkpoint" / "tcnlstm.pt"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TouchpadModelConfig(
        gesture_ckpt=Path(args.gesture_ckpt),
        motion_ckpt=Path(args.motion_ckpt),
    )
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = TouchpadAppWindow(host=args.host, port=args.port, config=config)
    window.show()
    sys.exit(app.exec())
