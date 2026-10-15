from __future__ import annotations

import argparse
import math
import os
import queue
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("Agg")

from apps.audio.low_latency_player import LowLatencyDrumPlayer
from apps.features.window_classifier.runtime import (
    ProbabilityChart,
    ProbabilityTimelineChart,
    RateTracker,
)
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from models.drum_gesture_recognizer import (
    DRUM_CLASS_IDS,
    DRUM_CLASS_LABELS,
    DrumGestureRecognizer,
    OnlineLinearAccelerationEstimator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoint"
    / "drum_gesture"
    / "my_v10.pth"
)
MODEL_OPTIONS = [
    (
        "right_p10_neg3_aug 6-class w21 (wxt val 95.8%, pos10-0 neg3 aug±12°+warp)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "right_p10_neg3_aug.pth",
        21,
    ),
    (
        "right_p14_neg3_aug 6-class w21 (wxt val 94.7%, pos14-4 neg3 aug±12°+warp)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "right_p14_neg3_aug.pth",
        21,
    ),
    (
        "multi_user_right_pos10neg3 6-class w21 (wxt val 96.6%, pos10-0 neg3+whr)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_right_pos10neg3.pth",
        21,
    ),
    (
        "multi_user_final_left 6-class w21 (LOO best=wxt 93.6%, pos14-4 neg2+whr)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_final_left.pth",
        21,
    ),
    (
        "multi_user_final_right 6-class w21 (LOO best=wxt 94.6%, pos14-4 neg2+whr)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_final_right.pth",
        21,
    ),
    (
        "5gesture 6-class w15 rawlin12hc (val 94.7%, 54 recordings)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "local_drum_5gesture_lead6_gap_raw6.pth",
        15,
    ),
    (
        "multi_user_wxtval2neg_left 6-class w21 (fast+slow+single+whr, wxt val, neg2)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_wxtval2neg_left.pth",
        21,
    ),
    (
        "multi_user_wxtval2neg_right 6-class w21 (fast+slow+single+whr, wxt val, neg2)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_wxtval2neg_right.pth",
        21,
    ),
    (
        "multi_user_left 6-class w21 (left hand only)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_left.pth",
        21,
    ),
    (
        "multi_user_right 6-class w21 (right hand only)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user_right.pth",
        21,
    ),
    (
        "multi_user 6-class w21 (datanew slow+single, no zhy)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "multi_user.pth",
        21,
    ),
    (
        "v11 lead18-4 gap3 rawlin12hc (no-test)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "my_v11.pth",
        15,
    ),
    (
        "v10 no-test all data (val acc 89.9%, all 47 groups)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "my_v10.pth",
        15,
    ),
    (
        "v9 switch labels fixed (test acc 91.3%, switch13 perfect)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "my_v9.pth",
        15,
    ),
    (
        "v8 1-per-class val (test acc 92.9%, balanced)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "my_v8.pth",
        15,
    ),
    (
        "v6 zone14-4 rapid+switch (acc 91.8%, 55K samples)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "my_v6.pth",
        15,
    ),
    (
        "v5p sliced_w15_rawlin12 (acc 96.6%, zone 22-16)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "sliced_w15_rawlin12_handcrafted.pth",
        15,
    ),
    (
        "v4p w15-17ch-zone22 (acc 90.8%)",
        PROJECT_ROOT
        / "checkpoint"
        / "drum_gesture"
        / "w15-17ch-zone22-group-v4p.pth",
        15,
    ),
]
SOUND_DIR = PROJECT_ROOT / "assets" / "drum_gesture" / "sounds"
IMU_SAMPLE_RATE_HZ = 200.0

DRUM_CLASS_COLORS = {
    0: QColor("#B0B8C1"),
    1: QColor("#5B8FF9"),
    2: QColor("#61DDAA"),
    3: QColor("#F6BD16"),
    4: QColor("#E8684A"),
    5: QColor("#8B5CF6"),
}
DRUM_SOUND_FILES = {
    1: SOUND_DIR / "gesture_1_closed_hat.wav",
    2: SOUND_DIR / "gesture_2_tom2.wav",
    3: SOUND_DIR / "gesture_3_snare.wav",
    4: SOUND_DIR / "gesture_4_tom1.wav",
    5: SOUND_DIR / "gesture_6_crash.wav",   # ring class 5 = crash
    6: SOUND_DIR / "gesture_5_kick.wav",    # foot kick (dedicated slot)
}

# ── 脚部踩踏检测默认参数 ──
FOOT_MODEL_PATH = (
    PROJECT_ROOT
    / "model_train"
    / "models"
    / "test_model_nodelay"
    / "hit_detector_inception.pth"
)
FOOT_OSC_PORT = 8888
FOOT_PROB_HIGH = 0.88
FOOT_PROB_LOW = 0.40
FOOT_COOLDOWN_MS = 200
FOOT_CLASS_ID = 6
FOOT_VOLUME_BOOST = 2.0  # 底鼓低频感知音量补偿

# ── 分轨音量补偿（保持音色不变,仅调大音量）──
PER_CLASS_VOLUME_BOOST = {
    1: 1.6,   # 踩镲 (gesture_1_closed_hat)
    2: 1.6,   # 通鼓2 (gesture_2_tom2)
    3: 1.6,   # 军鼓 (gesture_3_snare)
    4: 1.6,   # 通鼓1 (gesture_4_tom1)
    5: 1.6,   # 吊镲/crash (gesture_6_crash)
    6: FOOT_VOLUME_BOOST,  # 底鼓 (gesture_5_kick)
}

# ── 热路径打印限频: 每击一次 print(flush=True) 会同步写控制台, 阻塞 UI 主线程;
#    控制台滚动缓冲区持续增长时写操作越来越慢 → 打击越多越卡。改为限频且不 flush。
_PRINT_LAST: dict[str, float] = {}


def _rate_limited_print(key: str, text: str, interval: float = 0.5) -> None:
    now = time.perf_counter()
    if now - _PRINT_LAST.get(key, -1e9) >= interval:
        _PRINT_LAST[key] = now
        print(text)


@dataclass(frozen=True)
class DrumInferenceJob:
    address: str
    ring_name: str
    values: np.ndarray
    window_end_sample_index: int
    window_peak_norm: float
    # ── 延迟分解用时间戳 (窗口最新帧) ──
    newest_local_timestamp: float = 0.0
    newest_host_timestamp: float = 0.0
    newest_packet_end: float = 0.0
    newest_arrival_perf: float = 0.0
    submit_perf: float = 0.0


@dataclass(frozen=True)
class DrumInferenceResult:
    address: str
    ring_name: str
    prediction: object
    elapsed_ms: float
    sample_count: int
    window_end_sample_index: int
    window_peak_norm: float
    # ── 延迟分解用时间戳 (窗口最新帧) ──
    newest_local_timestamp: float = 0.0
    newest_host_timestamp: float = 0.0
    newest_packet_end: float = 0.0
    newest_arrival_perf: float = 0.0
    submit_perf: float = 0.0


@dataclass(frozen=True)
class DrumInferenceFailure:
    address: str
    ring_name: str
    error: str


class DrumInferenceWorker(QThread):
    ready = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: Path):
        super().__init__()
        self.model_path = Path(model_path)
        self._jobs: queue.Queue[DrumInferenceJob | None] = queue.Queue(maxsize=64)
        self._stopping = False

    def submit(self, job: DrumInferenceJob) -> None:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stopping = True
        try:
            if self._jobs.full():
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put_nowait(None)
        self.wait(1500)

    def _set_realtime_priority(self) -> None:
        """Set high thread priority and limit PyTorch threading to reduce jitter."""
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import ctypes.wintypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentThread()
            kernel32.SetThreadPriority(handle, 2)  # THREAD_PRIORITY_HIGHEST
        except Exception:
            pass

    def run(self) -> None:
        # ── 降低 CPU 调度抖动 ──
        self._set_realtime_priority()

        try:
            recognizer = DrumGestureRecognizer(self.model_path)
        except Exception as exc:
            self.fatal_error.emit(f"模型加载失败: {exc}")
            return

        self.ready.emit(recognizer.n_classes)
        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.job_failed.emit(DrumInferenceFailure(job.address, job.ring_name, str(exc)))
                continue
            self.result_ready.emit(
                DrumInferenceResult(
                    address=job.address,
                    ring_name=job.ring_name,
                    prediction=prediction,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    sample_count=int(job.values.shape[0]),
                    window_end_sample_index=job.window_end_sample_index,
                    window_peak_norm=job.window_peak_norm,
                    newest_local_timestamp=job.newest_local_timestamp,
                    newest_host_timestamp=job.newest_host_timestamp,
                    newest_packet_end=job.newest_packet_end,
                    newest_arrival_perf=job.newest_arrival_perf,
                    submit_perf=job.submit_perf,
                )
            )


class FootDetectWorker(QThread):
    """后台线程: UDP OSC 监听 + 脚部踩踏模型推理 (双阈值滞回)."""

    foot_hit_detected = pyqtSignal(float)
    foot_status_updated = pyqtSignal(str)
    foot_error = pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        osc_port: int = 8888,
        prob_high: float = 0.85,
        prob_low: float = 0.40,
        cooldown_ms: int = 400,
    ):
        super().__init__()
        self._model_path = Path(model_path)
        self._osc_port = int(osc_port)
        self._prob_high = float(prob_high)
        self._prob_low = float(prob_low)
        self._cooldown_ms = int(cooldown_ms)
        self._stopping = False

    def stop(self) -> None:
        self._stopping = True
        self.wait(3000)

    def run(self) -> None:
        import sys
        import traceback
        from pathlib import Path as _Path

        _project_root = _Path(__file__).resolve().parents[1]
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        try:
            self._run_impl(_project_root)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[FootDetectWorker] 未捕获异常:\n{tb}", flush=True)
            self.foot_error.emit(f"内部错误: {exc}")

    def _run_impl(self, _project_root) -> None:
        import sys
        import socket
        import threading as _threading
        import time as _time

        # ── 1. 导入检测模块 ──
        # realtime_detect.py 在模块级调用 matplotlib.use('TkAgg'),
        # 但 PyQt6 已占用 qt 框架, matplotlib 会拒绝切换并抛出异常.
        # 这里把 matplotlib.use 临时替换为空操作, 绕过冲突.
        print("[FootDetectWorker] 导入 realtime_detect 模块...", flush=True)
        import matplotlib as _mpl
        _mpl_use = _mpl.use
        _mpl.use = lambda *a, **kw: None
        try:
            from model_train.realtime_detect import (
                RealtimeDetector,
                load_model as _load_model,
                parse_osc_bundle,
            )
        except Exception as exc:
            print(f"[FootDetectWorker] 导入失败: {exc}", flush=True)
            self.foot_error.emit(f"导入脚部检测模块失败: {exc}")
            return
        finally:
            _mpl.use = _mpl_use
        print("[FootDetectWorker] 导入成功", flush=True)

        # ── 2. 加载模型 ──
        print(f"[FootDetectWorker] 加载模型: {self._model_path}", flush=True)
        try:
            model, mean, std, window_frames = _load_model(str(self._model_path))
        except Exception as exc:
            print(f"[FootDetectWorker] 模型加载失败: {exc}", flush=True)
            self.foot_error.emit(f"脚部模型加载失败: {exc}")
            return
        print(f"[FootDetectWorker] 模型加载成功 window_frames={window_frames} channels={len(mean)}", flush=True)

        self.foot_status_updated.emit(f"模型已加载 (窗={window_frames}帧), 启动 OSC 监听...")

        # ── 3. 创建检测器 ──
        detector = RealtimeDetector(
            model=model,
            mean=mean,
            std=std,
            window_frames=window_frames,
            prob_high=self._prob_high,
            prob_low=self._prob_low,
            cooldown_ms=self._cooldown_ms,
        )

        # ── 4. OSC 监听线程 ──
        osc_socket = None
        osc_stop = False
        osc_bound = False

        def osc_listener() -> None:
            nonlocal osc_socket, osc_stop, osc_bound
            try:
                osc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                osc_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                osc_socket.settimeout(0.1)
                osc_socket.bind(("0.0.0.0", self._osc_port))
                osc_bound = True
                print(f"[FootDetectWorker] OSC 监听已绑定端口 {self._osc_port}", flush=True)
                self.foot_status_updated.emit(f"OSC 端口 {self._osc_port} 已绑定，等待数据...")
            except OSError as exc:
                print(f"[FootDetectWorker] OSC 端口 {self._osc_port} 绑定失败: {exc}", flush=True)
                self.foot_error.emit(f"OSC 端口 {self._osc_port} 绑定失败: {exc}")
                detector.running = False
                return

            while not osc_stop and detector.running:
                try:
                    data, _addr = osc_socket.recvfrom(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    frame = parse_osc_bundle(data)
                except Exception:
                    continue
                if len(frame) >= 6:
                    detector.add_imu_frame(*frame)

            if osc_socket is not None:
                try:
                    osc_socket.close()
                except OSError:
                    pass

        osc_thread = _threading.Thread(target=osc_listener, daemon=True)
        osc_thread.start()

        # 等待 OSC 绑定完成 (最多 2 秒)
        _time.sleep(0.3)
        if not osc_bound and not self._stopping:
            # osc_listener 在绑定失败时会 emit foot_error 并设置 detector.running=False
            if not detector.running:
                return

        # ── 5. 主检测循环 ──
        print("[FootDetectWorker] 进入检测循环", flush=True)
        last_frame_count = 0
        last_buffer_report = _time.time()
        try:
            while not self._stopping and detector.running:
                _time.sleep(0.001)

                with detector.lock:
                    cur_count = detector.frame_count
                    buf_len = len(detector.buffer)

                if buf_len < detector.window_frames:
                    # 每秒报告一次缓冲状态
                    now = _time.time()
                    if now - last_buffer_report > 2.0:
                        last_buffer_report = now
                        if cur_count == 0:
                            self.foot_status_updated.emit(
                                f"等待 OSC 数据... (缓冲 {buf_len}/{detector.window_frames} 帧)"
                            )
                        else:
                            self.foot_status_updated.emit(
                                f"缓冲中... {buf_len}/{detector.window_frames} 帧"
                            )
                    continue

                if cur_count == last_frame_count:
                    continue
                last_frame_count = cur_count

                try:
                    prob = detector.predict()
                except Exception:
                    continue

                detector.total_frames += 1

                now = _time.time()
                in_cooldown = (now - detector.last_trigger_time) < detector.cooldown_s

                if prob > detector.prob_high:
                    if not in_cooldown:
                        self.foot_hit_detected.emit(prob)
                        detector.last_trigger_time = now
                        if not detector.armed:
                            detector.armed = True
                            detector.total_triggers += 1
                            _rate_limited_print("foot", f"[FootDetectWorker] 检测到踩踏! P={prob:.3f} #{detector.total_triggers}")
                elif prob < detector.prob_low:
                    detector.armed = False

                if detector.total_frames % 60 == 0:
                    cooldown_left = max(0.0, detector.cooldown_s - (now - detector.last_trigger_time))
                    cd_str = f"CD{cooldown_left:.1f}s" if cooldown_left > 0.01 else "ready"
                    state_str = "ARMED" if detector.armed else "idle"
                    self.foot_status_updated.emit(
                        f"帧={detector.total_frames} | P={prob:.3f} | {state_str} | {cd_str} | trig={detector.total_triggers}"
                    )
        finally:
            detector.running = False
            osc_stop = True
            osc_thread.join(timeout=1.0)
            print("[FootDetectWorker] 检测循环已退出", flush=True)
            self.foot_status_updated.emit("已停止")


@dataclass
class DrumRingState:
    ring_name: str
    estimator: OnlineLinearAccelerationEstimator = field(default_factory=OnlineLinearAccelerationEstimator)
    samples: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=512))
    norms: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    committed_counts: Counter[int] = field(default_factory=Counter)
    prediction_history: deque[int] = field(default_factory=deque)
    rate_tracker: RateTracker = field(default_factory=RateTracker)
    sample_count: int = 0
    frames_since_infer: int = 0
    last_timestamp: float | None = None
    last_trigger_sample: int = -100000
    # ── 时间戳缓冲 (用于延迟分解, 与 samples 一一对应) ──
    local_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    host_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    packet_end_counters: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    arrival_perfs: deque[float] = field(default_factory=lambda: deque(maxlen=512))


class DrumGestureWindow(RingStreamWindowBase):
    probability_changed = pyqtSignal(tuple)

    # 视觉刷新节流: 两个窗口同时高频重绘会导致概率图卡顿, 限制最大刷新率 ~30Hz
    UI_REFRESH_MIN_INTERVAL = 0.033

    def __init__(self, host: str, port: int, model_path: Path):
        super().__init__(host=host, port=port)
        self.setWindowTitle("Drum Gesture 实时识别")
        self.resize(1180, 780)

        self.model_path = Path(model_path).expanduser().resolve()
        self.window_size = self.window_size_for_model(self.model_path)
        self.states: dict[str, DrumRingState] = {}
        self.active_address = ""
        self.active_result_address = ""
        self.runtime_ready = False
        self.runtime_failed = False
        self.in_flight_addresses: set[str] = set()
        self.player: LowLatencyDrumPlayer | None = None
        self.timeline_window: ProbabilityTimelineWindow | None = None
        self._last_ui_refresh = 0.0
        self._last_frame_ui_refresh = 0.0
        self._latest_status = ""

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel("状态：准备启动")
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel(f"模型：{self.display_path(self.model_path)} | 加载中")
        self.active_ring_label = QLabel("当前显示：无")
        self.buffer_label = QLabel("缓冲区：0 帧")
        self.fps_label = QLabel("结果帧率：0.0 Hz")
        self.peak_label = QLabel("最近峰值：-")

        self.model_select = QComboBox()
        for label, path, window_size in MODEL_OPTIONS:
            self.model_select.addItem(label, (str(path.resolve()), int(window_size)))
        self.select_current_model_option()

        self.active_class_ids = DRUM_CLASS_IDS
        self.chart = ProbabilityChart(
            class_ids=self.active_class_ids,
            class_labels=DRUM_CLASS_LABELS,
            class_colors=DRUM_CLASS_COLORS,
        )

        self.confidence_threshold_input = QDoubleSpinBox()
        self.confidence_threshold_input.setRange(0.0, 1.0)
        self.confidence_threshold_input.setDecimals(2)
        self.confidence_threshold_input.setSingleStep(0.05)
        self.confidence_threshold_input.setValue(0.55)
        self.confidence_threshold_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.cooldown_input = QSpinBox()
        self.cooldown_input.setRange(1, 200)
        self.cooldown_input.setSingleStep(5)
        self.cooldown_input.setValue(24)
        self.cooldown_input.setSuffix(" 帧")
        self.cooldown_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.step_input = QSpinBox()
        self.step_input.setRange(1, 20)
        self.step_input.setValue(4)
        self.step_input.setSuffix(" 帧")
        self.step_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.stable_window_input = QDoubleSpinBox()
        self.stable_window_input.setRange(0.005, 0.50)
        self.stable_window_input.setDecimals(3)
        self.stable_window_input.setSingleStep(0.005)
        self.stable_window_input.setValue(0.02)
        self.stable_window_input.setSuffix(" s")
        self.stable_window_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.trigger_ratio_input = QDoubleSpinBox()
        self.trigger_ratio_input.setRange(0.50, 1.00)
        self.trigger_ratio_input.setDecimals(2)
        self.trigger_ratio_input.setSingleStep(0.05)
        self.trigger_ratio_input.setValue(0.67)
        self.trigger_ratio_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.volume_input = QDoubleSpinBox()
        self.volume_input.setRange(0.0, 1.0)
        self.volume_input.setDecimals(2)
        self.volume_input.setSingleStep(0.05)
        self.volume_input.setValue(0.80)
        self.volume_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.volume_input.valueChanged.connect(self.on_volume_changed)

        # ── 脚部踩踏检测控件 ──
        self.foot_enable_checkbox = QCheckBox("启用脚部踩踏检测 (OSC)")
        self.foot_prob_high_input = QDoubleSpinBox()
        self.foot_prob_high_input.setRange(0.50, 0.99)
        self.foot_prob_high_input.setDecimals(2)
        self.foot_prob_high_input.setSingleStep(0.05)
        self.foot_prob_high_input.setValue(FOOT_PROB_HIGH)
        self.foot_prob_high_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.foot_prob_low_input = QDoubleSpinBox()
        self.foot_prob_low_input.setRange(0.10, 0.80)
        self.foot_prob_low_input.setDecimals(2)
        self.foot_prob_low_input.setSingleStep(0.05)
        self.foot_prob_low_input.setValue(FOOT_PROB_LOW)
        self.foot_prob_low_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.foot_cooldown_input = QSpinBox()
        self.foot_cooldown_input.setRange(100, 2000)
        self.foot_cooldown_input.setSingleStep(50)
        self.foot_cooldown_input.setValue(FOOT_COOLDOWN_MS)
        self.foot_cooldown_input.setSuffix(" ms")
        self.foot_cooldown_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.foot_status_label = QLabel("脚部检测：未启用")
        self.foot_worker: FootDetectWorker | None = None

        # ── 阈值触发 (测延迟用) ──
        self.threshold_enable_checkbox = QCheckBox("阈值触发 (测延迟)")
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 500.0)
        self.threshold_input.setDecimals(1)
        self.threshold_input.setSingleStep(5.0)
        self.threshold_input.setValue(90.0)
        self.threshold_input.setSuffix(" m/s²")
        self.threshold_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.threshold_cooldown_input = QSpinBox()
        self.threshold_cooldown_input.setRange(10, 2000)
        self.threshold_cooldown_input.setSingleStep(10)
        self.threshold_cooldown_input.setValue(200)
        self.threshold_cooldown_input.setSuffix(" ms")
        self.threshold_cooldown_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.threshold_class_input = QSpinBox()
        self.threshold_class_input.setRange(1, 5)
        self.threshold_class_input.setValue(1)
        self.threshold_class_input.setSuffix(" 类音色")
        self.threshold_class_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.latency_label = QLabel("延迟分解：--")
        self.latency_label.setProperty("class", "sub-text")
        self.latency_label.setWordWrap(True)

        self.last_threshold_trigger_perf = 0.0

        self.load_sound_effects()
        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()

        self.worker: DrumInferenceWorker | None = None
        self.start_worker()
        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        panel = QFrame()
        panel.setProperty("class", "panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(10)

        title = QLabel("Drum Gesture 实时识别")
        title.setProperty("class", "title")
        desc = QLabel(
            "在线估计 linear acceleration，持续滑窗推理最近窗口，稳定判定后触发对应鼓件音色。"
        )
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)
        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        for widget in (
            self.stream_status_label,
            self.workflow_label,
            self.device_label,
            self.active_ring_label,
            self.buffer_label,
            self.fps_label,
            self.peak_label,
            self.foot_status_label,
            self.latency_label,
        ):
            panel_layout.addWidget(widget)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=150,
        )

        settings_panel = QFrame()
        settings_panel.setProperty("class", "settings-panel")
        settings = QGridLayout(settings_panel)
        settings.setContentsMargins(14, 12, 14, 12)
        settings.setHorizontalSpacing(16)
        settings.setVerticalSpacing(12)
        settings.addWidget(QLabel("模型"), 0, 0)
        settings.addWidget(self.model_select, 0, 1, 1, 3)
        settings.addWidget(QLabel("推理步长"), 1, 0)
        settings.addWidget(self.step_input, 1, 1)
        settings.addWidget(QLabel("置信阈值"), 1, 2)
        settings.addWidget(self.confidence_threshold_input, 1, 3)
        settings.addWidget(QLabel("冷却时间"), 2, 0)
        settings.addWidget(self.cooldown_input, 2, 1)
        settings.addWidget(QLabel("稳定窗口"), 2, 2)
        settings.addWidget(self.stable_window_input, 2, 3)
        settings.addWidget(QLabel("触发比例"), 3, 0)
        settings.addWidget(self.trigger_ratio_input, 3, 1)
        settings.addWidget(QLabel("音量"), 3, 2)
        settings.addWidget(self.volume_input, 3, 3)

        # ── 脚部检测参数行 ──
        settings.addWidget(QLabel("脚部检测"), 4, 0)
        settings.addWidget(self.foot_enable_checkbox, 4, 1)
        settings.addWidget(QLabel("触发阈值"), 4, 2)
        settings.addWidget(self.foot_prob_high_input, 4, 3)

        settings.addWidget(QLabel("释放阈值"), 5, 0)
        settings.addWidget(self.foot_prob_low_input, 5, 1)
        settings.addWidget(QLabel("冷却(ms)"), 5, 2)
        settings.addWidget(self.foot_cooldown_input, 5, 3)

        # ── 阈值触发参数行 ──
        settings.addWidget(QLabel("阈值触发"), 6, 0)
        settings.addWidget(self.threshold_enable_checkbox, 6, 1)
        settings.addWidget(QLabel("模长阈值"), 6, 2)
        settings.addWidget(self.threshold_input, 6, 3)
        settings.addWidget(QLabel("冷却(ms)"), 7, 0)
        settings.addWidget(self.threshold_cooldown_input, 7, 1)
        settings.addWidget(QLabel("音色"), 7, 2)
        settings.addWidget(self.threshold_class_input, 7, 3)

        panel_layout.addWidget(settings_panel)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setProperty("class", "btn-gray")
        self.timeline_btn = QPushButton("打开概率曲线")
        self.timeline_btn.setProperty("class", "btn-gray")
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.clear_btn)
        action_row.addWidget(self.timeline_btn)
        action_row.addStretch()
        panel_layout.addLayout(action_row)

        result_panel = QFrame()
        result_panel.setProperty("class", "panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_layout.setSpacing(10)
        result_title = QLabel("实时概率")
        result_title.setProperty("class", "title")
        self.result_label = QLabel("--")
        self.result_label.setProperty("class", "title")
        self.result_meta_label = QLabel("最近结果：无")
        self.result_meta_label.setProperty("class", "sub-text")
        self.result_meta_label.setWordWrap(True)
        self.count_label = QLabel(self.empty_count_text())
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.chart, 1)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_meta_label)
        result_layout.addWidget(self.count_label)

        root.addWidget(panel)
        root.addWidget(result_panel, 1)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.clear_btn.clicked.connect(self.clear_results)
        self.timeline_btn.clicked.connect(self.open_timeline_window)
        self.model_select.currentIndexChanged.connect(self.on_model_selected)
        self.foot_enable_checkbox.toggled.connect(self.on_foot_detection_toggled)
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])
        if self.runtime_failed:
            self.set_runtime_status("状态：已订阅 IMU，但推理不可用")
        elif self.runtime_ready:
            self.set_runtime_status("状态：已订阅 IMU，持续滑窗推理中")
        else:
            self.set_runtime_status("状态：已订阅 IMU，等待模型加载")

    def handle_stream_message(self, message: dict) -> None:
        if str(message.get("type", "")) != "imu":
            return
        self.handle_imu_message(message)

    def handle_imu_message(self, message: dict) -> None:
        handle_perf = time.perf_counter()
        address = str(message.get("address", ""))
        if not address:
            return
        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return

        state = self.state_for(address)
        timestamp = self.message_timestamp(message)
        dt = self.sample_dt(state, timestamp)
        linear_acc, _gravity = state.estimator.update(
            np.asarray(acc, dtype=np.float32),
            np.asarray(gyr, dtype=np.float32),
            dt,
        )
        raw_acc = np.asarray(acc, dtype=np.float32)
        raw_gyro = np.asarray(gyr, dtype=np.float32)
        sample = np.concatenate((raw_acc, raw_gyro, linear_acc)).astype(np.float32)
        state.samples.append(sample)
        state.norms.append(float(np.linalg.norm(linear_acc)))
        # ── 时间戳缓冲 (与 samples 一一对应, 用于延迟分解) ──
        state.local_timestamps.append(float(message.get("local_timestamp", 0.0) or 0.0))
        state.host_timestamps.append(float(message.get("host_timestamp", 0.0) or 0.0))
        state.packet_end_counters.append(
            float(message.get("packet_end_counter", 0) or 0) / 16384.0
        )
        state.arrival_perfs.append(float(message.get("arrival_perf", 0.0) or 0.0))
        state.sample_count += 1
        state.frames_since_infer += 1
        state.last_timestamp = timestamp

        raw_norm = float(np.linalg.norm(raw_acc))

        self.active_address = address
        # ── 每帧 UI 标签节流: 200Hz 的 setText 会拖慢 UI 线程, 限制到 ~30Hz ──
        now = time.perf_counter()
        if now - self._last_frame_ui_refresh >= self.UI_REFRESH_MIN_INTERVAL:
            self._last_frame_ui_refresh = now
            self.active_ring_label.setText(f"当前显示：{state.ring_name}")
            self.buffer_label.setText(
                f"缓冲区：{len(state.samples)} 帧 | raw模长 {raw_norm:.1f} | 线性模长 {state.norms[-1]:.1f}"
            )
            self.update_count_views(state.committed_counts)

        # ── 阈值触发模式 (独立于模型, 用于测量链路延迟) ──
        if self.threshold_enable_checkbox.isChecked():
            threshold = float(self.threshold_input.value())
            cooldown_s = float(self.threshold_cooldown_input.value()) / 1000.0
            if raw_norm > threshold and (handle_perf - self.last_threshold_trigger_perf) > cooldown_s:
                self.last_threshold_trigger_perf = handle_perf
                self.play_gesture_sound(int(self.threshold_class_input.value()))
                self._update_latency_breakdown(message, raw_norm, handle_perf)

        self.maybe_submit_window(address)

    def _update_latency_breakdown(
        self,
        message: dict,
        raw_norm: float,
        handle_perf: float,
    ) -> None:
        """计算并显示延迟分解。

        时钟域说明:
        - 锯齿: 设备时钟域 (精确, 无需校准)
        - 传输: 主进程 perf_counter 域 (arrival 与 host 同域做差, 精确)
        - 应用侧: 本进程 perf_counter 域 (handle 与 sound 同域做差, 精确)
        """
        local_ts = float(message.get("local_timestamp", 0.0) or 0.0)
        host_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        packet_end = float(message.get("packet_end_counter", 0) or 0) / 16384.0
        arrival_perf = float(message.get("arrival_perf", 0.0) or 0.0)

        # ① 锯齿(打包等待)
        if packet_end > 0 and local_ts > 0:
            sawtooth_ms = (packet_end - local_ts) * 1000.0
        else:
            sawtooth_ms = float("nan")

        # ② BLE传输(无线电+解析): arrival 与 host 均为主进程 perf 域
        if arrival_perf > 0 and host_ts > 0:
            device_to_arrival_ms = (arrival_perf - host_ts) * 1000.0
            transport_ms = (
                device_to_arrival_ms - sawtooth_ms
                if not math.isnan(sawtooth_ms)
                else device_to_arrival_ms
            )
        else:
            transport_ms = float("nan")

        # ③ 应用侧(WebSocket中转+UI+触发)
        sound_perf = time.perf_counter()
        app_side_ms = (sound_perf - handle_perf) * 1000.0

        parts = [
            ("锯齿", sawtooth_ms),
            ("传输", transport_ms),
            ("应用侧", app_side_ms),
        ]
        total_ms = sum(value for _name, value in parts if not math.isnan(value))
        text = (
            f"延迟分解: 锯齿={sawtooth_ms:.1f}ms | 传输={transport_ms:.1f}ms | "
            f"应用侧(含中转)={app_side_ms:.1f}ms | 合计(不含中转)≈{total_ms:.1f}ms | raw模长={raw_norm:.1f}"
        )
        self.latency_label.setText(text)
        _rate_limited_print("threshold", f"[阈值触发] {text}")

    def _update_model_latency_breakdown(
        self,
        result: DrumInferenceResult,
        queue_length: int,
    ) -> None:
        """模型路径触发时的延迟分解。

        与阈值路径共用「锯齿 + 传输」的时钟域计算方法, 并追加模型特有部分:
        - 推理: worker 实测 elapsed_ms
        - 投票: (queue_length - 1) 次额外推理, 每次间隔 step 帧
        """
        packet_end = result.newest_packet_end
        local_ts = result.newest_local_timestamp
        host_ts = result.newest_host_timestamp
        arrival_perf = result.newest_arrival_perf

        # ① 锯齿(打包等待): 设备时钟域
        if packet_end > 0 and local_ts > 0:
            sawtooth_ms = (packet_end - local_ts) * 1000.0
        else:
            sawtooth_ms = float("nan")

        # ② 传输(无线电+解析): 主进程 perf 域
        if arrival_perf > 0 and host_ts > 0:
            device_to_arrival_ms = (arrival_perf - host_ts) * 1000.0
            transport_ms = (
                device_to_arrival_ms - sawtooth_ms
                if not math.isnan(sawtooth_ms)
                else device_to_arrival_ms
            )
        else:
            transport_ms = float("nan")

        # ③ 推理: worker 实测
        inference_ms = float(result.elapsed_ms)

        # ④ 投票队列: 额外 (queue_length - 1) 次推理等待
        step = max(int(self.step_input.value()), 1)
        frame_ms = 1000.0 / IMU_SAMPLE_RATE_HZ
        voting_ms = float(max(0, queue_length - 1)) * step * frame_ms

        parts = [
            ("锯齿", sawtooth_ms),
            ("传输", transport_ms),
            ("推理", inference_ms),
            ("投票", voting_ms),
        ]
        total_ms = sum(value for _name, value in parts if not math.isnan(value))
        window_frames = int(self.window_size)
        text = (
            f"模型路径: 锯齿={sawtooth_ms:.1f}ms | 传输={transport_ms:.1f}ms | "
            f"推理={inference_ms:.1f}ms | 投票={voting_ms:.1f}ms | "
            f"合计≈{total_ms:.1f}ms | 窗长={window_frames}帧"
        )
        self.latency_label.setText(text)
        _rate_limited_print("model", f"[模型触发] {text}")

    def maybe_submit_window(self, address: str) -> None:
        if (
            self.worker is None
            or self.runtime_failed
            or not self.runtime_ready
            or address in self.in_flight_addresses
        ):
            return
        state = self.states.get(address)
        window_size = int(self.window_size)
        if state is None or len(state.samples) < window_size:
            return

        step = int(self.step_input.value())
        if state.frames_since_infer < step:
            return

        values = np.stack(list(state.samples)[-window_size:], axis=0).astype(np.float32, copy=False)
        if values.shape != (window_size, 9):
            return

        state.frames_since_infer = 0
        window_end = state.sample_count - 1
        window_peak = float(max(list(state.norms)[-window_size:]))
        # ── 捕获窗口最新帧的时间戳 (用于模型路径延迟分解) ──
        newest_local_ts = state.local_timestamps[-1] if state.local_timestamps else 0.0
        newest_host_ts = state.host_timestamps[-1] if state.host_timestamps else 0.0
        newest_packet_end = state.packet_end_counters[-1] if state.packet_end_counters else 0.0
        newest_arrival_perf = state.arrival_perfs[-1] if state.arrival_perfs else 0.0
        submit_perf = time.perf_counter()
        self.in_flight_addresses.add(address)
        self.worker.submit(
            DrumInferenceJob(
                address=address,
                ring_name=state.ring_name,
                values=values,
                window_end_sample_index=window_end,
                window_peak_norm=window_peak,
                newest_local_timestamp=newest_local_ts,
                newest_host_timestamp=newest_host_ts,
                newest_packet_end=newest_packet_end,
                newest_arrival_perf=newest_arrival_perf,
                submit_perf=submit_perf,
            )
        )
        self.set_runtime_status(f"状态：{state.ring_name} 滑窗推理中（步长 {step} 帧）")

    def on_worker_ready(self, class_count: int) -> None:
        self.runtime_ready = True
        self.active_class_ids = tuple(range(int(class_count)))
        self.chart.set_class_ids(self.active_class_ids)
        self.count_label.setText(self.empty_count_text())
        self.device_label.setText(
            f"模型：{self.current_model_label()} | window={self.window_size} | 已就绪"
        )
        self.set_runtime_status("状态：模型已就绪，等待 IMU 数据")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.in_flight_addresses.clear()
        self.device_label.setText(
            f"模型：{self.current_model_label()} | window={self.window_size} | 加载失败"
        )
        self.set_runtime_status(f"状态：{message}")

    def on_inference_result(self, result: DrumInferenceResult) -> None:
        self.in_flight_addresses.discard(result.address)
        state = self.state_for(result.address)
        self.active_result_address = result.address
        state.rate_tracker.mark()
        prediction = result.prediction

        # ── 关键路径: 触发判定 + 窗口提交 (不节流, 保证低延迟) ──
        decision_class, history_size, queue_length, class_count, threshold = self.update_prediction_history(
            state,
            int(prediction.class_id),
            float(prediction.confidence),
        )

        is_trigger = (
            decision_class is not None
            and result.window_end_sample_index - state.last_trigger_sample >= int(self.cooldown_input.value())
        )
        if is_trigger:
            class_id = int(decision_class)
            state.committed_counts[class_id] += 1
            state.last_trigger_sample = int(result.window_end_sample_index)
            state.prediction_history.clear()
            self.play_gesture_sound(class_id)
            self._update_model_latency_breakdown(result, queue_length)
            self._latest_status = (
                f"状态：{result.ring_name} 稳定触发 {class_id}:{DRUM_CLASS_LABELS[class_id]} "
                f"({prediction.confidence:.3f})"
            )
        else:
            self._latest_status = (
                f"状态：{result.ring_name} -> {prediction.class_id}:{prediction.label} "
                f"({prediction.confidence:.3f})，等待稳定触发"
            )

        self.maybe_submit_window(result.address)
        self.submit_ready_windows(exclude_address=result.address)

        # ── 所有 UI 刷新统一节流到 ~30Hz (图表 + 标签 + 状态栏) ──
        # 触发判定 / 音效 / 推理提交都在上方, 保持全速率, 不受节流影响。
        now = time.perf_counter()
        if now - self._last_ui_refresh < self.UI_REFRESH_MIN_INTERVAL:
            return
        self._last_ui_refresh = now

        self.chart.set_probabilities(prediction.probabilities)
        self.probability_changed.emit(prediction.probabilities)
        self.result_label.setText(f"{prediction.class_id} · {prediction.label}")
        self.peak_label.setText(
            f"窗口峰值：{result.window_peak_norm:.1f} | window end sample {result.window_end_sample_index}"
        )
        self.result_meta_label.setText(
            f"{result.ring_name} | 窗长 {result.sample_count} 帧 | 推理 {result.elapsed_ms:.1f} ms | "
            f"最高概率 {prediction.confidence:.3f} | 队列 {history_size}/{queue_length} | "
            f"当前类计数 {class_count}/{threshold}"
        )
        self.set_runtime_status(self._latest_status)
        self.update_count_views(state.committed_counts)

    def on_inference_failed(self, failure: DrumInferenceFailure) -> None:
        self.in_flight_addresses.discard(failure.address)
        self.set_runtime_status(f"状态：{failure.ring_name} 推理失败（{failure.error}）")
        self.submit_ready_windows()

    def submit_ready_windows(self, exclude_address: str = "") -> None:
        for address in list(self.states):
            if address == exclude_address:
                continue
            self.maybe_submit_window(address)

    def result_queue_length(self) -> int:
        # 固定 1 次投票 (立即触发): 与推理步长解耦, 避免 step 变化改变投票次数。
        # 历史队列仅保留最近 1 次推理结果, 置信度 ≥ 阈值即触发。
        return 1

    def min_trigger_count(self, queue_length: int) -> int:
        return max(1, int(math.ceil(float(self.trigger_ratio_input.value()) * queue_length)))

    def update_prediction_history(
        self,
        state: DrumRingState,
        class_id: int,
        confidence: float,
    ) -> tuple[int | None, int, int, int, int]:
        queue_length = self.result_queue_length()
        threshold = self.min_trigger_count(queue_length)
        stable_class_id = class_id
        if confidence < float(self.confidence_threshold_input.value()):
            stable_class_id = 0

        history = state.prediction_history
        history.append(stable_class_id)
        while len(history) > queue_length:
            history.popleft()

        history_size = len(history)
        class_count = history.count(stable_class_id)
        if history_size < queue_length:
            return None, history_size, queue_length, class_count, threshold

        for candidate in self.active_class_ids[1:]:
            count = history.count(candidate)
            if count >= threshold:
                return int(candidate), history_size, queue_length, count, threshold
        return None, history_size, queue_length, class_count, threshold

    def clear_results(self) -> None:
        self.states.clear()
        self.active_address = ""
        self.active_result_address = ""
        self.in_flight_addresses.clear()
        self.chart.clear()
        self.result_label.setText("--")
        self.result_meta_label.setText("最近结果：无")
        self.count_label.setText(self.empty_count_text())
        self.active_ring_label.setText("当前显示：无")
        self.buffer_label.setText("缓冲区：0 帧")
        self.fps_label.setText("结果帧率：0.0 Hz")
        self.peak_label.setText("最近峰值：-")
        self.latency_label.setText("延迟分解：--")
        self.last_threshold_trigger_perf = 0.0
        self.set_runtime_status("状态：结果已清空，等待 IMU 数据")

    def open_timeline_window(self) -> None:
        if self.timeline_window is not None and self.timeline_window.isVisible():
            self.timeline_window.raise_()
            self.timeline_window.activateWindow()
            return
        self.timeline_window = ProbabilityTimelineWindow(
            class_ids=self.active_class_ids,
            class_labels=DRUM_CLASS_LABELS,
            class_colors=DRUM_CLASS_COLORS,
        )
        self.probability_changed.connect(self.timeline_window.update_probabilities)
        self.timeline_window.destroyed.connect(self._on_timeline_window_closed)
        self.timeline_window.show()

    def _on_timeline_window_closed(self) -> None:
        if self.timeline_window is not None:
            self.probability_changed.disconnect(self.timeline_window.update_probabilities)
            self.timeline_window = None

    def start_worker(self) -> None:
        self.runtime_ready = False
        self.runtime_failed = False
        self.in_flight_addresses.clear()
        self.device_label.setText(
            f"模型：{self.current_model_label()} | window={self.window_size} | 加载中"
        )
        self.worker = DrumInferenceWorker(self.model_path)
        self.worker.ready.connect(self.on_worker_ready)
        self.worker.result_ready.connect(self.on_inference_result)
        self.worker.job_failed.connect(self.on_inference_failed)
        self.worker.fatal_error.connect(self.on_worker_fatal_error)
        self.worker.start()

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def on_model_selected(self) -> None:
        data = self.model_select.currentData()
        if not data:
            return
        path_text, window_size = data
        selected_path = Path(path_text).expanduser().resolve()
        if selected_path == self.model_path and int(window_size) == int(self.window_size):
            return

        self.stop_worker()
        self.model_path = selected_path
        self.window_size = int(window_size)
        self.clear_results()
        self.start_worker()

    def current_model_label(self) -> str:
        text = self.model_select.currentText()
        if text:
            return text
        return self.display_path(self.model_path)

    def select_current_model_option(self) -> None:
        resolved = self.model_path.resolve()
        for index in range(self.model_select.count()):
            data = self.model_select.itemData(index)
            if data and Path(data[0]).resolve() == resolved:
                self.model_select.setCurrentIndex(index)
                return

    def window_size_for_model(self, model_path: Path) -> int:
        resolved = Path(model_path).expanduser().resolve()
        for _label, path, window_size in MODEL_OPTIONS:
            if path.resolve() == resolved:
                return int(window_size)
        return 20

    def load_sound_effects(self) -> None:
        # 用低延迟播放器替代 QSoundEffect (消除 Qt 音频引擎大缓冲)
        if self.player is not None:
            self.player.close()
        self.player = LowLatencyDrumPlayer()
        for class_id, path in DRUM_SOUND_FILES.items():
            self.player.load_sound(class_id, path.resolve())

    def on_volume_changed(self, value: float) -> None:
        # 音量在 play_gesture_sound 中即时应用, 此处无需预置
        _ = value

    def play_gesture_sound(self, class_id: int) -> None:
        if self.player is None:
            return
        vol = float(self.volume_input.value())
        boost = PER_CLASS_VOLUME_BOOST.get(class_id, 1.0)
        self.player.play(class_id, min(1.0, vol * boost))

    # ── 脚部踩踏检测 ──

    def on_foot_detection_toggled(self, enabled: bool) -> None:
        if enabled:
            self.start_foot_worker()
        else:
            self.stop_foot_worker()

    def start_foot_worker(self) -> None:
        if self.foot_worker is not None:
            self.stop_foot_worker()

        model_path = Path(str(FOOT_MODEL_PATH)).expanduser().resolve()
        if not model_path.exists():
            self.foot_status_label.setText(f"脚部检测：模型文件不存在 {model_path}")
            self.foot_enable_checkbox.setChecked(False)
            return

        self.foot_worker = FootDetectWorker(
            model_path=model_path,
            osc_port=FOOT_OSC_PORT,
            prob_high=float(self.foot_prob_high_input.value()),
            prob_low=float(self.foot_prob_low_input.value()),
            cooldown_ms=int(self.foot_cooldown_input.value()),
        )
        self.foot_worker.foot_hit_detected.connect(self.on_foot_hit)
        self.foot_worker.foot_status_updated.connect(self.on_foot_status)
        self.foot_worker.foot_error.connect(self.on_foot_error)
        self.foot_worker.start()
        self.foot_status_label.setText("脚部检测：启动中，等待 OSC 数据...")

    def stop_foot_worker(self) -> None:
        if self.foot_worker is not None:
            self.foot_worker.stop()
            self.foot_worker = None
        self.foot_status_label.setText("脚部检测：未启用")

    def on_foot_hit(self, prob: float) -> None:
        self.play_gesture_sound(FOOT_CLASS_ID)

    def on_foot_status(self, status: str) -> None:
        self.foot_status_label.setText(f"脚部检测：{status}")

    def on_foot_error(self, error: str) -> None:
        print(f"[DrumGestureWindow] 脚部检测错误: {error}", flush=True)
        self.foot_status_label.setText(f"⚠ 脚部错误：{error}")
        self.foot_enable_checkbox.setChecked(False)

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        for address in removed_addresses:
            self.states.pop(address, None)
            self.in_flight_addresses.discard(address)
        for address in changed_connection_addresses:
            ring = self.ring_info.get(address)
            connected = bool(ring.get("connected", False)) if ring is not None else False
            if not connected:
                self.states.pop(address, None)
                self.in_flight_addresses.discard(address)

    def state_for(self, address: str) -> DrumRingState:
        state = self.states.get(address)
        ring_name = self.ring_name(address)
        if state is None:
            state = DrumRingState(ring_name=ring_name)
            self.states[address] = state
        elif state.ring_name != ring_name:
            state.ring_name = ring_name
        return state

    def update_count_views(self, counts: Counter[int]) -> None:
        text = " ".join(f"{class_id}={int(counts.get(class_id, 0))}" for class_id in self.active_class_ids[1:])
        self.count_label.setText(f"累计计数：{text}")
        if self.active_result_address:
            state = self.states.get(self.active_result_address)
            if state is not None:
                self.fps_label.setText(f"结果帧率：{state.rate_tracker.fps():.1f} Hz")

    def set_runtime_status(self, text: str) -> None:
        # 相同文本的重复 setText 只会触发无意义重绘 (如 50Hz 的"滑窗推理中"), 直接跳过
        if text == getattr(self, "_last_status_text", None):
            return
        self._last_status_text = text
        self.workflow_label.setText(text)

    def ring_name(self, address: str) -> str:
        ring = self.ring_info.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        return address[-8:] if address else "Unknown"

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path.resolve())

    def empty_count_text(self) -> str:
        text = " ".join(f"{class_id}=0" for class_id in self.active_class_ids[1:])
        return f"累计计数：{text}"

    def message_timestamp(self, message: dict) -> float | None:
        for key in ("local_timestamp", "timestamp", "host_timestamp"):
            try:
                value = float(message.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
        return None

    def sample_dt(self, state: DrumRingState, timestamp: float | None) -> float:
        if timestamp is None or state.last_timestamp is None:
            return 1.0 / IMU_SAMPLE_RATE_HZ
        dt = float(timestamp - state.last_timestamp)
        if dt > 1.0:
            dt = dt / 1000.0
        if not np.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            return 1.0 / IMU_SAMPLE_RATE_HZ
        return dt

    def closeEvent(self, event) -> None:
        self.stop_worker()
        self.stop_foot_worker()
        if self.timeline_window is not None:
            self.timeline_window.close()
            self.timeline_window = None
        if self.player is not None:
            self.player.close()
            self.player = None
        self.client.stop()
        self.client.wait(1500)
        event.accept()


class ProbabilityTimelineWindow(QWidget):
    """Standalone floating window showing 5-class probability history over time."""

    def __init__(
        self,
        *,
        class_ids: tuple[int, ...],
        class_labels: dict[int, str],
        class_colors: dict[int, QColor],
    ):
        super().__init__()
        self.setWindowTitle("5 分类概率时间曲线")
        self.resize(900, 420)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._chart = ProbabilityTimelineChart(
            class_ids=class_ids,
            class_labels=class_labels,
            class_colors=class_colors,
            history_seconds=10.0,
        )
        layout.addWidget(self._chart)

    def update_probabilities(
        self,
        probabilities: list[float] | tuple[float, ...],
    ) -> None:
        self._chart.set_probabilities(probabilities)

    def clear_chart(self) -> None:
        self._chart.clear()


def create_parser() -> argparse.ArgumentParser:
    parser = create_base_parser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    return run_window(
        lambda: DrumGestureWindow(host=args.host, port=args.port, model_path=args.model_path),
        style_sheet=APP_STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
