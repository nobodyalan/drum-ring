from __future__ import annotations

import argparse
import json
import queue
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from whisper.whisper_with_pretrain_recognizer import WhisperWithPretrainRecognizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this single value to switch the experiment UI language: "en" or "zh".
UI_LANGUAGE = "en"

SURFACES_BY_LANGUAGE = {
    "en": ("Desk", "Palm", "Thigh", "Chest pocket", "Air"),
    "zh": ("桌面", "手心", "大腿", "上衣口袋", "空中"),
}

UI_TEXT = {
    "en": {
        "window_title": "Handwriting Experiment",
        "stream_disconnected": "Stream: disconnected",
        "stream_connected": "Stream: connected",
        "stream_error": "Stream: error ({message})",
        "status_initial": "Ready",
        "phase_wait_initial": "Waiting",
        "capture_none": "Capture: none",
        "last_result_none": "Last result: none",
        "result_title": "Result",
        "experiment": "Experiment",
        "user_id": "User ID",
        "output_dir": "Output",
        "rebuild": "New session",
        "apply_subscription": "Apply rings",
        "ring_selection": "Rings",
        "exp1_title": "Experiment 1: characters",
        "exp2_title": "Experiment 2: baseline recognition",
        "exp3_title": "Experiment 3: personalized recognition",
        "load_number_model": "Loading number model",
        "load_word_model": "Loading word model",
        "model_load_failed": "Model load failed: {error}",
        "model_ready": "Model ready",
        "loading_models": "Loading models",
        "capture_mode": "Capture mode",
        "session_ready_log": "New {title}: {count} trials",
        "subscribed": "Subscribed to IMU",
        "cancelled": "Capture cancelled",
        "retake": "Retake",
        "capture_waiting": "Capture: waiting for IMU | 0 frames | 0.00 s",
        "recording": "Recording",
        "no_imu": "No IMU samples",
        "last_capture": "Last capture: {samples} frames | {seconds:.2f} s",
        "saved_log": "{content} / {surface} -> saved {samples} frames",
        "inferring": "Recognizing",
        "target_frames": "Target: {target} | {samples} frames",
        "model_unavailable": "Model unavailable",
        "saved_inferring": "Saved, recognizing",
        "result_empty": "(empty)",
        "inference_failed": "Recognition failed",
        "result_meta": "Target: {target} | Edit: {edit_distance} | Confidence: {confidence:.3f} | Time: {elapsed_ms:.1f} ms{error_text}",
        "result_error": " | Error: {error}",
        "result_log": "{content} / {surface} -> {result} | edit={edit_distance}",
        "review_done": "Done",
        "done": "Complete",
        "all_done": "All trials complete",
        "experiment_done": "Complete",
        "current_surface": "Surface: {surface} (surface_id={surface_id})",
        "start_surface": "Start on: {surface}",
        "same_surface": "Stay on: {surface}",
        "switch_surface": "Switch surface: {previous} -> {current}",
        "capture_live": "Capture: {ring_name} | {samples} frames | {elapsed:.2f} s",
        "waiting_imu": "Waiting for IMU",
        "phase_recording": "RECORDING",
        "phase_inference": "Recognizing",
        "phase_done": "Complete",
        "phase_review": "WAITING",
        "phase_ready": "WAITING",
        "phase_ready_short": "Waiting",
        "phase_recording_short": "Recording",
        "phase_inference_short": "Recognizing",
        "phase_review_short": "Review",
        "phase_done_short": "Complete",
        "ring_online": "online",
        "ring_offline": "offline",
    },
    "zh": {
        "window_title": "手写识别实验系统",
        "stream_disconnected": "数据流状态：未连接",
        "stream_connected": "数据流状态：已连接",
        "stream_error": "数据流状态：异常（{message}）",
        "status_initial": "状态：准备启动",
        "phase_wait_initial": "等待开始：按空格开始",
        "capture_none": "当前录制：无",
        "last_result_none": "最近结果：无",
        "result_title": "识别结果",
        "experiment": "实验",
        "user_id": "用户 ID",
        "output_dir": "输出目录",
        "rebuild": "新建 / 重建实验",
        "apply_subscription": "应用订阅",
        "ring_selection": "选择订阅戒指",
        "exp1_title": "实验 1：单字母 / 单数字采集",
        "exp2_title": "实验 2：单词 / 数字串基线识别",
        "exp3_title": "实验 3：单词 / 数字串用户微调识别",
        "load_number_model": "状态：正在加载数字串模型",
        "load_word_model": "状态：正在加载单词模型",
        "model_load_failed": "模型加载失败：{error}",
        "model_ready": "状态：模型已就绪",
        "loading_models": "状态：正在加载模型",
        "capture_mode": "状态：采集模式",
        "session_ready_log": "重建 {title}，共 {count} 条",
        "subscribed": "状态：已订阅 IMU",
        "cancelled": "状态：当前录制已取消",
        "retake": "状态：已退回，等待重新书写",
        "capture_waiting": "当前录制：等待 IMU | 0 帧 | 0.00 s",
        "recording": "状态：正在录制",
        "no_imu": "状态：没有采到 IMU",
        "last_capture": "最近录制：{samples} 帧 | {seconds:.2f} s",
        "saved_log": "{content} / {surface} -> 保存 {samples} 帧",
        "inferring": "推理中",
        "target_frames": "目标：{target} | {samples} 帧",
        "model_unavailable": "模型不可用",
        "saved_inferring": "状态：已保存原始数据，正在推理",
        "result_empty": "(空)",
        "inference_failed": "推理失败",
        "result_meta": "目标：{target} | 编辑距离：{edit_distance} | 置信度：{confidence:.3f} | 推理：{elapsed_ms:.1f} ms{error_text}",
        "result_error": " | 错误：{error}",
        "result_log": "{content} / {surface} -> {result} | edit={edit_distance}",
        "review_done": "状态：推理完成",
        "done": "完成",
        "all_done": "所有内容已完成",
        "experiment_done": "状态：实验完成",
        "current_surface": "当前书写平面：{surface}（surface_id={surface_id}）",
        "start_surface": "开始前请确认：现在写在【{surface}】上",
        "same_surface": "继续保持在【{surface}】上书写",
        "switch_surface": "请先切换书写平面：从【{previous}】换到【{current}】",
        "capture_live": "当前录制：{ring_name} | {samples} 帧 | {elapsed:.2f} s",
        "waiting_imu": "等待 IMU",
        "phase_recording": "正在书写 / 录制中",
        "phase_inference": "已结束书写：正在识别，请等待",
        "phase_done": "实验已完成",
        "phase_review": "等待进入下一条",
        "phase_ready": "等待开始",
        "phase_ready_short": "等待空格开始",
        "phase_recording_short": "录制中",
        "phase_inference_short": "推理中",
        "phase_review_short": "等待空格进入下一条",
        "phase_done_short": "已完成",
        "ring_online": "在线",
        "ring_offline": "离线",
    },
}

if UI_LANGUAGE not in UI_TEXT:
    raise ValueError(f"Unsupported UI_LANGUAGE: {UI_LANGUAGE!r}")

SURFACES = SURFACES_BY_LANGUAGE[UI_LANGUAGE]
RECORDED_SURFACES = SURFACES_BY_LANGUAGE["zh"]
SPEED_ID = 0

EXP1_LETTERS = tuple(string.ascii_lowercase)
EXP1_DIGITS = tuple(string.digits)
NUMBER_STRING_LENGTH = 6
NUMBER_STRINGS_PER_SURFACE = 5
WORDS_3K_PER_SURFACE = 5
WORDS_7K_PER_SURFACE = 5
MAX_WORD_LENGTH = 15

DATASET_BY_KIND = {
    "letter": "letters",
    "digit": "digits",
    "word": "words",
    "number": "numbers",
}


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title_key: str
    needs_inference: bool
    finetuned: bool = False


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("exp1", "exp1_title", needs_inference=False),
    ExperimentSpec("exp2", "exp2_title", needs_inference=True),
    ExperimentSpec("exp3", "exp3_title", needs_inference=True, finetuned=True),
)


def tr(key: str, **kwargs: Any) -> str:
    text = UI_TEXT[UI_LANGUAGE][key]
    return text.format(**kwargs) if kwargs else text


def experiment_title(spec: ExperimentSpec) -> str:
    return tr(spec.title_key)


def surface_name(surface_id: int) -> str:
    return SURFACES[int(surface_id)]


@dataclass(frozen=True)
class Trial:
    content: str
    kind: str
    surface_id: int


@dataclass
class SavedRecord:
    trial_index: int
    content: str
    kind: str
    surface_id: int
    values: np.ndarray
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceJob:
    trial_index: int
    content: str
    kind: str
    values: np.ndarray
    sample_count: int
    capture_ms: float


@dataclass(frozen=True)
class ModelSource:
    config_dir: Path
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class InferenceResult:
    trial_index: int
    content: str
    kind: str
    prediction: str
    average_confidence: float
    elapsed_ms: float
    sample_count: int
    capture_ms: float
    was_resampled: bool
    error: str = ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def load_words(path: Path, *, limit_length: int) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if not word or len(word) > limit_length or not word.isalpha():
            continue
        words.append(word)
    if not words:
        raise ValueError(f"No usable words loaded from {path}")
    return words


def build_single_trials() -> list[Trial]:
    trials: list[Trial] = []
    for surface_id in range(len(SURFACES)):
        trials.extend(Trial(content=letter, kind="letter", surface_id=surface_id) for letter in EXP1_LETTERS)
        trials.extend(Trial(content=digit, kind="digit", surface_id=surface_id) for digit in EXP1_DIGITS)
    return trials


def build_sequence_trials(user_id: int, seed: int) -> list[Trial]:
    rng = random.Random((int(seed) * 1009) + int(user_id))
    words_3k = load_words(PROJECT_ROOT / "data" / "text" / "words" / "3kwords.txt", limit_length=MAX_WORD_LENGTH)
    words_7k = load_words(PROJECT_ROOT / "data" / "text" / "words" / "7kwords.txt", limit_length=MAX_WORD_LENGTH)

    trials: list[Trial] = []
    for surface_id in range(len(SURFACES)):
        for _ in range(NUMBER_STRINGS_PER_SURFACE):
            content = "".join(rng.choice(string.digits) for _ in range(NUMBER_STRING_LENGTH))
            trials.append(Trial(content=content, kind="number", surface_id=surface_id))
        for word in rng.sample(words_3k, k=min(WORDS_3K_PER_SURFACE, len(words_3k))):
            trials.append(Trial(content=word, kind="word", surface_id=surface_id))
        for word in rng.sample(words_7k, k=min(WORDS_7K_PER_SURFACE, len(words_7k))):
            trials.append(Trial(content=word, kind="word", surface_id=surface_id))
    return trials


class ExperimentDataStore:
    def __init__(
        self,
        *,
        root_dir: Path,
        user_id: int,
        experiment_key: str,
        session_id: str,
        dataset_names: tuple[str, ...],
    ):
        self.root_dir = Path(root_dir).resolve()
        self.user_id = int(user_id)
        self.experiment_key = experiment_key
        self.session_id = session_id
        self.records_by_dataset: dict[str, list[SavedRecord]] = {name: [] for name in dataset_names}

    @property
    def experiment_dir(self) -> Path:
        return self.root_dir / f"person_{self.user_id}" / self.session_id / self.experiment_key

    def add_record(self, record: SavedRecord) -> None:
        dataset_name = DATASET_BY_KIND[record.kind]
        if dataset_name not in self.records_by_dataset:
            self.records_by_dataset[dataset_name] = []
        bucket = self.records_by_dataset[dataset_name]
        bucket[:] = [item for item in bucket if item.trial_index != record.trial_index]
        bucket.append(record)
        bucket.sort(key=lambda item: item.trial_index)
        self.save()

    def update_result(self, trial_index: int, result: dict[str, Any]) -> None:
        for bucket in self.records_by_dataset.values():
            for item in bucket:
                if item.trial_index == trial_index:
                    item.result = result
                    self.save()
                    return

    def delete_trial(self, trial_index: int) -> None:
        changed = False
        for bucket in self.records_by_dataset.values():
            original_len = len(bucket)
            bucket[:] = [item for item in bucket if item.trial_index != trial_index]
            changed = changed or len(bucket) != original_len
        if changed:
            self.save()

    def save(self) -> None:
        for dataset_name, records in self.records_by_dataset.items():
            dataset_dir = self.experiment_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            arrays = {str(index): record.values.astype(np.float32, copy=False) for index, record in enumerate(records)}
            np.savez_compressed(dataset_dir / "x.npz", **arrays)
            labels = [
                [record.content, self.user_id, SPEED_ID, int(record.surface_id)]
                for record in records
            ]
            (dataset_dir / "y.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results = [record.result for record in records if record.result is not None]
            results_path = dataset_dir / "results.json"
            if results:
                results_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            elif results_path.exists():
                results_path.unlink()


class ExperimentInferenceWorker(QThread):
    status_changed = pyqtSignal(str)
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, *, number_source: ModelSource, word_source: ModelSource):
        super().__init__()
        self.number_source = number_source
        self.word_source = word_source
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue()
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        self._jobs.put(job)

    def stop(self) -> None:
        self._stopping = True
        self._jobs.put(None)
        self.wait(1500)

    def run(self) -> None:
        try:
            self.status_changed.emit(tr("load_number_model"))
            number_recognizer = WhisperWithPretrainRecognizer(
                self.number_source.config_dir,
                device="cpu",
                checkpoint_path=self.number_source.checkpoint_path,
            )
            self.status_changed.emit(tr("load_word_model"))
            word_recognizer = WhisperWithPretrainRecognizer(
                self.word_source.config_dir,
                device="cpu",
                checkpoint_path=self.word_source.checkpoint_path,
            )
        except Exception as exc:
            self.fatal_error.emit(tr("model_load_failed", error=exc))
            return

        self.ready.emit()
        self.status_changed.emit(tr("model_ready"))
        while not self._stopping:
            try:
                job = self._jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is None:
                return

            recognizer = word_recognizer if job.kind == "word" else number_recognizer
            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.result_ready.emit(
                    InferenceResult(
                        trial_index=job.trial_index,
                        content=job.content,
                        kind=job.kind,
                        prediction=prediction.text,
                        average_confidence=prediction.average_confidence,
                        elapsed_ms=elapsed_ms,
                        sample_count=job.sample_count,
                        capture_ms=job.capture_ms,
                        was_resampled=prediction.was_resampled,
                    )
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.result_ready.emit(
                    InferenceResult(
                        trial_index=job.trial_index,
                        content=job.content,
                        kind=job.kind,
                        prediction="",
                        average_confidence=0.0,
                        elapsed_ms=elapsed_ms,
                        sample_count=job.sample_count,
                        capture_ms=job.capture_ms,
                        was_resampled=False,
                        error=str(exc),
                    )
                )


class HandwritingExperimentWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int, *, output_dir: Path, seed: int):
        super().__init__(host=host, port=port)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.session_id = time.strftime("session_%Y%m%d_%H%M%S")

        self.trials: list[Trial] = []
        self.current_index = 0
        self.phase = "ready"
        self.current_samples: list[list[float]] = []
        self.recording_started_at = 0.0
        self.recording_address = ""
        self.worker: ExperimentInferenceWorker | None = None
        self.runtime_ready = False
        self.runtime_failed = False
        self.data_store: ExperimentDataStore | None = None

        self.setWindowTitle(tr("window_title"))
        self.resize(1180, 780)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.stream_status_label = QLabel(tr("stream_disconnected"))
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel(tr("status_initial"))
        self.workflow_label.setWordWrap(True)
        self.phase_dot = QFrame()
        self.phase_dot.setFixedSize(24, 24)
        self.phase_state_label = QLabel(tr("phase_wait_initial"))
        self.phase_state_label.setWordWrap(True)
        self.phase_state_label.setStyleSheet("font-size: 24px; font-weight: 900; color: #233142;")
        self.prompt_label = QLabel("--")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.prompt_label.setStyleSheet("font-size: 64px; font-weight: 800; color: #2f3e52;")
        self.surface_label = QLabel("--")
        self.surface_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.surface_label.setStyleSheet("font-size: 36px; font-weight: 900; color: #233142;")
        self.surface_change_label = QLabel("--")
        self.surface_change_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.surface_change_label.setWordWrap(True)
        self.surface_change_label.setStyleSheet(
            "font-size: 30px; font-weight: 900; color: #7a1f1f; "
            "background-color: #fff0d6; border: 3px solid #f08c00; "
            "border-radius: 8px; padding: 18px;"
        )
        self.progress_label = QLabel("--")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.capture_label = QLabel(tr("capture_none"))
        self.result_label = QLabel("--")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result_label.setStyleSheet("font-size: 42px; font-weight: 800; color: #2f3e52;")
        self.result_meta_label = QLabel(tr("last_result_none"))
        self.result_meta_label.setWordWrap(True)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.document().setMaximumBlockCount(240)

        self.experiment_combo = QComboBox()
        for spec in EXPERIMENTS:
            self.experiment_combo.addItem(experiment_title(spec), spec.key)
        self.user_id_spin = QSpinBox()
        self.user_id_spin.setRange(0, 10000)
        self.user_id_spin.setValue(0)
        self.output_dir_edit = QLineEdit(str(self.output_dir))

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(200)
        self.capture_timer.timeout.connect(self.refresh_capture_status)

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()
        self.rebuild_experiment()
        self.capture_timer.start()
        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        left_panel = QFrame()
        left_panel.setProperty("class", "panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        title = QLabel(tr("window_title"))
        title.setProperty("class", "title")
        left_layout.addWidget(title)
        left_layout.addWidget(self.stream_status_label)
        left_layout.addWidget(self.workflow_label)

        settings = QGridLayout()
        settings.addWidget(QLabel(tr("experiment")), 0, 0)
        settings.addWidget(self.experiment_combo, 0, 1)
        settings.addWidget(QLabel(tr("user_id")), 1, 0)
        settings.addWidget(self.user_id_spin, 1, 1)
        settings.addWidget(QLabel(tr("output_dir")), 2, 0)
        settings.addWidget(self.output_dir_edit, 2, 1)
        left_layout.addLayout(settings)

        actions = QHBoxLayout()
        self.rebuild_btn = QPushButton(tr("rebuild"))
        self.rebuild_btn.setProperty("class", "btn-primary")
        self.apply_btn = QPushButton(tr("apply_subscription"))
        self.apply_btn.setProperty("class", "btn-primary")
        actions.addWidget(self.rebuild_btn)
        actions.addWidget(self.apply_btn)
        actions.addStretch()
        left_layout.addLayout(actions)

        self.add_ring_selection_panel(left_layout, title_text=tr("ring_selection"), max_height=150)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.surface_change_label)
        left_layout.addWidget(self.surface_label)
        left_layout.addWidget(self.prompt_label, 1)
        left_layout.addWidget(self.capture_label)
        phase_row = QHBoxLayout()
        phase_row.setSpacing(10)
        phase_row.addWidget(self.phase_dot)
        phase_row.addWidget(self.phase_state_label, 1)
        left_layout.addLayout(phase_row)

        right_panel = QFrame()
        right_panel.setProperty("class", "panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        result_title = QLabel(tr("result_title"))
        result_title.setProperty("class", "title")
        right_layout.addWidget(result_title)
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_meta_label)

        root.addWidget(left_panel, 3)
        root.addWidget(right_panel, 2)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.rebuild_btn.clicked.connect(self.rebuild_experiment)
        self.experiment_combo.currentIndexChanged.connect(self.rebuild_experiment)
        self.bind_stream_client_signals()

    def current_spec(self) -> ExperimentSpec:
        key = str(self.experiment_combo.currentData())
        return next(spec for spec in EXPERIMENTS if spec.key == key)

    def set_connection_status(self, connected: bool) -> None:
        if self.status_label is not None:
            self.status_label.setText(tr("stream_connected" if connected else "stream_disconnected"))

    def set_stream_error(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(tr("stream_error", message=message))

    def format_ring_text(self, ring: dict[str, Any]) -> str:
        ring_name = str(ring.get("name", "Unknown"))
        address = str(ring.get("address", ""))
        connected = bool(ring.get("connected", False))
        state = tr("ring_online" if connected else "ring_offline")
        return f"{ring_name} | {address} | {state}"

    def rebuild_experiment(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.session_id = time.strftime("session_%Y%m%d_%H%M%S")
        self.runtime_ready = False
        self.runtime_failed = False
        self.phase = "ready"
        self.current_index = 0
        self.current_samples.clear()
        self.recording_address = ""
        self.result_label.setText("--")
        self.result_meta_label.setText(tr("last_result_none"))

        self.output_dir = Path(self.output_dir_edit.text()).expanduser()
        user_id = int(self.user_id_spin.value())
        spec = self.current_spec()
        self.trials = build_single_trials() if spec.key == "exp1" else build_sequence_trials(user_id=user_id, seed=self.seed)
        self.data_store = ExperimentDataStore(
            root_dir=self.output_dir,
            user_id=user_id,
            experiment_key=spec.key,
            session_id=self.session_id,
            dataset_names=("letters", "digits") if spec.key == "exp1" else ("words", "numbers"),
        )
        self.data_store.save()

        if spec.needs_inference:
            number_source, word_source = self.model_sources_for(spec, user_id)
            self.worker = ExperimentInferenceWorker(number_source=number_source, word_source=word_source)
            self.worker.status_changed.connect(self.set_runtime_status)
            self.worker.ready.connect(self.on_worker_ready)
            self.worker.result_ready.connect(self.on_inference_result)
            self.worker.fatal_error.connect(self.on_worker_fatal_error)
            self.worker.start()
            self.set_runtime_status(tr("loading_models"))
        else:
            self.runtime_ready = True
            self.set_runtime_status(tr("capture_mode"))

        self.refresh_prompt()
        self.append_log(tr("session_ready_log", title=experiment_title(spec), count=len(self.trials)))
        self.setFocus()

    def model_sources_for(self, spec: ExperimentSpec, user_id: int) -> tuple[ModelSource, ModelSource]:
        number_config_dir = PROJECT_ROOT / "checkpoint" / "whisper-number-exp"
        word_config_dir = PROJECT_ROOT / "checkpoint" / "whisper-word-exp"
        if spec.finetuned:
            base = PROJECT_ROOT / "checkpoint" / "whisper-exp" / f"person_{int(user_id)}"
            return (
                ModelSource(
                    config_dir=number_config_dir,
                    checkpoint_path=base / "models" / "whisper_with_pretrain_best_number.pt",
                ),
                ModelSource(
                    config_dir=word_config_dir,
                    checkpoint_path=base / "models" / "whisper_with_pretrain_best_word.pt",
                ),
            )
        return ModelSource(config_dir=number_config_dir), ModelSource(config_dir=word_config_dir)

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])
        self.set_runtime_status(f"{tr('subscribed')} | {self.phase_text()}")
        self.setFocus()

    def handle_stream_message(self, message: dict) -> None:
        if str(message.get("type", "")) != "imu" or self.phase != "recording":
            return
        address = str(message.get("address", ""))
        if not address:
            return
        selected = set(self.selected_ring_addresses())
        if selected and address not in selected:
            return
        if self.recording_address and address != self.recording_address:
            return
        self.recording_address = address

        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return
        self.current_samples.append([float(value) for value in acc] + [float(value) for value in gyr])

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.handle_space()
            event.accept()
            return
        if key == Qt.Key.Key_Backspace:
            self.handle_backspace()
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_space(self) -> None:
        if not self.trials or self.phase == "done":
            return
        if self.phase == "recording":
            self.finish_capture()
            return
        if self.phase == "review":
            self.current_index += 1
            self.phase = "ready"
            self.result_label.setText("--")
            self.result_meta_label.setText(tr("last_result_none"))
            self.refresh_prompt()
            return
        if self.phase == "inference":
            return
        self.start_capture()

    def handle_backspace(self) -> None:
        if self.phase == "recording":
            self.phase = "ready"
            self.current_samples.clear()
            self.recording_address = ""
            self.set_runtime_status(tr("cancelled"))
            self.refresh_prompt()
            return

        completed_cursor = self.current_index + 1 if self.phase in {"review", "inference"} else self.current_index
        if completed_cursor <= 0:
            return
        new_index = completed_cursor - 1
        if self.data_store is not None:
            self.data_store.delete_trial(new_index)
        self.current_index = new_index
        self.phase = "ready"
        self.result_label.setText("--")
        self.result_meta_label.setText(tr("last_result_none"))
        self.set_runtime_status(tr("retake"))
        self.refresh_prompt()

    def start_capture(self) -> None:
        if self.current_index >= len(self.trials):
            self.phase = "done"
            self.refresh_prompt()
            return
        self.phase = "recording"
        self.current_samples.clear()
        self.recording_started_at = time.time()
        self.recording_address = ""
        self.capture_label.setText(tr("capture_waiting"))
        self.update_phase_indicator()
        self.set_runtime_status(tr("recording"))

    def finish_capture(self) -> None:
        trial = self.trials[self.current_index]
        values = np.asarray(self.current_samples, dtype=np.float32)
        capture_ms = max((time.time() - self.recording_started_at) * 1000.0, 0.0)
        sample_count = int(values.shape[0])
        self.current_samples.clear()
        self.recording_address = ""

        if sample_count <= 0:
            self.phase = "ready"
            self.set_runtime_status(tr("no_imu"))
            self.capture_label.setText(tr("capture_none"))
            self.update_phase_indicator()
            return

        record = SavedRecord(
            trial_index=self.current_index,
            content=trial.content,
            kind=trial.kind,
            surface_id=trial.surface_id,
            values=values,
        )
        if self.data_store is not None:
            self.data_store.add_record(record)

        self.capture_label.setText(tr("last_capture", samples=sample_count, seconds=capture_ms / 1000.0))
        spec = self.current_spec()
        if not spec.needs_inference:
            self.append_log(
                tr("saved_log", content=trial.content, surface=surface_name(trial.surface_id), samples=sample_count)
            )
            self.current_index += 1
            self.phase = "ready"
            self.refresh_prompt()
            return

        self.phase = "inference"
        self.result_label.setText(tr("inferring"))
        self.result_meta_label.setText(tr("target_frames", target=trial.content, samples=sample_count))
        self.update_phase_indicator()
        if self.worker is None or self.runtime_failed:
            self.on_inference_result(
                InferenceResult(
                    trial_index=self.current_index,
                    content=trial.content,
                    kind=trial.kind,
                    prediction="",
                    average_confidence=0.0,
                    elapsed_ms=0.0,
                    sample_count=sample_count,
                    capture_ms=capture_ms,
                    was_resampled=False,
                    error=tr("model_unavailable"),
                )
            )
            return
        self.worker.submit(
            InferenceJob(
                trial_index=self.current_index,
                content=trial.content,
                kind=trial.kind,
                values=values,
                sample_count=sample_count,
                capture_ms=capture_ms,
            )
        )
        self.set_runtime_status(tr("saved_inferring"))

    def on_worker_ready(self) -> None:
        self.runtime_ready = True
        self.runtime_failed = False
        self.set_runtime_status(f"{tr('model_ready')} | {self.phase_text()}")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_ready = False
        self.runtime_failed = True
        self.set_runtime_status(message)
        self.append_log(message)

    def on_inference_result(self, result: InferenceResult) -> None:
        if self.phase != "inference" or result.trial_index != self.current_index:
            return
        if result.trial_index >= len(self.trials):
            return
        trial = self.trials[result.trial_index]
        edit_distance = levenshtein(trial.content, result.prediction)
        metrics = {
            "trial_index": result.trial_index,
            "target": trial.content,
            "prediction": result.prediction,
            "edit_distance": int(edit_distance),
            "exact_match": bool(result.prediction == trial.content),
            "cer": float(edit_distance / max(len(trial.content), 1)),
            "surface_id": int(trial.surface_id),
            "surface": RECORDED_SURFACES[trial.surface_id],
            "kind": trial.kind,
            "sample_count": int(result.sample_count),
            "capture_ms": float(result.capture_ms),
            "inference_ms": float(result.elapsed_ms),
            "average_confidence": float(result.average_confidence),
            "was_resampled": bool(result.was_resampled),
            "error": result.error,
        }
        if self.data_store is not None:
            self.data_store.update_result(result.trial_index, metrics)

        self.phase = "review"
        display_text = result.prediction if result.prediction else tr("result_empty")
        if result.error:
            display_text = tr("inference_failed")
        self.result_label.setText(display_text)
        error_text = tr("result_error", error=result.error) if result.error else ""
        self.result_meta_label.setText(
            tr(
                "result_meta",
                target=trial.content,
                edit_distance=edit_distance,
                confidence=result.average_confidence,
                elapsed_ms=result.elapsed_ms,
                error_text=error_text,
            )
        )
        self.append_log(
            tr(
                "result_log",
                content=trial.content,
                surface=surface_name(trial.surface_id),
                result=display_text,
                edit_distance=edit_distance,
            )
        )
        self.set_runtime_status(tr("review_done"))
        self.refresh_prompt()

    def refresh_prompt(self) -> None:
        if self.current_index >= len(self.trials):
            self.phase = "done"
            self.prompt_label.setText(tr("done"))
            self.surface_label.setText(tr("all_done"))
            self.surface_change_label.setVisible(False)
            self.progress_label.setText(f"{len(self.trials)} / {len(self.trials)}")
            self.capture_label.setText(tr("capture_none"))
            self.update_phase_indicator()
            self.set_runtime_status(tr("experiment_done"))
            return

        trial = self.trials[self.current_index]
        self.prompt_label.setText(trial.content)
        self.surface_label.setText(
            tr("current_surface", surface=surface_name(trial.surface_id), surface_id=trial.surface_id)
        )
        self.surface_change_label.setText(self.surface_instruction_text(self.current_index))
        self.surface_change_label.setVisible(self.is_surface_boundary(self.current_index))
        self.progress_label.setText(f"{self.current_index + 1} / {len(self.trials)}")
        self.update_phase_indicator()

    def is_surface_boundary(self, trial_index: int) -> bool:
        if trial_index <= 0:
            return True
        if trial_index >= len(self.trials):
            return False
        return self.trials[trial_index].surface_id != self.trials[trial_index - 1].surface_id

    def surface_instruction_text(self, trial_index: int) -> str:
        if trial_index >= len(self.trials):
            return ""
        trial = self.trials[trial_index]
        current_surface = surface_name(trial.surface_id)
        if trial_index <= 0:
            return tr("start_surface", surface=current_surface)

        previous_surface = surface_name(self.trials[trial_index - 1].surface_id)
        if previous_surface == current_surface:
            return tr("same_surface", surface=current_surface)
        return tr("switch_surface", previous=previous_surface, current=current_surface)

    def refresh_capture_status(self) -> None:
        if self.phase != "recording":
            return
        elapsed = max(time.time() - self.recording_started_at, 0.0)
        ring_name = self.ring_name(self.recording_address) if self.recording_address else tr("waiting_imu")
        self.capture_label.setText(
            tr("capture_live", ring_name=ring_name, samples=len(self.current_samples), elapsed=elapsed)
        )

    def update_phase_indicator(self) -> None:
        if self.phase == "recording":
            color = "#1f9d55"
            text = tr("phase_recording")
        elif self.phase == "inference":
            color = "#f08c00"
            text = tr("phase_inference")
        elif self.phase == "done":
            color = "#596579"
            text = tr("phase_done")
        elif self.phase == "review":
            color = "#c92a2a"
            text = tr("phase_review")
        else:
            color = "#c92a2a"
            text = tr("phase_ready")
        self.phase_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 12px; border: 2px solid #ffffff;"
        )
        self.phase_state_label.setText(text)

    def phase_text(self) -> str:
        return {
            "ready": tr("phase_ready_short"),
            "recording": tr("phase_recording_short"),
            "inference": tr("phase_inference_short"),
            "review": tr("phase_review_short"),
            "done": tr("phase_done_short"),
        }.get(self.phase, self.phase)

    def ring_name(self, address: str) -> str:
        ring = self.ring_info.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        return address[-8:] if address else "Unknown"

    def set_runtime_status(self, text: str) -> None:
        self.workflow_label.setText(text)

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser("Handwriting recognition experiment system")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "handwriting_experiments"))
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: HandwritingExperimentWindow(
            host=args.host,
            port=args.port,
            output_dir=Path(args.output_dir),
            seed=args.seed,
        ),
        style_sheet=APP_STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
