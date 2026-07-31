"""
离线鼓击手势 → 音频合成流水线

用法:
  # 从 .npy.enc 加密录制文件处理
  python scripts/offline_drum.py process data/xxx.npy.enc --model checkpoint/drum_gesture/xxx.pth

  # 从 JSONL 文件处理
  python scripts/offline_drum.py process recording.jsonl --model checkpoint/drum_gesture/xxx.pth

  # 仅输出触发事件 (不合成音频)
  python scripts/offline_drum.py process data/xxx.npy.enc --model checkpoint/drum_gesture/xxx.pth --dump-triggers triggers.json

流程:
  1. 加载录制数据 (.npy.enc 或 .jsonl)
  2. 逐帧运行与实时 App 完全相同的 pipeline:
     OnlineLinearAccelerationEstimator → 滑窗 → DrumGestureRecognizer.predict()
     → 稳定投票 → 冷却检查 → 触发事件
  3. 按触发时间混合音效 WAV → 输出合成音频
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import wave
from collections import deque
from pathlib import Path
from typing import Union

import numpy as np

# ── 项目路径 ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from models.drum_gesture_recognizer import (
    DRUM_CLASS_IDS,
    DRUM_CLASS_LABELS,
    DrumGestureRecognizer,
    OnlineLinearAccelerationEstimator,
)

# ── 常量 (与 apps/drum_gesture_app.py 保持一致) ─────────────────
SOUND_DIR = _PROJECT_ROOT / "assets" / "drum_gesture" / "sounds"
IMU_SAMPLE_RATE_HZ = 200.0
DEFAULT_STEP = 4
DEFAULT_STABLE_WINDOW = 0.02
DEFAULT_TRIGGER_RATIO = 0.67
DEFAULT_COOLDOWN_FRAMES = 40
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

DRUM_SOUND_FILES: dict[int, Path] = {
    1: SOUND_DIR / "gesture_1_closed_hat.wav",
    2: SOUND_DIR / "gesture_2_tom2.wav",
    3: SOUND_DIR / "gesture_3_snare.wav",
    4: SOUND_DIR / "gesture_4_tom1.wav",
    5: SOUND_DIR / "gesture_5_kick.wav",
}

PER_CLASS_VOLUME_BOOST: dict[int, float] = {
    1: 1.6,  # 踩镲
    2: 1.6,  # 通鼓2
    3: 1.6,  # 军鼓
    4: 1.6,  # 通鼓1
    5: 2.0,  # 底鼓
}


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def _load_npy_enc(filepath: Path) -> np.ndarray:
    """解密 .npy.enc 文件并返回 numpy 数组 (N, 8)。

    列顺序: [local_timestamp, host_timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    """
    from apps.shared.file_saver import materialize

    temp_path = materialize(filepath)
    try:
        data = np.load(temp_path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"期望 (N, 8) 的数组，收到: {data.shape}")
    return data


def _load_jsonl(filepath: Path) -> list[dict]:
    """加载 JSONL 文件，返回消息列表。"""
    messages: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages.append(json.loads(line))
    return messages


def _load_recording(filepath: Path) -> np.ndarray:
    """统一加载接口：根据后缀自动选择加载器。

    返回 (N, 8) 的 numpy 数组，列 = [local_ts, host_ts, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]。
    JSONL 格式会从消息中提取相同 8 列。
    """
    suffix = filepath.suffix.lower()
    if suffix == ".enc":
        return _load_npy_enc(filepath)
    if suffix == ".jsonl":
        msgs = _load_jsonl(filepath)
        rows = []
        for msg in msgs:
            acc = msg.get("acc")
            gyr = msg.get("gyr")
            if not isinstance(acc, list) or not isinstance(gyr, list):
                continue
            if len(acc) != 3 or len(gyr) != 3:
                continue
            local_ts = float(msg.get("local_timestamp", 0.0) or 0.0)
            host_ts = float(msg.get("host_timestamp", 0.0) or 0.0)
            rows.append([local_ts, host_ts, acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2]])
        return np.array(rows, dtype=np.float64)
    # 尝试作为普通 .npy 加载
    data = np.load(filepath)
    if data.ndim == 2 and data.shape[1] >= 6:
        return data
    raise ValueError(f"不支持的文件格式: {filepath.suffix}")


# ═══════════════════════════════════════════════════════════════
# 离线 Pipeline
# ═══════════════════════════════════════════════════════════════

class OfflineDrumPipeline:
    """离线鼓击手势识别流水线。

    与 DrumGestureWindow 实时流水线的逻辑完全一致：
      OnlineLinearAccelerationEstimator → 滑窗 → 模型推理 → 稳定投票 → 冷却 → 触发。
    """

    def __init__(
        self,
        recognizer: DrumGestureRecognizer,
        *,
        window_size: int | None = None,
        step: int = DEFAULT_STEP,
        stable_window: float = DEFAULT_STABLE_WINDOW,
        trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.recognizer = recognizer
        self.window_size = window_size or recognizer.window_size
        self.step = max(step, 1)
        self.stable_window = stable_window
        self.trigger_ratio = trigger_ratio
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = confidence_threshold

        # 稳定投票参数
        self.queue_length = max(1, int(round(stable_window * IMU_SAMPLE_RATE_HZ / self.step)))
        self.min_trigger_count = max(1, int(math.ceil(trigger_ratio * self.queue_length)))

    def run(self, frames: np.ndarray) -> list[tuple[int, int, float]]:
        """在录制帧上运行完整流水线。

        Args:
            frames: (N, 8) 数组，列 = [local_ts, host_ts, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]

        Returns:
            list of (class_id, sample_index, confidence) 触发事件，按时间排序
        """
        estimator = OnlineLinearAccelerationEstimator()

        # 缓冲区
        samples: deque[np.ndarray] = deque(maxlen=self.window_size + self.step)
        norms: deque[float] = deque(maxlen=self.window_size + self.step)

        # 稳定投票
        prediction_history: deque[int] = deque()

        last_trigger_sample = -100_000
        frames_since_infer = 0
        triggers: list[tuple[int, int, float]] = []

        prev_local_ts: float | None = None

        for i in range(len(frames)):
            row = frames[i]
            acc = np.asarray(row[2:5], dtype=np.float32)
            gyr = np.asarray(row[5:8], dtype=np.float32)
            local_ts = float(row[0])

            # ── dt 计算 ──
            if prev_local_ts is not None and local_ts > prev_local_ts:
                dt = local_ts - prev_local_ts
                if dt <= 0.0 or dt > 0.1:
                    dt = 1.0 / IMU_SAMPLE_RATE_HZ
            else:
                dt = 1.0 / IMU_SAMPLE_RATE_HZ
            prev_local_ts = local_ts

            # ── 重力补偿 ──
            linear_acc, _gravity = estimator.update(acc, gyr, dt)

            # ── 构建 9 通道样本 ──
            sample = np.concatenate([acc, gyr, linear_acc]).astype(np.float32)
            samples.append(sample)
            norms.append(float(np.linalg.norm(linear_acc)))

            frames_since_infer += 1

            # ── 是否提交窗口 ──
            if frames_since_infer < self.step or len(samples) < self.window_size:
                continue

            frames_since_infer = 0

            # 提取窗口 (window_size, 9)
            window = np.stack(
                [samples[-self.window_size + k] for k in range(self.window_size)],
                axis=0,
            )

            # ── 模型推理 ──
            prediction = self.recognizer.predict(window)
            class_id = int(prediction.class_id)
            confidence = float(prediction.confidence)

            # ── 稳定投票 ──
            stable_cid = class_id if confidence >= self.confidence_threshold else 0
            prediction_history.append(stable_cid)
            while len(prediction_history) > self.queue_length:
                prediction_history.popleft()

            if len(prediction_history) < self.queue_length:
                continue

            decision_class: int | None = None
            for candidate in DRUM_CLASS_IDS[1:]:  # skip class 0 (空)
                count = prediction_history.count(candidate)
                if count >= self.min_trigger_count:
                    decision_class = int(candidate)
                    break

            if decision_class is None:
                continue

            # ── 冷却检查 (使用 window_end_sample_index) ──
            window_end_idx = i
            if window_end_idx - last_trigger_sample < self.cooldown_frames:
                continue

            # ── 触发! ──
            last_trigger_sample = window_end_idx
            prediction_history.clear()
            triggers.append((decision_class, window_end_idx, confidence))

        return triggers

    def summary(self) -> str:
        return (
            f"Pipeline: window_size={self.window_size}, step={self.step}, "
            f"queue_length={self.queue_length}, min_trigger_count={self.min_trigger_count}, "
            f"cooldown={self.cooldown_frames}frames, confidence_threshold={self.confidence_threshold}"
        )


# ═══════════════════════════════════════════════════════════════
# 音频合成
# ═══════════════════════════════════════════════════════════════

def _load_wav(filepath: Path) -> tuple[np.ndarray, int, int]:
    """加载 WAV 文件。

    Returns:
        (audio_samples, sample_rate, num_channels)
        audio_samples 形状: (N,) 单声道 或 (N, C) 多声道，dtype=int16
    """
    with wave.open(str(filepath), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nf = wf.getnframes()
        raw = wf.readframes(nf)
    audio = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        audio = audio.reshape(-1, nch)
    return audio, sr, nch


def _resample_to_mono(
    audio: np.ndarray,
    src_sr: int,
    dst_sr: int,
) -> np.ndarray:
    """重采样并转为单声道 float64 波形。"""
    # 转单声道
    if audio.ndim > 1:
        mono = audio.mean(axis=1).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    if src_sr == dst_sr:
        return mono

    # 线性插值重采样
    n_src = len(mono)
    n_dst = max(1, int(n_src * dst_sr / src_sr))
    src_t = np.arange(n_src, dtype=np.float64)
    dst_t = np.linspace(0.0, float(n_src - 1), n_dst)
    return np.interp(dst_t, src_t, mono)


def synthesize_audio(
    triggers: list[tuple[int, int, float]],
    output_path: Path,
    output_sr: int = 44100,
) -> None:
    """将触发事件合成为单个 WAV 文件。

    Args:
        triggers: list of (class_id, sample_index, confidence)
        output_path: 输出 WAV 路径
        output_sr: 输出采样率 (Hz)
    """
    # ── 加载所有音效 ──
    sound_cache: dict[int, tuple[np.ndarray, int]] = {}
    #   value: (waveform_float64_mono, sample_rate)

    for cid, wav_path in DRUM_SOUND_FILES.items():
        if not wav_path.exists():
            print(f"  ⚠ 音效文件不存在: {wav_path}")
            continue
        audio, src_sr, _nch = _load_wav(wav_path)
        mono = _resample_to_mono(audio, src_sr, output_sr)
        sound_cache[cid] = (mono, output_sr)

    if not triggers:
        print("没有触发事件，输出 1 秒静音。")
        silence = np.zeros(output_sr, dtype=np.int16)
        _write_wav(output_path, silence, output_sr)
        return

    # ── 计算总时长 ──
    last_sample_idx = max(t[1] for t in triggers)
    max_trigger_s = last_sample_idx / IMU_SAMPLE_RATE_HZ

    max_sound_s = 0.0
    for _cid, (waveform, sr) in sound_cache.items():
        max_sound_s = max(max_sound_s, len(waveform) / sr)

    total_duration = max_trigger_s + max_sound_s + 0.5  # 0.5s 尾部 padding
    total_samples = int(total_duration * output_sr)

    # ── 混合 ──
    mixed = np.zeros(total_samples, dtype=np.float64)

    for class_id, trigger_sample_idx, _conf in triggers:
        trigger_time = trigger_sample_idx / IMU_SAMPLE_RATE_HZ
        if class_id not in sound_cache:
            print(f"  ⚠ class_id={class_id} 无对应音效，跳过")
            continue

        waveform, _sr = sound_cache[class_id]
        boost = PER_CLASS_VOLUME_BOOST.get(class_id, 1.0)

        start_sample = int(trigger_time * output_sr)
        available = min(len(waveform), total_samples - start_sample)
        if available <= 0:
            continue

        mixed[start_sample : start_sample + available] += waveform[:available] * boost

    # ── 归一化 ──
    peak = float(np.abs(mixed).max())
    if peak > 0:
        mixed = mixed / peak * 0.9 * 32767.0

    mixed_i16 = mixed.astype(np.int16)
    _write_wav(output_path, mixed_i16, output_sr)

    duration_s = len(mixed_i16) / output_sr
    print(f"音频已保存: {output_path}")
    print(f"  时长: {duration_s:.2f}s, 触发事件: {len(triggers)}, 输出采样率: {output_sr}Hz")


def _write_wav(filepath: Path, audio: np.ndarray, sample_rate: int) -> None:
    """写入单声道 int16 WAV 文件。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _find_default_model() -> Path:
    """查找默认模型 checkpoint。"""
    candidates = [
        _PROJECT_ROOT / "checkpoint" / "drum_gesture" / "w15-17ch-zone22-group-v4p.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def cmd_process(args: argparse.Namespace) -> None:
    """process 子命令：加载数据 → 运行流水线 → 合成音频。"""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    model_path = Path(args.model).expanduser().resolve() if args.model else _find_default_model()
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()

    # ── 1. 加载数据 ──
    t0 = time.perf_counter()
    print(f"加载数据: {input_path}")
    frames = _load_recording(input_path)
    load_s = time.perf_counter() - t0
    print(f"  帧数: {len(frames)}, 加载耗时: {load_s:.2f}s")

    if len(frames) < 10:
        print("错误: 数据帧数过少 (< 10)，无法处理")
        sys.exit(1)

    # ── 2. 加载模型 ──
    t0 = time.perf_counter()
    print(f"加载模型: {model_path}")
    # 匹配实时 App 的推理配置
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    recognizer = DrumGestureRecognizer(model_path)
    load_model_s = time.perf_counter() - t0
    print(f"  模型 window_size={recognizer.window_size}, "
          f"input_channels={recognizer.input_channels}, head={recognizer.model_head}")
    print(f"  模型加载耗时: {load_model_s:.2f}s")

    # ── 3. 运行流水线 ──
    pipeline = OfflineDrumPipeline(
        recognizer,
        window_size=args.window_size,
        step=args.step,
        stable_window=args.stable_window,
        trigger_ratio=args.trigger_ratio,
        cooldown_frames=args.cooldown,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"运行流水线: {pipeline.summary()}")

    t0 = time.perf_counter()
    triggers = pipeline.run(frames)
    pipeline_s = time.perf_counter() - t0

    duration_s = len(frames) / IMU_SAMPLE_RATE_HZ
    print(f"  录制时长: {duration_s:.2f}s, 触发事件: {len(triggers)}, "
          f"处理耗时: {pipeline_s:.2f}s ({pipeline_s / max(duration_s, 0.01):.1f}x 实时)")

    # ── 打印触发事件 ──
    if triggers:
        print(f"\n触发事件 ({len(triggers)}):")
        print(f"  {'时间':>8s}  {'帧':>6s}  {'类别':>6s}  {'标签':>6s}  {'置信度':>8s}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
        for cid, sample_idx, conf in triggers:
            label = DRUM_CLASS_LABELS.get(cid, f"class_{cid}")
            time_s = sample_idx / IMU_SAMPLE_RATE_HZ
            print(f"  {time_s:8.3f}  {sample_idx:6d}  {cid:6d}  {label:>6s}  {conf:8.4f}")
    else:
        print("\n⚠ 未检测到任何触发事件。尝试调低 --trigger-ratio 或 --confidence-threshold。")

    # ── 4. 保存触发事件 (可选) ──
    if args.dump_triggers:
        dump_path = Path(args.dump_triggers).expanduser().resolve()
        trigger_data = [
            {
                "class_id": cid,
                "label": DRUM_CLASS_LABELS.get(cid, str(cid)),
                "sample_index": int(si),
                "time_s": round(si / IMU_SAMPLE_RATE_HZ, 4),
                "confidence": round(conf, 4),
            }
            for cid, si, conf in triggers
        ]
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as fh:
            json.dump(trigger_data, fh, indent=2, ensure_ascii=False)
        print(f"\n触发事件已保存: {dump_path}")

    # ── 5. 合成音频 ──
    if not args.no_audio:
        print(f"\n合成音频 → {output_path}")
        t0 = time.perf_counter()
        synthesize_audio(triggers, output_path, output_sr=args.sample_rate)
        synth_s = time.perf_counter() - t0
        print(f"  合成耗时: {synth_s:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线鼓击手势 → 音频合成流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ── process ──
    p = sub.add_parser("process", help="处理录制数据并合成音频")
    p.add_argument("input", type=str, help="输入文件 (.npy.enc / .jsonl / .npy)")
    p.add_argument("--model", type=str, default=None,
                   help="模型 checkpoint 路径 (默认自动查找)")
    p.add_argument("--output", type=str, default="output.wav",
                   help="输出 WAV 文件路径 (默认: output.wav)")
    p.add_argument("--window-size", type=int, default=None,
                   help="滑窗大小（帧），默认从模型元数据自动检测")
    p.add_argument("--step", type=int, default=DEFAULT_STEP,
                   help=f"推理步长（帧），默认: {DEFAULT_STEP}")
    p.add_argument("--stable-window", type=float, default=DEFAULT_STABLE_WINDOW,
                   help=f"稳定投票窗口（秒），默认: {DEFAULT_STABLE_WINDOW}")
    p.add_argument("--trigger-ratio", type=float, default=DEFAULT_TRIGGER_RATIO,
                   help=f"触发比例，默认: {DEFAULT_TRIGGER_RATIO}")
    p.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN_FRAMES,
                   help=f"冷却帧数，默认: {DEFAULT_COOLDOWN_FRAMES}")
    p.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                   help=f"置信度阈值（低于此值视为空手势），默认: {DEFAULT_CONFIDENCE_THRESHOLD}")
    p.add_argument("--sample-rate", type=int, default=44100,
                   help="输出音频采样率 (Hz)，默认: 44100")
    p.add_argument("--dump-triggers", type=str, default=None,
                   help="将触发事件保存为 JSON 文件")
    p.add_argument("--no-audio", action="store_true",
                   help="仅输出触发事件，不合成音频")
    p.set_defaults(func=cmd_process)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
