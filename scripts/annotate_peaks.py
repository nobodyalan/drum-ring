"""
纯负例峰值检测 → 伪串口标注工具

从无 Arduino 标注的录制中, 基于 |linear_acc| 检测峰值, 生成伪 serial CSV,
使录制可以被正常切片+训练流程使用。

用法:
  python scripts/annotate_peaks.py data/xxx_whrfast1_0.npy.enc --label 1 --output data_simulate
  python scripts/annotate_peaks.py data/xxx_whrfast1_0.npy.enc --label 1 --threshold 80 --gap-ms 60

流程:
  1. 解密 .npy.enc
  2. OnlineLinearAccelerationEstimator → linear_acc
  3. |linear_acc| > threshold 峰值检测 (gap_ms 去重)
  4. 生成 serial_{stamp}_{user}_{label}.csv
  5. 解密 IMU .npy 和 serial .csv 一起输出到 output-dir
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize, stash
from models.drum_gesture_recognizer import OnlineLinearAccelerationEstimator


def detect_peaks(
    t_rel: np.ndarray,
    acc_mag: np.ndarray,
    threshold: float = 100.0,
    gap_s: float = 0.040,
) -> np.ndarray:
    """Detect peaks where |acc| > threshold, at most one per gap_s."""
    peaks: list[float] = []
    i = 0
    while i < len(t_rel):
        if acc_mag[i] > threshold:
            peaks.append(float(t_rel[i]))
            # Skip ahead
            skip_to = t_rel[i] + gap_s
            while i < len(t_rel) and t_rel[i] < skip_to:
                i += 1
        else:
            i += 1
    return np.array(peaks, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="纯负例峰值检测 → 伪串口标注")
    parser.add_argument("input", type=str, help="输入 .npy.enc 文件路径")
    parser.add_argument("--label", type=int, required=True,
                        help="手势标签 (1-4), 用于命名输出文件")
    parser.add_argument("--output", type=str, default="data_simulate",
                        help="输出目录 (默认: data_simulate)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="|linear_acc| 峰值阈值 (默认: 100)")
    parser.add_argument("--gap-ms", type=float, default=100.0,
                        help="峰值最小间隔 ms (默认: 100)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[X] 文件不存在: {input_path}")
        return 1

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 解析文件名 ──
    import re
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy)?\.enc$", input_path.name)
    if not match:
        print(f"[X] 无法解析文件名: {input_path.name}")
        print("    期望格式: ..._{10位时间戳}_{用户名}_{标签}.npy.enc")
        return 1
    stamp, user, _old_label = match.groups()

    label = args.label
    print(f"  录制: stamp={stamp} user={user} label={label}")

    # ── 解密并加载 ──
    print("  解密...")
    tmp = materialize(input_path)
    imu = np.load(tmp)
    tmp.unlink()

    if imu.ndim != 2 or imu.shape[1] < 8:
        print(f"[X] IMU shape 错误: {imu.shape}")
        return 1

    t = imu[:, 1] if np.nanmax(imu[:, 1]) > 0 else imu[:, 0]
    t = t.astype(np.float64)
    t_rel = t - t[0]
    acc = imu[:, 2:5].astype(np.float32)
    gyr = imu[:, 5:8].astype(np.float32)

    # ── 在线重力补偿 ──
    print("  重力补偿...")
    estimator = OnlineLinearAccelerationEstimator()
    linear_accs = []
    dt = 1.0 / 200.0
    for i in range(len(acc)):
        la, _ = estimator.update(acc[i], gyr[i], dt)
        linear_accs.append(la)
    linear_acc = np.array(linear_accs, dtype=np.float32)
    lin_mag = np.linalg.norm(linear_acc, axis=1)

    # ── 峰值检测 ──
    gap_s = args.gap_ms / 1000.0
    peaks = detect_peaks(t_rel, lin_mag, threshold=args.threshold, gap_s=gap_s)
    print(f"  检测到 {len(peaks)} 个峰值 (|linear_acc|>{args.threshold}, gap≥{args.gap_ms}ms)")
    print(f"  录制时长: {t_rel[-1]:.1f}s, 密度: {len(peaks)/max(t_rel[-1],0.01):.1f} 峰/秒")

    # ── 相邻峰值间隔统计 ──
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        print(f"  峰值间隔: mean={intervals.mean():.3f}s std={intervals.std():.3f}s "
              f"min={intervals.min():.3f}s max={intervals.max():.3f}s")

    if len(peaks) == 0:
        print("[X] 未检测到任何峰值, 降低 --threshold 试试")
        return 1

    # ── 保存 IMU (.npy.enc) ──
    import io
    imu_buf = io.BytesIO()
    np.save(imu_buf, imu)
    imu_name = f"BCL6035F45_{stamp}_{user}_{label}.npy.enc"
    imu_path = output_dir / imu_name
    stash(imu_path, imu_buf.getvalue(), original_name=imu_name.replace(".enc", ""))
    print(f"  IMU 已保存: {imu_path}")

    # ── 生成伪 serial CSV (.csv.enc) ──
    now_wall = time.time()
    csv_lines = ["wall_time,elapsed_s,line"]
    for i, peak_t in enumerate(peaks, 1):
        csv_lines.append(
            f"{now_wall + peak_t:.6f},"
            f"{peak_t:.6f},"
            f'"Sensor 0 FALLING count={i} raw=0 delta=0 min=0"'
        )
    csv_data = "\n".join(csv_lines) + "\n"
    serial_name = f"serial_{stamp}_{user}_{label}.csv.enc"
    serial_path = output_dir / serial_name
    stash(serial_path, csv_data.encode("utf-8"), original_name=serial_name.replace(".enc", ""))
    print(f"  Serial 已保存: {serial_path}")

    print(f"\n  输出目录: {output_dir}/")
    print(f"    {imu_name}")
    print(f"    {serial_name}")
    print(f"\n  确认无误后, 复制到 data/ 目录:")
    print(f"    copy {output_dir}\\{imu_name} data\\")
    print(f"    copy {output_dir}\\{serial_name} data\\")
    print(f"  然后重新切片即可。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
