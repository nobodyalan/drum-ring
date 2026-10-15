"""批量峰值检测质量检查 (非 switch 录制, 自适应阈值)

对所有非 switch 录制 (gesture 为单个数字) 运行峰值检测 (gap=50ms),
并通过自适应降低阈值让检测峰数匹配预期的 action_count。

判定规则:
  1. 从 --threshold 开始 (默认 100), gap=--gap-ms (默认 50ms)
  2. 实际峰数 == 预期 → 检查间隔规整性:
       通过:  span(max-min) < 85ms 且 std < 40ms
       报告:  间隔不规整 (span 或 std 超限)
  3. 实际峰数 < 预期 → 阈值每次减 --step (默认 5), 重新检测
  4. 过程中实际峰数 > 预期 → 报告 "过检 overshoot"
  5. 阈值降到 --floor 仍未达标 → 报告

用法:
  python scripts/check_peaks_batch.py --data-dir datanew/hyb
  python scripts/check_peaks_batch.py --data-dir datanew/hyb --threshold 120 --step 5 --floor 20
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize
from models.drum_gesture_recognizer import OnlineLinearAccelerationEstimator
from scripts.annotate_peaks import _has_chinese, detect_peaks


def parse_hyb_name(name: str) -> dict[str, str] | None:
    """解析 hyb IMU 文件名 → 各字段。

    格式: {MAC}_{UUID}_{hand}_{stamp}_{user}_{gesture}_{group}_{trial}_{count}.npy.enc
    """
    if not name.endswith(".npy.enc"):
        return None
    stem = name[: -len(".npy.enc")]
    parts = stem.split("_")
    if len(parts) < 9:
        return None
    return {
        "mac": parts[0],
        "uuid": parts[1],
        "hand": parts[2],
        "stamp": parts[3],
        "user": parts[4],
        "gesture": parts[5],
        "group": parts[6],
        "trial": parts[7],
        "count": parts[8],
    }


def load_and_compensate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解密 → 重力补偿 → 返回 (t_rel, |linear_acc|)。"""
    imu = np.load(path)
    if imu.ndim != 2 or imu.shape[1] < 8:
        raise ValueError(f"IMU shape 错误: {imu.shape}")

    t = imu[:, 1] if np.nanmax(imu[:, 1]) > 0 else imu[:, 0]
    t = t.astype(np.float64)
    t_rel = t - t[0]
    acc = imu[:, 2:5].astype(np.float32)
    gyr = imu[:, 5:8].astype(np.float32)

    estimator = OnlineLinearAccelerationEstimator()
    linear_accs = []
    dt = 1.0 / 200.0
    for i in range(len(acc)):
        la, _ = estimator.update(acc[i], gyr[i], dt)
        linear_accs.append(la)
    lin_mag = np.linalg.norm(np.array(linear_accs, dtype=np.float32), axis=1)
    return t_rel, lin_mag


def adaptive_check(
    t_rel: np.ndarray,
    lin_mag: np.ndarray,
    expected: int,
    gap_s: float,
    start: float,
    step: float,
    floor: float,
    max_span_ms: float,
    max_std_ms: float,
) -> dict:
    """自适应降低阈值直到峰数匹配预期, 返回判定结果。

    阈值单调降低 → 峰数单调增加, 因此 count 会依次经过
    < expected → == expected → > expected。
    """
    threshold = start
    history: list[tuple[float, int, float, float]] = []
    while threshold >= floor:
        peaks, _ = detect_peaks(t_rel, lin_mag, threshold=threshold, gap_s=gap_s)
        count = len(peaks)
        if count >= 2:
            intervals = np.diff(peaks)
            span_ms = float((intervals.max() - intervals.min()) * 1000)
            std_ms = float(intervals.std() * 1000)
        else:
            span_ms = float("nan")
            std_ms = float("nan")
        history.append((threshold, count, span_ms, std_ms))

        if count == expected:
            if std_ms < max_std_ms and span_ms < max_span_ms:
                return {
                    "status": "pass",
                    "threshold": threshold,
                    "count": count,
                    "span_ms": span_ms,
                    "std_ms": std_ms,
                    "history": history,
                }
            return {
                "status": "fail",
                "reason": "count 达标但间隔不规整",
                "threshold": threshold,
                "count": count,
                "span_ms": span_ms,
                "std_ms": std_ms,
                "history": history,
            }
        if count > expected:
            return {
                "status": "fail",
                "reason": "过检 overshoot",
                "threshold": threshold,
                "count": count,
                "span_ms": span_ms,
                "std_ms": std_ms,
                "history": history,
            }
        threshold -= step
    return {
        "status": "fail",
        "reason": f"阈值降到 floor={floor} 仍未达标",
        "threshold": threshold,
        "count": history[-1][1] if history else 0,
        "span_ms": history[-1][2] if history else float("nan"),
        "std_ms": history[-1][3] if history else float("nan"),
        "history": history,
    }


def _fmt_ms(v: float) -> str:
    return "  nan" if np.isnan(v) else f"{v:6.1f}"


def _history_str(history: list[tuple[float, int, float, float]]) -> str:
    return " ".join(f"{int(t)}→{c}" for t, c, _, _ in history)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量峰值检测质量检查 (非 switch, 自适应阈值)")
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "datanew" / "hyb"),
                        help="数据目录 (默认: datanew/hyb)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="起始 |linear_acc| 峰值阈值 (默认: 100)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="阈值每次降低的步长 (默认: 5)")
    parser.add_argument("--floor", type=float, default=20.0,
                        help="阈值下限, 低于此值停止 (默认: 20)")
    parser.add_argument("--gap-ms", type=float, default=50.0,
                        help="峰值最小间隔 ms (默认: 50)")
    parser.add_argument("--max-span-ms", type=float, default=85.0,
                        help="通过标准: 间隔 max-min 上限 ms (默认: 85)")
    parser.add_argument("--max-std-ms", type=float, default=40.0,
                        help="通过标准: 间隔标准差上限 ms (默认: 40)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"[X] 目录不存在: {data_dir}")
        return 1

    # 收集非 switch 录制
    files = []
    for p in sorted(data_dir.glob("*.npy.enc")):
        if p.name.startswith("phone_imu_"):
            continue
        if _has_chinese(p.name):
            continue
        info = parse_hyb_name(p.name)
        if info is None:
            continue
        if not info["gesture"].isdigit():  # switch 手势含 "-", 跳过
            continue
        files.append((p, info))

    print(f"  非 switch 录制: {len(files)} 个")
    print(f"  参数: gap={args.gap_ms}ms 起始阈值={args.threshold} 步长={args.step} "
          f"floor={args.floor} 标准: span<{args.max_span_ms}ms 且 std<{args.max_std_ms}ms\n")

    tmp = tempfile.TemporaryDirectory(prefix="check_peaks_")
    dec = Path(tmp.name)

    passed, failed = [], []
    for path, info in files:
        expected = int(info["count"])
        label = (f"{info['hand']:>5} {info['gesture']:>4} {info['group']:<6} "
                 f"trial={info['trial']} expect={expected}")
        try:
            dec_path = materialize(path, output_dir=dec)
            t_rel, lin_mag = load_and_compensate(dec_path)
            res = adaptive_check(
                t_rel, lin_mag, expected,
                gap_s=args.gap_ms / 1000.0,
                start=args.threshold, step=args.step, floor=args.floor,
                max_span_ms=args.max_span_ms, max_std_ms=args.max_std_ms,
            )
        except Exception as e:
            failed.append((label, f"解析失败: {e}"))
            continue

        if res["status"] == "pass":
            row = (f"thr={res['threshold']:.0f} n={res['count']} "
                   f"span={_fmt_ms(res['span_ms'])}ms std={_fmt_ms(res['std_ms'])}ms")
            passed.append((label, row))
        else:
            row = (f"{res['reason']} | n={res['count']}/{expected} "
                   f"span={_fmt_ms(res['span_ms'])}ms std={_fmt_ms(res['std_ms'])}ms "
                   f"| 轨迹: {_history_str(res['history'])}")
            failed.append((label, row))

    print(f"{'='*100}")
    print(f" 通过: {len(passed)} / {len(files)}")
    for label, row in passed:
        print(f"   [PASS] {label:<38} {row}")

    print(f"\n{'='*100}")
    print(f" 失败: {len(failed)} 个 (需人工对照)")
    for label, row in failed:
        print(f"   [FAIL] {label:<38} {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
