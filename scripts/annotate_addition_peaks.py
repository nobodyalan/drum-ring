"""对 data/addition_data/ 的纯负例录制做峰值标注 (阈值 90, gap 100ms)。

峰数校验:
- 预期 200: 容忍 {192, 196, 200, 204, 208} (即 ±4 或 ±8)
- 预期 80 : 必须严格等于 80

若 90 阈值下峰数不符, 按步长 2 向两侧调整阈值 (90→92→88→94→86...),
直到找到满足校验的阈值; 找不到则标记为无解。

输出: data_simulate/addition_data/ 下生成 serial_{stamp}_{user}_0.csv.enc + 复制 IMU .enc
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from apps.shared.file_saver import materialize, stash  # noqa: E402
import annotate_peaks as ap  # noqa: E402

DATA_DIR = ROOT / "data" / "addition_data"
OUTPUT_DIR = ROOT / "data_simulate" / "addition_data"

# (文件名, 预期峰数)
TARGETS: list[tuple[str, int]] = [
    ("BCL6035F45_ED_B1_A5_E5_5F_45_1788186528_whr80switch25_0.npy.enc", 200),
    ("BCL6035F45_ED_B1_A5_E5_5F_45_1788188195_whr5rapid100_0.npy.enc", 200),
]

START_THRESHOLD = 90.0
STEP = 2.0
GAP_MS = 100.0
# 阈值调整范围: 在 90 附近按步长 2 搜索, 防止跌到噪声区(如 18)或高到丢弃真实峰(如 172)
LO, HI = 60.0, 130.0


def acceptable(count: int, expected: int) -> bool:
    if expected == 80:
        return count == 80
    if expected == 200:
        return count in (192, 196, 200, 204, 208)
    return count == expected


def threshold_order(start: float, step: float, lo: float, hi: float):
    """按与 start 的距离递增返回阈值: 90, 92, 88, 94, 86, ..."""
    yield start
    down = start - step
    up = start + step
    while down >= lo or up <= hi:
        if up <= hi:
            yield up
            up += step
        if down >= lo:
            yield down
            down -= step


def parse_name(filename: str) -> tuple[str, str, str] | None:
    m = re.fullmatch(r"(.+)_(\d{10})_([^_]+)_0\.npy\.enc", filename)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main() -> int:
    if not DATA_DIR.exists():
        print(f"[X] 目录不存在: {DATA_DIR}")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="addition_annotate_")
    decrypted = Path(tmp.name)

    print(f"输入: {DATA_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"参数: threshold 起始 {START_THRESHOLD:.0f}, 步长 {STEP:.0f}, gap {GAP_MS:.0f}ms")
    print(f"校验: 200 → {{192,196,200,204,208}}; 80 → 严格 80\n")

    passed, failed = [], []
    for filename, expected in TARGETS:
        src = DATA_DIR / filename
        name = filename[:36]
        info = parse_name(filename)
        stamp = user = "?"
        if info:
            stamp, user = info[1], info[2]
        if not src.exists():
            failed.append((filename, expected, "文件不存在"))
            print(f"[MISS] {filename}")
            continue
        try:
            dec = materialize(src, output_dir=decrypted)
            t_rel, lin_mag, _ = ap._load_linear(dec)
        except Exception as exc:
            failed.append((filename, expected, f"加载失败: {exc}"))
            print(f"[ERR ] {name} 加载失败: {exc}")
            continue

        gap_s = GAP_MS / 1000.0
        # 起点阈值 90
        peaks0, _ = ap.detect_peaks(t_rel, lin_mag, threshold=START_THRESHOLD, gap_s=gap_s)
        count0 = len(peaks0)

        chosen_thr = None
        chosen_peaks = None
        tried: list[tuple[float, int]] = []
        if acceptable(count0, expected):
            chosen_thr, chosen_peaks = START_THRESHOLD, peaks0
        else:
            for thr in threshold_order(START_THRESHOLD, STEP, LO, HI):
                peaks, _ = ap.detect_peaks(t_rel, lin_mag, threshold=thr, gap_s=gap_s)
                tried.append((thr, len(peaks)))
                if acceptable(len(peaks), expected):
                    chosen_thr, chosen_peaks = thr, peaks
                    break

        if chosen_peaks is None:
            # 记录与目标最接近的阈值供参考
            best = min(tried, key=lambda t: abs(t[1] - expected)) if tried else (START_THRESHOLD, count0)
            failed.append((filename, expected, f"无解 (90→{count0}, 最佳 {best[0]:.0f}→{best[1]})"))
            print(f"[FAIL] {name}  expect={expected}  thr90={count0}  "
                  f"范围内 [{LO:.0f},{HI:.0f}] 无符合阈值 (最佳 {best[0]:.0f}→{best[1]})")
            continue

        count = len(chosen_peaks)
        # 生成 serial + 复制 IMU
        serial_path = ap._write_serial(chosen_peaks, stamp, user, "0", OUTPUT_DIR)
        imu_dst = OUTPUT_DIR / filename
        imu_dst.write_bytes(src.read_bytes())
        span_txt = "-"
        if count >= 2:
            iv = np.diff(chosen_peaks)
            span_txt = f"{iv.mean()*1000:.1f}ms"
        passed.append((filename, expected, chosen_thr, count, serial_path.name))
        print(f"[PASS] {name}  expect={expected}  thr={chosen_thr:.0f}  n={count}  "
              f"间隔均值={span_txt}  → {serial_path.name}")

    print(f"\n{'='*100}")
    print(f"通过: {len(passed)}/{len(TARGETS)}")
    for filename, expected, thr, count, sname in passed:
        print(f"  [OK ] thr={thr:.0f} n={count} (期望{expected})  {filename}  →  {OUTPUT_DIR / sname}")
        print(f"        以及 {OUTPUT_DIR / filename}")
    print(f"\n失败: {len(failed)}")
    for filename, expected, reason in failed:
        print(f"  [XX ] 期望{expected}  {filename}  ({reason})")
    print(f"\n输出目录: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
