"""把 label-5 相关录制从 addition_data 目录改标签后写入 data/ 顶层并重新加密。

源:
  IMU    : data/addition_data/BCL..._{stamp}_{user}_0.npy.enc
  serial : data_simulate/addition_data/serial_{stamp}_{user}_0.csv.enc
输出:
  data/{BCL...|serial}_{stamp}_{user}_{label}.{npy|csv}.enc

- whr5 / lqw5 / whr5rapid100 → label 5 (单手势 5 / 吊镲)
- whr80switch15 → label 1 (switch 1-5)
- whr80switch25 → label 2 (switch 2-5)
- whr80switch35 → label 3 (switch 3-5)
- whr80switch45 → label 4 (switch 4-5)

切片程序 collect_recordings 只扫 data/*.enc (顶层), 因此必须写入 data/ 顶层。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize, stash  # noqa: E402

OUT_DIR = ROOT / "data"
IMU_SRC_DIR = ROOT / "data" / "addition_data"
SERIAL_SRC_DIR = ROOT / "data_simulate" / "addition_data"
IMU_PREFIX = "BCL6035F45_ED_B1_A5_E5_5F_45"

# (stamp, user, new_label)
TARGETS: list[tuple[str, str, int]] = [
    ("1788185645", "whr5", 5),
    ("1788185941", "lqw5", 5),
    ("1788188195", "whr5rapid100", 5),
    ("1788186294", "whr80switch15", 1),
    ("1788186528", "whr80switch25", 2),
    ("1788186621", "whr80switch25", 2),
    ("1788186672", "whr80switch35", 3),
    ("1788186759", "whr80switch35", 3),
    ("1788186798", "whr80switch35", 3),
    ("1788186837", "whr80switch35", 3),
    ("1788186878", "whr80switch45", 4),
    ("1788186915", "whr80switch45", 4),
    ("1788186957", "whr80switch45", 4),
]


def main() -> int:
    tmp = tempfile.TemporaryDirectory(prefix="label5_relabel_")
    dec_dir = Path(tmp.name)

    for stamp, user, label in TARGETS:
        src_imu = IMU_SRC_DIR / f"{IMU_PREFIX}_{stamp}_{user}_0.npy.enc"
        src_serial = SERIAL_SRC_DIR / f"serial_{stamp}_{user}_0.csv.enc"
        new_imu_name = f"{IMU_PREFIX}_{stamp}_{user}_{label}.npy"
        new_serial_name = f"serial_{stamp}_{user}_{label}.csv"

        if not src_imu.exists():
            print(f"[skip] 缺少 IMU 源: {src_imu.name}")
            continue
        if not src_serial.exists():
            print(f"[skip] 缺少 serial 源: {src_serial.name}")
            continue

        dec = materialize(src_imu, output_dir=dec_dir)
        stash(OUT_DIR / (new_imu_name + ".enc"), dec.read_bytes(), original_name=new_imu_name)

        dec = materialize(src_serial, output_dir=dec_dir)
        stash(OUT_DIR / (new_serial_name + ".enc"), dec.read_bytes(), original_name=new_serial_name)

        print(f"[ok] {user} → label {label}  ({stamp})")

    print("\n写入 data/ 顶层的文件:")
    for stamp, user, label in TARGETS:
        print(f"  {IMU_PREFIX}_{stamp}_{user}_{label}.npy.enc")
        print(f"  serial_{stamp}_{user}_{label}.csv.enc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
