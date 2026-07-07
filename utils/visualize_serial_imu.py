import argparse
import csv
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize


RAW_RE = re.compile(r"\braw=(-?\d+(?:\.\d+)?)")
EDGE_RE = re.compile(r"\b(FALLING|RISING)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Arduino serial events on recorded IMU waveforms."
    )
    parser.add_argument(
        "--batch",
        default="",
        help="Recording timestamp suffix, e.g. 1783340555. Defaults to latest serial batch.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    batch = args.batch or latest_serial_batch(data_dir)
    if not batch:
        raise SystemExit("No serial_<batch>.csv.enc found.")

    serial_path = serial_file_for_batch(data_dir, batch)
    imu_paths = imu_files_for_batch(data_dir, batch)
    if serial_path is None:
        raise SystemExit(f"Missing serial file for batch: {batch}")
    if not imu_paths:
        raise SystemExit(f"No IMU .npy.enc files found for batch {batch}")

    output = Path(args.output) if args.output else data_dir / f"imu_serial_overlay_{batch}.png"
    serial_rows = load_serial_rows(serial_path)
    imu_records = [(path, load_npy(path)) for path in imu_paths]
    plot_overlay(imu_records, serial_rows, output, batch)
    print(output)
    return 0


def latest_serial_batch(data_dir: Path) -> str:
    batches = []
    for path in data_dir.glob("serial_*.csv.enc"):
        name = path.name
        match = re.fullmatch(r"serial_(\d+)(?:_[A-Za-z0-9._-]+)?\.csv\.enc", name)
        if match:
            batches.append(match.group(1))
    return sorted(batches)[-1] if batches else ""


def serial_file_for_batch(data_dir: Path, batch: str) -> Path | None:
    matches = sorted(
        path
        for path in data_dir.glob(f"serial_{batch}*.csv.enc")
        if re.fullmatch(rf"serial_{re.escape(batch)}(?:_[A-Za-z0-9._-]+)?\.csv\.enc", path.name)
    )
    return matches[-1] if matches else None


def imu_files_for_batch(data_dir: Path, batch: str) -> list[Path]:
    return sorted(
        path
        for path in data_dir.glob(f"*_{batch}*.npy.enc")
        if re.fullmatch(rf".+_{re.escape(batch)}(?:_[A-Za-z0-9._-]+)?\.npy\.enc", path.name)
    )


def load_npy(path: Path) -> np.ndarray:
    materialized = materialize(path)
    try:
        return np.load(materialized)
    finally:
        materialized.unlink(missing_ok=True)


def load_serial_rows(path: Path) -> list[dict[str, float | str]]:
    materialized = materialize(path)
    try:
        text = materialized.read_text(encoding="utf-8")
    finally:
        materialized.unlink(missing_ok=True)

    rows: list[dict[str, float | str]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        line = str(row.get("line", ""))
        elapsed = float(row.get("elapsed_s", 0.0) or 0.0)
        raw_value = parse_serial_value(line)
        edge_match = EDGE_RE.search(line)
        rows.append(
            {
                "elapsed": elapsed,
                "line": line,
                "value": np.nan if raw_value is None else raw_value,
                "edge": edge_match.group(1) if edge_match else "",
            }
        )
    return rows


def parse_serial_value(line: str) -> float | None:
    raw_match = RAW_RE.search(line)
    if raw_match:
        return float(raw_match.group(1))

    values: list[float] = []
    for part in line.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values[-1] if values else None


def plot_overlay(
    imu_records: list[tuple[Path, np.ndarray]],
    serial_rows: list[dict[str, float | str]],
    output: Path,
    batch: str,
) -> None:
    n_rows = len(imu_records) + 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, max(4, 3.1 * n_rows)),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    event_times = [float(row["elapsed"]) for row in serial_rows]
    edge_rows = [row for row in serial_rows if row["edge"]]

    max_time = max(event_times, default=0.0)
    for ax, (path, data) in zip(axes[:-1], imu_records):
        if data.ndim != 2 or data.shape[1] < 8:
            ax.set_title(f"{path.name} unsupported shape={data.shape}")
            continue

        t = data[:, 1]
        if not np.isfinite(t).any() or np.nanmax(t) <= 0:
            t = data[:, 0]
        t = t - t[0]
        acc = data[:, 2:5]
        gyr = data[:, 5:8]
        acc_norm = np.linalg.norm(acc, axis=1)
        gyr_norm = np.linalg.norm(gyr, axis=1)
        max_time = max(max_time, float(t[-1]) if t.size else 0.0)

        ax.plot(t, acc_norm, label="|acc|", color="#dc2626", linewidth=1.0)
        ax.plot(t, gyr_norm, label="|gyr|", color="#2563eb", linewidth=1.0)
        draw_serial_lines(ax, edge_rows or serial_rows)
        ax.set_ylabel("norm")
        ax.set_title(path.name.removesuffix(".enc"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    ax = axes[-1]
    serial_t = np.asarray([float(row["elapsed"]) for row in serial_rows], dtype=float)
    serial_v = np.asarray([float(row["value"]) for row in serial_rows], dtype=float)
    finite = np.isfinite(serial_v)
    if finite.sum() >= 2:
        ax.plot(serial_t[finite], serial_v[finite], color="#059669", linewidth=1.0)
    draw_serial_lines(ax, edge_rows or serial_rows)
    ax.set_title("Arduino serial values and event times")
    ax.set_ylabel("serial")
    ax.set_xlabel("elapsed seconds")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, max(max_time, 1.0))

    fig.suptitle(f"IMU + Arduino serial overlay | batch {batch}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def draw_serial_lines(ax, rows: list[dict[str, float | str]]) -> None:
    for row in rows:
        elapsed = float(row["elapsed"])
        edge = str(row.get("edge", ""))
        if edge == "FALLING":
            color = "#dc2626"
            label = "F"
        elif edge == "RISING":
            color = "#059669"
            label = "R"
        else:
            color = "#64748b"
            label = ""
        ax.axvline(elapsed, color=color, linewidth=0.8, alpha=0.5)
        if label:
            top = ax.get_ylim()[1]
            ax.text(elapsed, top, label, color=color, fontsize=8, va="top", ha="left")


if __name__ == "__main__":
    raise SystemExit(main())
