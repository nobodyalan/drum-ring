from __future__ import annotations

"""End-to-end local drum gesture training.

Default reproduction command:

    python scripts/train_local_drum_gestures.py \
      --run-name local_drum_4gesture_lead6_gap_raw6

Pipeline:
1. decrypt encrypted recordings from data/*.enc into a temporary directory;
2. estimate online-compatible linear acceleration;
3. align serial edge events to nearby linear-acceleration peaks;
4. sample lead6 positive windows and same-recording hard negatives;
5. split by recording group, compute train-only acc/gyro z-score;
6. train RegularizedInceptionTime and save checkpoint/eval summaries.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.callback.schedule import fit_one_cycle  # noqa: F401
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.losses import LabelSmoothingCrossEntropyFlat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tsai.data.core import Categorize, TSDataLoaders, TSDatasets
from tsai.data.transforms import TSCutOut, TSGaussianNoise, TSInputDropout
from tsai.metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize
from models.drum_gesture_recognizer import (  # noqa: E402
    OnlineLinearAccelerationEstimator,
    RegularizedInceptionTime,
)


WINDOW = 15
LEAD_END_OFFSETS = (6, 4, 2, 0)
PEAK_SEARCH_EACH_SIDE = 2 * WINDOW
EXCLUDE_BEFORE_PEAK = 30
EXCLUDE_AFTER_PEAK = 24
HARD_NEGATIVES_PER_FILE = 160
SEED = 42
LABEL_IDS = (1, 2, 3, 4)

_TEMP_DIR_HANDLES: list[tempfile.TemporaryDirectory] = []


@dataclass(frozen=True)
class Recording:
    stamp: str
    user: str
    label: int
    imu_path: Path
    serial_path: Path


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 5-class local drum gesture model from encrypted data/*.enc recordings."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--work-dir", default=str(ROOT / "data_processed_local_drum_gesture"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoint" / "drum_gesture"))
    parser.add_argument("--analysis-dir", default=str(ROOT / "analysis_outputs"))
    parser.add_argument("--run-name", default="local_drum_4gesture_lead6_gap_raw6")
    parser.add_argument("--labels", type=parse_int_tuple, default=LABEL_IDS)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--lead-end-offsets", type=parse_int_tuple, default=LEAD_END_OFFSETS)
    parser.add_argument("--peak-search-samples", type=int, default=PEAK_SEARCH_EACH_SIDE)
    parser.add_argument("--exclude-before-peak", type=int, default=EXCLUDE_BEFORE_PEAK)
    parser.add_argument("--exclude-after-peak", type=int, default=EXCLUDE_AFTER_PEAK)
    parser.add_argument("--serial-edge", choices=["FALLING", "RISING"], default="FALLING")
    parser.add_argument(
        "--feature-mode",
        choices=["lin6", "rawlin12", "rawlin12_handcrafted"],
        default="lin6",
        help=(
            "lin6 uses linear_acc xyz + gyro xyz; rawlin12 prepends raw acc xyz + raw gyro xyz; "
            "rawlin12_handcrafted appends common per-frame IMU derived channels."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gaussian-noise", type=float, default=0.02)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--cutout", type=float, default=0.05)
    parser.add_argument("--hard-negatives-per-file", type=int, default=HARD_NEGATIVES_PER_FILE)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-windows", action="store_true")
    parser.add_argument(
        "--keep-decrypted",
        action="store_true",
        help="Keep decrypted files under work-dir/decrypted for inspection.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def decrypt_dataset(data_dir: Path, work_dir: Path, keep_decrypted: bool) -> Path:
    if keep_decrypted:
        decrypted_dir = work_dir / "decrypted"
        if decrypted_dir.exists():
            shutil.rmtree(decrypted_dir)
        decrypted_dir.mkdir(parents=True)
        for path in sorted(data_dir.glob("*.enc")):
            materialize(path, output_dir=decrypted_dir)
        return decrypted_dir

    tmp = tempfile.TemporaryDirectory(prefix="drum_gesture_decrypted_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        materialize(path, output_dir=decrypted_dir)
    _TEMP_DIR_HANDLES.append(tmp)
    return decrypted_dir


def parse_recording_name(name: str) -> tuple[str, str, int] | None:
    match = re.search(r"_(\d+)_([A-Za-z0-9]+)_(\d+)(?:\.npy|\.csv)$", name)
    if not match:
        return None
    stamp, user, label = match.groups()
    return stamp, user, int(label)


def collect_recordings(
    decrypted_dir: Path,
    label_ids: tuple[int, ...],
) -> tuple[list[Recording], list[dict[str, str]]]:
    imu_by_stamp = {}
    serial_by_stamp = {}
    skipped = []

    for path in sorted(decrypted_dir.glob("*.npy")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "imu name has no configured label"})
            continue
        stamp, user, label = parsed
        if label not in label_ids:
            skipped.append({"file": path.name, "reason": f"imu label {label} not in {label_ids}"})
            continue
        imu_by_stamp[stamp] = (path, user, label)

    for path in sorted(decrypted_dir.glob("serial_*.csv")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "serial name has no configured label"})
            continue
        stamp, user, label = parsed
        if label not in label_ids:
            skipped.append({"file": path.name, "reason": f"serial label {label} not in {label_ids}"})
            continue
        serial_by_stamp[stamp] = (path, user, label)

    recordings: list[Recording] = []
    for stamp, (imu_path, imu_user, imu_label) in sorted(imu_by_stamp.items()):
        serial_item = serial_by_stamp.get(stamp)
        if serial_item is None:
            skipped.append({"file": imu_path.name, "reason": "missing serial csv"})
            continue
        serial_path, serial_user, serial_label = serial_item
        if (imu_user, imu_label) != (serial_user, serial_label):
            skipped.append(
                {
                    "file": imu_path.name,
                    "serial": serial_path.name,
                    "reason": "imu and serial label/user mismatch",
                }
            )
            continue
        recordings.append(
            Recording(
                stamp=stamp,
                user=imu_user,
                label=imu_label,
                imu_path=imu_path,
                serial_path=serial_path,
            )
        )
    return recordings, skipped


def imu_time_seconds(imu: np.ndarray) -> np.ndarray:
    if imu.shape[1] > 1 and np.nanmax(imu[:, 1]) > 0:
        t = imu[:, 1]
    else:
        t = imu[:, 0]
    return t.astype(np.float64) - float(t[0])


def estimate_linear_recording(imu: np.ndarray) -> np.ndarray:
    t = imu_time_seconds(imu)
    acc = imu[:, 2:5].astype(np.float32)
    gyro = imu[:, 5:8].astype(np.float32)
    estimator = OnlineLinearAccelerationEstimator()
    rows = np.zeros((imu.shape[0], 6), dtype=np.float32)
    last_t = float(t[0])
    for index in range(imu.shape[0]):
        current_t = float(t[index])
        dt = current_t - last_t if index else 1.0 / 200.0
        linear_acc, _gravity = estimator.update(acc[index], gyro[index], dt)
        rows[index, :3] = linear_acc
        rows[index, 3:] = gyro[index]
        last_t = current_t
    return rows


def serial_event_times(serial_path: Path, edge: str) -> np.ndarray:
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def peak_indices(linear_imu: np.ndarray, imu_t: np.ndarray, event_times: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for event_t in event_times:
        event_idx = int(np.argmin(np.abs(imu_t - event_t)))
        start = event_idx - PEAK_SEARCH_EACH_SIDE
        end = event_idx + PEAK_SEARCH_EACH_SIDE + 1
        if start < 0 or end > linear_imu.shape[0]:
            continue
        norm = np.linalg.norm(linear_imu[start:end, :3], axis=1)
        peaks.append(start + int(np.argmax(norm)))
    return peaks


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=0, prepend=values[:1]).astype(np.float32)


def handcrafted_channels(raw6: np.ndarray, linear_imu: np.ndarray) -> np.ndarray:
    raw_acc = raw6[:, :3]
    raw_gyro = raw6[:, 3:6]
    linear_acc = linear_imu[:, :3]
    gyro = linear_imu[:, 3:6]

    raw_acc_norm = _norm(raw_acc)
    linear_acc_norm = _norm(linear_acc)
    raw_gyro_norm = _norm(raw_gyro)
    gyro_norm = _norm(gyro)
    raw_acc_delta_norm = _norm(_delta(raw_acc))
    linear_acc_delta_norm = _norm(_delta(linear_acc))
    gyro_delta_norm = _norm(_delta(gyro))

    raw_linear_dot = np.sum(raw_acc * linear_acc, axis=1, keepdims=True).astype(np.float32)
    denom = np.maximum(raw_acc_norm * linear_acc_norm, 1e-6)
    raw_linear_cos = (raw_linear_dot / denom).astype(np.float32)

    return np.concatenate(
        [
            raw_acc_norm,
            linear_acc_norm,
            raw_gyro_norm,
            gyro_norm,
            raw_acc_delta_norm,
            linear_acc_delta_norm,
            gyro_delta_norm,
            raw_linear_dot,
            raw_linear_cos,
        ],
        axis=1,
    ).astype(np.float32)


def feature_recording(raw_imu: np.ndarray, linear_imu: np.ndarray, feature_mode: str) -> np.ndarray:
    raw6 = raw_imu[:, 2:8].astype(np.float32)
    if feature_mode == "lin6":
        return linear_imu.astype(np.float32)
    if feature_mode == "rawlin12":
        return np.concatenate([raw6, linear_imu], axis=1).astype(np.float32)
    if feature_mode == "rawlin12_handcrafted":
        base = np.concatenate([raw6, linear_imu], axis=1).astype(np.float32)
        return np.concatenate([base, handcrafted_channels(raw6, linear_imu)], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def positive_windows(feature_imu: np.ndarray, peaks: list[int]) -> np.ndarray:
    segments = []
    channel_count = int(feature_imu.shape[1])
    for peak in peaks:
        for lead in LEAD_END_OFFSETS:
            end_exclusive = peak - lead + 1
            start = end_exclusive - WINDOW
            if start < 0 or end_exclusive > feature_imu.shape[0]:
                continue
            segment = feature_imu[start:end_exclusive, :]
            if segment.shape == (WINDOW, channel_count):
                segments.append(segment)
    if not segments:
        return np.empty((0, channel_count, WINDOW), dtype=np.float32)
    return np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32)


def safe_negative_end_indices(n_samples: int, peaks: list[int]) -> np.ndarray:
    valid = np.ones(n_samples, dtype=bool)
    valid[: WINDOW - 1] = False
    for peak in peaks:
        forbidden_start = max(0, peak - EXCLUDE_BEFORE_PEAK)
        forbidden_end = min(n_samples - 1, peak + EXCLUDE_AFTER_PEAK + WINDOW - 1)
        valid[forbidden_start : forbidden_end + 1] = False
    return np.where(valid)[0]


def negative_windows(
    feature_imu: np.ndarray,
    peaks: list[int],
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, int]:
    channel_count = int(feature_imu.shape[1])
    end_indices = safe_negative_end_indices(feature_imu.shape[0], peaks)
    if end_indices.size == 0:
        return np.empty((0, channel_count, WINDOW), dtype=np.float32), 0
    chosen = rng.choice(end_indices, size=count, replace=end_indices.size < count)
    segments = []
    for end_inclusive in chosen:
        end_exclusive = int(end_inclusive) + 1
        start = end_exclusive - WINDOW
        segments.append(feature_imu[start:end_exclusive, :])
    return np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32), int(end_indices.size)


def build_dataset(
    recordings: list[Recording],
    work_dir: Path,
    seed: int,
    hard_negatives_per_file: int,
    serial_edge: str,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    linear_dir = work_dir / "linear_acc"
    linear_dir.mkdir(parents=True, exist_ok=True)

    x_parts = []
    y_parts = []
    groups: list[str] = []
    summary = {
        "window": WINDOW,
        "lead_end_offsets": list(LEAD_END_OFFSETS),
        "peak_search_each_side": PEAK_SEARCH_EACH_SIDE,
        "serial_edge": serial_edge,
        "feature_mode": feature_mode,
        "negative_exclusion": {
            "exclude_before_peak": EXCLUDE_BEFORE_PEAK,
            "exclude_after_peak": EXCLUDE_AFTER_PEAK,
            "window_overlap_extra_after": EXCLUDE_AFTER_PEAK + WINDOW - 1,
        },
        "recordings": [],
    }

    for recording in recordings:
        raw_imu = np.load(recording.imu_path)
        if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
            raise ValueError(f"Unsupported IMU shape for {recording.imu_path}: {raw_imu.shape}")

        linear_imu = estimate_linear_recording(raw_imu)
        np.save(linear_dir / f"{recording.stamp}_{recording.user}_{recording.label}_linear.npy", linear_imu)

        imu_t = imu_time_seconds(raw_imu)
        events = serial_event_times(recording.serial_path, serial_edge)
        peaks = peak_indices(linear_imu, imu_t, events)
        feature_imu = feature_recording(raw_imu, linear_imu, feature_mode)

        pos = positive_windows(feature_imu, peaks)
        if pos.shape[0] > 0:
            group = f"{recording.user}_{recording.stamp}_{recording.label}"
            x_parts.append(pos)
            y_parts.append(np.full(pos.shape[0], recording.label, dtype=np.int64))
            groups.extend([group] * pos.shape[0])

        neg, safe_candidates = negative_windows(
            feature_imu,
            peaks,
            rng,
            hard_negatives_per_file,
        )
        if neg.shape[0] > 0:
            group = f"{recording.user}_{recording.stamp}_{recording.label}"
            x_parts.append(neg)
            y_parts.append(np.zeros(neg.shape[0], dtype=np.int64))
            groups.extend([group] * neg.shape[0])

        peak_minus_event = []
        for event_t, peak in zip(events[: len(peaks)], peaks):
            peak_minus_event.append(float(imu_t[peak] - event_t))
        summary["recordings"].append(
            {
                "stamp": recording.stamp,
                "user": recording.user,
                "label": recording.label,
                "imu_file": recording.imu_path.name,
                "serial_file": recording.serial_path.name,
                "raw_samples": int(raw_imu.shape[0]),
                "serial_events": int(events.size),
                "usable_peaks": int(len(peaks)),
                "positive_windows": int(pos.shape[0]),
                "negative_windows": int(neg.shape[0]),
                "safe_negative_end_candidates": safe_candidates,
                "peak_minus_event_samples_mean": (
                    float(np.mean(np.asarray(peak_minus_event) * 200.0))
                    if peak_minus_event
                    else None
                ),
            }
        )

    if not x_parts:
        raise ValueError("No training windows were built.")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    group_arr = np.asarray(groups)
    summary["x_shape"] = list(x.shape)
    summary["label_counts"] = {
        str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))
    }
    return x, y, group_arr, summary


def split_by_group(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    group_labels = []
    for idx in range(len(unique_groups)):
        labels = y[group_indices == idx]
        positives = labels[labels > 0]
        group_labels.append(int(positives[0]) if positives.size else 0)
    group_labels = np.asarray(group_labels, dtype=np.int64)
    train_val_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        stratify=group_labels,
        random_state=seed,
    )
    train_val_labels = np.array(
        [group_labels[np.where(unique_groups == group)[0][0]] for group in train_val_groups]
    )
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=len(np.unique(train_val_labels)),
        stratify=train_val_labels,
        random_state=seed,
    )
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)
    split_info = {
        "mode": "group_stratified",
        "test_size": test_size,
        "train_groups": sorted(set(groups[train_mask].astype(str).tolist())),
        "val_groups": sorted(set(groups[val_mask].astype(str).tolist())),
        "test_groups": sorted(set(groups[test_mask].astype(str).tolist())),
    }
    return x[train_mask], y[train_mask], x[val_mask], y[val_mask], x[test_mask], y[test_mask], split_info


def standardize_channel_groups(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if feature_mode == "lin6":
        groups = {
            "linear_acc": slice(0, 3),
            "gyro": slice(3, 6),
        }
    elif feature_mode in {"rawlin12", "rawlin12_handcrafted"}:
        groups = {
            "raw_acc": slice(0, 3),
            "raw_gyro": slice(3, 6),
            "linear_acc": slice(6, 9),
            "gyro": slice(9, 12),
        }
        if feature_mode == "rawlin12_handcrafted":
            handcrafted_names = [
                "raw_acc_norm",
                "linear_acc_norm",
                "raw_gyro_norm",
                "gyro_norm",
                "raw_acc_delta_norm",
                "linear_acc_delta_norm",
                "gyro_delta_norm",
                "raw_linear_dot",
                "raw_linear_cos",
            ]
            for offset, name in enumerate(handcrafted_names, start=12):
                groups[name] = slice(offset, offset + 1)
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    stats: dict[str, float | str] = {"mode": "channel_groups", "feature_mode": feature_mode}
    for name, channel_slice in groups.items():
        stats[f"{name}_mean"] = float(x_train[:, channel_slice, :].mean())
        stats[f"{name}_std"] = float(max(x_train[:, channel_slice, :].std(), 1e-6))

    def apply(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for name, channel_slice in groups.items():
            mean = float(stats[f"{name}_mean"])
            std = float(stats[f"{name}_std"])
            out[:, channel_slice, :] = (out[:, channel_slice, :] - mean) / std
        return out.astype(np.float32)

    return apply(x_train), apply(x_val), apply(x_test), stats


@torch.inference_mode()
def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    logits = model(torch.from_numpy(x).float())
    pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
    labels = sorted(np.unique(y).tolist())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
    }


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    run_name: str,
    epochs: int,
    seed: int,
    checkpoint_dir: Path,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    gaussian_noise: float,
    input_dropout: float,
    cutout: float,
    test_size: float,
    feature_mode: str,
) -> dict:
    set_seed(seed)
    x_train, y_train, x_val, y_val, x_test, y_test, split_info = split_by_group(
        x,
        y,
        groups,
        seed,
        test_size,
    )
    x_train, x_val, x_test, stats = standardize_channel_groups(
        x_train,
        x_val,
        x_test,
        feature_mode,
    )

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    train_idx = list(range(len(y_train)))
    valid_idx = list(range(len(y_train), len(y_all)))
    dsets = TSDatasets(
        x_all,
        y_all,
        tfms=[None, [Categorize()]],
        splits=(train_idx, valid_idx),
        inplace=True,
    )
    batch_tfms = []
    if gaussian_noise > 0:
        batch_tfms.append(TSGaussianNoise(gaussian_noise))
    if input_dropout > 0:
        batch_tfms.append(TSInputDropout(input_dropout))
    if cutout > 0:
        batch_tfms.append(TSCutOut(cutout))
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        bs=batch_size,
        batch_tfms=batch_tfms,
        num_workers=0,
        device=torch.device("cpu"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        fastai_model_dir = checkpoint_dir.relative_to(ROOT)
    except ValueError:
        fastai_model_dir = checkpoint_dir
    model = RegularizedInceptionTime(c_in=dls.vars, c_out=dls.c, head="gap").cpu()
    learn = Learner(
        dls,
        model,
        loss_func=LabelSmoothingCrossEntropyFlat(eps=label_smoothing),
        metrics=accuracy,
        wd=weight_decay,
        path=ROOT,
        model_dir=fastai_model_dir,
    )
    best_name = f"{run_name}_best"
    learn.fit_one_cycle(
        epochs,
        lr_max=lr,
        cbs=[SaveModelCallback(monitor="valid_loss", fname=best_name)],
    )
    learn.load(best_name)

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    torch.save({"model": learn.model.state_dict()}, checkpoint_path)

    summary = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "classes": {"0": "negative", **{str(label): f"gesture_{label}" for label in LABEL_IDS}},
        "split_info": split_info,
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "train_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_train, return_counts=True))
        },
        "val_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_val, return_counts=True))
        },
        "test_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_test, return_counts=True))
        },
        "standardize_mode": "channel_groups",
        "standardize_stats": stats,
        "feature_mode": feature_mode,
        "model": "RegularizedInceptionTime",
        "model_head": "gap",
        "input_shape": [int(x_train.shape[1]), WINDOW],
        "loss": "label_smoothing",
        "label_smoothing": label_smoothing,
        "wd": weight_decay,
        "gaussian_noise": gaussian_noise,
        "input_dropout": input_dropout,
        "cutout": cutout,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "best_valid_loss": float(min(row[1] for row in learn.recorder.values)),
        "best_val_accuracy": float(max(row[2] for row in learn.recorder.values)),
        "test_metrics": evaluate(learn.model.eval(), x_test, y_test),
    }
    eval_path = checkpoint_dir / f"{run_name}_eval.json"
    eval_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    global WINDOW
    global LEAD_END_OFFSETS
    global PEAK_SEARCH_EACH_SIDE
    global EXCLUDE_BEFORE_PEAK
    global EXCLUDE_AFTER_PEAK
    global LABEL_IDS

    WINDOW = int(args.window)
    LEAD_END_OFFSETS = tuple(args.lead_end_offsets)
    PEAK_SEARCH_EACH_SIDE = int(args.peak_search_samples)
    EXCLUDE_BEFORE_PEAK = int(args.exclude_before_peak)
    EXCLUDE_AFTER_PEAK = int(args.exclude_after_peak)
    LABEL_IDS = tuple(args.labels)

    data_dir = Path(args.data_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    decrypted_dir = decrypt_dataset(data_dir, work_dir, args.keep_decrypted)
    recordings, skipped = collect_recordings(decrypted_dir, LABEL_IDS)
    if len(recordings) < 10:
        raise SystemExit(f"Too few labeled recordings found: {len(recordings)}")

    x, y, groups, dataset_summary = build_dataset(
        recordings,
        work_dir,
        seed=args.seed,
        hard_negatives_per_file=args.hard_negatives_per_file,
        serial_edge=args.serial_edge,
        feature_mode=args.feature_mode,
    )
    dataset_summary["skipped_files"] = skipped
    dataset_summary["recording_count"] = len(recordings)
    dataset_summary["hard_negatives_per_file"] = args.hard_negatives_per_file
    dataset_summary["label_ids"] = list(LABEL_IDS)

    if args.save_windows:
        windows_dir = work_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        np.save(windows_dir / f"{args.run_name}_x.npy", x)
        np.save(windows_dir / f"{args.run_name}_y.npy", y)
        np.save(windows_dir / f"{args.run_name}_groups.npy", groups)
        dataset_summary["windows"] = {
            "x": str(windows_dir / f"{args.run_name}_x.npy"),
            "y": str(windows_dir / f"{args.run_name}_y.npy"),
            "groups": str(windows_dir / f"{args.run_name}_groups.npy"),
        }

    analysis_dir.mkdir(exist_ok=True)
    dataset_path = analysis_dir / f"{args.run_name}_dataset_summary.json"
    dataset_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))

    train_summary = train_model(
        x,
        y,
        groups,
        args.run_name,
        args.epochs,
        args.seed,
        checkpoint_dir,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.label_smoothing,
        args.gaussian_noise,
        args.input_dropout,
        args.cutout,
        args.test_size,
        args.feature_mode,
    )
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
