"""
Lead timing analysis with drum-gesture streaming voting mechanism.

Simulates the EXACT same inference pipeline as apps/drum_gesture_app.py:
  - Window slides by --step frames
  - History queue of length = stable_window * 200 / step
  - Trigger when class count >= trigger_ratio * queue_length
  - Cooldown after trigger

For each Arduino-annotated peak, records:
  - When the stable voting would trigger the correct gesture
  - The actual sound delay relative to the peak
  - Missed triggers and false triggers

Usage:
  python scripts/analyze_lead_timing_v2.py \
    --model checkpoint/drum_gesture/local_drum_all_w15_lead22201816_rawlin12hc_weighted_balanced_random.pth \
    --window 15 --lead-offsets 22,20,18,16 --user zhy --label 1
"""

import sys, json, base64, re, shutil, tempfile, argparse, numpy as np, torch
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from models.drum_gesture_recognizer import (
    DrumGestureRecognizer, OnlineLinearAccelerationEstimator,
    build_rawlin12_handcrafted_window, DRUM_CLASS_LABELS, HANDCRAFTED_STATS,
)
import pandas as pd

# ── Decryption ──
_INDEX_PATH = ROOT / "apps" / "shared" / ".ui_index.json"

def _load_index():
    return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))

def _derive_key(seed, salt, rounds):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=rounds)
    return base64.urlsafe_b64encode(kdf.derive(seed.encode("utf-8")))

def decrypt_dataset(data_dir):
    tmp = tempfile.TemporaryDirectory(prefix="drum_lead_v2_")
    dec_dir = Path(tmp.name)
    idx = _load_index()
    phrase = "".join(chr(a ^ b) for a, b in zip(idx["left"], idx["right"]))
    stamp_raw = base64.b64decode(idx["stamp"])
    for path in sorted(data_dir.glob("*.enc")):
        blob = path.read_bytes()
        candidates = [stamp_raw]
        if stamp_raw.endswith(b"\n"):
            candidates.append(stamp_raw[:-1] + b"\r\n")
        for st in candidates:
            if blob.startswith(st):
                payload_bytes = blob[len(st):]
                break
        payload = json.loads(payload_bytes.decode("utf-8"))
        salt = base64.b64decode(payload["salt"])
        key = _derive_key(phrase, salt, int(payload["rounds"]))
        data = Fernet(key).decrypt(str(payload["blob"]).encode("ascii"))
        (dec_dir / Path(str(payload["name"]).replace(":", "_")).name).write_bytes(data)
    return dec_dir, tmp


def parse_recording_name(name):
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy|\.csv)$", name)
    return (match.groups()[0], match.groups()[1], int(match.groups()[2])) if match else None


def imu_time_seconds(imu):
    if imu.shape[1] > 1 and np.nanmax(imu[:, 1]) > 0:
        t = imu[:, 1]
    else:
        t = imu[:, 0]
    return t.astype(np.float64) - float(t[0])


def estimate_linear_acceleration(imu):
    t_arr = imu_time_seconds(imu)
    acc = imu[:, 2:5].astype(np.float32)
    gyro = imu[:, 5:8].astype(np.float32)
    estimator = OnlineLinearAccelerationEstimator()
    rows = np.zeros((imu.shape[0], 6), dtype=np.float32)
    last_t = float(t_arr[0])
    for idx in range(imu.shape[0]):
        dt = float(t_arr[idx]) - last_t if idx else 1.0 / 200.0
        la, _ = estimator.update(acc[idx], gyro[idx], dt)
        rows[idx, :3] = la
        rows[idx, 3:] = gyro[idx]
        last_t = float(t_arr[idx])
    return rows


def serial_event_times(serial_path, edge="FALLING"):
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def align_peaks(linear_imu, imu_t, event_times, search_each_side=30):
    peaks = []
    for et in event_times:
        ei = int(np.argmin(np.abs(imu_t - et)))
        s, e = ei - search_each_side, ei + search_each_side + 1
        if s >= 0 and e <= linear_imu.shape[0]:
            peaks.append(s + int(np.argmax(np.linalg.norm(linear_imu[s:e, :3], axis=1))))
    return peaks


# ═══════════════════════════════════════════════════════════════════
# Streaming inference simulator (matches drum_gesture_app.py)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TriggerEvent:
    """Record of a sound trigger."""
    trigger_frame: int       # window end frame when vote triggered
    trigger_time_s: float    # time in seconds
    class_id: int           # which gesture triggered
    peak_idx: int            # which peak this corresponds to (-1 if false trigger)
    delay_ms: float          # trigger_time - peak_time in ms (negative = before peak)


def simulate_streaming_inference(
    recognizer: DrumGestureRecognizer,
    raw_imu: np.ndarray,
    linear_imu: np.ndarray,
    t_rel: np.ndarray,
    peaks: list[int],
    true_label: int,
    window: int,
    step: int = 5,
    stable_window_s: float = 0.04,
    trigger_ratio: float = 0.67,
    cooldown_frames: int = 24,
) -> tuple[list[TriggerEvent], list[int], list[int]]:
    """
    Simulate the drum-gesture streaming inference pipeline.

    Returns:
      triggers: list of TriggerEvent for each sound trigger
      missed_peaks: peak indices where no correct trigger happened
      false_trigger_peaks: peak indices where wrong gesture triggered first
    """
    fps = 200.0
    queue_length = max(1, int(stable_window_s * fps / step))  # ~2

    # Build feature array: (n_frames, 21)
    raw6 = raw_imu[:, 2:8].astype(np.float32)
    lin6 = linear_imu[:, :].astype(np.float32)
    raw_linear_12_full = np.concatenate([raw6, lin6], axis=1)  # (n_frames, 12)

    model = recognizer.model
    model.eval()
    device = recognizer.device

    # ── Batch inference every `step` frames ──
    n_frames = raw_imu.shape[0]
    inference_end_frames = list(range(window - 1, n_frames, step))
    
    predictions = []  # list of (end_frame, class_id, probs)
    
    batch_windows = []
    batch_frames = []
    BATCH = 256
    
    for end_inclusive in inference_end_frames:
        start = end_inclusive - window + 1
        win_12 = raw_linear_12_full[start:end_inclusive + 1, :]  # (W, 12)
        win_21 = build_rawlin12_handcrafted_window(win_12, window)  # (W, 21)
        win = win_21.T.copy()  # (21, W)
        stats = recognizer.standardize_stats
        win[0:3, :] = (win[0:3, :] - stats["raw_acc_mean"]) / stats["raw_acc_std"]
        win[3:6, :] = (win[3:6, :] - stats["raw_gyro_mean"]) / stats["raw_gyro_std"]
        win[6:9, :] = (win[6:9, :] - stats["linear_acc_mean"]) / stats["linear_acc_std"]
        win[9:12, :] = (win[9:12, :] - stats["gyro_mean"]) / stats["gyro_std"]
        for ch, (_name, mean, std) in HANDCRAFTED_STATS.items():
            cm = stats.get(f"{_name}_mean", mean)
            cs = stats.get(f"{_name}_std", std)
            win[ch, :] = (win[ch, :] - cm) / cs
        batch_windows.append(win)
        batch_frames.append(end_inclusive)
        
        if len(batch_windows) >= BATCH:
            inputs = torch.from_numpy(np.stack(batch_windows)).to(device)
            with torch.inference_mode():
                probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
            for i, (ef, pr) in enumerate(zip(batch_frames, probs)):
                cls = int(np.argmax(pr))
                predictions.append((ef, cls, pr))
            batch_windows.clear()
            batch_frames.clear()
    
    # Flush remaining
    if batch_windows:
        inputs = torch.from_numpy(np.stack(batch_windows)).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
        for i, (ef, pr) in enumerate(zip(batch_frames, probs)):
            cls = int(np.argmax(pr))
            predictions.append((ef, cls, pr))
    
    # ── Streaming voting ──
    history: deque = deque(maxlen=queue_length)
    last_trigger_frame = -999
    triggers: list[TriggerEvent] = []
    
    # Map each peak to the trigger event
    peak_triggered: dict[int, TriggerEvent] = {}  # peak_idx → first correct trigger
    peak_wrong_first: dict[int, tuple[int, int, float]] = {}  # peak_idx → (frame, wrong_class, conf)
    
    for end_frame, class_id, probs in predictions:
        history.append((class_id, probs))
        
        if len(history) < queue_length:
            continue
        
        # ── Vote counting ──
        vote_counts = {}
        for cls_i, _ in history:
            vote_counts[cls_i] = vote_counts.get(cls_i, 0) + 1
        
        winner = max(vote_counts, key=vote_counts.get)
        winner_count = vote_counts[winner]
        
        if winner_count >= trigger_ratio * queue_length:
            # Check cooldown
            if end_frame - last_trigger_frame < cooldown_frames:
                continue
            
            # Find which peak this trigger belongs to
            # A trigger is "for peak P" if P is the nearest future peak
            # (i.e., the trigger happened before P but after the previous trigger)
            nearest_peak_idx = -1
            for pi in range(len(peaks)):
                if peaks[pi] > end_frame + cooldown_frames // 2:
                    nearest_peak_idx = pi
                    break
                elif peaks[pi] <= end_frame < peaks[pi] + cooldown_frames:
                    nearest_peak_idx = pi
                    break
            
            trigger = TriggerEvent(
                trigger_frame=end_frame,
                trigger_time_s=float(t_rel[end_frame]),
                class_id=winner,
                peak_idx=nearest_peak_idx,
                delay_ms=float(t_rel[end_frame] - t_rel[peaks[nearest_peak_idx]]) * 1000.0
                if nearest_peak_idx >= 0 and nearest_peak_idx < len(peaks) else float("nan"),
            )
            triggers.append(trigger)
            last_trigger_frame = end_frame
            
            # Record per-peak
            if nearest_peak_idx >= 0 and nearest_peak_idx < len(peaks):
                if winner == true_label:
                    if nearest_peak_idx not in peak_triggered:
                        peak_triggered[nearest_peak_idx] = trigger
                else:
                    if nearest_peak_idx not in peak_triggered:
                        peak_wrong_first[nearest_peak_idx] = (end_frame, winner, 
                            float(max(probs)))
    
    # ── Classify peaks ──
    missed_peaks = []
    false_trigger_peaks = []
    
    for pi in range(len(peaks)):
        if pi not in peak_triggered:
            if pi in peak_wrong_first:
                false_trigger_peaks.append(pi)
            else:
                missed_peaks.append(pi)
    
    return triggers, missed_peaks, false_trigger_peaks


# ═══════════════════════════════════════════════════════════════════
# Output & CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lead timing with streaming voting")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--lead-offsets", type=str, default="22,20,18,16")
    parser.add_argument("--step", type=int, default=5, help="Inference stride (frames)")
    parser.add_argument("--stable-window", type=float, default=0.04, help="Stable window seconds")
    parser.add_argument("--trigger-ratio", type=float, default=0.67)
    parser.add_argument("--cooldown", type=int, default=24, help="Cooldown frames")
    parser.add_argument("--user", default="")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--stamp", default="")
    parser.add_argument("--per-peak", action="store_true")
    args = parser.parse_args()

    lead_offsets = tuple(int(x.strip()) for x in args.lead_offsets.split(","))
    window = int(args.window)
    model_path = Path(args.model).expanduser().resolve()

    print("=" * 80, flush=True)
    print("  LEAD TIMING v2 — Streaming Voting Simulation", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print(f"  Params: W={window} lead={lead_offsets} step={args.step} "
          f"stable={args.stable_window}s ratio={args.trigger_ratio} cd={args.cooldown}", flush=True)
    print("=" * 80, flush=True)

    # ── Load model ──
    print("\nLoading model...", flush=True)
    recognizer = DrumGestureRecognizer(str(model_path), device="cpu")
    print(f"Loaded: {recognizer.input_channels}ch, {recognizer.window_size}frames", flush=True)

    # ── Decrypt & collect ──
    print("Decrypting...", flush=True)
    dec_dir, tmp = decrypt_dataset(Path(args.data_dir))
    
    recordings = {}
    for path in sorted(dec_dir.glob("*.npy")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            continue
        stamp, user, label = parsed
        recordings.setdefault(stamp, {})["imu_path"] = path
        recordings[stamp]["user"] = user
        recordings[stamp]["label"] = label
    
    for path in sorted(dec_dir.glob("serial_*.csv")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            continue
        stamp, user, label = parsed
        recordings.setdefault(stamp, {})["serial_path"] = path
    
    # Filter
    candidates = []
    for stamp, rec in recordings.items():
        if "imu_path" not in rec or "serial_path" not in rec:
            continue
        if args.stamp and args.stamp not in stamp:
            continue
        if args.user and rec["user"] != args.user:
            continue
        if args.label is not None and rec["label"] != args.label:
            continue
        rec["stamp"] = stamp
        candidates.append(rec)
    
    if not candidates:
        print("No matching recordings!", flush=True)
        return 1
    
    # Exclude switch recordings for clean analysis
    non_switch = [r for r in candidates if "switch" not in r.get("user", "").lower()]
    if non_switch:
        print(f"\nNote: excluding {len(candidates) - len(non_switch)} switch recordings for clean analysis", flush=True)
        candidates = non_switch
    
    print(f"\nAnalyzing {len(candidates)} recording(s)...\n", flush=True)

    all_delays: dict[int, list[float]] = {}  # label → [delay_ms]
    all_missed: dict[int, int] = {}           # label → count
    all_false: dict[int, int] = {}            # label → count
    all_total: dict[int, int] = {}            # label → total peaks

    for rec in candidates:
        user, label, stamp = rec["user"], rec["label"], rec["stamp"]
        print(f"  ── {user}_{label} (stamp={stamp}) ──", flush=True)
        
        raw_imu = np.load(rec["imu_path"])
        if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
            continue
        
        t_rel = imu_time_seconds(raw_imu)
        print(f"    Estimating linear acc ({raw_imu.shape[0]} frames)...", flush=True)
        linear_imu = estimate_linear_acceleration(raw_imu)
        events = serial_event_times(rec["serial_path"])
        peaks = align_peaks(linear_imu, t_rel, events)
        
        print(f"    {len(peaks)} peaks, simulating streaming...", flush=True)
        
        triggers, missed, false_trig = simulate_streaming_inference(
            recognizer, raw_imu, linear_imu, t_rel, peaks, label,
            window=window, step=args.step,
            stable_window_s=args.stable_window,
            trigger_ratio=args.trigger_ratio,
            cooldown_frames=args.cooldown,
        )
        
        # Collect delays for correctly triggered peaks
        correct_triggers = [t for t in triggers if t.class_id == label and t.peak_idx >= 0 and t.peak_idx not in missed]
        delays = [t.delay_ms for t in correct_triggers]
        
        all_delays.setdefault(label, []).extend(delays)
        all_missed[label] = all_missed.get(label, 0) + len(missed)
        all_false[label] = all_false.get(label, 0) + len(false_trig)
        all_total[label] = all_total.get(label, 0) + len(peaks)
        
        if delays:
            arr = np.array(delays)
            print(f"    Triggers: {len(correct_triggers)}/{len(peaks)} correct, "
                  f"{len(missed)} missed, {len(false_trig)} wrong", flush=True)
            print(f"    Delay: mean={arr.mean():+.1f}ms std={arr.std():.1f}ms "
                  f"median={np.median(arr):+.1f}ms "
                  f"min={arr.min():+.1f}ms max={arr.max():+.1f}ms", flush=True)
        else:
            print(f"    NO correct triggers! {len(missed)} missed, {len(false_trig)} wrong", flush=True)
        
        if args.per_peak and missed:
            print(f"    Missed peaks (index): {missed[:10]}{'...' if len(missed) > 10 else ''}", flush=True)
    
    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SUMMARY — Streaming Voting Results")
    print("=" * 80)
    print(f"  {'手势':<10} {'总Peak':>7} {'触发':>7} {'漏判':>7} {'误触':>7} "
          f"{'延迟均值':>9} {'延迟中位':>9} {'P10':>8} {'P90':>8} | 判定")
    print("  " + "-" * 85)
    
    for label in sorted(all_total.keys()):
        name = DRUM_CLASS_LABELS.get(label, str(label))
        total = all_total[label]
        missed = all_missed.get(label, 0)
        false_trig = all_false.get(label, 0)
        triggered = total - missed
        delays = all_delays.get(label, [])
        
        if delays:
            arr = np.array(delays)
            min_lead = min(lead_offsets)
            max_lead = max(lead_offsets)
            # In streaming, delay = trigger_time - peak_time (negative = before peak)
            # Training window ends at peak-lead → natural_lead ~= -delay/5
            natural_lead_frames = -np.array(delays) / 5.0
            mean_lead = float(natural_lead_frames.mean())
            
            if mean_lead > max_lead:
                verdict = "Lead偏小（检测在窗口外）"
            elif mean_lead < min_lead:
                verdict = "Lead偏大（检测很晚）"
            else:
                verdict = "✓ Lead合适"
            
            print(f"  {name:<10} {total:>7} {triggered:>7} {missed:>7} {false_trig:>7} "
                  f"{arr.mean():>+9.1f} {np.median(arr):>+9.1f} "
                  f"{np.percentile(arr,10):>+8.1f} {np.percentile(arr,90):>+8.1f} | {verdict} "
                  f"(natural_lead={mean_lead:.0f}frames)")
        else:
            print(f"  {name:<10} {total:>7} {0:>7} {missed:>7} {false_trig:>7} "
                  f"{'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>8} | 无正确触发")
    
    print()
    total_all = sum(all_total.values())
    total_trig = sum(len(all_delays.get(l, [])) for l in all_total)
    total_missed = sum(all_missed.values())
    print(f"  总计: {total_all} peaks | 正确触发: {total_trig} ({total_trig/total_all*100:.1f}%) "
          f"| 漏判: {total_missed} ({total_missed/total_all*100:.1f}%)", flush=True)
    
    # ── Lead recommendation ──
    all_delays_flat = []
    for delays in all_delays.values():
        all_delays_flat.extend(delays)
    if all_delays_flat:
        arr = np.array(all_delays_flat)
        natural_leads = -arr / 5.0  # convert ms → frames
        print(f"\n  实际 natural lead: mean={natural_leads.mean():.0f} frames "
              f"({natural_leads.mean()*5:.0f}ms)", flush=True)
        print(f"  当前训练 lead: {lead_offsets}", flush=True)
        
        p25 = int(np.percentile(natural_leads, 25))
        p75 = int(np.percentile(natural_leads, 75))
        print(f"  建议 lead: ({p25+2}, {p75-2}, ...) 覆盖 P25-P75 区间", flush=True)
    
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
