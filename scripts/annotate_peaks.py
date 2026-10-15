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
import json
import re
import sys
import tempfile
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
) -> tuple[np.ndarray, np.ndarray]:
    """检测 |acc| 越过 threshold 的 burst 内的真实局部峰值, 每个 gap_s 最多一个。

    与训练流水线 peak_indices 对齐: 对每个 burst 报告 argmax(|acc|) 的时刻,
    而不是第一个越过阈值的样本 (首过阈值会偏向手势起始沿而非峰值)。

    返回 (peak_times, peak_values)。
    """
    peak_times: list[float] = []
    peak_values: list[float] = []
    n = len(t_rel)
    i = 0
    while i < n:
        if acc_mag[i] > threshold:
            # 找到当前 burst 的结束位置 (回落至阈值以下)
            j = i
            while j < n and acc_mag[j] > threshold:
                j += 1
            # 报告 burst 内真实峰值 (argmax)
            peak_idx = i + int(np.argmax(acc_mag[i:j]))
            peak_times.append(float(t_rel[peak_idx]))
            peak_values.append(float(acc_mag[peak_idx]))
            # 从峰值起跳过 gap_s, 避免同一动作被重复检测
            skip_to = t_rel[peak_idx] + gap_s
            i = peak_idx + 1
            while i < n and t_rel[i] < skip_to:
                i += 1
        else:
            i += 1
    return (
        np.array(peak_times, dtype=np.float64),
        np.array(peak_values, dtype=np.float64),
    )


def plot_peaks(
    t_rel: np.ndarray,
    lin_mag: np.ndarray,
    raw_mag: np.ndarray,
    peaks: np.ndarray,
    peak_vals: np.ndarray,
    threshold: float,
    out_png: Path,
    show: bool = False,
    title: str | None = None,
    labels: np.ndarray | None = None,
) -> None:
    """保存/显示峰值检测预览图: |linear_acc| 波形 + 阈值线 + 检测峰标记。"""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title or f"Peak detection preview — threshold={threshold:.0f}, n_peaks={len(peaks)}")

    ax1.plot(t_rel, lin_mag, color="tab:blue", lw=0.8, label="|linear_acc|")
    ax1.plot(t_rel, raw_mag, color="tab:gray", lw=0.6, alpha=0.55, label="|raw_acc|")
    ax1.axhline(threshold, color="tab:red", ls="--", lw=1.2, label=f"threshold={threshold:.0f}")
    for p in peaks:
        ax1.axvline(p, color="tab:red", lw=0.7, alpha=0.45)
    if len(peaks):
        if labels is not None and len(labels) == len(peaks):
            unique = sorted({int(l) for l in labels})
            cmap = plt.get_cmap("tab10")
            for lbl in unique:
                idx = [i for i, l in enumerate(labels) if int(l) == lbl]
                ax1.scatter(peaks[idx], peak_vals[idx], s=30, zorder=5,
                            color=cmap(lbl % 10), label=f"gesture {lbl}")
        else:
            ax1.scatter(peaks, peak_vals, color="tab:red", s=30, zorder=5, label="detected peaks")
    ax1.set_ylabel("|acc| (m/s²)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    if len(peak_vals):
        ax2.hist(peak_vals, bins=30, color="tab:orange", alpha=0.75)
        ax2.axvline(threshold, color="tab:red", ls="--", lw=1.2)
        ax2.set_xlabel("peak |linear_acc| (m/s²)")
        ax2.set_ylabel("count")
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  峰值预览图已保存: {out_png}")
    if show:
        _size_to_screen(fig)
        fig.tight_layout()
        plt.show()
    plt.close(fig)


def _size_to_screen(fig, height_margin_px: int = 100) -> None:
    """按屏幕分辨率重设图形尺寸并最大化窗口, 让预览图铺满整个屏幕宽度。

    跨后端兼容 (TkAgg / QtAgg): 直接用图形自身的窗口取屏幕尺寸, 并在 Tk
    下先 win.update() 映射窗口再最大化, 避免 Windows 高 DPI 缩放下 fig.dpi
    被拉到 ~150 导致画布被裁剪到左侧。
    """
    mng = getattr(getattr(fig, "canvas", None), "manager", None)
    win = getattr(mng, "window", None) if mng is not None else None
    if win is None:
        return

    if hasattr(win, "winfo_screenwidth"):
        # ── TkAgg ──
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        fig.set_size_inches(sw / fig.dpi, max(sh - height_margin_px, 600) / fig.dpi)
        win.update()  # 先映射窗口, 让后续 zoom 的 Configure 事件按正确 dpi 处理
        try:
            win.geometry(f"{sw}x{sh}+0+0")
        except Exception:
            pass
        try:
            win.state("zoomed")
        except Exception:
            pass
    else:
        # ── QtAgg (QMainWindow) ──
        try:
            screen = win.screen()
            geom = screen.availableGeometry()
            sw, sh = geom.width(), geom.height()
        except Exception:
            sw, sh = 1920, 1080
        fig.set_size_inches(sw / fig.dpi, max(sh - height_margin_px, 600) / fig.dpi)
        show_max = getattr(win, "showMaximized", None)
        if callable(show_max):
            show_max()
        else:
            resize = getattr(win, "resize", None)
            if callable(resize):
                try:
                    resize(sw, sh)
                except Exception:
                    pass


def _has_chinese(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in text)


def _parse_lcy_name(name: str) -> tuple[str, str, str, str, str] | None:
    """解析 lcy IMU 文件名 → (stamp, user, gesture, group, mac)。"""
    mac = name.split("_", 1)[0]
    m = re.search(
        r"_(\d{10})_([A-Za-z0-9]+?)_([0-9]+(?:-[0-9]+)?)_([A-Za-z]+)_\d+_\d+\.npy$",
        name,
    )
    if not m:
        return None
    stamp, user, gesture, group = m.groups()
    return stamp, user, gesture, group, mac


def _lcy_hand(meta_path: Path, mac: str, filename: str) -> str:
    """手部: recording_meta rings[].hand → 文件名 → unknown。"""
    if meta_path is not None and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for ring in meta.get("rings", []):
                if str(ring.get("name", "")) == mac:
                    hand = str(ring.get("hand", ""))
                    if hand in ("left", "right"):
                        return hand
        except Exception:
            pass
    lower = filename.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return "unknown"


def batch_lcy(data_dir: Path, output_dir: Path) -> int:
    """批量处理 lcy 数据, 按人物/手部分类输出到 data_simulate/{user}/{hand}/。

    跳过 phone_imu 和中文文件名。保留原始文件名 (meta/serial/imu 按 stamp 匹配)。
    """
    if not data_dir.exists():
        print(f"[X] 目录不存在: {data_dir}")
        return 1

    tmp = tempfile.TemporaryDirectory(prefix="lcy_annotate_")
    decrypted = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        if path.name.startswith("phone_imu_"):
            continue
        if _has_chinese(path.name):
            continue
        materialize(path, output_dir=decrypted)

    meta_by_stamp: dict[str, Path] = {}
    for mp in sorted(decrypted.glob("recording_meta_*.json")):
        m = re.search(r"_(\d{10})_", mp.name)
        if m:
            meta_by_stamp[m.group(1)] = mp

    count = 0
    for npy in sorted(decrypted.glob("*.npy")):
        if npy.name.startswith("phone_imu_"):
            continue
        info = _parse_lcy_name(npy.name)
        if not info:
            continue
        stamp, user, gesture, group, mac = info
        meta_path = meta_by_stamp.get(stamp)
        hand = _lcy_hand(meta_path, mac, npy.name)

        sub = output_dir / user / hand
        sub.mkdir(parents=True, exist_ok=True)

        # IMU
        stash(sub / (npy.name + ".enc"), npy.read_bytes(), original_name=npy.name)
        # serial + meta (同 stamp)
        for p in sorted(decrypted.glob(f"serial_{stamp}_*.csv")):
            stash(sub / (p.name + ".enc"), p.read_bytes(), original_name=p.name)
        for p in sorted(decrypted.glob(f"recording_meta_{stamp}_*.json")):
            stash(sub / (p.name + ".enc"), p.read_bytes(), original_name=p.name)

        label_display = gesture if "-" in gesture else f"gesture_{gesture}"
        print(f"  [{hand:>5}] {user}  {label_display:<8} ({group})  stamp={stamp}")
        count += 1

    print(f"\n  完成: {count} 个录制 → {output_dir}/ (按 {user}/{hand} 分类)")
    return 0


def _parse_expected(name: str) -> int | None:
    """从文件名提取 action_count (预期次数)。"""
    m = re.search(r"_(\d+)\.npy$", name)
    return int(m.group(1)) if m else None


_TEMP_HANDLES: list = []


def _resolve_candidates(data_dir: Path, args) -> list[dict]:
    """解密并解析 data_dir 下所有录制, 按 user/hand/label/group 过滤。"""
    tmp = tempfile.TemporaryDirectory(prefix="annotate_sel_")
    _TEMP_HANDLES.append(tmp)
    decrypted = Path(tmp.name)

    # 解密 meta (用于读手部), 记录 stamp → (decrypted meta, 原始 meta .enc)
    meta_by_stamp: dict[str, Path] = {}
    meta_enc_by_stamp: dict[str, Path] = {}
    for path in sorted(data_dir.glob("recording_meta_*.json.enc")):
        if _has_chinese(path.name):
            continue
        dec = materialize(path, output_dir=decrypted)
        m = re.search(r"_(\d{10})_", dec.name)
        if m:
            meta_by_stamp[m.group(1)] = dec
            meta_enc_by_stamp[m.group(1)] = path

    candidates: list[dict] = []
    for path in sorted(data_dir.glob("*.npy.enc")):
        if path.name.startswith("phone_imu_"):
            continue
        if _has_chinese(path.name):
            continue
        dec = materialize(path, output_dir=decrypted)
        info = _parse_lcy_name(dec.name)
        if not info:
            continue
        stamp, user, gesture, group, mac = info
        hand = _lcy_hand(meta_by_stamp.get(stamp), mac, dec.name)
        candidates.append({
            "path": dec, "src_enc": path,
            "meta_enc": meta_enc_by_stamp.get(stamp),
            "stamp": stamp, "user": user,
            "gesture": gesture, "group": group, "hand": hand,
            "expected": _parse_expected(dec.name),
        })

    if args.user:
        candidates = [c for c in candidates if c["user"] == args.user]
    if args.hand:
        candidates = [c for c in candidates if c["hand"] == args.hand]
    if args.label is not None:
        candidates = [c for c in candidates if c["gesture"] == args.label]
    if args.group:
        candidates = [c for c in candidates if c["group"] == args.group]
    if args.single_only:
        candidates = [c for c in candidates
                      if c["gesture"].isdigit() and c["group"] != "combo"]
    return candidates


def _load_linear(imu_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解密后 IMU → 重力补偿 → (t_rel, |linear_acc|, |raw_acc|)。"""
    imu = np.load(imu_path)
    if imu.ndim != 2 or imu.shape[1] < 8:
        raise ValueError(f"IMU shape 错误: {imu.shape}")
    t = imu[:, 1] if np.nanmax(imu[:, 1]) > 0 else imu[:, 0]
    t = t.astype(np.float64)
    t_rel = t - t[0]
    acc = imu[:, 2:5].astype(np.float32)
    gyr = imu[:, 5:8].astype(np.float32)
    raw_mag = np.linalg.norm(acc, axis=1)
    estimator = OnlineLinearAccelerationEstimator()
    linear_accs = []
    dt = 1.0 / 200.0
    for i in range(len(acc)):
        la, _ = estimator.update(acc[i], gyr[i], dt)
        linear_accs.append(la)
    lin_mag = np.linalg.norm(np.array(linear_accs, dtype=np.float32), axis=1)
    return t_rel, lin_mag, raw_mag


def adaptive_detect(t_rel, lin_mag, expected, gap_s, start, step, floor):
    """自适应降阈值直到峰数匹配预期。"""
    threshold = start
    last_threshold = threshold
    history: list = []
    peaks: np.ndarray = np.array([], dtype=np.float64)
    while threshold >= floor:
        last_threshold = threshold
        peaks, _ = detect_peaks(t_rel, lin_mag, threshold=threshold, gap_s=gap_s)
        count = len(peaks)
        if count >= 2:
            iv = np.diff(peaks)
            span_ms = float((iv.max() - iv.min()) * 1000)
            std_ms = float(iv.std() * 1000)
        else:
            span_ms = float("nan")
            std_ms = float("nan")
        history.append((threshold, count, span_ms, std_ms))
        if count == expected:
            return {"status": "hit", "threshold": threshold, "peaks": peaks,
                    "count": count, "span_ms": span_ms, "std_ms": std_ms,
                    "history": history}
        if count > expected:
            return {"status": "overshoot", "threshold": threshold, "peaks": peaks,
                    "count": count, "span_ms": span_ms, "std_ms": std_ms,
                    "history": history}
        threshold -= step
    return {"status": "floor", "threshold": last_threshold, "peaks": peaks,
            "count": history[-1][1] if history else 0,
            "span_ms": history[-1][2] if history else float("nan"),
            "std_ms": history[-1][3] if history else float("nan"),
            "history": history}


def build_peak_labels(gesture: str, group: str, n_peaks: int) -> np.ndarray:
    """逐峰标签: 单手势全同; switch → slow: a-b-a-b, fast: a-b-b-a。"""
    if "-" not in gesture:
        return np.full(n_peaks, int(gesture), dtype=np.int64)
    a, b = (int(x) for x in gesture.split("-"))
    pattern = (a, b, b, a) if group == "fast" else (a, b)
    return np.array([pattern[i % len(pattern)] for i in range(n_peaks)], dtype=np.int64)


def _write_serial(peaks, stamp: str, user: str, label: str, output_dir: Path,
                  labels=None) -> Path:
    """生成伪 serial CSV (加密) 到输出目录。labels 提供逐峰标签。"""
    now_wall = time.time()
    lines = ["wall_time,elapsed_s,line"]
    for i, pt in enumerate(peaks, 1):
        tag = f"label={labels[i-1]} " if labels is not None else ""
        lines.append(f"{now_wall + pt:.6f},{pt:.6f},"
                     f"\"Sensor 0 FALLING count={i} {tag}raw=0 delta=0 min=0\"")
    csv_data = "\n".join(lines) + "\n"
    name = f"serial_{stamp}_{user}_{label}.csv.enc"
    path = output_dir / name
    stash(path, csv_data.encode("utf-8"), original_name=name.replace(".enc", ""))
    return path


def batch_annotate(args, output_dir: Path) -> int:
    """批量峰值检测+拆分: 达标 (峰数=预期 且 span<阈值 且非 combo) 自动放行生成 serial。"""
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"[X] 目录不存在: {data_dir}")
        return 1
    candidates = _resolve_candidates(data_dir, args)
    if not candidates:
        print("[X] 无匹配录制")
        return 1
    print(f"  匹配 {len(candidates)} 个录制 (含左右手, 共 {len({c['stamp'] for c in candidates})} 个 session)")
    print(f"  放行标准: 峰数=预期 且 span<{args.max_span_ms}ms 且 group≠combo")
    print(f"  自适应阈值: {args.threshold} → {args.floor} (步长 {args.step})  gap={args.gap_ms}ms\n")

    gap_s = args.gap_ms / 1000.0
    passed, failed = [], []
    for c in candidates:
        label = (f"[{c['hand']:>5}] {c['user']}  gesture={c['gesture']:<4} "
                 f"group={c['group']:<6} expect={c['expected']}")
        try:
            t_rel, lin_mag, _ = _load_linear(c["path"])
            res = adaptive_detect(t_rel, lin_mag, c["expected"], gap_s,
                                  args.threshold, args.step, args.floor)
        except Exception as e:
            failed.append((label, f"解析失败: {e}"))
            continue

        is_combo = c["group"] == "combo"
        if res["status"] == "hit" and res["span_ms"] < args.max_span_ms and not is_combo:
            sub = output_dir / c["user"] / c["hand"]
            sub.mkdir(parents=True, exist_ok=True)
            p = _write_serial(res["peaks"], c["stamp"], c["user"], c["gesture"], sub)
            # 复制原始 IMU .enc 到对应文件夹
            (sub / c["src_enc"].name).write_bytes(c["src_enc"].read_bytes())
            # 复制 meta .enc (同 stamp 只复制一次)
            if c["meta_enc"] is not None:
                dst = sub / c["meta_enc"].name
                if not dst.exists():
                    dst.write_bytes(c["meta_enc"].read_bytes())
            passed.append((label, f"thr={res['threshold']:.0f} span={res['span_ms']:.1f}ms "
                                 f"→ {c['user']}/{c['hand']}/{p.name}"))
        elif res["status"] == "hit" and is_combo:
            failed.append((label, f"combo 不放行 (span={res['span_ms']:.1f}ms)"))
        elif res["status"] == "hit":
            failed.append((label, f"count 达标但 span={res['span_ms']:.1f}ms ≥ {args.max_span_ms}ms"))
        elif res["status"] == "overshoot":
            failed.append((label, f"过检 overshoot n={res['count']}/{c['expected']}"))
        else:
            failed.append((label, f"阈值降到 {args.floor} 未达标 n={res['count']}/{c['expected']}"))

    print(f"{'='*96}")
    print(f" 放行: {len(passed)} / {len(candidates)}")
    for label, row in passed:
        print(f"   [PASS] {label:<44} {row}")
    print(f"\n{'='*96}")
    print(f" 需人工审查: {len(failed)} 个")
    for label, row in failed:
        print(f"   [REVIEW] {label:<44} {row}")
    print(f"\n  输出: {output_dir}/{{user}}/{{hand}}/ (IMU + serial + meta)")
    return 0


def view_matching(args, output_dir: Path) -> int:
    """按过滤器找匹配录制并展示峰值检测预览 (供人工审查)。"""
    data_dir = Path(args.data_dir).expanduser().resolve()
    candidates = _resolve_candidates(data_dir, args)
    if not candidates:
        print("[X] 无匹配录制 (可用 --user/--hand/--label/--group 过滤)")
        return 1
    if len(candidates) > 1:
        print(f"  {len(candidates)} 个匹配, 使用第一个:")
        for c in candidates:
            print(f"    [{c['hand']:>5}] {c['user']}  gesture={c['gesture']}  "
                  f"group={c['group']}  stamp={c['stamp']}  expect={c['expected']}")

    c = candidates[0]
    t_rel, lin_mag, raw_mag = _load_linear(c["path"])
    gap_s = args.gap_ms / 1000.0
    peaks, peak_vals = detect_peaks(t_rel, lin_mag, threshold=args.threshold, gap_s=gap_s)
    print(f"  加载: [{c['hand']}] {c['user']} gesture={c['gesture']} group={c['group']} "
          f"expect={c['expected']} stamp={c['stamp']}")
    print(f"  检测到 {len(peaks)} 个峰值 (阈值={args.threshold}, gap={args.gap_ms}ms)")
    if len(peaks) >= 2:
        iv = np.diff(peaks)
        print(f"  间隔: mean={iv.mean()*1000:.1f}ms span={(iv.max()-iv.min())*1000:.1f}ms "
              f"std={iv.std()*1000:.1f}ms")
    if args.plot or args.show:
        png = output_dir / f"peaks_{c['stamp']}_{c['user']}_{c['gesture']}.png"
        plot_peaks(t_rel, lin_mag, raw_mag, peaks, peak_vals,
                   threshold=args.threshold, out_png=png, show=args.show)
    return 0


def export_review_plots(args, output_dir: Path) -> int:
    """批量生成 REVIEW 录制预览图, 展示检出问题时的阈值状态。"""
    data_dir = Path(args.data_dir).expanduser().resolve()
    candidates = _resolve_candidates(data_dir, args)
    if not candidates:
        print("[X] 无匹配录制")
        return 1
    plot_dir = Path(args.plot_dir).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    gap_s = args.gap_ms / 1000.0
    made, skipped = 0, 0
    for c in candidates:
        try:
            t_rel, lin_mag, raw_mag = _load_linear(c["path"])
            res = adaptive_detect(t_rel, lin_mag, c["expected"], gap_s,
                                  args.threshold, args.step, args.floor)
        except Exception as e:
            print(f"  [skip] [{c['hand']}] {c['user']} {c['gesture']} {c['group']}: {e}")
            continue

        is_combo = c["group"] == "combo"
        if res["status"] == "hit" and res["span_ms"] < args.max_span_ms and not is_combo:
            skipped += 1  # 已放行, 无需审查
            continue

        if res["status"] == "overshoot":
            reason = "overshoot"
        elif res["status"] == "hit" and is_combo:
            reason = "combo"
        elif res["status"] == "hit":
            reason = "span"
        else:
            reason = "undercount"

        thr = res["threshold"]
        peaks, peak_vals = detect_peaks(t_rel, lin_mag, threshold=thr, gap_s=gap_s)
        span_txt = f"{res['span_ms']:.0f}" if not np.isnan(res["span_ms"]) else "-"
        title = (f"{c['user']} {c['hand']} gesture={c['gesture']} {c['group']} "
                 f"expect={c['expected']} | {reason} | thr={thr:.0f} "
                 f"n={res['count']} span={span_txt}ms")
        name = (f"{c['user']}_{c['hand']}_{c['gesture']}_{c['group']}_"
                f"{c['stamp']}_{reason}.png")
        out = plot_dir / name
        plot_peaks(t_rel, lin_mag, raw_mag, peaks, peak_vals,
                   threshold=thr, out_png=out, show=False, title=title)
        made += 1

    print(f"\n  生成 {made} 张 REVIEW 预览图 → {plot_dir}/ (跳过已放行 {skipped} 个)")
    return 0


def annotate_all(args, output_dir: Path) -> int:
    """全量标注: 固定 threshold/gap 对所有非中文录制做尖峰+标签处理。

    单手势 → 全同标签; switch → slow: a-b-a-b, fast: a-b-b-a。
    serial(带标签)+IMU+meta → {user}/{hand}/; 可视化 → plot-dir。
    """
    data_dir = Path(args.data_dir).expanduser().resolve()
    candidates = _resolve_candidates(data_dir, args)
    if not candidates:
        print("[X] 无匹配录制")
        return 1
    plot_dir = Path(args.plot_dir).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)
    gap_s = args.gap_ms / 1000.0

    n_done = 0
    for c in candidates:
        try:
            t_rel, lin_mag, raw_mag = _load_linear(c["path"])
            peaks, peak_vals = detect_peaks(t_rel, lin_mag,
                                            threshold=args.threshold, gap_s=gap_s)
        except Exception as e:
            print(f"  [skip] [{c['hand']}] {c['user']} {c['gesture']} {c['group']}: {e}")
            continue
        labels = build_peak_labels(c["gesture"], c["group"], len(peaks))

        sub = output_dir / c["user"] / c["hand"]
        sub.mkdir(parents=True, exist_ok=True)
        _write_serial(peaks, c["stamp"], c["user"], c["gesture"], sub, labels=labels)
        (sub / c["src_enc"].name).write_bytes(c["src_enc"].read_bytes())
        if c["meta_enc"] is not None:
            dst = sub / c["meta_enc"].name
            if not dst.exists():
                dst.write_bytes(c["meta_enc"].read_bytes())

        title = (f"{c['user']} {c['hand']} gesture={c['gesture']} {c['group']} "
                 f"n={len(peaks)} thr={args.threshold} gap={args.gap_ms}ms")
        out = plot_dir / f"{c['user']}_{c['hand']}_{c['gesture']}_{c['group']}_{c['stamp']}.png"
        plot_peaks(t_rel, lin_mag, raw_mag, peaks, peak_vals,
                   threshold=args.threshold, out_png=out, show=False,
                   title=title, labels=labels)
        n_done += 1

    print(f"\n  完成 {n_done} 个录制")
    print(f"  serial+IMU+meta → {output_dir}/{{user}}/{{hand}}/")
    print(f"  可视化 → {plot_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="纯负例峰值检测 → 伪串口标注 / 批量拆分")
    parser.add_argument("input", nargs="?", type=str, help="输入 .npy.enc 文件路径 (单文件模式)")
    parser.add_argument("--batch-lcy", action="store_true",
                        help="批量模式: 处理 lcy 目录并按人物/手部分类")
    parser.add_argument("--lcy-dir", type=str, default=str(ROOT / "datanew" / "lcy"),
                        help="lcy 数据目录 (默认: datanew/lcy/)")
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "datanew" / "hyb"),
                        help="批量/过滤模式数据目录 (默认: datanew/hyb/)")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式: 峰值检测+拆分, 达标自动放行生成 serial")
    parser.add_argument("--review-plots", action="store_true",
                        help="批量生成 REVIEW 录制预览图 (展示检出问题时的阈值)")
    parser.add_argument("--annotate-all", action="store_true",
                        help="全量标注: 固定阈值/gap 对所有非中文录制做尖峰+标签处理")
    parser.add_argument("--plot-dir", type=str, default="review_plots",
                        help="REVIEW 预览图输出目录 (默认: review_plots)")
    parser.add_argument("--label", type=str, default=None,
                        help="手势标签 (过滤/命名), 如 1, 2, 1-2")
    parser.add_argument("--user", default="", help="过滤: 用户名 (hyb/lcy)")
    parser.add_argument("--hand", default="", help="过滤: 手部 (left/right)")
    parser.add_argument("--group", default="", help="过滤: 节奏组 (slow/fast/combo)")
    parser.add_argument("--single-only", action="store_true",
                        help="只处理单一手势 (排除 1-2 等 switch), 且排除 combo")
    parser.add_argument("--output", type=str, default="data_simulate",
                        help="输出目录 (默认: data_simulate)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="起始 |linear_acc| 峰值阈值 (默认: 100)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="批量模式阈值每次降低步长 (默认: 5)")
    parser.add_argument("--floor", type=float, default=20.0,
                        help="批量模式阈值下限 (默认: 20)")
    parser.add_argument("--gap-ms", type=float, default=50.0,
                        help="峰值最小间隔 ms (默认: 50)")
    parser.add_argument("--max-span-ms", type=float, default=100.0,
                        help="放行标准: 间隔 max-min 上限 ms (默认: 100)")
    parser.add_argument("--plot", action="store_true",
                        help="保存峰值检测预览图 (PNG) 到输出目录")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示预览图 (默认仅保存 PNG)")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_lcy:
        return batch_lcy(Path(args.lcy_dir).expanduser().resolve(), output_dir)
    if args.batch:
        return batch_annotate(args, output_dir)
    if args.review_plots:
        return export_review_plots(args, output_dir)
    if args.annotate_all:
        return annotate_all(args, output_dir)
    if args.input is None and (args.user or args.hand or args.label is not None or args.group):
        return view_matching(args, output_dir)
    if args.input is None:
        print("[X] 单文件模式需要输入文件路径; 或 --batch 批量; 或 --user/--hand/--label/--group + --show 过滤查看")
        return 1
    if args.label is None:
        print("[X] 单文件模式需要 --label")
        return 1

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[X] 文件不存在: {input_path}")
        return 1

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
    raw_mag = np.linalg.norm(acc, axis=1)

    # ── 峰值检测 ──
    gap_s = args.gap_ms / 1000.0
    peaks, peak_vals = detect_peaks(t_rel, lin_mag, threshold=args.threshold, gap_s=gap_s)
    print(f"  检测到 {len(peaks)} 个峰值 (|linear_acc|>{args.threshold}, gap≥{args.gap_ms}ms)")
    print(f"  录制时长: {t_rel[-1]:.1f}s, 密度: {len(peaks)/max(t_rel[-1],0.01):.1f} 峰/秒")
    if len(peaks) >= 1:
        print(f"  峰值强度 |linear_acc|: mean={peak_vals.mean():.1f} "
              f"max={peak_vals.max():.1f} min={peak_vals.min():.1f}")

    # ── 相邻峰值间隔统计 ──
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        print(f"  峰值间隔: mean={intervals.mean():.3f}s std={intervals.std():.3f}s "
              f"min={intervals.min():.3f}s max={intervals.max():.3f}s")

    # ── 可视化预览 (在保存前生成, 便于确认阈值/gap 是否合适) ──
    if args.plot or args.show:
        png_path = output_dir / f"peaks_{stamp}_{user}_{label}.png"
        plot_peaks(t_rel, lin_mag, raw_mag, peaks, peak_vals,
                   threshold=args.threshold, out_png=png_path, show=args.show)

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
