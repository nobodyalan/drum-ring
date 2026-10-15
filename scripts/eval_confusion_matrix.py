"""用训练好的 checkpoint 在验证集(与训练相同的划分)上计算混淆矩阵。

复现 scripts/train_local_drum_gestures.py 的切片+split 流程 (seed 固定 42, 划分确定),
加载 checkpoint 后在 val (和 train) 上评估。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_local_drum_gestures as ap  # noqa: E402
from models.drum_gesture_recognizer import DrumGestureRecognizer, DRUM_CLASS_LABELS  # noqa: E402

# ── 与训练命令一致的参数 ──
RUN_NAME = "local_drum_5gesture_lead6_gap_raw6"
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "data_processed_local_drum_gesture"
CHECKPOINT = ROOT / "checkpoint" / "drum_gesture" / f"{RUN_NAME}.pth"
FEATURE_MODE = "rawlin12_handcrafted"
SEED = 42
WINDOW = 15
TRANSITION_START = 50
TRANSITION_END = 17
TRANSITION_NEGS = 4
HARD_NEGS = 80
SERIAL_EDGE = "FALLING"


def main() -> int:
    # 覆盖模块全局 (与训练一致)
    ap.LABEL_IDS = (1, 2, 3, 4, 5)
    ap.TRANSITION_START_OFFSET = TRANSITION_START
    ap.TRANSITION_END_OFFSET = TRANSITION_END

    print("解密 + 切片 ...")
    decrypted = ap.decrypt_dataset(DATA_DIR, WORK_DIR, keep_decrypted=False)
    recordings, _skipped = ap.collect_recordings(decrypted, ap.LABEL_IDS)
    x, y, groups, _summary = ap.build_dataset(
        recordings,
        WORK_DIR,
        seed=SEED,
        hard_negatives_per_file=HARD_NEGS,
        serial_edge=SERIAL_EDGE,
        feature_mode=FEATURE_MODE,
        transition_negatives_per_peak=TRANSITION_NEGS,
        use_peak_exclusion=False,
    )
    x_train, y_train, x_val, y_val, x_test, y_test, _split = ap.split_by_group(
        x, y, groups, SEED, test_size=0.25, val_size=0.20,
        val_one_per_class=True, no_test=True,
    )
    x_train, x_val, x_test, _stats = ap.standardize_channel_groups(
        x_train, x_val, x_test, FEATURE_MODE,
    )

    # 加载模型 (复用 DrumGestureRecognizer 的 checkpoint 解析逻辑)
    recognizer = DrumGestureRecognizer(CHECKPOINT)
    model = recognizer.model.eval()
    c_out = recognizer.n_classes

    # 评估 val + train
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    labels = list(range(c_out))
    label_names = [DRUM_CLASS_LABELS.get(i, str(i)) for i in labels]

    for tag, xx, yy in [("验证集 val", x_val, y_val), ("训练集 train", x_train, y_train)]:
        with torch.inference_mode():
            logits = model(torch.from_numpy(xx).float())
            pred = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(yy, pred)
        cm = confusion_matrix(yy, pred, labels=labels)
        print(f"\n===== {tag}  accuracy={acc:.4f} =====")
        header = "        " + "".join(f"{name:>8s}" for name in label_names)
        print(header)
        for i, row in enumerate(cm):
            print(f"{label_names[i]:>7s} " + "".join(f"{v:>8d}" for v in row))
        print("\n分类报告:")
        print(classification_report(yy, pred, labels=labels, target_names=label_names, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
