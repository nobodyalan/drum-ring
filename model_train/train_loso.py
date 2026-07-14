"""
Leave-One-Session-Out (LOSO) 交叉验证训练
==========================================
每次留一个 session 作为独立验证集, 其余用于训练,
评估模型在每个 session 上的泛化能力.

参考: Müller et al. (2024) "IMU-Based Fitness Activity Recognition"
      — 使用 leave-one-subject-out 评估真实泛化能力

用法:
  python train_loso.py --data_dir multi_sliced_data --save_dir models_loso
  python train_loso.py --data_dir multi_sliced_data --save_dir models_loso --aug-rotate 12
  python train_loso.py --data_dir multi_sliced_data --sessions fast_0,slow_0  # 只测试指定session
"""

import os
import sys
import argparse
import numpy as np

# 复用 train_model 的加载和训练逻辑
from train_model import load_session_slices, train


def main():
    parser = argparse.ArgumentParser(
        description="LOSO (Leave-One-Session-Out) 交叉验证训练"
    )
    parser.add_argument("--data_dir", default="multi_sliced_data",
                        help="切片数据目录")
    parser.add_argument("--save_dir", default="models_loso",
                        help="模型保存目录")
    parser.add_argument("--model", default="inception",
                        choices=["inception", "fcn", "lite", "tst"])
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大训练轮数")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug-rotate", type=float, default=0,
                        help="IMU 旋转增强角度 (度), 建议 10-15")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout 概率 (默认 0.4)")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="L2 权重衰减 (默认 1e-3)")
    parser.add_argument("--neg-sessions", type=str, default="",
                        help="负样本优先 session")
    parser.add_argument("--focal-loss", type=float, default=0)
    parser.add_argument("--sessions", type=str, default="",
                        help="只对逗号分隔的 session 做 LOSO (默认: 全部)")
    parser.add_argument("--quick", action="store_true",
                        help="快速诊断模式: 每session只跑1 epoch, 不做增强, 不做早停")
    parser.add_argument("--balance-mode", type=str, default="posthit",
                        choices=["posthit", "random"],
                        help="负例平衡策略: posthit(默认) / random")

    args = parser.parse_args()
    sessions_path = os.path.join(args.data_dir, "sessions.txt")
    if not os.path.exists(sessions_path):
        print(f"[X] sessions.txt 未找到: {sessions_path}")
        return

    with open(sessions_path) as f:
        all_sessions = [l.strip() for l in f if l.strip()]

    # 只对指定的 session 做 LOSO
    if args.sessions:
        target = set(s.strip() for s in args.sessions.split(",") if s.strip())
        test_sessions = [s for s in all_sessions if s in target]
        unknown = target - set(all_sessions)
        for u in unknown:
            print(f"[WARN] session 不存在: {u}")
    else:
        test_sessions = all_sessions

    if not test_sessions:
        print("[X] 没有可测试的 session")
        return

    neg_sessions = [s.strip() for s in args.neg_sessions.split(",") if s.strip()] \
                   if args.neg_sessions else None

    results = {}
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  LOSO 交叉验证 ({len(test_sessions)} sessions)")
    print(f"  增强: {'IMU旋转 max=' + str(args.aug_rotate) + '°' if args.aug_rotate > 0 else '无'}")
    print("=" * 60)

    for val_name in test_sessions:
        # 快速模式: 1 epoch, 无增强
        n_epochs = 1 if args.quick else args.epochs
        aug_deg = 0 if args.quick else args.aug_rotate

        print(f"\n{'#' * 60}")
        print(f"  LOSO: 验证集 = [{val_name}], 训练集 = 其余 {len(all_sessions)-1} 个")
        if args.quick:
            print(f"  快速模式: {n_epochs} epoch, 无增强")
        print(f"{'#' * 60}")

        # 加载数据: 当前 session 作为验证集
        data = load_session_slices(
            args.data_dir,
            val_sessions=[val_name],
            neg_sessions=neg_sessions,
            balance_mode=args.balance_mode,
        )

        if data["X_train"] is None or len(data["X_train"]) == 0:
            print(f"  [SKIP] {val_name}: 无训练数据")
            continue
        if data["X_val"] is None or len(data["X_val"]) == 0:
            print(f"  [SKIP] {val_name}: 无验证数据")
            continue

        # 训练
        model_save_dir = os.path.join(args.save_dir, f"loso_{val_name}")
        try:
            learn, model = train(
                X=data["X_train"], y=data["y_train"],
                model_name=args.model,
                epochs=n_epochs, batch_size=args.bs, lr=args.lr,
                save_dir=model_save_dir, focal_gamma=args.focal_loss,
                X_val=data["X_val"], y_val=data["y_val"],
                preproc_mean=data["mean"], preproc_std=data["std"],
                aug_rotate_deg=aug_deg,
                dropout=args.dropout, weight_decay=args.weight_decay,
            )

            # 收集结果 — 用模型直接计算验证集 Recall (需手动归一化)
            import torch
            # 用与训练相同的 mean/std 归一化
            _mean = data["mean"].reshape(1, -1, 1)
            _std = data["std"].reshape(1, -1, 1)
            X_val_norm = (data["X_val"] - _mean) / (_std + 1e-8)

            model_cpu = model.cpu()
            X_val_t = torch.from_numpy(X_val_norm).float()
            with torch.no_grad():
                logits = model_cpu(X_val_t)
                preds = logits.argmax(dim=1).numpy()
            recall = float((preds[data["y_val"] == 1] == 1).mean()) if (data["y_val"] == 1).sum() > 0 else float('nan')
            precision = float((preds[preds == 1] == data["y_val"][preds == 1]).mean()) if preds.sum() > 0 else 0.0
            acc = float((preds == data["y_val"]).mean())

            results[val_name] = {
                "recall": float(recall), "precision": float(precision), "accuracy": float(acc),
                "val_hit": int((data["y_val"] == 1).sum()),
                "val_bg": int((data["y_val"] == 0).sum()),
            }
        except Exception as e:
            print(f"  [FAIL] {val_name}: {e}")
            import traceback; traceback.print_exc()
            results[val_name] = {"error": str(e)}

    # ── 汇总 (按 Recall 降序) ──
    print(f"\n{'=' * 70}")
    print(f"  LOSO 结果汇总 (按 Recall 降序)")
    print(f"{'=' * 70}")
    # 按 recall 排序
    sorted_results = sorted(
        [(n, i) for n, i in results.items() if "error" not in i],
        key=lambda x: x[1]["recall"], reverse=True,
    )
    failed = [(n, i) for n, i in results.items() if "error" in i]

    print(f"  {'session':20s} {'Recall':>7s} {'Prec':>7s} {'Acc':>7s} {'val_hit':>7s} {'val_bg':>7s}")
    print(f"  {'-' * 60}")
    for name, info in sorted_results:
        print(f"  {name:20s} {info['recall']:7.1%} {info['precision']:7.1%} "
              f"{info['accuracy']:7.1%} {info['val_hit']:7d} {info['val_bg']:7d}")
    for name, info in failed:
        print(f"  {name:20s} ERROR: {info['error']}")

    # 统计
    recalls = [i["recall"] for _, i in sorted_results]
    if recalls:
        print(f"\n  Recall: min={min(recalls):.1%} median={np.median(recalls):.1%} max={max(recalls):.1%}")
        print(f"  Recall<60%: {sum(1 for r in recalls if r < 0.6)}/{len(recalls)} sessions")

    # 保存汇总
    summary_path = os.path.join(args.save_dir, "loso_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'session':25s} {'recall':>8s} {'precision':>8s} {'accuracy':>8s} {'val_hit':>7s} {'val_bg':>7s}\n")
        for name, info in sorted_results:
            f.write(f"{name:25s} {info['recall']:8.1%} {info['precision']:8.1%} "
                    f"{info['accuracy']:8.1%} {info['val_hit']:7d} {info['val_bg']:7d}\n")
    print(f"\n  汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
