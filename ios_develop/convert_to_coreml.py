"""
================================================================================
 PyTorch → CoreML 模型转换脚本（PC 端一次性运行）
================================================================================

 输入: model_train/models/hit_detector_inception.pth
 输出: ios_develop/HitDetector.mlpackage  (拖入 Xcode)
       ios_develop/DrumHitApp/preproc_params.json  (mean/std, App 读取)

 用法:
   cd ios_develop
   python convert_to_coreml.py

 依赖:
   pip install torch coremltools numpy
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. 加载 checkpoint
# ==============================================================================

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_train", "final_model",
                         "hit_detector_inception.pth")
OUTPUT_MLMODEL = os.path.join(os.path.dirname(__file__), "HitDetector.mlmodel")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "DrumHitApp", "preproc_params.json")


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = ["model_state", "mean", "std", "channels", "classes",
                "window_frames", "target_fs"]
    missing = [k for k in required if k not in ckpt or ckpt[k] is None]
    if missing:
        raise ValueError(f"Checkpoint 缺少字段: {missing}")
    return ckpt


# ==============================================================================
# 2. 重建 WeightedInceptionTime（与 train_model.py 完全一致）
#   关键改动：加权池化层固化为常量（去掉 nn.Parameter 和 softmax）
# ==============================================================================

# ── 架构参数（与训练时一致）──
KS = [3, 5, 7]
NF = 32
DEPTH = 4
OUT_CH = NF * (len(KS) + 1)  # 128


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks):
        super().__init__()
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks
        ])
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        branches = [conv(z) for conv in self.convs]
        branches.append(self.maxpool(x))
        return self.act(self.bn(torch.cat(branches, dim=1)))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf, depth, ks):
        super().__init__()
        out_ch = nf * (len(ks) + 1)
        self.inceptions = nn.ModuleList([
            InceptionModule(ni if i == 0 else out_ch, nf, ks)
            for i in range(depth)
        ])
        self.shortcut = nn.Sequential(
            nn.Conv1d(ni, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        for inc in self.inceptions:
            x = inc(x)
        return self.act(x + residual)


class WeightedInceptionTimeExport(nn.Module):
    """
    与训练时的 WeightedInceptionTime 架构完全一致，
    唯一区别：time_weights 固化为常量（非 nn.Parameter），forward 中不做 softmax。
    coremltools trace 时只会看到 multiply + sum，100% 兼容。
    """

    def __init__(self, c_in: int, c_out: int, time_weights_fixed: torch.Tensor):
        super().__init__()
        self.backbone = InceptionBlock(c_in, NF, DEPTH, KS)

        # ── 关键：固化为 buffer（非 Parameter），避免 coremltools 困惑 ──
        self.register_buffer("time_weights", time_weights_fixed)  # (15,)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(OUT_CH, c_out),
        )

    def forward(self, x):
        # x: (N, 6, 15) — 固定15帧，无需动态切片
        feat = self.backbone(x)                       # → (N, 128, 15)

        # 加权池化：乘 + 求和（w 已经是训练好的 softmax 结果，固定 15 维）
        pooled = (feat * self.time_weights.view(1, 1, -1)).sum(dim=-1)  # (N, 128)

        return self.head(pooled)


# ==============================================================================
# 3. 转换主函数
# ==============================================================================

def convert():
    print("=" * 55)
    print("  PyTorch → CoreML 模型转换")
    print("=" * 55)

    # 3.1 加载 checkpoint
    print(f"\n[1/5] 加载 checkpoint: {CKPT_PATH}")
    ckpt = load_checkpoint(CKPT_PATH)
    c_in = ckpt["channels"]
    c_out = ckpt["classes"]
    print(f"      架构: inception  |  输入: ({c_in}, {ckpt['window_frames']})  |  输出: {c_out} 类")

    # 3.2 固化 time_weights
    print("\n[2/5] 固化加权池化参数...")
    raw_weights = ckpt["model_state"]["time_weights"]  # (15,)
    # 训练时的处理：softmax(time_weights)
    time_weights_fixed = torch.softmax(raw_weights, dim=0).detach().clone()
    print(f"      训练后权重 (softmax 归一化):")
    print(f"      {time_weights_fixed.numpy().round(3)}")
    print(f"      权重比 (末帧/首帧): {time_weights_fixed[-1].item() / time_weights_fixed[0].item():.1f}x")

    # 3.3 重建模型并加载权重
    print("\n[3/5] 重建模型 + 加载训练权重...")
    model = WeightedInceptionTimeExport(c_in, c_out, time_weights_fixed)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3.4 验证输出一致
    print("\n[4/5] 验证 PyTorch 推理...")
    example = torch.randn(1, c_in, ckpt["window_frames"])
    with torch.no_grad():
        logits = model(example)
        prob = torch.softmax(logits, dim=1)
    print(f"      P(hit)={prob[0, 1].item():.3f}  P(bg)={prob[0, 0].item():.3f}")

    # 3.5 trace + 转 CoreML
    print("\n[5/5] TorchScript trace → CoreML...")
    traced = torch.jit.trace(model, example)

    try:
        import coremltools as ct

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="imu_window", shape=(1, c_in, ckpt["window_frames"]))],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.iOS13,
        )
        # 兼容旧版 coremltools / Windows：保存为 .mlmodel 而非 .mlpackage
        mlmodel.save(OUTPUT_MLMODEL)
        print(f"      已保存: {OUTPUT_MLMODEL}")

        # 打印输入输出信息
        spec = mlmodel.get_spec()
        print(f"      输入: {[i.name for i in spec.description.input]}")
        print(f"      输出: {[o.name for o in spec.description.output]}")

    except ImportError:
        print("      [!] coremltools 未安装，跳过 CoreML 转换")
        print("      请运行: pip install coremltools")
        print(f"\n      手动转换：将 traced 模型保存为 ONNX 后再转")
        torch.onnx.export(model, example, "model.onnx",
                          input_names=["imu_window"],
                          output_names=["logits"],
                          dynamic_axes={"imu_window": {0: "batch"},
                                        "logits": {0: "batch"}})
        print(f"      已导出 ONNX: model.onnx（可用 coremltools 转换）")

    # 3.6 导出 preproc_params.json
    mean_arr = ckpt["mean"].squeeze()  # (6,)
    std_arr = ckpt["std"].squeeze()    # (6,)

    # 确保是 numpy
    if isinstance(mean_arr, torch.Tensor):
        mean_arr = mean_arr.numpy()
    if isinstance(std_arr, torch.Tensor):
        std_arr = std_arr.numpy()

    params = {
        "mean": mean_arr.tolist(),
        "std": std_arr.tolist(),
        "channel_order": ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"],
        "window_frames": ckpt["window_frames"],
        "target_fs": ckpt["target_fs"],
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"\n      已保存: {OUTPUT_JSON}")
    print(f"      mean: {params['mean']}")
    print(f"      std:  {params['std']}")

    print("\n" + "=" * 55)
    print("  转换完成！")
    print(f"  1. 把 {OUTPUT_MLMODEL} 拖入 Xcode")
    print(f"  2. {OUTPUT_JSON} 会被 App 自动读取")
    print("=" * 55)


if __name__ == "__main__":
    convert()
