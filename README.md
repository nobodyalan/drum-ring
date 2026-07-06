# drum-ring

鼓击检测项目 — 踩踏 IMU 训练与实时检测。

## 项目概述

利用 iPhone ZigSim 发送的 IMU 数据 (UDP OSC)，通过阈值检测算法实时识别脚底踩踏动作，触发音效反馈。同时支持训练数据录制与深度学习模型训练。

## 实时检测程序

| 程序 | 信号 | 检测方式 | 用途 |
|---|---|---|---|
| `realtime_simple.py` | 合加速度 `√(ax²+ay²+az²)` | STA/LTA ratio + jerk 双通道 | 通用踩踏检测 |
| `realtime_mag_jerk.py` | X 轴加速度 `ax` | STA/LTA ratio (abs) + jerk 双向 | **预判模式**, 提腿即触发 |
| `measure_latency.py` | 双手机 | 同上 (双通道独立) | **延迟测量**工具 |

### 快速开始

```bash
# 终端实时显示
python realtime_mag_jerk.py -v

# 带波形图
python realtime_mag_jerk.py --plot

# 调参
python realtime_mag_jerk.py --mag-high 0.6 --jerk-thresh 1.0 --jerk-down-thresh 2.0 -v

# 双手机延迟测量
python measure_latency.py --plot
```

### 检测策略

```text
主通道 (STA/LTA):
  short = mean(最近2帧 abs(ax))
  long  = mean(全部30帧 abs(ax))
  ratio = short / long
  触发: ratio > 2.0 AND ax > 0.6g

快速通道 (jerk 双向):
  触发: (jerk > +1.0 OR jerk < -2.0) AND ratio > 1.2
        ↑ 向上提腿 (预判)     ↑ 向下踩踏 (确认)

加速度双阈值滞回:
  ax > mag_high (0.6g)  → 触发
  ax < mag_low  (0.15g) → 释放, 重新就绪
```

### 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ratio-thresh` | 2.0 | STA/LTA 触发比值 |
| `--mag-high` | 0.6 | ax 高阈值 (触发) |
| `--mag-low` | 0.15 | ax 低阈值 (释放) |
| `--jerk-thresh` | 2.0 | 正向 jerk 阈值 (向上提腿) |
| `--jerk-down-thresh` | 2.0 | 负向 jerk 阈值 (向下踩踏, 绝对值) |
| `--fast-ratio` | 1.2 | 快速通道最低 ratio |
| `--cooldown` | 120 | 冷却时间 ms |
| `--quiet-frames` | 5 | ratio < reset 释放所需持续帧数 |

## 模型训练 (model_train/)

```bash
cd model_train

# 录制数据
python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/w_X

# 训练
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --save_dir models_margin

# 实时检测 (模型)
python realtime_detect.py --model models_margin/hit_detector_inception.pth -v
```

详见 `model_train/CLAUDE.md`。

## 延迟分析

| 环节 | 延迟 | 说明 |
|---|---|---|
| 提腿 → 裤兜手机 | ~100ms | 物理传导 |
| WiFi UDP 传输 | ~40ms (±1.2ms) | 可忽略 |
| 检测算法 | ~17ms (1帧) | 快速通道, 零 STA/LTA 积分 |
| 声卡播放 | <5ms | winsound SND_ASYNC |

用 `measure_latency.py` 双手机方案可精确测量物理延迟 (WiFi 抵消)。

## 目录结构

```text
.
├── realtime_simple.py          # 合加速度检测
├── realtime_mag_jerk.py        # X 轴加速度双向 jerk 预判
├── measure_latency.py          # 双手机延迟测量工具
├── 1.wav                       # 触发音效
├── main.py                     # BLE 戒指主程序
├── model_train/                # 模型训练子项目
│   ├── train_model.py
│   ├── realtime_detect.py
│   ├── export_training_data.py
│   └── CLAUDE.md
├── core/                       # BLE 协议
├── apps/                       # 戒指应用
└── latency_data/               # 延迟测量原始数据
```
