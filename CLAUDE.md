# drum-ring — 智能戒指手势识别的鼓击应用

## 项目概览

drum-ring 是一个完整的智能戒指手势识别系统，核心功能是**通过戒指 IMU 数据实时检测鼓击手势**并播放对应音色。项目包含：

- **桌面端** (PyQt6 + PyTorch): 实时推理 + 可视化 + 模型训练
- **iOS 端** (Swift + CoreML): 独立运行的手势识别 App
- **硬件**: 蓝牙戒指 (IMU 200Hz) + Arduino 压电传感器 (训练数据标注用)
- **传输**: WebSocket 中转 (BLE → PC → OSC / JSONL)

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      硬件层                                   │
│  Ring (BLE IMU 200Hz)  │  Arduino (压电传感器, USB)           │
│  加速度 + 陀螺仪 6轴      │  踩踏/敲击事件检测                 │
└──────────┬──────────────────┬──────────────────────────────┘
           ↓                  ↓
┌─────────────────────────────────────────────────────────────┐
│                    传输层 (stream_hub.py)                     │
│  WebSocket Server (127.0.0.1:17888)                          │
│  JSONL pub/sub: 戒指 IMU 数据分发                              │
└──────────┬──────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────┐
│                  桌面应用层 (main.py → QApplication)           │
│  MainWindow → apps/drum_gesture_app.py                       │
│  │  DrumGestureWindow (实时推理 + 音效)                        │
│  │  measure_latency.py (延迟对比)                              │
│  │  realtime_detect.py (踩踏检测)                              │
│  └── apps/features/                                          │
│      gesture/ (通用手势识别)                                   │
│      window_classifier/ (概率图表 UI 组件)                      │
│      airmouse/ touchpad/ typering/ whisper/                   │
└──────────┬──────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────┐
│                   模型层 (models/)                            │
│  drum_gesture_recognizer.py — 鼓击手势识别                     │
│  cnn.py / tcn_lstm.py — 其他手势模型                          │
└──────────┬──────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────┐
│              训练流水线 (model_train/)                         │
│  export_training_data.py → filter_close_clicks.py             │
│  → slice_dataset.py → train_model.py → analyze_badcases.py   │
│  → visualize_session.py                                       │
└─────────────────────────────────────────────────────────────┘
```

## main.py 入口

```
main.py → QApplication (PyQt6 + Fusion 样式)
  → MainWindow (desktop/main_window.py)
    → start_background_services()
      → 启动 WebSocket 客户端订阅 IMU 流
      → 各 Feature App 注册
```

`qt_bootstrap.py` 负责设置 Qt 插件路径（兼容 PyInstaller 打包）。

## Drum Gesture 模块 (apps/drum_gesture_app.py)

### 核心类

| 类 | 职责 |
|---|------|
| `DrumGestureWindow` | 主窗口: 参数面板 + 概率图表 + 音效播放 |
| `DrumInferenceWorker` | 后台 QThread: 接收 IMU 窗口 → 模型推理 → 返回结果 |
| `DrumRingState` | 单戒指状态: IMU 缓冲区、预测历史队列、触发冷却 |
| `OnlineLinearAccelerationEstimator` | 实时重力补偿: 陀螺仪积分 + 加速度校正 → 线性加速度 |

### 数据流

```
Ring BLE → WebSocket Server → stream_hub → DrumGestureWindow
  │
  ├─ handle_imu_message(): 接收 acc[3] + gyr[3]
  │   → OnlineLinearAccelerationEstimator.update() → linear_acc
  │   → 拼接 [raw_acc, raw_gyro, linear_acc, raw_gyro] = 12 通道
  │   → 滑窗 (window_size 帧)
  │
  ├─ maybe_submit_window(): 每 step 帧提交一次
  │   → DrumInferenceWorker.submit(job)
  │   → recognizer.predict(window_values)
  │
  └─ on_inference_result(): 稳定判定 + 音效触发
      → update_prediction_history(): 历史队列投票
      → 超过 trigger_ratio → 触发音效 + 冷却
```

### 触发策略: 历史队列稳定投票

```
参数:
  --step: 推理步长 (默认 5 帧 = 25ms)
  --stable-window: 稳定窗口时长 (默认 0.04s)
  --trigger-ratio: 触发比例 (默认 0.67)
  --cooldown: 冷却帧数 (默认 24 帧 = 120ms)

队列长度 = stable_window * 200Hz / step = ~2
触发条件: 队列中某手势类别计数 >= trigger_ratio * 队列长度
```

### DrumGestureRecognizer 模型 (models/drum_gesture_recognizer.py)

#### 模型架构: RegularizedInceptionTime

```
输入: (1, 21, 20) — 21 通道 × 20 帧窗口 (100ms @200Hz)

特征构建 (build_rawlin12_handcrafted_window):
  ┌─────────────────────────────────────────────┐
  │ 原始 12 通道:                                │
  │   raw_acc[3] + raw_gyro[3]                   │
  │   + linear_acc[3] + gyro[3]                  │
  ├─────────────────────────────────────────────┤
  │ 手工特征 9 通道:                              │
  │   raw_acc_norm    — 原始加速度模长            │
  │   linear_acc_norm — 线性加速度模长 (关键!)     │
  │   raw_gyro_norm   — 陀螺仪模长               │
  │   gyro_norm       — 去重力陀螺仪模长          │
  │   raw_acc_delta   — 加速度帧间差分模长        │
  │   linear_acc_delta— 线性加速度帧间差分模长     │
  │   gyro_delta      — 陀螺仪帧间差分模长        │
  │   raw_linear_dot  — raw_acc·linear_acc 内积  │
  │   raw_linear_cos  — 两者夹角余弦             │
  └─────────────────────────────────────────────┘
  总计 21 通道

InceptionTime Backbone:
  ks=40 (大核, 覆盖 200ms 全窗口)
  nf=32, depth=6, bottleneck=True
  输出: (128, 20)

Head (GAP 模式):
  AdaptiveAvgPool1d → Dropout → Linear(128, 5)
  
5 类输出: 空(0) / 踩镲(1) / 通鼓2(2) / 军鼓(3) / 通鼓1(4)
```

#### OnlineLinearAccelerationEstimator (重力补偿)

```
原理: 互补滤波器 (陀螺仪积分 + 加速度校正)

1. 初始化: 用加速度方向估计重力方向 (假设静止)
2. 每帧更新:
   orientation += gyro * dt               (陀螺仪积分)
   用加速度方向校正 (trust 系数自适应):
     trust = 0.025 * exp(-(||acc||-9.81)²/2.5²)
     - 加速度接近 9.81 → trust 高 → 多信加速度
     - 加速度远离 9.81 (运动中) → trust 低 → 主要靠陀螺仪
3. linear_acc = acc - gravity_sensor      (世界坐标系去重力)
```

## 📊 两套模型训练体系

### 体系 A: 踩踏检测 (model_train/) — 时序二分类

```
目标: 检测脚踩踏 → 手机 IMU 脉冲
数据类型: 手机裤兜 IMU (60Hz 13轴) + Arduino 踩踏时刻
模型: WeightedInceptionTime (ks=[3,5,7], 指数加权池化)

流水线:
  1. export_training_data.py  录制 (Arduino + ZigSim iPhone)
  2. filter_close_clicks.py   清理异常标注
  3. slice_dataset.py         峰值对齐 + 滑动窗口切片
  4. train_model.py           训练 WeightedInceptionTime
  5. analyze_badcases.py      Badcase 分析
  6. realtime_detect.py       实时推理 (概率双阈值滞回)
```

### 体系 B: 鼓击手势识别 (apps/drum_gesture_app + models/) — 5 类分类

```
目标: 识别鼓击手势类型 (空/踩镲/通鼓/军鼓...)
数据类型: 戒指 IMU (200Hz 6轴) + 人工标注
模型: RegularizedInceptionTime (ks=40, GAP head)

流水线:
  1. 数据录制 (桌面应用录制 + 人工标注手势类别)
  2. 离线训练 (外部 Python 脚本, 非本仓库)
  3. 导出 checkpoint (.pth)
  4. convert_to_coreml.py → CoreML (.mlmodel) → iOS
```

## 数据标注与训练 (体系 A 踩踏检测)

### 标注策略

```
原始 Arduino 点击 → 峰值对齐 [0, +100ms] → hit_zone = [peak - PRE, peak + POST]
                                                       ↓
滑动窗口 [t-W, t] → 标签 = 窗口末帧是否在 hit_zone 内
```

### 峰值对齐

```
问题: Arduino 触发和 IMU 加速度峰值之间有时延 (0-100ms)
方法: 在每个 click 后 [click, click+100ms] 内找 accel_mag 全局最大值
修复历史: argmax(全局最大) → 最近局部最大值 → [0,120ms] 全局最大 → [0,100ms] 时间掩码
```

### 正负例平衡

```
posthit 模式: 优先保留击打后 200ms 的负例 (反弹峰区域)
random 模式: 均匀随机抽取
bg:正例 ≤ 2.5:1 (posthit) / 2:1 (random)
```

### Badcase 分析

```
FP (误触发): 背景被预测为击打 — 主要集中在 slow 类型 session
FN (漏检):   击打被预测为背景 — post=-20 时严重
根因: 慢踩踏的反弹峰波形和首次冲击相似, 150ms 窗口内无法区分
修复: 窗口扩至 250ms + 峰值对齐改局部最大 + post-hit 扩至 200ms
```

## 目录结构

```
drum-ring/
├── main.py                    # 桌面应用入口
├── qt_bootstrap.py            # Qt 插件路径
├── project.yml                # XcodeGen iOS 项目配置
│
├── apps/                      # PyQt6 应用层
│   ├── drum_gesture_app.py    # 鼓击手势主应用 ★
│   ├── recorder_app.py        # 数据录制
│   ├── framework/              # UI 框架 (样式/流窗口/Bootstrap)
│   ├── features/               # 手势/空中鼠标/触摸板/键盘/Whisper
│   └── shared/                 # 常量 / 文件保存
│
├── models/                    # 推理模型
│   ├── drum_gesture_recognizer.py  # 鼓击手势识别器 ★
│   ├── cnn.py                 # 通用 CNN
│   └── tcn_lstm.py            # TCN+LSTM
│
├── core/                      # 核心传输层
│   ├── stream_hub.py          # WebSocket pub/sub 服务
│   └── ble_ring_v2.py         # BLE 戒指连接
│
├── model_train/               # 训练流水线 ★
│   ├── train_model.py         # WeightedInceptionTime 训练
│   ├── slice_dataset.py       # 峰值对齐 + 滑动窗口切片
│   ├── filter_close_clicks.py # 标注清理工具
│   ├── realtime_detect.py     # 实时踩踏检测 (模型模式)
│   ├── analyze_badcases.py    # Badcase 分析 + 可视化
│   ├── visualize_session.py   # 单 session IMU 可视化
│   ├── latency_compare.py     # Arduino vs iPhone 延迟对比
│   ├── debug_osc.py           # ZigSim OSC 诊断
│   └── README.md              # 工具链文档
│
├── checkpoint/drum_gesture/    # 训练好的模型权重
├── ios_develop/               # iOS CoreML App
└── desktop/                   # 桌面 MainWindow
```

## 常用命令

```bash
# ── 桌面鼓击手势 App ──
python main.py

# ── 踩踏检测训练流水线 ──
# 1. 确认 ZigSim 连接
python model_train/debug_osc.py
# 2. 测量延迟
python model_train/latency_compare.py --port COM6 --drop-mode --no-plot
# 3. 录制数据
python model_train/export_training_data.py --port COM6 --wifi
# 4. 标注清理
python model_train/filter_close_clicks.py --data_dir train_data --list-clicks slow_0
# 5. 切片
python model_train/slice_dataset.py --data_dir train_data --save sliced_data --pre 120 --post 30
# 6. 训练
python model_train/train_model.py --data_dir sliced_data --neg-sessions neg_a_0,neg_b_0
# 7. Badcase 分析
python model_train/analyze_badcases.py --model models/hit_detector_inception.pth --data_dir sliced_data
# 8. 实时检测
python model_train/realtime_detect.py --model models/hit_detector_inception.pth -v

# ── iOS CoreML 导出 ──
python ios_develop/convert_to_coreml.py
```
