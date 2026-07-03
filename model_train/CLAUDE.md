# 鼓击检测项目 — 踩踏 IMU 训练

## 硬件连接
- Arduino + 压电传感器 → USB 串口 → PC
- iPhone ZigSim → OSC UDP → PC (端口 8888)
- 传感器贴脚底/踩踏面，手机放裤兜

## 数据流
```
录制:  export_training_data.py  (纯录制全量数据)
          ↓
      train_data/w_*/  (imu + click_timestamps)
          ↓
切片:  slice_dataset.py / train_model.py 内置 (滑动窗口 + 标签生成)
          ↓
      (内部: N×6×15 数组 + 标签)
          ↓
训练:  train_model.py  (定制 WeightedInceptionTime)
          ↓
      models_margin/hit_detector_inception.pth
          ↓
实时:  realtime_detect.py  (logit margin + 滑动平均)
```

## 当前模型架构 (2026-07 重构)

### WeightedInceptionTime (89K 参数, 替代 tsai 默认 456K)
- 卷积核: [3, 5, 7] (替代 tsai 默认 [39, 19, 9])
- 深度: 4 层 (替代 6 层)
- 池化: 可学习加权池化 (替代 GAP 全局平均池化)
  - 时间权重初始化为指数递增, softmax 归一化
  - 学到的权重: 前 5 帧 ~0, 后 10 帧均分
- 架构与 realtime_detect.py 中的 WeightedInceptionTime 完全一致
- 旧 tsai 模型自动检测: state_dict 含 "inceptionblock" → 自动切换

### 训练参数
- Loss: tsai FocalLoss (gamma=2.0) / 标准 CrossEntropyLoss
- 优化器: AdamW + One-Cycle LR (lr_max=1e-3)
- Batch size: 64, Max epochs: 50, EarlyStopping patience=10
- 标准化: 每通道 Z-score (mean/std 保存到 checkpoint)
- 窗口: 250ms (15帧@60Hz), 训练 stride=2帧
- Hit zone: [-10ms, +190ms]
- 背景:击打 = 2:1 平衡 (只对训练集)

### 训练数据分割
- `--val-sessions w_5,f_3,strong_0` → session 级独立验证集
- 其余 20 sessions 为训练集 (含 shaking_0/1 作为负样本)
- 验证集不采样, 保持原始分布
- window 级随机分割已弃用 (跨 session 泄露)

## 实时检测 (realtime_detect.py)

### 策略: 纯概率双阈值滞回
```
每帧推理 → P(hit) = softmax(logits)[1]
  → P(hit) > prob_high + 不在冷却期 → 触发提示音 + 进入冷却
  → P(hit) < prob_low → 释放滞回 (允许再次触发)
```

### 参数
- `--prob-high 0.85`: 概率高阈值 (超过触发)
- `--prob-low 0.4`: 概率低阈值 (低于释放)
- `--cooldown 400`: 触发后冷却 ms
- `-v / --verbose`: 单行动态刷新, 实时显示概率条
- `--calibrate`: 30s 校准模式, 不触发, 输出概率分布和建议阈值
- `--debug`: 每秒打印 OSC 原始数据

### 为什么用 softmax 概率
- CE Loss 训练出的 softmax 概率跨 session 语义一致
- 概率 ∈ [0,1] 有直观物理含义 (模型对踩踏的把握程度)
- 双阈值滞回: P>0.85 触发 (高置信), P<0.4 释放 (安全复位)
- 冷却期防连击, 不依赖滑动窗口平滑

### 支持架构
- 新 WeightedInceptionTime (自动检测 state_dict key 含 "backbone")
- 旧 tsai InceptionTime (key 含 "inceptionblock")
- FCN (model_name="fcn")
- LiteCNN, TST

## 延迟分析
- 总延迟: 踩踏→蜂鸣 ~114-244ms
- 物理延迟 (脚→手机): ~143ms (中位), 波动 ±62ms ← 主导
- WiFi 传输: ~40ms, 抖动仅 1.2ms ← 可忽略
- 模型推理: 2.6ms CPU / 2.4ms GPU (P99: 3.5ms)
- 检测窗口: avg_4 ≈ 67ms
- 部署手机后: 减掉 WiFi 40ms, 延迟 ~100-200ms

## 踩踏 vs 鼓棒的区别
- 鼓棒: 强冲击, gyro 为主, 峰值在点击后 5-20ms, SNR > 10x
- 踩踏: 低频晃动, accel 为主, 峰值在点击后 40-170ms, SNR 1-8x
- latency_compare 的 onset 阈值检测对踩踏无效，只当录制工具用

## 常用命令

### 训练
```bash
# 标准 CE
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --save_dir models_margin

# Focal Loss
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --focal-loss 2.0 --save_dir models_margin
```

### 实时检测
```bash
# 校准
python realtime_detect.py --model models_margin/hit_detector_inception.pth --calibrate

# 检测
python realtime_detect.py --model models_margin/hit_detector_inception.pth -v --prob-high 0.85 --prob-low 0.4

# 调试 OSC 数据
python realtime_detect.py --debug
```

### 录制数据
```bash
python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/w_X
```

## 已弃用/删除
- export 的自动切窗口功能 (extract_pending, save(X.npy))
- IMU SNR 过滤器 (filter_clicks_by_imu)
- merged 功能 (旧窗口格式)
- latency_compare 的 onset 阈值调试 — 对踩踏无意义
- tsai 默认 InceptionTime (ks=[39,19,9], GAP, 456K)
- 窗口级随机训练/验证分割 (跨 session 泄露)
- softmax 概率 + 连续命中 触发策略 → 改为 logit margin + 滑动平均
- logit margin + 滑动平均 → 改为纯概率双阈值滞回 (P>0.85 触发 / P<0.4 释放)
- 能量积分 (Σ max(0, P-0.5)) 策略

## Arduino
- data_collection_leg/data_collection_leg.ino — 文本协议, 只激活 A1
- data_collection_fast/data_collection_fast.ino — 二进制协议 (未在用)
- 格式: "Sensor 0 FALLING count=1 raw=45 delta=120 min=42"
