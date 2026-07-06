# 鼓击检测项目 — 实时检测 & 模型训练

## 实时检测程序 (项目根目录)

### realtime_mag_jerk.py (当前主力)
- **信号**: X 轴加速度 `ax` (正=向上提腿, 负=向下踩踏)
- **检测**: STA/LTA ratio (用 `abs(ax)` 计算) + jerk 双向快速通道
- **预判**: 向上提腿时 `jerk > +jerk_thresh` 即触发, 比踩踏早 50-150ms
- **确认**: 向下踩踏时 `jerk < -jerk_down_thresh` 也触发
- **默认**: ratio=2.0, mag_high=0.6, mag_low=0.15, jerk_thresh=2.0, jerk_down_thresh=2.0, cooldown=120ms

### realtime_simple.py
- **信号**: 合加速度 `√(ax²+ay²+az²)`
- **检测**: STA/LTA ratio + jerk 单向快速通道
- **默认**: ratio=2.5, mag_high=0.8, mag_low=0.3, jerk_thresh=0.35, fast_ratio=1.2, cooldown=150ms

### measure_latency.py
- **功能**: 双手机同时监听 (端口 8888 + 8889), 测量物理延迟
- **原理**: 同 WiFi 下两路 OSC 延迟相减, 消去网络抖动
- **配对**: foot 先触发 → pocket 后触发 → 计算时间差
- **用法**: `python measure_latency.py --plot`

## 模型训练 (model_train/)

详见 model_train/ 目录下的 CLAUDE.md 和 PROGRAMS_REFERENCE.md。

### 核心文件
- `train_model.py` — 训练 WeightedInceptionTime (89K 参数)
- `realtime_detect.py` — 模型推理实时检测 (softmax 概率双阈值)
- `export_training_data.py` — 录制训练数据 (Arduino + iPhone)

## 硬件连接
- Arduino + 压电传感器 → USB 串口 → PC
- iPhone ZigSim → OSC UDP → PC (端口 8888)
- 传感器贴脚底/踩踏面，手机放裤兜

## 常用命令

```bash
# X轴预判检测
python realtime_mag_jerk.py -v
python realtime_mag_jerk.py --plot --jerk-thresh 0.8 --jerk-down-thresh 1.5

# 合加速度检测
python realtime_simple.py -v --ratio-thresh 2.5 --mag-high 0.8

# 延迟测量
python measure_latency.py --plot

# 录制数据 (model_train/)
python model_train/export_training_data.py --port COM6 --wifi --strength weak

# 训练模型 (model_train/)
python model_train/train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0

# 实时检测 (模型) (model_train/)
python model_train/realtime_detect.py --model models_margin/hit_detector_inception.pth -v
```

## 踩踏 vs 鼓棒

- 鼓棒: 强冲击, gyro 为主, 峰值在点击后 5-20ms, SNR > 10x
- 踩踏: 低频晃动, accel 为主, 峰值在点击后 40-170ms, SNR 1-8x
- latency_compare 的 onset 阈值检测对踩踏无效，只当录制工具用
