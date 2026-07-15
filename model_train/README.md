# model_train — 鼓击检测模型训练 & 分析

## ZigSim 热点连接

**核心**: ZigSim 始终向 PC 发送数据, 所以 ZigSim 中填的 IP 永远是 **PC 的 IP**。
PC 端程序 `realtime_detect.py` 绑定 `0.0.0.0:8888` (监听所有网卡), 无需指定 IP。

### 方法 1: PC 开热点 (推荐, IP 固定)

```
PC:  设置 → 网络和 Internet → 移动热点 → 开启
iPhone: WiFi 连接 PC 的热点

PC IP: 192.168.137.1 (Windows 默认, 基本不变)
```

ZigSim 设置:
```
IP:   192.168.137.1
Port: 8888
```

### 方法 2: iPhone 开热点

```
iPhone: 设置 → 个人热点 → 开启
PC:     WiFi 连接 iPhone 的热点
```

连接后, **在 PC 上查看本机 IP**:

```powershell
# PowerShell / CMD
ipconfig

# 找到连接 iPhone 热点的网卡, 查看 IPv4 地址, 如:
# 172.20.10.2 或 172.20.10.3
```

把查到的 PC IP 填入 ZigSim:
```
IP:   172.20.10.2    (填 PC 的 IP, 不是 iPhone 的)
Port: 8888
```

---

## 工具链

| 程序 | 用途 |
|------|------|
| `export_training_data.py` | 录制训练数据 (Arduino + iPhone) |
| `filter_close_clicks.py` | 清理踩踏标注 (过近/异常偏移/手动删除) |
| `slice_dataset.py` | IMU 数据切片 → 训练样本 |
| `train_model.py` | 训练 InceptionTime 模型 |
| `realtime_detect.py` | 模型实时推理 + 踩踏检测 |
| `analyze_badcases.py` | 模型 Badcase 分析 + 可视化 |
| `visualize_session.py` | 可视化单个 session 的 IMU 波形 |
| `latency_compare.py` | Arduino vs iPhone 延迟对比 (实时绘图) |
| `debug_osc.py` | ZigSim OSC 数据诊断 (查看所有发送的地址) |

---

## 工作流程

### 第 1 步: 确认 ZigSim 连接 (debug_osc)

```bash
# 确保 iPhone ZigSim 正在发送, PC 能收到
python debug_osc.py

# 指定端口和自动停止时间
python debug_osc.py --port 8888 --duration 10
```

输出:
```
✅ 监听 UDP 0.0.0.0:8888
等待 ZigSim 数据... (按 Ctrl+C 停止)

  🆕 [/gyro] → 3 个值, 样例: [0.001, -0.003, 0.002]
  🆕 [/accel] → 3 个值, 样例: [0.01, -0.98, 0.05]
  🆕 [/quaternion] → 4 个值, 样例: [0.5, 0.5, 0.5, 0.5]
  🆕 [/gravity] → 3 个值, 样例: [0.0, -1.0, 0.0]
  📦 包 #1: 4 条消息, 地址: ['/gyro', '/accel', '/quaternion', '/gravity']

汇总: 共收到 600 个 OSC 包
发现 4 种 OSC 地址:
  /accel           | 3 个值 | 出现 600 次 | 样例: [0.01, -0.98, 0.05]
  /gravity         | 3 个值 | 出现 600 次 | 样例: [0.0, -1.0, 0.0]
  /gyro            | 3 个值 | 出现 600 次 | 样例: [0.001, -0.003, 0.002]
  /quaternion      | 4 个值 | 出现 600 次 | 样例: [0.5, 0.5, 0.5, 0.5]

📊 通道扩展建议:
  基础: gyro(3) + accel(3) = 6 通道
  ✅ 四元数 (4) → +4 通道, 总计 10
  ✅ 重力向量 (3) → +3 通道, 总计 13
  衍生特征 (从现有6轴计算): accel_mag(1) + gyro_mag(1) + jerk(1) = +3
  最大可用: 16 通道
```

### 第 2 步: 测量延迟 (latency_compare)

```bash
# 同时连接 Arduino + iPhone, 实时对比延迟 (弹窗)
python latency_compare.py --port COM6 --osc-port 8888

# 纯控制台模式 (无 GUI)
python latency_compare.py --port COM6 --osc-port 8888 --no-plot

# 只看 IMU, 不连 Arduino
python latency_compare.py --osc-port 8888 --no-arduino

# 调灵敏度
python latency_compare.py --port COM6 --gyro-thresh 2.5 --accel-thresh 0.5

# 保存延迟数据
python latency_compare.py --port COM6 --osc-port 8888 --save delay.csv
```

图表输出:
```
┌──────────────────────────────────────────────┐
│ 加速度计 [g]: 红/绿/蓝线 = X/Y/Z            │
│ 红色竖线 = Arduino 检测到敲击               │
├──────────────────────────────────────────────┤
│ 陀螺仪 [rad/s]: 红/绿/蓝线 = X/Y/Z          │
│ 红色竖线 = Arduino 敲击                     │
│ 蓝色虚线 = IMU onset 检测                   │
├──────────────────────────────────────────────┤
│ 延迟测量 [ms]: 散点 = 每次测量, 红线 = 移动平均│
│ 绿线=50ms, 橙线=100ms, 红线=200ms           │
└──────────────────────────────────────────────┘
```

控制台输出:
```
  🥁 Arduino 敲击 #1 @ t=1700000123.456
  ⏱ 延迟测量: 85.3ms (Arduino→IMU onset, 敲击#1)
  最近5次延迟: ['82ms', '88ms', '79ms', '91ms', '85ms']
```

### 第 3 步: 录制 & 训练

```bash
# 录制
python export_training_data.py --port COM6 --wifi

# 预览标注
python visualize_session.py w_0 --data-dir train_data

# 清理异常标注
python filter_close_clicks.py --data_dir train_data --list-clicks slow_0
python filter_close_clicks.py --data_dir train_data --remove-low-shift 20 --apply

# 切片
python slice_dataset.py --data_dir train_data --save sliced_data --pre 120 --post 30

# 训练
python train_model.py --data_dir sliced_data --save_dir models \
    --neg-sessions neg_b_1,neg_wxt_0 --no-augment

# 分析
python analyze_badcases.py --model models/hit_detector_inception.pth \
    --data_dir sliced_data --save badcase_analysis

# 实时检测
python realtime_detect.py --model models/hit_detector_inception.pth -v
```
