# IMU Streamer — iOS IMU → OSC 实时发送器

类似 ZigSim 的 iOS App，以最高 200Hz 将传感器数据通过 UDP OSC 发送至电脑。

## 发送的数据

| OSC 地址 | 通道数 | 来源 | 单位 |
|----------|--------|------|------|
| `/gyro` | 3 | `CMDeviceMotion.rotationRate` | rad/s |
| `/accel` | 3 | `CMDeviceMotion.userAcceleration` | g (已去重力) |
| `/quaternion` | 4 | `CMDeviceMotion.attitude.quaternion` | — |
| `/gravity` | 3 | `CMDeviceMotion.gravity` | g |

OSC Bundle 格式与 ZigSim 完全兼容，可直接被以下程序接收：
- `python model_train/debug_osc.py --port 8888`
- `python realtime_mag_jerk.py --port 8888 -v`
- `python realtime_simple.py --port 8888 -v`
- `python measure_latency.py`

## 性能

- 目标: 200Hz (真机实测 iPhone 12+ 可达 ~190-200Hz)
- 每包大小: ~156 bytes
- 带宽: ~31 KB/s @ 200Hz
- UDP 编码耗时: < 10 µs (微秒级)
- 后台线程运行，不阻塞 UI

## 应用到 iPhone — 方式一：Xcode 直接编译 (需要 Mac)

### 1. 添加为新 Target

在 `drum-ring/ios_develop/` 下用 XcodeGen 生成项目：

```bash
cd ios_develop
xcodegen generate
```

这会生成 `DrumHit.xcodeproj`，其中包含两个 Target：
- **DrumHit** — 原有鼓击检测 App
- **IMUStreamer** — 新增 IMU 发送器

然后在 Xcode 中选 IMUStreamer scheme，连 iPhone 真机 `⌘R` 运行。

### 2. 或者手动添加到现有 Xcode 项目

如果你已有自己的 Xcode 项目：

1. 把 `ios_develop/IMUStreamer/` 下所有 `.swift` 文件拖入项目
2. 把 `Info.plist` 内容合并到你的 Info.plist
3. 确保 Target 的 Deployment Target ≥ iOS 16.0
4. 连 iPhone 真机 `⌘R` 运行（模拟器无真实 IMU）

### 3. 首次运行

iPhone 会弹出两个权限请求：
- **运动与健身** → 允许（传感器数据）
- **本地网络** → 允许（UDP 发送）

都点"允许"后即可正常使用。

## 应用到 iPhone — 方式二：CI + AltStore (无需 Mac)

参照 DrumHit 的 CI 流程，在 GitHub Actions 中新增 IMUStreamer Target 的构建。
修改 `.github/workflows/` 中的 build 脚本，对 IMUStreamer Target 执行编译。

## PC 端接收验证

App 启动后，在 PC 上运行：

```bash
# 验证数据到达
python model_train/debug_osc.py --port 8888

# 或直接接入实时检测
python realtime_mag_jerk.py --port 8888 -v
```

## 文件说明

```
IMUStreamer/
├── StreamerApp.swift      # @main 入口
├── ContentView.swift       # SwiftUI 界面 + ViewModel
├── SensorManager.swift     # CMMotionManager 封装 (200Hz)
├── OSCEncoder.swift        # ZigSim 兼容的 OSC Bundle 编码
├── OSCSender.swift         # POSIX UDP 发送器
├── Info.plist              # 应用元数据 + 权限描述
└── README.md               # 本文件
```
