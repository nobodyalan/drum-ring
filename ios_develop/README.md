# iOS 鼓击检测 App — 操作指南

## 目录结构

```
ios_develop/
  ├── convert_to_coreml.py          # ① PC 端运行：PyTorch → CoreML 转换脚本
  ├── HitDetector.mlpackage         # ② 脚本生成的 CoreML 模型（拖入 Xcode）
  ├── DrumHitApp/                   # ③ iOS App 源码
  │   ├── DrumHitApp.swift          // @main 入口
  │   ├── ContentView.swift         // UI + ViewModel（串联整个管线）
  │   ├── SensorManager.swift       // CMMotionManager 封装
  │   ├── InferenceEngine.swift     // 缓冲 + Z-score 标准化 + CoreML 推理
  │   ├── DetectionStateMachine.swift // 双阈值滞回 + 冷却
  │   ├── AudioEngine.swift         // AVAudioEngine 音效播放
  │   ├── RingBuffer.swift          // 泛型环形缓冲
  │   ├── preproc_params.json       // ① 脚本生成的标准化参数
  │   ├── hit.wav                   // 触发音效
  │   └── Info.plist                // 权限描述
  └── README.md                     // 本文件
```

## 操作步骤（共 6 步）

### 步骤 1：PC 端转换模型（只需跑一次）

```bash
# 在 drum-ring/ 根目录下
cd ios_develop

# 安装依赖（如果还没有）
pip install torch coremltools numpy

# 运行转换
python convert_to_coreml.py
```

转换成功后：
- 生成 `ios_develop/HitDetector.mlpackage` — CoreML 模型
- 更新 `ios_develop/DrumHitApp/preproc_params.json` — 标准化参数

### 步骤 2：新建 Xcode 项目

1. 打开 Xcode → File → New → Project
2. 选择 **iOS → App**
3. 配置：
   - Product Name: `DrumHit`
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Minimum Deployment: **iOS 16.0**
4. 保存到任意位置

### 步骤 3：导入文件到 Xcode

把以下文件**拖入** Xcode 项目（拖到 Project Navigator 的 DrumHit 文件夹下）：

**所有 .swift 文件**：
- `DrumHitApp.swift`
- `ContentView.swift`
- `SensorManager.swift`
- `InferenceEngine.swift`
- `DetectionStateMachine.swift`
- `AudioEngine.swift`
- `RingBuffer.swift`

**资源文件**：
- `HitDetector.mlpackage` — CoreML 模型
- `preproc_params.json` — 标准化参数
- `hit.wav` — 触发音效

> ⚠️ 拖入时勾选 "Copy items if needed" 和 target "DrumHit"

**关于 Info.plist**：
Xcode 新建项目会自动生成 Info.plist。你需要手动添加运动传感器权限——在 Xcode 中打开 Info.plist，添加一行：
- Key: `Privacy - Motion Usage Description`
- Value: `使用运动传感器检测鼓击动作`

### 步骤 4：删除 Xcode 自动生成的默认文件

Xcode 模板会生成默认的 `ContentView.swift` 和 `DrumHitApp.swift`。因为我们已导入同名文件，选择 **Replace**（替换）。

### 步骤 5：连 iPhone 真机测试

1. 用数据线连接 iPhone
2. Xcode 顶部选择你的 iPhone（不是模拟器——模拟器没有真实 IMU）
3. **重要**：在 iPhone 上信任开发者证书（设置 → 通用 → VPN与设备管理）
4. 按 `⌘R` Build & Run
5. 把 iPhone 放裤兜，踩踏测试

### 步骤 6：调优参数

如果检测效果不理想，在 `ContentView.swift` 的 `DrumHitViewModel` 中调整：

```swift
let probHigh: Float = 0.85    // 调低 → 更容易触发（但可能误触发更多）
let probLow: Float = 0.40     // 调高 → 滞回释放更快
let cooldownMs: Int = 400     // 调小 → 连击检测更灵敏
```

## App 工作原理

```
iPhone IMU (60Hz)
    ↓
SensorManager — CMMotionManager.deviceMotion 每 16.7ms 回调一次
    ↓
InferenceEngine — 15 帧滑动窗口 → Z-score 标准化 → CoreML 推理
    ↓
DetectionStateMachine — P(hit) > 0.85 触发 → 进入 ARMED 状态
                     — P(hit) < 0.40 释放 → 回到就绪
                     — 触发后 400ms 冷却，防止连击
    ↓
AudioEngine — AVAudioPlayerNode 播放 hit.wav
    ↓
ContentView — UI 更新（计数 + 概率条）
```

## 文件对应关系

| iOS 文件 | 参考的 Python 代码 |
|---------|-------------------|
| `SensorManager.swift` | 替代 `realtime_detect.py:666-710` (OSC 监听) |
| `InferenceEngine.swift` | `realtime_detect.py:338-358` (predict 方法) |
| `DetectionStateMachine.swift` | `realtime_detect.py:405-420` (双阈值滞回) |
| `AudioEngine.swift` | `realtime_detect.py:268-279` (winsound → AVAudioEngine) |
| `convert_to_coreml.py` | `train_model.py:302-379` (重建架构 + 固化权重) |
| `RingBuffer.swift` | `collections.deque(maxlen=N)` |

## 常见问题

**Q: convert_to_coreml.py 报错 "No module named 'coremltools'"**
```bash
pip install coremltools
```

**Q: Xcode 找不到 HitDetector，报 "Cannot find 'HitDetector' in scope"**
确认 `HitDetector.mlpackage` 已拖入项目，且 Target Membership 勾选了 DrumHit。

**Q: 运行后一直显示 P(hit)=0.000**
检查 iPhone 是否平放不动。模型只在检测到类似踩踏的信号时才会输出高概率。尝试晃动手机看概率是否变化。

**Q: 模拟器能跑吗？**
模拟器没有真实 IMU 硬件，CMMotionManager 只返回零值。必须真机测试。
