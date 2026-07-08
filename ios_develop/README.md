# iOS 鼓击检测 App — 操作指南

## 目录结构

```
ios_develop/
  ├── convert_to_coreml.py          # ① PC 端运行：PyTorch → CoreML 转换脚本
  ├── HitDetector.mlmodel           # ② 脚本生成的 CoreML 模型（neuralnetwork 格式）
  ├── Generated/                    # ③ CI 自动生成目录（本地为空，.gitkeep 占位）
  │   └── .gitkeep
  ├── DrumHitApp/                   # ④ iOS App 源码
  │   ├── DrumHitApp.swift          // @main 入口
  │   ├── ContentView.swift         // UI + ViewModel（串联整个管线）
  │   ├── SensorManager.swift       // CMMotionManager 封装 (60Hz IMU)
  │   ├── InferenceEngine.swift     // 缓冲 + Z-score 标准化 + CoreML 推理
  │   ├── DetectionStateMachine.swift // 双阈值滞回 + 冷却
  │   ├── AudioEngine.swift         // AVAudioEngine 音效播放
  │   ├── RingBuffer.swift          // 泛型环形缓冲
  │   ├── preproc_params.json       // ① 脚本生成的标准化参数
  │   ├── Info.plist                // 权限描述 + 必填字段
  │   └── README.md                 // 本文件
  └── README.md
```

## 方式一：CI 自动编译 + AltStore 安装（推荐，无需 Mac/Xcode）

### CI 流程（6 步自动化）

1. **触发 GitHub Actions** → Actions → "Build iOS App" → Run workflow
2. **编译 CoreML 模型** → `xcrun coremlc generate` 生成 Swift 类 + `coremlc compile` 编译 `.mlmodelc`
3. **xcodegen 生成 Xcode 项目** + **xcodebuild 编译** (不签名)
4. **手动复制资源** → `1.wav` + `preproc_params.json` + `.mlmodelc` 进 `.app`
5. **ad-hoc 签名** → `codesign --sign -`
6. **打包 IPA** → 上传到 GitHub Artifacts（保留 30 天）

### 安装步骤

1. 下载 CI 产出的 `DrumHit-XX.zip`，解压得到 `DrumHit.ipa`
2. 传到 iPhone（AirDrop / 微信 / 邮件）
3. iPhone 打开 **AltStore** → My Apps → **+** → 选择 `DrumHit.ipa`
4. 首次打开需信任证书：设置 → 通用 → VPN 与设备管理
5. **⚠️ 每 7 天在 AltStore 中 Refresh 续签**（iPhone 连电脑 AltServer 自动刷）

### 关键设计说明

- **不需要 Apple 开发者账号**：CI 用 ad-hoc 签名编译 .app，AltStore 安装时用免费 Apple ID 重签
- **不需要 Mac/Xcode**：所有编译在 GitHub 免费 macOS runner 上完成
- **绕过 Xcode 26 限制**：
  - Xcode 26.5 禁止 `CODE_SIGN_IDENTITY="-"`（ad-hoc）→ CI 改为 `codesign --sign -` 后补签
  - Xcode 26.5 不再编译旧 `.mlmodel`（neuralnetwork）格式 → CI 手动 `coremlc generate` + `compile`
  - `CODE_SIGNING_ALLOWED=NO` 导致 xcodebuild 跳过资源复制 → CI 手动 `cp` 资源文件
- **Info.plist 必须包含所有必填字段**：`CFBundleExecutable`、`CFBundleIdentifier`、`CFBundleVersion`、`CFBundleShortVersionString`、`CFBundlePackageType`、`CFBundleName`。Xcode 只**展开**已有的 `$(...)` 变量，不会自动注入缺失的 key。缺失必填字段 → AltStore 报 "the app is in an invalid format"

## 方式二：本地 Xcode 开发（需要 Mac）

### 步骤 1：PC 端转换模型（只需跑一次）

```bash
cd ios_develop
pip install torch coremltools numpy
python convert_to_coreml.py
```

成功后生成 `HitDetector.mlmodel` 和 `preproc_params.json`。

### 步骤 2-6（同原有步骤 2-6，见下方）

2. Xcode 新建 iOS App 项目（SwiftUI, iOS 16.0+）
3. 拖入所有 `.swift` 文件 + `HitDetector.mlmodel` + `preproc_params.json` + `1.wav`
4. Info.plist 添加 `Privacy - Motion Usage Description`
5. 替换默认文件
6. 连 iPhone 真机 `⌘R` 运行（模拟器无真实 IMU）

## App 工作原理

```
iPhone IMU (60Hz)
    ↓
SensorManager — CMMotionManager.deviceMotion 每 16.7ms 回调一次
    ↓
InferenceEngine — 15 帧滑动窗口 → Z-score 标准化 → CoreML 推理
    ↓
DetectionStateMachine — P(hit) > 0.85 触发 → ARMED → 400ms 冷却
    ↓
AudioEngine — AVAudioPlayerNode 播放 1.wav
    ↓
ContentView — UI 更新（计数 + 概率条）
```

## 调优参数

在 `ContentView.swift` 的 `DrumHitViewModel` 中：

```swift
let probHigh: Float = 0.85    // 调低 → 更易触发
let probLow: Float = 0.40     // 调高 → 滞回释放更快
let cooldownMs: Int = 400     // 调小 → 连击更灵敏
```

## 常见问题

**Q: AltStore 安装报 "the app is in an invalid format"**
确认 Info.plist 包含所有必填字段（见"关键设计说明"）

**Q: App 安装后闪退**
- 检查 `.mlmodelc`、`preproc_params.json`、`1.wav` 是否在 `.app` bundle 中
- Xcode 26.5 生成 Swift 接口中 `init()` 可能已于 iOS 26 废弃，需改用 `init(configuration:)`

**Q: 模拟器能跑吗？**
模拟器无真实 IMU，必须真机测试

**Q: CI 编译失败**
查看 "Compile CoreML Model" 步骤输出，确认 `HitDetector.mlmodel` 文件存在且能被 `coremlc` 读取
