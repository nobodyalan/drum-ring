---
name: ios-ci-altstore-pipeline
description: iOS CI build + AltStore install pipeline — Xcode 26 workarounds, CoreML compilation, code signing, AltStore compatibility
metadata:
  type: project
---

## CI Pipeline Overview

GitHub Actions (`build-ios.yml`) on `macos-latest` compiles DrumHit iOS app without Apple ID, producing unsigned IPA for AltStore side-loading.

## Xcode 26.5 Workarounds (Three Critical Issues)

### 1. Code Signing
- **Problem**: Xcode 26.5 SDK blocks `CODE_SIGN_IDENTITY="-"` (ad-hoc) → error "Ad Hoc code signing is not allowed with SDK 'iOS 26.5'"
- **Solution**: Build with `CODE_SIGN_IDENTITY="" CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO`, then use `codesign --force --sign - --timestamp=none` after build

### 2. CoreML Model Compilation
- **Problem**: Xcode 26.5 no longer auto-compiles old `.mlmodel` files in `neuralnetwork` format (frozen/deprecated). Build log shows no CoreML compilation step.
- **Solution**: Pre-compile in CI using `xcrun coremlc generate` (Swift source) + `xcrun coremlc compile` (`.mlmodelc`), place in `ios_develop/Generated/`. Note: `coremlcompiler` does NOT exist — correct tool is `coremlc`.

### 3. Bundle Resources Not Copied
- **Problem**: `CODE_SIGNING_ALLOWED=NO` causes xcodebuild to skip "Copy Bundle Resources" phase. `.app` contains only executable + Info.plist + PkgInfo.
- **Solution**: Manually copy resources after build: `1.wav`, `preproc_params.json`, `HitDetector.mlmodelc`

### 4. Info.plist Missing Required Keys
- **Problem**: Source `Info.plist` only had `NSMotionUsageDescription` + `UIApplicationSceneManifest`. Xcode's `-expandbuildsettings` only expands existing `$(...)` variables, does NOT inject missing keys. AltStore rejects IPA with "the app is in an invalid format" if `CFBundleExecutable`, `CFBundleIdentifier` etc. are absent.
- **Solution**: Source `Info.plist` must explicitly contain ALL required keys with `$(...)` variable placeholders: `CFBundleExecutable=$(EXECUTABLE_NAME)`, `CFBundleIdentifier=$(PRODUCT_BUNDLE_IDENTIFIER)`, `CFBundleVersion=$(CURRENT_PROJECT_VERSION)`, `CFBundleShortVersionString=$(MARKETING_VERSION)`, `CFBundlePackageType=APPL`, `CFBundleName=$(PRODUCT_NAME)`, `CFBundleInfoDictionaryVersion=6.0`

## AltStore Compatibility Checklist
- `.app` must have `_CodeSignature/` directory (ad-hoc signing creates this)
- Info.plist must have all required keys (see #4 above)
- IPA structure: `Payload/DrumHit.app/...` (standard zip)
- AltStore refreshes every 7 days via AltServer on companion computer

## Key Files
- `.github/workflows/build-ios.yml` — CI pipeline
- `project.yml` — xcodegen project spec
- `ios_develop/DrumHitApp/Info.plist` — must contain all required keys
- `ios_develop/HitDetector.mlmodel` — old neuralnetwork format CoreML model (364KB)
- `ios_develop/Generated/` — CI-populated directory (`.gitkeep` in repo, `.gitignore` excludes content)
- `1.wav` — audio file at repo root
- `ios_develop/DrumHitApp/AudioEngine.swift` — looks for "1.wav" via `Bundle.main.url(forResource: "1", withExtension: "wav")`

## Model Conversion
`convert_to_coreml.py` uses `convert_to="neuralnetwork"` with `minimum_deployment_target=ct.target.iOS13`, outputting old single-file `.mlmodel` format. Should eventually switch to `convert_to="mlprogram"` with `.mlpackage` for Xcode 26 compatibility. [[ios-mlmodel-upgrade]]

## AltStore Install Steps for End Users
1. GitHub Actions → Build iOS App → Run workflow → download artifact
2. Unzip → transfer `DrumHit.ipa` to iPhone
3. AltStore → + → select IPA
4. Trust certificate: Settings → General → VPN & Device Management
5. Refresh every 7 days
