# ai-ring

AI-Ring project，基于 `PyQt6 + qasync + bleak`，用于：

- 扫描并连接智能戒指设备
- 进行设备时间同步（校准）
- 录制 IMU 数据

## TODO

- 同步录制音频数据
- 添加应用内单独页面单独进程以实现不同功能

## 环境要求

- Python `3.12+`
- 支持 BLE 的系统与硬件（建议 macOS / Linux / Windows 桌面环境）
- 蓝牙权限已授予（尤其是 macOS）

## 安装

本项目使用 `pyproject.toml` 管理依赖，推荐用 `uv`。

```bash
# 安装依赖（会根据 uv.lock 解析）
uv sync
```

## 运行

### 使用 uv 运行

```bash
uv run python main.py
```

启动后可在界面中：
1. 点击“开始扫描”发现设备
2. 对单个设备执行连接 / 同步 / 开始录制
3. 关闭窗口时会自动尝试停止录制并断开设备

## 数据输出

程序会在项目根目录自动创建 `records/`：

- IMU 录制数据：`records/<设备名>_<时间戳>.npy`
  - 数据 shape 为 `N x 7`
  - 列顺序：`acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp`
- 时间同步校准数据：`records/calib_time_<时间戳>.json`

## 开发指南

### 目录结构

```text
.
├── main.py                # GUI 入口（扫描、连接、同步、录制）
├── core/
│   └── ble_ring_v2.py     # BLE 协议与 IMU 数据处理
├── pyproject.toml         # 项目与依赖配置
└── uv.lock                # 锁定依赖版本
```

### 开发运行

```bash
uv run python main.py
```

### 常见开发操作

```bash
# 新增依赖（示例）
uv add <package>

# 锁文件更新
uv lock

# 同步环境
uv sync
```

### 调试建议

- 蓝牙扫描不到设备时，先确认设备已开机且广播名称为 `BCL*`
- macOS 下若首次运行失败，检查系统“隐私与安全性”中的蓝牙权限
- 录制文件较大时，优先检查磁盘空间（录制使用 memmap 临时文件再转为 `.npy`）

## 已知限制

- 当前无自动化测试配置（未提供 `tests/`）
- UI 字体默认使用 `Microsoft YaHei`，在部分系统可能回退到其他字体

## License

暂未声明。
