# ai-ring

基于 `PyQt6 + qasync + bleak` 的智能戒指桌面管理工具，支持：

- BLE 扫描与连接（`BCL*` 设备）
- 时间同步校准
- IMU 录制
- 独立应用进程订阅 IMU 流并处理/可视化

## 功能概览

- 主程序：设备扫描、连接、同步、录制、状态显示
- 数据流服务：主程序内置 JSONL over TCP 发布服务
- 应用系统：点击主界面“应用”按钮启动独立进程
- 示例应用：
  - `apps/demo_processor_app.py`（处理模板）
  - `apps/imu_visualizer_app.py`（多戒指高性能可视化）

## 环境要求

- Python `3.12+`
- 支持 BLE 的系统与硬件（macOS / Linux / Windows）
- 系统蓝牙权限已授权

## 安装

推荐使用 `uv`：

```bash
uv sync
```

## 运行

```bash
uv run python main.py
```

启动后你可以：

1. 点击“开始扫描”发现设备
2. 对单个设备执行连接 / 同步 / 开始录制
3. 在主界面底部“应用”栏启动独立应用
4. 在顶部 `Port` 输入框切换应用数据流端口（运行中可切换）

## 数据输出

程序会在项目根目录创建 `records/`：

- IMU 数据：`records/<设备名>_<时间戳>.npy`
  - shape: `N x 7`
  - 列顺序：`acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp`
- 时间同步数据：`records/calib_time_<时间戳>.json`

## 目录结构

```text
.
├── main.py                         # 主 GUI（扫描/连接/同步/录制/启动应用）
├── core/
│   ├── ble_ring_v2.py              # BLE 协议、设备连接、数据解析
│   └── stream_hub.py               # 应用流服务（JSONL over TCP）
├── apps/
│   ├── demo_processor_app.py       # 示例应用：数据处理模板
│   ├── imu_visualizer_app.py       # 示例应用：IMU 可视化
│   └── common/
│       ├── __init__.py
│       └── stream_client.py        # 应用侧通用流客户端
├── pyproject.toml
└── uv.lock
```

## 应用开发指南

这一节是开发新独立应用（独立页面 + 独立进程）的标准方式。

### 1. 应用运行模型

- 每个应用是单独 Python 进程
- 主程序负责采集戒指 IMU 等数据并推送到流服务
- 应用通过 `host/port` 连接并订阅指定戒指地址
- 主程序关闭不强制杀掉应用进程，应用自行处理重连/退出

### 2. 数据流协议（JSONL over TCP）

- 传输格式：每条消息一行 JSON（`\n` 分隔）
- 默认地址：`127.0.0.1:17888`（可在主界面改 port）

服务端下行消息：

```json
{"type":"hello","version":1}
{"type":"status","rings":[{"address":"...","name":"BCL-01","connected":true}]}
{"type":"imu","address":"...","name":"BCL-01","timestamp":1710000000.0,"acc":[0.1,0.2,0.3],"gyr":[1.0,2.0,3.0]}
{"type":"subscribed","rings":["addr1","addr2"]}
{"type":"error","message":"unsupported_command"}
```

应用上行指令：

```json
{"cmd":"status"}
{"cmd":"subscribe","rings":["addr1","addr2"]}
```

说明：

- `rings` 为空列表时默认接收全部
- 当前可视化应用为了“未订阅不显示”，会发送占位值避免接收全部

### 3. 推荐开发步骤

1. 复制 `apps/demo_processor_app.py` 作为新应用起点
2. 复用 `apps/common/stream_client.py` 处理连接、重连、消息分发
3. 在应用内实现：
   - 设备列表渲染（消费 `status`）
   - 订阅逻辑（发送 `subscribe`）
   - 数据处理逻辑（消费 `imu`）
4. 在 `main.py` 的应用栏新增启动按钮

### 4. 最小应用模板

```python
from common.stream_client import JsonLineStreamClient

client = JsonLineStreamClient(host="127.0.0.1", port=17888)
client.sig_connected.connect(lambda ok: client.send_command({"cmd": "status"}) if ok else None)
client.sig_message.connect(on_message)
client.start()

# 订阅指定戒指
client.send_command({"cmd": "subscribe", "rings": ["ring_addr"]})
```

### 5. 在主界面接入新应用

在 `main.py` 中复用现有 `launch_app(...)` 方式新增按钮：

- 启动参数统一传入 `--host` 和 `--port`
- 确保脚本支持命令行参数：

```bash
uv run python apps/your_app.py --host 127.0.0.1 --port 17888
```

### 6. 应用开发建议

- 不在应用里直接连 BLE；BLE 统一由主程序管理
- 处理高频流时使用环形缓冲和定时重绘
- 订阅变更时及时清理无效状态
- UI 与处理解耦：网络线程只负责收包，UI 线程做展示

## 开发常用命令

```bash
# 运行主程序
uv run python main.py

# 运行示例应用（需主程序已启动）
uv run python apps/demo_processor_app.py --host 127.0.0.1 --port 17888
uv run python apps/imu_visualizer_app.py --host 127.0.0.1 --port 17888

# 检查语法
python -m compileall main.py core apps
```

## 调试建议

- 扫描不到设备：确认蓝牙权限已授权且设备有电
- 应用收不到数据：检查应用连接 `port` 是否与主界面一致
- 卡顿：减小窗口时间、提高下采样、检查 CPU 占用
- 录制文件大：关注磁盘空间（录制使用 memmap 临时文件）

## 已知限制

- 当前尚未配置自动化测试（无 `tests/`）
- 应用进程生命周期目前由用户手动管理

## License

暂未声明。
