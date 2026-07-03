# 鼓击检测项目 — 程序使用手册

> 生成日期: 2026-07-03 | 共 6 个 Python 程序 + 1 个 Arduino 固件

---

## 目录

1. [数据流概览](#1-数据流概览)
2. [export_training_data.py — 训练数据采集](#2-export_training_datapy--训练数据采集)
3. [slice_dataset.py — 滑动窗口切片](#3-slice_datasetpy--滑动窗口切片)
4. [train_model.py — 模型训练](#4-train_modelpy--模型训练)
5. [realtime_detect.py — 实时踩踏检测](#5-realtime_detectpy--实时踩踏检测)
6. [latency_compare.py — 延迟对比工具](#6-latency_comparepy--延迟对比工具)
7. [visualize_session.py — 录制数据可视化](#7-visualize_sessionpy--录制数据可视化)
8. [Arduino 固件](#8-arduino-固件)

---

## 1. 数据流概览

```
录制:  export_training_data.py  (纯录制全量数据)
          ↓
      train_data/w_*/  (imu + click_timestamps)
          ↓
切片:  slice_dataset.py / train_model.py 内置 (滑动窗口 + 标签生成)
          ↓
      (内部: N×6×15 数组 + 标签)
          ↓
训练:  train_model.py  (WeightedInceptionTime / FCN / LiteCNN / TST)
          ↓
      models_margin/hit_detector_inception.pth
          ↓
实时:  realtime_detect.py  (纯概率双阈值滞回)
```

---

## 2. export_training_data.py — 训练数据采集

**用途**: 实时采集 Arduino 敲击事件 + iPhone IMU (ZigSim OSC)，保存完整录制供后续切片训练。

### 前置依赖

```bash
pip install pyserial numpy
```

### 硬件要求

- Arduino + 压电传感器 → USB 串口 → PC
- iPhone ZigSim → OSC UDP → PC (端口 8888)
- 传感器贴脚底/踩踏面，手机放裤兜

### 输出格式

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_imu_60.npy` | `(T, 7) float64` | 列: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, timestamp_us |
| `*_click_timestamps_60.npy` | `(N,) float64` | Arduino 敲击时刻 (Unix 秒) |

### 使用示例

```bash
# 采集 weak 手势
python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/w_0

# 采集 strong 手势
python export_training_data.py --port COM6 --wifi --strength strong --save_dir train_data/s_0

# 定时 30 秒自动停止
python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/w_1 --duration 30

# 查看已录制数据
python export_training_data.py --inspect train_data/w_0

# 列出可用串口
python export_training_data.py --list-ports
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--port`, `-p` | str | 自动检测 | Arduino 串口号 (如 COM6, /dev/ttyUSB0) |
| `--baud`, `-b` | int | 115200 | 串口波特率 |
| `--wifi` | flag | False | WiFi 热点 OSC 模式 (启用 OSC 监听) |
| `--strength` | str | `"strong"` | 力度标签 (`weak` / `strong`)，用于目录命名 |
| `--save_dir`, `-s` | str | `train_data` | 输出目录 |
| `--duration`, `-d` | float | 0 | 采集时长 (秒)，0 = 手动按 Enter 停止 |
| `--gesture-id` | int | 0 | 手势编号 (用于文件名) |
| `--session` | int | 0 | 采集会话编号 |
| `--sub` | int | 0 | 子会话编号 |
| `--inspect` | str | None | 查看已录制数据目录 (画波形图) |
| `--list-ports` | flag | False | 列出所有可用串口 |

### 运行中操作

- 采集过程中敲击传感器，控制台实时显示 `🥁 Hit #N | delta=XXX`
- 每 5 秒输出状态 `[状态] IMU帧=XXX | 敲击=N`
- 5 秒未收到 IMU 数据会警告
- 按 **Enter** 停止录制并保存

---

## 3. slice_dataset.py — 滑动窗口切片

**用途**: 从连续 IMU 录制中生成 `(N, C, T)` 训练样本，适配 tsai 训练框架。

### 标签策略

- 每个 click 前后扩展为 hit_zone: `[click - 10ms, click + 190ms]`
- 窗口**末尾帧**落在 hit_zone 内 → label = 击打类别
- 窗口末尾帧落在 hit_zone 外 → label = 0 (背景)
- 训练与推理方向一致 (只见过往数据预测当前)

### 默认切片参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 窗口长度 | 250ms (15帧@60Hz) | 足够看到完整波动 |
| 滑动步长 | 1帧 (~16.7ms) | 默认密集采样 |
| Hit zone | [-10ms, +190ms] | 击打标注窗口 |

### 使用示例

```bash
# 自动扫描 train_data/ 下所有 session (从目录名推断 weak/strong)
python slice_dataset.py --data_dir train_data --save sliced_data

# 手动指定 session 和标签
python slice_dataset.py --sessions train_data/w_0 train_data/s_0 --labels 1 2 --save sliced_data

# 自定义窗口参数 (500ms 窗口, 步长 2 帧)
python slice_dataset.py --data_dir train_data --window 0.5 --stride 0.033 --save sliced_data

# 保持原始分布不采样
python slice_dataset.py --data_dir train_data --save sliced_data --no-balance
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | None | 自动扫描目录 (从子目录名推断 label) |
| `--sessions` | list[str] | None | 手动指定 session 目录列表 |
| `--labels` | list[int] | None | 对应标签列表 (需与 --sessions 一一对应) |
| `--save`, `-s` | str | `sliced_data` | 输出目录 |
| `--window` | float | 0.25 | 窗口长度 (秒) |
| `--stride` | float | 1/60 | 滑动步长 (秒)，训练建议 2 帧 = 0.033 |
| `--pre` | int | 10 | 击打前扩展 (ms) |
| `--post` | int | 190 | 击打后扩展 (ms) |
| `--no-balance` | flag | False | 不对背景类下采样 |
| `--bg-ratio` | float | 2.0 | 背景:击打的最大比例 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `X.npy` | `(M, 6, W)` 训练样本 |
| `y.npy` | `(M,)` 标签 (0=背景, 1=weak, 2=strong) |
| `info.npz` | 切片参数元信息 |

> **注意**: 也可以跳过此步骤，直接用 `train_model.py --data_dir train_data` 内置切片。

---

## 4. train_model.py — 模型训练

**用途**: 训练 IMU 击打检测模型，支持 4 种架构。

### 模型架构

| 架构名 | 参数 | 参数量 | 特点 |
|--------|------|--------|------|
| `inception` | WeightedInceptionTime | **89K** | 🏆 **推荐** — ks=[3,5,7], depth=4, 指数加权池化 |
| `fcn` | FCN | ~20K | 快速 baseline, 三层 1D 卷积 + GAP |
| `lite` | LiteCNN | ~5K | 极轻量, 深度可分离卷积, 适合移动端 |
| `tst` | TST (Transformer) | ~200K | 自注意力对比实验, 小数据集易过拟合 |

### WeightedInceptionTime 架构特点

- 卷积核 [3, 5, 7] (替代 tsai 默认 [39, 19, 9])，适配 15 帧短窗口
- 4 层 InceptionBlock + 残差连接
- **可学习指数加权池化** (替代 GAP)：窗口末尾帧权重更大 (4.5x)，训练/推理方向一致
- 自动检测旧版 tsai InceptionTime 并兼容加载

### 使用示例

```bash
# 标准 CE Loss 训练
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --save_dir models_margin

# Focal Loss (gamma=2.0, 聚焦难样本)
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --focal-loss 2.0

# 使用 FCN 快速训练
python train_model.py --data_dir train_data --model fcn --epochs 30

# 使用预切片数据
python train_model.py --data_dir sliced_data --save_dir models

# 指定负样本 session (shaking/no_move 的背景样本优先保留)
python train_model.py --data_dir train_data --neg-sessions shaking_0,shaking_1
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `train_data` | 数据目录 (原始录制 或 预切片) |
| `--model` | str | `inception` | 模型架构: `inception` / `fcn` / `lite` / `tst` |
| `--epochs` | int | 70 | 最大训练轮数 (EarlyStopping patience=10) |
| `--bs` | int | 64 | 批次大小 |
| `--lr` | float | 1e-3 | 初始学习率 (One-Cycle 策略自动调整) |
| `--save_dir` | str | `models` | 模型保存目录 |
| `--focal-loss` | float | 0 | Focal Loss gamma, 0=标准 CE, 推荐 2.0 |
| `--val-sessions` | str | `""` | 逗号分隔的验证集 session 名 (如 `w_5,f_3,w_7`) |
| `--neg-sessions` | str | `""` | 逗号分隔的负样本 session (如 `shaking_0,no_move_0`) |

### 训练配置 (内置)

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW + One-Cycle LR (lr_max=1e-3) |
| 标准化 | 每通道 Z-score (mean/std 保存到 checkpoint) |
| 窗口 | 250ms (15帧@60Hz) |
| Hit zone | [-10ms, +190ms] |
| 背景:击打平衡比 | ≤ 2:1 (仅训练集, 验证集保持原始分布) |
| EarlyStopping | valid_loss 10 轮不降即停 |
| 验证分割 | Session 级独立 (推荐) 或窗口级随机 20% |

### 输出文件

| 文件 | 说明 |
|------|------|
| `hit_detector_{model}.pth` | 完整 checkpoint (权重 + mean/std + 元信息) |
| `curve_{model}.png` | 训练曲线图 (Loss + Accuracy) |

### 训练数据分割策略

- `--val-sessions w_5,f_3,strong_0` → 这 3 个 session 作为验证集
- 其余 session 为训练集 (含 shaking_0/1 作为负样本)
- 验证集**不采样**，保持原始分布
- 窗口级随机分割 (不指定 `--val-sessions` 时) 已弃用 (跨 session 泄露)

---

## 5. realtime_detect.py — 实时踩踏检测

**用途**: 接收 iPhone ZigSim OSC 流，用训练好的模型实时预测踩踏事件。

### 检测策略: 纯概率双阈值滞回

```
每帧推理 → P(hit) = softmax(logits)[1]
  → P(hit) > prob_high + 不在冷却期 → 触发提示音 + 进入冷却
  → P(hit) < prob_low → 释放滞回 (允许再次触发)
```

**为什么用 softmax 概率**：
- CE Loss 训练的 softmax 概率跨 session 语义一致
- 概率 ∈ [0,1] 有直观物理含义
- 双阈值滞回: P > 0.85 触发 (高置信), P < 0.4 释放 (安全复位)
- 冷却期防连击

### 使用示例

```bash
# 默认参数 (概率双阈值滞回)
python realtime_detect.py

# 校准模式: 采集 30s 背景数据, 输出建议阈值
python realtime_detect.py --calibrate

# 根据校准结果调参 + 实时显示
python realtime_detect.py --prob-high 0.85 --prob-low 0.4 -v

# 使用其他模型
python realtime_detect.py --model models_margin/hit_detector_fcn.pth

# 降低冷却时间 (更快的连击响应)
python realtime_detect.py --cooldown 300

# 调试 OSC 数据 (每秒打印原始值)
python realtime_detect.py --debug

# 实时绘图模式
python realtime_detect.py --plot
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `models/hit_detector_inception.pth` | 模型 .pth 路径 |
| `--port` | int | 8888 | OSC UDP 监听端口 |
| `--prob-high` | float | 0.85 | 概率高阈值 (超过触发) |
| `--prob-low` | float | 0.4 | 概率低阈值 (低于释放滞回) |
| `--cooldown` | int | 400 | 触发后冷却时间 (ms) |
| `--freq` | int | 800 | 提示音频率 (Hz) |
| `--sound-ms` | int | 100 | 提示音持续时间 (ms) |
| `--debug` | flag | False | 调试模式: 每秒打印 OSC 原始数据 |
| `--verbose`, `-v` | flag | False | 实时模式: 单行动态刷新概率条 |
| `--plot` | flag | False | 实时绘图: P(hit) 动态曲线 + 双阈值线 |
| `--calibrate` | flag | False | 校准模式: 30s 采集后输出建议阈值 |

### 运行模式

#### 正常检测模式 (默认)
- 窗口填满 15 帧后开始推理
- 每 120 帧 (~2s) 输出一次状态
- `-v` 模式: 单行动态刷新 `[====>     ] P=0.723 | .    | trig=3`

#### 校准模式 (`--calibrate`)
- 采集 30 秒真实 IMU 数据
- 不触发提示音
- 输出 P(hit) 分布统计 (Mean, Std, P90, P95, P99, Max)
- 给出建议的 `--prob-high` / `--prob-low` 阈值
- 画概率分布直方图 (含阈值标记)

#### 实时绘图 (`--plot`)
- matplotlib 动态显示 P(hit) 时间曲线
- 标注 prob_high (红线) 和 prob_low (橙线)

### 延迟分析

| 环节 | 延迟 | 说明 |
|------|------|------|
| 物理延迟 (脚→手机) | ~143ms (中位), 波动 ±62ms | 主导因素 |
| WiFi 传输 | ~40ms, 抖动 1.2ms | 可忽略 |
| 模型推理 | 2.6ms CPU / 2.4ms GPU | P99: 3.5ms |
| 检测窗口 | ~67ms (avg_4) | 已移除滑动平均 |
| **总延迟** | **~114-244ms** | 部署手机后可减 WiFi 40ms |

### 支持架构

自动检测 state_dict 适配:
- 新 WeightedInceptionTime (`backbone` key)
- 旧 tsai InceptionTime (`inceptionblock` key)
- FCN (`model_name="fcn"`)
- LiteCNN (`model_name="lite"`)
- TST (`model_name="tst"`)

---

## 6. latency_compare.py — 延迟对比工具

**用途**: 同时读取 Arduino 传感器和 iPhone IMU，以 PC 时钟为公共参考，在同一张图上叠加显示，分析延迟。

### 延迟分析方法

| 方法 | 说明 |
|------|------|
| 方法A (实时) | Arduino 敲击 → 红色竖线标记 → 在 IMU 波形图上肉眼观察滞后 |
| 方法B (事件) | IMU onset 检测 → 自动计算 Arduino 事件与 IMU 突变的时间差 |
| 方法C (互相关) | 离线分析 — 滑动窗口互相关估计延迟 |

> **踩踏场景注意**: 踩踏是低频晃动 (accel 为主)，onset 阈值检测对踩踏效果有限，此工具主要用于鼓棒场景的延迟标定。

### 使用示例

```bash
# 标准模式 (同时连接 Arduino + OSC)
python latency_compare.py --port COM6 --osc-port 8888

# 纯控制台模式 (不弹窗)
python latency_compare.py --port COM6 --osc-port 8888 --no-plot

# 只测 IMU, 不连 Arduino
python latency_compare.py --osc-port 8888 --no-arduino

# 调整 onset 灵敏度
python latency_compare.py --port COM6 --osc-port 8888 --gyro-thresh 2.5 --accel-thresh 0.5

# 保存延迟数据
python latency_compare.py --port COM6 --osc-port 8888 --save latency_results.csv

# 调试前 10 个 OSC 包
python latency_compare.py --port COM6 --osc-port 8888 --debug

# 列出可用串口
python latency_compare.py --list-ports
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--port`, `-p` | str | 自动检测 | Arduino 串口号 |
| `--baud`, `-b` | int | 115200 | Arduino 波特率 |
| `--osc-port`, `-o` | int | 8888 | OSC UDP 监听端口 |
| `--no-plot` | flag | False | 纯控制台模式 (不弹窗) |
| `--no-arduino` | flag | False | 不连接 Arduino (仅显示 IMU) |
| `--gyro-thresh` | float | 1.5 | 陀螺仪 onset 检测阈值 (rad/s) |
| `--accel-thresh` | float | 0.3 | 加速度计 onset 检测阈值 (g) |
| `--onset-cooldown` | float | 0.15 | onset 检测冷却时间 (秒) |
| `--list-ports` | flag | False | 列出可用串口后退出 |
| `--save` | str | None | 保存延迟数据到 CSV |
| `--debug` | flag | False | 打印前 10 个 OSC 包的原始地址和数值 |

### 图表界面

三行子图实时显示:
1. **加速度计 [g]** — Accel X/Y/Z 波形 + Arduino 敲击红色竖线
2. **陀螺仪 [rad/s]** — Gyro X/Y/Z 波形 + 敲击/IMU onset 标记
3. **延迟统计** — 每次测量的延迟散点图 + 移动平均线

### CSV 输出格式

| 列 | 说明 |
|-----|------|
| index | 测量序号 |
| arduino_time | Arduino 事件 PC 时间 |
| imu_onset_time | IMU onset PC 时间 |
| delay_ms | 延迟 (ms) |

---

## 7. visualize_session.py — 录制数据可视化

**用途**: 读取 `train_data/` 下的单个 session，画出 6 通道 IMU 波形 + Arduino 踩踏标注。

### 使用示例

```bash
# 列出所有可用 session
python visualize_session.py --list

# 画 w_0 全部数据
python visualize_session.py w_0

# 只看 10-30 秒
python visualize_session.py w_0 --range 10 30

# 只看第 5-15 次踩踏
python visualize_session.py w_0 --clicks 5 15

# 只看加速度 3 通道 (不画陀螺仪)
python visualize_session.py w_0 --accel-only

# 保存到文件
python visualize_session.py w_0 --save plot.png

# 自定义标题
python visualize_session.py w_0 --title "Weak Stomp - Session 0"

# 指定数据目录
python visualize_session.py w_0 --data-dir train_data
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | str | (必填) | Session 名称 (如 `w_0`, `f_2`, `shaking_0`) |
| `--data-dir` | str | `train_data` | 数据目录 |
| `--range` | float float | None | 显示的时间范围 (秒) `T_MIN T_MAX` |
| `--clicks` | int int | None | 显示第 START 到 END 次踩踏 |
| `--accel-only` | flag | False | 只看加速度 3 通道 |
| `--save` | str | None | 保存图片到文件 (PNG) |
| `--title` | str | None | 自定义标题 |
| `--list` | flag | False | 列出所有可用 session |

### 图表内容

- **Gyro 子图**: gyro_x/y/z 三通道波形 + 踩踏红色竖线
- **Accel 子图**: accel_x/y/z 三通道波形 + 踩踏红色竖线
- **Accel Mag 子图**: 合加速度幅值 `sqrt(ax²+ay²+az²)` + 踩踏编号标注 + 粉色 hit zone 半透明区域
- 踩踏红色竖线 + 粉色阴影 = +190ms hit zone
- 时间轴自适应分辨率 (秒级/亚秒级刻度)
- 踩踏编号 + 相对偏移时间标注

---

## 8. Arduino 固件

### data_collection_leg.ino

**位置**: `data_collection_leg/data_collection_leg/data_collection_leg.ino`

**功能**: 压电传感器数据采集，通过串口输出敲击事件。

**协议**: 文本协议
```
Sensor 0 FALLING count=1 raw=45 delta=120 min=42
Sensor 0 RISING  count=1 raw=210 delta=55 min=42
```

**硬件**: 只激活 A1 引脚 (单传感器模式)

**串口参数**: 115200 baud

### 字段说明

| 字段 | 说明 |
|------|------|
| Sensor `<N>` | 传感器编号 |
| FALLING / RISING | 边沿类型 (FALLING = 敲击检测) |
| count | 累计敲击计数 |
| raw | 当前 ADC 原始值 |
| delta | 峰值-基线差值 (反映敲击力度) |
| min | 窗口内最小值 (基线) |

---

## 常用命令速查

### 完整工作流

```bash
# 1. 录制训练数据
python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/w_0
python export_training_data.py --port COM6 --wifi --strength strong --save_dir train_data/s_0

# 2. 查看录制数据
python visualize_session.py w_0

# 3. 训练模型
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --save_dir models_margin

# 4. 校准阈值
python realtime_detect.py --model models_margin/hit_detector_inception.pth --calibrate

# 5. 实时检测
python realtime_detect.py --model models_margin/hit_detector_inception.pth -v --prob-high 0.85 --prob-low 0.4
```

### 训练变体

```bash
# Focal Loss
python train_model.py --data_dir train_data --val-sessions w_5,f_3,strong_0 --focal-loss 2.0

# 快速实验 (FCN)
python train_model.py --data_dir train_data --model fcn --epochs 30

# 指定负样本优先保留
python train_model.py --data_dir train_data --val-sessions w_5,f_3 --neg-sessions shaking_0,shaking_1
```

### 调试命令

```bash
# 调试 OSC 数据
python realtime_detect.py --debug

# 列出串口
python latency_compare.py --list-ports

# 列出可用的 training session
python visualize_session.py --list

# 延迟对比 (控制台模式)
python latency_compare.py --port COM6 --osc-port 8888 --no-plot
```
