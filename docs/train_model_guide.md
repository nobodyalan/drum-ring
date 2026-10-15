# 鼓击手势模型训练指南

本文档说明鼓击手势识别模型的**训练程序、完整流程、关键参数、模型产物位置，以及如何接入 `main.py` 做实时推理**。

## 总览

| 项目 | 内容 |
|---|---|
| 训练入口 | `scripts/train_local_drum_gestures.py` |
| 模型定义 | `models/drum_gesture_recognizer.py` → `RegularizedInceptionTime` |
| 训练框架 | fastai + tsai（`Learner` / `TSDataLoaders` / `SaveModelCallback`） |
| 特征 | `rawlin12_handcrafted` = 17 通道 × 15 帧（详见 `docs/17channel_feature_pipeline.md`） |
| 输出目录 | `checkpoint/drum_gesture/` |
| 推理入口 | `models/drum_gesture_recognizer.py` → `DrumGestureRecognizer` |
| GUI 接入 | `apps/drum_gesture_app.py` → `DrumInferenceWorker`（后台 QThread） |

---

## 一、训练程序与完整流程

`scripts/train_local_drum_gestures.py` 是一个**端到端脚本**，一条命令跑完从原始加密数据到模型产物的全部步骤：

```
① 解密数据            decrypt_dataset() → data_processed_local_drum_gesture/
   │
   ▼
② 估计线性加速度        estimate_linear_recording() → OnlineLinearAccelerationEstimator
   │                    (四元数互补滤波, 见 17 通道文档)
   ▼
③ 串口事件对齐到峰值    serial_event_times() → peak_indices()
   │                    每个 Arduino 事件 ±PEAK_SEARCH_EACH_SIDE 帧内找 |linear_acc| 最大值
   ▼
④ 构建特征              feature_recording(raw_imu, linear_imu, feature_mode)
   │                    rawlin12_handcrafted → 17 通道
   ▼
⑤ 切片窗口              positive_windows / negative_windows / transition_negative_windows
   │                    正例: 窗口末帧落在 [peak-14, peak-4]
   │                    负例: 过渡区 [peak-60, peak-(14+3)) + 随机硬负例
   ▼
⑥ 划分训练/验证/测试     split_by_group() 或 split_random()
   │
   ▼
⑦ 训练 + 保存           RegularizedInceptionTime + fit_one_cycle
                        → checkpoint/drum_gesture/{run_name}.pth + _eval.json
```

### 正/负例切片策略（关键）

| 概念 | 值 | 含义 |
|---|---|---|
| `WINDOW` | 15 帧（75ms @200Hz） | 模型输入窗口长度 |
| `POS_ZONE_END_FIRST` | 14 | 正例窗口末帧最早在 `peak-14`（峰前 70ms） |
| `POS_ZONE_END_LAST` | 4 | 正例窗口末帧最晚在 `peak-4`（峰前 20ms） |
| `TRANSITION_START_OFFSET` | 60 | 过渡区负例起点 `peak-60` |
| `TRANSITION_GAP` | 3 | 过渡区与正例区之间隔 3 帧，防重叠 |
| `HARD_NEGATIVES_PER_FILE` | 180 | 每个录制采样 180 个硬负例 |
| `PEAK_SEARCH_EACH_SIDE` | 30 | 事件→峰值对齐搜索半径（±30 帧） |

---

## 二、模型架构：RegularizedInceptionTime

定义在 `models/drum_gesture_recognizer.py`，是 tsai `InceptionTime` 骨架 + 池化头的封装：

```
输入 (B, 17, 15)
   │
   ▼ InceptionTime backbone (inceptionblock)
   │   nf=32, ks=15, depth=4, bottleneck=False
   ▼
head:
  - "gap"      (默认): AdaptiveAvgPool1d → Linear(128, c_out)
  - "weighted" (可选): 可学习时间加权池化 time_weights → Linear(128, c_out)
   ▼
输出 logits (B, c_out) → softmax → 5 类概率
```

5 类：`0=空, 1=踩镲, 2=通鼓2, 3=军鼓, 4=通鼓1`（`DRUM_CLASS_LABELS`）。

> 注意：`CLAUDE.md` 里写的 `ks=40, depth=6, bottleneck=True` 是**旧描述**，当前代码默认 `nf=32, ks=15, depth=4, bottleneck=False`。

---

## 三、关键参数

### 命令行参数（`parse_args()`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | `data` | 原始加密数据目录 |
| `--work-dir` | `data_processed_local_drum_gesture` | 解密/中间产物目录 |
| `--checkpoint-dir` | `checkpoint/drum_gesture` | 模型输出目录 |
| `--analysis-dir` | `analysis_outputs` | 数据集摘要输出 |
| `--run-name` | `local_drum_4gesture_lead6_gap_raw6` | 本次训练的名字（决定产物文件名） |
| `--window` | 15 | 窗口帧数（75ms） |
| `--feature-mode` | `lin6` | `lin6` / `rawlin12` / `rawlin12_handcrafted`（17ch 用最后一个） |
| `--lead-end-offsets` | `14,12,10,8,6,4` | 正例窗口末帧相对峰值的偏移采样点 |
| `--epochs` | 40 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 1e-3 | 学习率（`--lr-find` 可自动找） |
| `--weight-decay` | 1e-2 | 权重衰减（fastai Adam，等价 AdamW） |
| `--label-smoothing` | 0.05 | 标签平滑 |
| `--gaussian-noise` | 0.05 | 高斯噪声增强 |
| `--input-dropout` | 0.10 | 输入 Dropout 增强 |
| `--cutout` | 0.10 | CutOut 增强 |
| `--split-mode` | `group` | `group`（按录制分组）/ `random` |
| `--val-size` | 0.20 | 验证集比例 |
| `--test-size` | 0.25 | 测试集比例 |
| `--no-test` | off | 跳过测试集，全部用于 train+val |
| `--lr-find` | off | 训练前跑 LR finder 自动选学习率 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--seed` | 42 | 随机种子 |
| `--balance-classes` | off | 类别均衡采样 |
| `--save-windows` / `--load-sliced` / `--slice-only` | — | 切片缓存/复用 |
| `--no-switch` / `--no-pure-negative` | off | 数据过滤开关 |

### 内部常量

`WINDOW=15`, `LABEL_IDS=(1,2,3,4)`, `SEED=42`, `TRANSITION_NEGATIVES_PER_PEAK=8`。

---

## 四、训练命令示例

### 从头训练（解密 → 切片 → 训练）

```bash
python scripts/train_local_drum_gestures.py \
  --data-dir datanew/hyb \
  --run-name my_v12 \
  --feature-mode rawlin12_handcrafted \
  --window 15 \
  --batch-size 64 \
  --epochs 40 \
  --lr-find \
  --no-test
```

### 复用已切好的窗口（跳过解密/切片）

```bash
python scripts/train_local_drum_gestures.py \
  --load-sliced my_sliced_data/windows \
  --run-name my_v11 \
  --feature-mode rawlin12_handcrafted \
  --batch-size 64 \
  --lr-find \
  --no-test
```

### 只切片不训练（调参用）

```bash
python scripts/train_local_drum_gestures.py \
  --data-dir datanew/hyb \
  --run-name debug_slice \
  --feature-mode rawlin12_handcrafted \
  --slice-only
```

---

## 五、模型产物位置

训练完成后在 `checkpoint/drum_gesture/` 下生成三件套：

| 文件 | 内容 |
|---|---|
| `{run_name}.pth` | 最终模型权重（`torch.save({"model": state_dict})`） |
| `{run_name}_best.pth` | fastai `SaveModelCallback` 保存的验证集最优权重 |
| `{run_name}_eval.json` | 训练摘要：划分信息、各类别样本数、标准化统计量、best loss/acc、test 指标、`input_shape` 等 |

### 现有主要版本

| 文件 | 说明 |
|---|---|
| `my_v11.pth` | **当前默认**：17ch + lead18-4 + gap3 + rawlin12hc（no-test） |
| `my_v10.pth` | v10：全量数据 no-test（val acc 89.9%） |
| `my_v9.pth` | v9：switch 标签修正版（test acc 91.3%） |
| `my_v8.pth` | v8：1-per-class 验证（test acc 92.9%） |
| `sliced_w15_rawlin12_handcrafted.pth` | 17 通道早期版本 |

`my_v11.pth` 的权重结构：`c_in=17, seq_len=15, head=gap`，加载时由 `DrumGestureRecognizer` 自动推断，无需手动指定。

---

## 六、如何接入 main.py（实时推理）

### 加载链路

```
main.py → QApplication → MainWindow (desktop/main_window.py)
   │  start_background_services() → 注册 Feature App
   ▼
apps/drum_gesture_app.py → DrumGestureWindow
   │  MODEL_OPTIONS 下拉框选择模型 → model_path
   ▼
DrumInferenceWorker (QThread.run)
   │  recognizer = DrumGestureRecognizer(self.model_path)   ← 加载 .pth + _eval.json
   ▼
每帧滑窗 → recognizer.predict(window) → DrumGesturePrediction
```

### 1. 模型注册（`apps/drum_gesture_app.py`）

```python
MODEL_OPTIONS = [
    ("v11 lead18-4 gap3 rawlin12hc (no-test)",
     PROJECT_ROOT / "checkpoint" / "drum_gesture" / "my_v11.pth", 15),
    ("v10 no-test all data ...",
     PROJECT_ROOT / "checkpoint" / "drum_gesture" / "my_v10.pth", 15),
    ...
]
```

每一项是 `(显示名, checkpoint 路径, window_size)`。把新训练出的 `.pth` 加进这个列表，GUI 下拉框即可选中。

### 2. 模型加载（`DrumGestureRecognizer.__init__`）

加载时**自动推断**结构，无需改代码：

```python
recognizer = DrumGestureRecognizer("checkpoint/drum_gesture/my_v11.pth")
# 内部自动:
#   c_in   ← state_dict 里 bottleneck.weight 的通道数 (17)
#   c_out  ← fc.weight 的类别数 (5)
#   window ← 同名 _eval.json 的 input_shape[1] (15)
#   head   ← 是否含 time_weights (gap / weighted)
#   标准化统计量 ← _eval.json 的 standardize_stats
```

### 3. 推理（`predict()`）

```python
prediction = recognizer.predict(window_values)   # window_values: (15, 9) 原始通道
# 内部: prepare() 构建 17 通道 + 逐通道 z-score → (1, 17, 15) → model → softmax
# 返回: class_id / label / confidence / probabilities
```

`DrumInferenceWorker` 在后台 QThread 里循环取滑窗任务 → `predict()` → 通过信号 `result_ready` 把结果发回主线程触发音效。

### 4. 换用新模型的完整步骤

1. 训练：`python scripts/train_local_drum_gestures.py --run-name my_new --feature-mode rawlin12_handcrafted ...`
2. 确认产物：`checkpoint/drum_gesture/my_new.pth` + `my_new_eval.json`
3. 在 `apps/drum_gesture_app.py` 的 `MODEL_OPTIONS` 里加一项 `("my_new ...", ... / "my_new.pth", 15)`
4. 重启 `python main.py`，在下拉框选择 `my_new`

---

## 七、注意事项

1. **训练/推理线程配置一致**：训练脚本会 `torch.set_num_threads(1)`（与推理侧 `DrumInferenceWorker._set_realtime_priority()` 一致），避免 BatchNorm/Linear 浮点累加顺序差异导致概率偏移。
2. **`_eval.json` 必须与 `.pth` 同名同目录**：模型加载依赖它读取 window 与标准化统计量；缺失时回退到默认值（可能导致精度下降）。
3. **17 通道 vs 21 通道**：当前训练用 `rawlin12_handcrafted`（17ch）。加载旧 21ch checkpoint 时，`prepare()` 会自动做通道兼容转换。
4. **推理仅支持 CPU**：`DrumGestureRecognizer` 明确拒绝非 CPU 设备（GUI 实时场景单线程 CPU 更快）。
