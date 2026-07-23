# 从 5 篇已发表论文看 drum-ring 的投稿路径

> 基于 CHI.md 中对 AirSticks 2.0、sEMG Guitar、CMMR 2025、Augmented Violin、CustomFX 等论文的深度分析，
> 提炼出能中稿论文的共同特征，映射到 CHI/UIST 审稿维度，并规划 drum-ring 的补全路径。

---

## 一、5 篇论文的核心画像

| 论文 | 会议 | 类型 | ML? | 延迟 | 准确率 | 用户研究 | 开源 |
|------|------|------|:--:|:--:|:--:|:--:|:--:|
| **AirSticks 2.0** | NIME'22 | Full | ❌ 纯规则 | 未报告 | N/A | ❌ 仅实践验证 | ❌ |
| **Augmented Violin** | ArtsIT'25 | Full | ✅ DTW | **52ms** | **88.6%** | ❌ 仅专家反馈 | ✅ GitHub |
| **sEMG Guitar** | NIME'24 | Full | ✅ BLSTM×2 | **306ms** | 0.77 | ✅ 7人,9维问卷 | ✅ 代码+数据 |
| **CMMR 2025** | CMMR'25 | Full | ✅ MLP | ~100-170ms | 未报告 | ❌ 仅作者自用 | 🟡 仅PDF |
| **CustomFX** | CS231N'24 | 课程 | ✅ MLP | **14.5ms** | **94%** | ❌ 无 | ❌ |

---

## 二、能中稿论文的共性特征

### 特征 1：创新点不在模型架构，在交互范式

**5 篇论文没有一篇是靠 SOTA 模型发出来的。**

| 论文 | 真正的创新 |
|------|-----------|
| AirSticks 2.0 | 离散触发+连续调制双通道统一 + 5条设计原则框架 |
| Augmented Violin | \$30 ESP32+加速度计实现非侵入式小提琴增强 |
| sEMG Guitar | 用肌肉信号替代脚踩踏板切换效果器 |
| CMMR 2025 | MLPClassifier→MLPRegressor迁移，置信度替代硬分类 |
| CustomFX | 精简 MediaPipe landmarks 实现 14.5ms 极低延迟 |

> **drum-ring 的对应**：你的创新点不是 InceptionTime（它是已有架构），而是**首次在单 IMU 戒指上实现鼓击手势的多分类+力度映射**。把这一点讲清楚。

### 特征 2：完成度的"及格线"惊人地低

| 维度 | 最低标准（有论文如此还能中） | 最高标准 |
|------|------|------|
| **量化评估** | 无（AirSticks, CMMR） | 混淆矩阵+RMSE+per-user 报告（sEMG Guitar） |
| **用户研究** | 无，仅作者自用（CMMR） | 7人+9维Likert+4开放问题（sEMG Guitar） |
| **延迟数字** | 不报告（AirSticks） | 逐环节分解 256+50=306ms（sEMG Guitar） |
| **baseline 对比** | 无（全部 5 篇都没有 formal baseline） | — |
| **开源代码** | 无（AirSticks） | GitHub+Colab+数据集（sEMG Guitar） |
| **跨用户泛化** | 无（全部 per-user 或未验证） | — |

**关键发现**：这 5 篇论文加起来覆盖了"什么都没有"到"该有的都有"。**最低门槛比你想象的低得多。**

### 特征 3：延迟的处理策略比延迟数字本身更重要

```
                 延迟递增 →
AirSticks    Augmented     CMMR         sEMG Guitar
(未报告)     Violin(52ms)  (~100-170ms) (306ms)
   │              │            │              │
   └── 策略A ──→  │            │              │
        直接报告    └── 策略B ──→│              │
        专业验证        不讨论   └── 策略C ──→  │
                       用置信度      诚实+角色  │
                       掩盖感知      重新定义   │
```

**三种成功的延迟策略**：

1. **直接报告 + 专业验证**（Augmented Violin, 52ms）：数字好就直接说，用专家背书
2. **不讨论数字，用交互设计掩盖**（CMMR 2025）：置信度融合让延迟变成音色渐变
3. **诚实报告 + 重新定位角色**（sEMG Guitar, 306ms）："不是替代，是补充"

**对你的启示**：你先测延迟 → 根据数字选策略。不管数字多少，都有对应的已发表论文可以引用作为 precedent。

---

## 三、CHI/UIST 审稿维度 × 各论文的得分

### CHI 审稿的核心维度

| 维度 | 含义 | 对你论文的影响 |
|------|------|:--:|
| **Novelty** | 想法是否新？之前没人做过？ | 🔴 核心 |
| **Contribution Clarity** | 贡献是否清晰陈述？一句话能说清？ | 🔴 核心 |
| **Technical Depth** | 技术是否有深度？还是 trivial？ | 🟡 Demo可浅/Full需深 |
| **Evaluation Rigor** | 评估是否规范？用户研究是否合理？ | 🟡 Demo可弱/Full需强 |
| **System Completeness** | 系统是否完整可运行？ | 🔴 Demo尤其重要 |
| **Significance** | 是不是 important problem？ | 🟡 对 CHI 社区的意义 |
| **Presentation** | 写得好不好？视频好不好？ | 🔴 第一印象 |

### 各论文在这些维度上的得分

| 维度 | AirSticks | Aug. Violin | sEMG Guitar | CMMR 2025 | **drum-ring 现状** |
|------|:--:|:--:|:--:|:--:|:--:|
| **Novelty** | 🟢 强 | 🟡 中 | 🟢 强 | 🟢 强 | 🟢 **强**（首个 IMU 戒指鼓击多分类） |
| **Contribution Clarity** | 🟢 5条原则 | 🟡 模糊 | 🟢 清楚 | 🟡 模糊 | 🔴 **需定义** |
| **Technical Depth** | 🟡 阈值规则 | 🟡 DTW | 🟢 BLSTM×2 | 🟡 MLP | 🟢 **InceptionTime+互补滤波** |
| **Evaluation Rigor** | 🔴 无 | 🔴 仅专家 | 🟢 7人规范 | 🔴 无 | 🔴 **核心短板** |
| **System Completeness** | 🟢 全链路 | 🟢 全链路 | 🟢 全链路 | 🟢 全链路 | 🟢 **PyQt6+音效+BLE** |
| **Significance** | 🟢 NIME社区 | 🟡 小众 | 🟢 吉他手群体 | 🟡 创作者社区 | 🟡 需定位 |
| **Presentation** | 🟢 视频+清晰 | 🟡 代码好 | 🟢 开源+数据 | 🟡 仅PDF | 🔴 **无视频+无写作** |

### 解读

**drum-ring 的强项**：Novelty（首个）、Technical Depth（InceptionTime+互补滤波）、System Completeness（PyQt6完整应用）——这三项已经达到甚至超过已发表论文的水平。

**drum-ring 的致命短板**：Contribution Clarity（说不清贡献）、Evaluation Rigor（无评估）、Presentation（无视频无写作）。这三个恰是审稿人最先看到的东西。

---

## 四、从"已有系统"到"可投论文"的最小路径

### 基于 5 篇论文的"及格线"，你需要完成的任务清单

```
🔴 = 必须做（不做无法投稿）
🟡 = 强烈建议（显著提升录用概率）
🟢 = 加分项
```

### 阶段 1：定义贡献 + 测量数据（独立完成，2 周）

| 任务 | 状态 | 参照论文 |
|------|:--:|------|
| 🔴 一句话贡献陈述 | ❌ | sEMG Guitar 的 "替代脚踩踏板" |
| 🔴 混淆矩阵 + per-class F1 | ❌ | sEMG Guitar 的 Fig.8 |
| 🔴 端到端延迟分解（每环节 ms） | ❌ | sEMG Guitar 的 "256+50=306ms" |
| 🟡 规则 baseline 对比（阈值 vs InceptionTime） | ❌ | — |
| 🟡 跨用户泛化准确率（如果数据够） | ❌ | sEMG Guitar per-user 模式 |

### 阶段 2：用户试用（需招募，2-3 周）

| 任务 | 最低标准 | 理想标准 |
|------|------|------|
| 🔴 3-5 人非正式试用 | Demo 投稿够用 | — |
| 🟡 10-15 人正式用户研究 | Short Paper 够用 | Full Paper 需要 |
| 🔴 定性反馈（至少几句话，最好录视频） | Demo 够用 | — |
| 🟡 9维 DMI 量表（来自 sEMG Guitar） | Short Paper 加分 | Full Paper 需要 |

### 阶段 3：论文写作 + 视频（3-4 周）

| 任务 | 要点 |
|------|------|
| 🔴 Demo 视频（2-3 分钟） | 戴戒指→拍腿→听到鼓声→看到 GUI 反馈 |
| 🔴 论文初稿 | 借 AirSticks 5条原则框架 + sEMG Guitar 结构 |
| 🟡 设计原则陈述 | 从 AirSticks 的 5 条适配为你的 5 条 |

---

## 五、一句话贡献陈述模板（最重要的事）

从 5 篇论文的共性来看，**一篇能中稿的论文必须能用一句话说清贡献**。

下面是基于你的项目，三个可选的方向：

### 方向 A：系统导向（适合 Demo）

> "我们构建了首个基于单 IMU 戒指的实时鼓击手势分类系统，支持 5 种鼓击手势识别 + 力度映射音量控制。"

- 贡献：system building
- 对标：Augmented Violin
- 风险：审稿人可能问 "so what?"

### 方向 B：知识导向（适合 Short Paper）

> "通过构建单 IMU 戒指鼓击系统，我们发现：① 单点 IMU 可可靠区分 N 种拍击模式但无法区分空间位置；② 深度学习分类（85% 准确率）显著优于规则阈值方法；③ 50-100ms 端到端延迟在非专业场景下是可接受的。"

- 贡献：deriving design knowledge
- 对标：sEMG Guitar（他们的贡献也是"发现肌肉信号的潜力和局限"）
- 风险：需要足够的实验数据支撑每个"发现"

### 方向 C：交互范式导向（适合 Full Paper / UIST Short）

> "我们提出了一种"离散手势分类 + 连续力度映射 + 置信度融合输出"的三层交互范式，使得单 IMU 戒指可以同时作为鼓击触发器、力度表达器和音色混合器。"

- 贡献：new interaction paradigm
- 对标：CMMR 2025（置信度输出范式）+ AirSticks 2.0（离散+连续范式）
- 风险：需要实现置信度融合输出，不只是 argmax

### 🎯 建议

**先选方向 B（知识导向）作为 Short Paper 目标，方向 A 作为 Demo 保底。**

理由：方向 B 的叙事已经有 3 篇论文可以支撑（sEMG Guitar 的 trade-off 叙事 + CMMR 的置信度思路 + Augmented Violin 的延迟先例），且你的系统已经天然在产生这些 knowledge（延迟局限、空间定位局限、ML vs 规则对比）。

---

## 六、推荐的投稿目标

| 目标 | 截止时间 | 篇幅 | 录取率 | 需要额外投入 |
|------|------|------|:--:|------|
| **CHI 2027 Demo** | 约 2027.1 | 4页 EA | ~40-50% | 2-3个月（评测+视频+写作） |
| **UIST 2027 Short Paper** | 约 2027.3 | 5页 | ~22% | 4-5个月（+用户研究+baseline对比） |
| **UIST 2027 Demo** | 约 2027.7 | 2页 | **65%** | 2-3个月（保底选项） |
| **NIME 2027** | 约 2027.1-4 | Full | ~30-40% | 3-5个月（NIME 对延迟+完成度要求最宽松） |

---

## 七、最终结论

1. **你的系统完成度已经超过 5 篇已发表论文中的至少 2 篇**（AirSticks 没有 ML，CMMR 没有评估）。不是系统不够好，是你还没"包装"成论文。

2. **延迟不是发表障碍**。306ms（sEMG Guitar）和"不报告延迟"（AirSticks, CMMR）都有论文发表。关键是诚实 + 策略性处理。

3. **创新点的定义是你当前最重要的事**。5 篇论文的共同特征：模型简单、评估弱、但**创新点清晰、一句话能说清**。你的 InceptionTime 比他们的 MLP/DTW/阈值都强，但你还没定义清楚"为什么需要它"。

4. **最小可行投稿**：混淆矩阵 + 延迟数字 + 5 人试用 + Demo 视频 + 4 页短文 = CHI/UIST Demo。**2-3 个月，独立可完成。**
