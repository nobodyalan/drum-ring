# Novelty 定义 × 用户实验设计：5 篇已发表论文的做法

> 聚焦两个核心维度：论文如何定义创新点、如何设计（或不设计）用户实验。
> 每篇论文给出直接引用原文的方式来说明它们的具体做法。

---

## 一、Novelty：创新点是怎么定义的

### 1.1 AirSticks 2.0 (NIME 2022) — "填补二分法之间的空白"

**创新点一句话**：首次在单个系统中同时实现离散触发和连续调制。

**论文原文怎么说的**：

> *"Few instruments combine discrete and continuous control in a unified manner, with most tending to favour either discrete or continuous control."*

> *"The AirSticks 2.0 is a new gestural musical instrument that combines both discrete and continuous control to facilitate expressive gestural musical performances."*

**Novelty 是怎么构建的**（三步法）：

```
Step 1: 指出领域存在一个二分法问题
  → "Few instruments combine discrete and continuous control"
  → 举证: Freedrum/Aerodrums (纯离散) vs Mi.Mu gloves/Wave ring (纯连续)

Step 2: 说明这个二分法导致了什么局限
  → 纯离散系统 "merely mimic... electronic drum kit"（缺乏音色控制）
  → 纯连续系统 "rarely used to trigger precisely timed, discrete sound events"

Step 3: 声明自己填补了空白
  → "combines both discrete and continuous control"
  → 给出具体的实现方式（5条设计原则 + 3个映射案例）
```

**关键技巧**：这篇论文没有宣称自己技术强——它的 novelty 是**找到社区公认的问题（离散/连续二分法），然后给出第一个统一的解决方案**。

---

### 1.2 sEMG Guitar (NIME 2024) — "用一个具体痛点切入"

**创新点一句话**：用肌肉电信号 + 深度学习替代脚踩踏板切换吉他效果器。

**论文原文怎么说的**：

> *"Often, guitarists are distracted by the need to press multiple pedals while playing, to shape the guitar's sound. This motivated our research in exploring a strategy for interacting with sound effects that is different from the conventional one."*

> *"In this paper, we investigate the relation between guitarists' sonic intentions and their muscular contractions, to ease the interaction with sound effects without the use of pedals."*

**Novelty 是怎么构建的**：

```
Step 1: 指出一个具体的、每个吉他手都有的痛点
  → "guitarists are distracted by the need to press multiple pedals while playing"
  → 这不是抽象的理论问题，是实际演奏中人人都遇到过的问题

Step 2: 提出替代方案
  → "without the use of pedals" — 用肌肉信号自动识别演奏技法
  → 离散分类 (7技法→3预设) + 连续回归 (力度→6参数)

Step 3: 明确研究问题
  → "Are muscle contractions useful to musicians, as a new form of 
     interaction with a pedal-board, to encourage exploration and 
     foster creativity?"
  → 一整篇论文就是在回答这个问题
```

**关键技巧**：Novelty 不是一个抽象声明，而是一个**可回答的研究问题**。审稿人看完 Introduction 就知道"如果这篇论文能回答这个问题，它就有贡献"。

---

### 1.3 CMMR 2025 — "改变输出方式就是创新"

**创新点一句话**：用置信度向量替代硬分类标签，让模型的不确定性变成音色融合的审美特征。

**论文原文怎么说的**：

> *"This approach embraces the guesswork of the model itself to smoothly interpolate between audio outputs... Outputting confidence ratings in place of predicted classes."*

> *"Our work shares some goals with previous sound tracing research, seeking approaches to gesture-sound pairing that move beyond the indicative or symbolic (e.g. raising a hand to trigger playback) to the descriptive or intuitive (e.g. arm movements that trace the shape of a phrase)."*

**Novelty 是怎么构建的**：

```
Step 1: 指出传统做法的问题
  → 硬分类 (symbolic mapping): "raising a hand to trigger playback"
  → 问题: 类间过渡生硬、模型猜错时没有优雅的 fallback

Step 2: 提出新做法
  → 置信度输出 (descriptive/intuitive mapping)
  → MLPClassifier → MLPRegressor 迁移
  → "embraces the guesswork of the model"

Step 3: 展示新做法带来的新能力
  → 类间平滑过渡
  → 模糊手势产生融合音色
  → 模型不确定性变成特征而非 bug
```

**关键技巧**：这篇论文的 novelty **不在模型架构**（就是最普通的 MLP），而在**输出策略**。你不需要发明新算法，只需要换一种方式使用现有算法的输出。

---

### 1.4 Augmented Violin (ArtsIT 2025) — "成本+非侵入性 = Novelty"

**创新点一句话**：只用 $30 的 ESP32 + 加速度计，非侵入式安装在小提琴上，实现 5 种手势的实时分类。

**Novelty 是怎么构建的**：

```
Step 1: 指出小提琴增强的传统做法的问题
  → 通常需要昂贵设备或侵入式改造

Step 2: 提出"足够好"的替代方案
  → $30 成本 + 5-8 example per gesture
  → 88.6% accuracy at 52ms latency
  → "without compromising the instrument's acoustic identity"

Step 3: 用专业音乐家验证
  → "responsive, intuitive, and non-interference with traditional technique"
```

**关键技巧**：这篇的 novelty 是**工程约束下的充分性证明**——不需要最好，只需要"足够好 + 足够便宜 + 足够简单"。

---

### 1.5 总结：四种 Novelty 构建模式

| 模式 | 代表论文 | 做法 | 适合 drum-ring? |
|------|---------|------|:--:|
| **填补二分法空白** | AirSticks 2.0 | 找到社区公认的对立（离散/连续），提供统一方案 | 🟢 你有 "IMU分类+力度映射"，可定位为 "ML vs 规则" 之间的统一 |
| **解决具体痛点** | sEMG Guitar | 从一个每个人都能理解的痛点出发（踩踏板分心） | 🟢 "电子鼓太贵/太大，戒指太小但能否做更多？" |
| **改变输出方式** | CMMR 2025 | 不改变模型，改变模型输出的使用方式 | 🔥 **直接可用！** 你可以做置信度输出 |
| **成本+充分性** | Augmented Violin | $30 做到 88.6%，证明"足够好" | 🟢 你的 BLE 戒指本身就是低成本方案 |

---

## 二、用户实验：怎么做的（和不做的）

### 2.1 sEMG Guitar (NIME 2024) — 最规范的用户研究

这是 5 篇中唯一有完整用户研究的。逐环节拆解：

**参与者**（Section 5）：

```
7 名有经验的男性吉他手
年龄: 21-37 岁 (M=28.25, SD=5.27)
平均音乐经验: 15 年
全部自评 intermediate 或 expert
无上肢受伤史
```

**实验流程**（三阶段）：

```
阶段 1: 数据集创建
  → 每人按 3.3.1 的步骤创建自己的训练数据
  → 7 种吉他技法 × 2 次采集 × 60 秒 = 14 分钟/人

阶段 2: 系统试用
  → 参与者用 backing track 即兴演奏
  → 交替使用各种技法
  → 系统实时分类并切换效果器预设

阶段 3: 问卷评估
  → 9 个封闭问题 (11-point Likert)
  → 4 个开放问题
```

**问卷设计**（基于 NIME 社区的 DMI 评估框架）：

```
封闭问题（11-point Likert）:
  1. Enjoyability    — "How enjoyable was your experience?"
  2. Playability     — "How much control did you feel?"
  3. Learnability    — "How easy was learning to use it?"
  4. Expressiveness  — "How much did it help you enhance expressiveness?"
  5. Novelty         — "How much novelty does it introduce?"
  6. Effectiveness   — "How much was it able to track your sonic intentions?"
  7. Wearability     — "How not invasive do you consider the system?"
  8. Adaptability    — "How much was the system able to improve following your feedback?"
  9. Usability       — "How confident would you feel using it in a live performance?"

开放问题:
  1. Main Limitations — "What are the main limitations for live use?"
  2. Overall Experience — "How was using muscular contractions? Good way to enhance expressiveness?"
  3. Added Value — "What is the added value vs standard interaction?"
  4. Improvements — "How would you improve the system?"
```

**结果展示方式**：

```
- Fig 9: 箱线图，按 median 降序排列，一目了然看出最强和最弱维度
- 每个维度摘录 1-2 句参与者的直接引语
- 正负面反馈都报告，不筛选
- 负面反馈报告得特别详细（"extreme muscle contraction",
  "distracting me from the instrument", "injecting lots of latency"）
```

**量化指标放在用户研究前面**（Section 4.3 Technical Evaluation）：

```
Table 1: 分类器 accuracy + 回归器 RMSE
Fig 8(a): 分类准确率分布 (violin plot across 7 datasets)
Fig 8(b): 回归 RMSE 分布 + residuals
```

> 这种做法是：**先展示系统性能数据（Section 4），再展示用户怎么看（Section 5）。两个互相验证。**

---

### 2.2 AirSticks 2.0 (NIME 2022) — 零用户实验，全靠实践验证

**没有做**：无 N、无 Likert、无统计、无 interview、无 questionnaire。

**取而代之的是什么**：

```
1. "ecologically validated through hundreds of real-time music 
    performances including several contributions to NIME conferences 
    since 2014"
   → 用 8 年演出历史代替用户实验

2. "practice-based research" 方法论
   → "using short design cycles and deadlines in the form of public 
      workshops and performances to 'ecologically validate' the 
      developments"

3. 三个映射的视频记录
   → 15s demo 视频 × 3 = 总共 45 秒的视频证据

4. "bodystorming" 作为设计方法
   → 声音设计师自己通过身体体验来验证设计决策
```

**论文如何处理"没有用户实验"这件事**：

论文根本没提"我们没有用户实验"——它把 **practice-based research** 作为一种同等合法的方法论来陈述：

> *"As practice-based researchers we will use videos of these mappings to illustrate our findings, and argue that this documentation is vital in sharing some of the tacit knowledge embedded in the physical experience of playing the instrument."*

> *"We used a collaborative practice-based approach... using short design cycles and deadlines in the form of public workshops and performances to 'ecologically validate' the developments."*

**对你的启示**：如果你的用户实验时间不够，你可以明确声明你是 **practice-based research**，用"系统已经在真实场景中经过多轮迭代验证"替代正式实验。NIME 审稿人接受这种做法。

---

### 2.3 CMMR 2025 — 零用户实验，全是自我报告

**没有做**：无任何第三方参与者。所有测试都是作者自己。

**取而代之的是什么**：

```
1. "First Application and Testing" (Section 3.4)
   → 作者一自己即兴演奏测试系统
   → 报告: "audio playback triggered by the model accurately 
     reflected the trained gesture categories"
   → 报告: "responding reliably enough to allow for 
     improvisation and play"
   → 报告: "some false predictions occurred, practice with 
     the system quickly reduced these"

2. "Creative Applications" (Section 3.5)
   → 作者一将系统整合进创作工作流
   → 2025 年秋季多场公开演出的实践验证
   → 报告: "we ultimately found these shifting background 
     mappings compelling enough to use on their own"

3. 讨论部分用了第一人称主观判断
   → "as performers we found that..."
   → "we found that the use of confidence ratings... resulted 
     in smoother transitions"
```

**论文如何处理"没有用户实验"这件事**：

论文**没有把"没有用户实验"当作一个 limitation 来讨论**。它直接声明自己是 practice-based work（section 5 标题就是 "a practice-based application"），不在 Conclusion 里道歉。论文的 limitation discussion（Section 4）讨论的是技术问题（特征选择、手势多样性），而不是方法论问题。

**对你的启示**：CMMR 这篇证明，在音乐 HCI 领域，**仅作者自我验证 + 公开演出实践，是可以发表 Full Paper 的**。你如果有 3-5 人非正式试用，已经超过这篇的标准。

---

### 2.4 Augmented Violin — 专业音乐家个人反馈

**参与者**：1 名专业小提琴手。

**做了什么**：非正式反馈——"responsive, intuitive, and non-interference with traditional technique" + "positive audience reception"。

**没有做什么**：无问卷、无 Likert、无统计、无多人比较。

---

### 2.5 用户实验的梯度：从零到完整

```
Level 0: 无用户实验
  └── AirSticks 2.0: practice-based research + 8年演出历史
  └── CMMR 2025: 作者自我测试 + 多场公开演出

Level 1: 个人专家反馈
  └── Augmented Violin: 1名专业小提琴手口头反馈

Level 2: 小规模非正式试用
  └── (你目前没有，但这是 Demo 投稿的最低可行标准)
  └── 3-5 人, 定性反馈, 录视频

Level 3: 正式用户研究
  └── sEMG Guitar: 7人, 3阶段实验, 9 Likert + 4 open

哪个 level 够投什么:
  Level 0-1 → NIME/CMMR Full Paper（在音乐 HCI 社区可以接受）
  Level 2   → CHI/UIST Demo/Poster
  Level 3   → CHI/UIST Short Paper / Full Paper
```

---

## 三、对你 drum-ring 的直接行动指南

### 3.1 Novelty 怎么做：选一个模式

基于你的系统现状，**推荐 AirSticks 2.0 模式**（填补空白）：

```
你的版本:

Step 1: 指出空白
  "Existing ring-based musical controllers (e.g., Wave ring) offer 
   only continuous control. Existing IMU-based drum systems (e.g., 
   Freedrum, Aerodrums) offer only discrete triggering. No system 
   combines ML-based multi-class gesture recognition with 
   continuous velocity mapping on a single IMU ring."

Step 2: 说明为什么这个空白重要
  "Without gesture classification, ring controllers cannot 
   distinguish between drum types (hi-hat vs. snare vs. tom). 
   Without continuous velocity mapping, drum triggers lose the 
   expressive nuance of acoustic drumming."

Step 3: 声明你的贡献
  "We present the first single-IMU ring system that combines:
   (1) deep learning-based 5-class drum gesture classification 
   (InceptionTime, 85% accuracy), and (2) continuous linear 
   acceleration-to-velocity mapping, within (3) a complete 
   PyQt6 desktop application with real-time visualization."
```

### 3.2 用户实验怎么做：根据投稿目标选择

```
如果你投 CHI/UIST Demo（2-3个月准备时间）:

  最低方案（对标 Augmented Violin）:
    - 3-5 人非正式试用
    - 每人 15-20 分钟
    - 收集: "好不好玩？延迟能接受吗？手势区分合理吗？"
    - 录像作为 Demo 视频素材
    - 不要求 Likert / 统计

  理想方案（对标 sEMG Guitar，但降级）:
    - 5-8 人
    - 简化版问卷: 从 sEMG Guitar 的 9 维选 5 维
      (Enjoyability, Playability, Effectiveness, Wearability, Novelty)
    - 5-point Likert（比 11-point 更简单）
    - 1-2 个开放问题
    - 报告: 描述统计 (median/mean + 参与者引语)

如果你投 Short Paper（4-5个月准备时间）:

   对标 sEMG Guitar 完整版:
    - 10-15 人
    - 9 维 DMI 问卷 (11-point Likert)
    - 4 开放问题
    - 混淆矩阵 + per-class F1 + 延迟 profiling
    - 规则 baseline 对比
```

### 3.3 问卷模板（直接可用）

基于 sEMG Guitar 的 9 维度，适配你的 drum-ring：

```
封闭问题 (5-point Likert: 1=Strongly Disagree, 5=Strongly Agree):

1. Enjoyability: 
   "Playing drums with the ring was enjoyable."
2. Playability: 
   "I felt in control of which drum sound was triggered."
3. Learnability: 
   "It was easy to learn which gesture produces which sound."
4. Expressiveness: 
   "The velocity mapping (harder hit = louder sound) felt natural."
5. Effectiveness: 
   "The system correctly recognized my intended drum gestures."
6. Wearability: 
   "The ring did not interfere with my natural playing movement."
7. Latency Perception:
   "The delay between my movement and the sound was acceptable."
8. Novelty:
   "This way of playing drums feels innovative."
9. Usability:
   "I could imagine using this for practice or performance."

开放问题:
1. "What gesture was hardest to trigger reliably?"
2. "Was the delay noticeable? If yes, did it affect your playing?"
3. "What would you change or improve?"
```

---

## 四、总结

| 维度 | 学 AirSticks 2.0 | 学 sEMG Guitar | 学 CMMR 2025 |
|------|:--:|:--:|:--:|
| **Novelty 构建** | 🔥 "填补空白"的三步叙事 | "具体痛点"驱动的研究问题 | "改变输出方式就是创新" |
| **用户实验** | 可以不做的替代方案 (practice-based) | 做的标准模板 (9维问卷) | 更低的门槛 (自我验证) |
| **延迟处理** | 不报告数字 | 诚实+角色重定义 | 不讨论，用置信度掩盖 |
| **最值得你学的** | Introduction 的叙事结构 | 问卷设计 + 量化+定性双报告 | 置信度输出思路 |

**核心结论**：你的系统在技术上已经超过 AirSticks 2.0（他们无 ML）和 CMMR 2025（他们仅有 MLP），在用户实验上你可以对标 sEMG Guitar 做简化版。Novelty 方面，用 AirSticks 的"填补空白"叙事框架 + CMMR 的置信度输出思路，你的论文就有明确的贡献陈述。
