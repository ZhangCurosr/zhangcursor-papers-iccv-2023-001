---
title: "MAtch-eXpand-and-Improve-Unsupervised-Finetuning-for-Zero-Sh"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lin_MAtch_eXpand_and_Improve_Unsupervised_Finetuning_for_Zero-Shot_Action_Recognition_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:00:11"
field: "视频动作识别"
keywords: ["zero-shot action recognition", "vision-language model", "multiple instance learning", "unsupervised finetuning", "CLIP", "text bag"]
innovations: ["提出无监督微调框架MAXI，利用无标签视频和多种语言知识源构建文本袋并进行MIL微调", "改进MIL-NCE损失，增加共享最优匹配文本的语义一致性约束", "首次在不依赖ground truth标注的情况下，使CLIP在零样本和少样本动作识别上超越全监督基线"]
benchmarks: ["UCF101", "HMDB51", "Kinetics-600", "MiniSSv2", "Charades", "UAV Human", "Moments-in-Time"]
---

# 论文速读：MAtch, eXpand and Improve: Unsupervised Finetuning for Zero-Shot Action Recognition with Language Knowledge

## 一句话总结
本文提出 MAXI，一种利用无标签视频数据和多种语言知识源（动作词典、GPT-3 文本扩展、BLIP 图像描述）构建文本袋，并通过多实例学习（MIL）目标对 CLIP 进行无监督微调的方法，显著提升了零样本和少样本动作识别的跨数据集迁移能力。

## 研究问题与动机
- 大规模视觉-语言（VL）模型在零样本动作识别上表现不佳，原因在于 VL 模型过度偏向表示物体（名词）而忽视动词/动作，需额外在视频数据上微调。
- 现有微调方法依赖大规模全标注动作数据集（如 K400 的完整标签），标注成本高且模型仅接触有限动作词汇（如 400 类），限制了零样本迁移到未见动作类别的能力。
- 无标注视频数据获取成本低，但如何有效利用无标签视频 + 语言知识源来微调 VL 模型仍缺乏研究。

## 核心贡献（创新点）
1. **首个基于无标签视频 + 多语言知识源的零样本动作识别微调框架**：通过动作词典、GPT-3 和 BLIP 三种语言来源为每个视频构建"文本袋"，无需任何 ground truth 标注即可完成微调。
2. **提出基于 MIL-NCE 的改进损失函数**：在标准 MIL-NCE 基础上，额外鼓励具有相同最优匹配文本的视频和文本袋在表示空间中靠近，增强语义一致性。
3. **充分验证了该方法在零样本和少样本迁移上的优势**：在七个下游基准上，相比原始 CLIP 最高提升 14% 绝对精度，且在部分任务上超越同样基于 K400 全监督微调的 ViFi-CLIP 基线；少样本场景下以 MAXI 预训练权重为初始化也达到最优结果。
4. **系统揭示了文本袋中动词注入的价值**：消融实验表明，在文本袋中加入 GPT-3 和 BLIP 提取的动词短语能显著提升零样本性能，而对象名词的收益相对较小，印证了 VL 模型对动词表征不足的固有缺陷。

## 方法详解
- **视频编码器适配**：沿用 ViFi-CLIP 范式，将 CLIP 图像编码器用于视频——随机采样 M 帧送入视觉编码器，取帧特征均值作为视频表示，不引入额外可学习参数，直接利用预训练权重初始化。
- **文本袋构建（Text Bag Construction）**：对每个无标签视频 $x_i$，通过三步构建文本袋 $T_i$：
  1. **CLIP 匹配（Match）**：用原始 CLIP 计算视频与动作词典 D 中所有文本的余弦相似度，取 Top-1 匹配文本 $\hat{t}_i = \arg\max_{t \in D} \text{sim}(\phi_v(x_i), \phi_t(t))$ 加入文本袋，作为从预训练 VL 模型蒸馏知识的机制。
  2. **GPT-3 文本扩展（Expand）**：以 $\hat{t}_i$ 为指令输入 LLM（GPT-3），利用其世界知识生成替代描述，从中提取动词及动词短语（含 lemma 和 gerund 形式）加入文本袋。
  3. **BLIP 视频→文本扩展**：用 BLIP 对视频各帧生成图像描述，从中提取动词和动词短语加入文本袋。
  - **文本袋过滤**：设置相似度阈值 $\delta_p$，保留 $p \times 100\%$ 最高置信度的视频及其文本袋，过滤噪声。
- **多实例学习损失（Improve）**：采用 MIL-NCE 框架，每轮随机从文本袋中采样 $N_{\text{bag}}$ 个文本，定义带共享最优匹配指示符的损失函数：
$$
\mathcal{L} = -\frac{1}{|I_B|} \sum_i \log \frac{\sum_j \sum_n \exp(\bar{\phi}_v(x_i)^\top \bar{\phi}_t(t_{j,n})/\sigma) \cdot \mathbb{1}(\hat{t}_i = \hat{t}_j)}{\sum_k \sum_n \exp(\bar{\phi}_v(x_i)^\top \bar{\phi}_t(t_{k,n})/\sigma)}
$$
该损失同时实现两项目标：(1) 视频与其文本袋内的正样本文本对齐（instance-level match）；(2) 具有相同 CLIP 最优匹配文本的视频-文本袋对被拉近（shared-label consistency）。
- **推理增强**：采用线性权重空间集成（weight-space ensembling），以 0.2 比例混合原始 CLIP 权重与微调后权重。

## 实验与结果
- **微调数据**：无标签 Kinetics-400（K400），约 24 万视频，无 ground truth 标注。
- **评估基准**：UCF101、HMDB51、K600、MiniSSv2、Charades、UAV Human、Moments-in-Time（MiT），七个完全未见过的下游数据集，涵盖不同域偏移。
- **主要零样本结果**（UCF101/HMDB51/K600 Top1/Top5）：
  - 原始 CLIP（ViT-B/16）：UCF 69.9 / HMDB 38.0 / K600 Top1 63.5 / Top5 86.8。
  - MAXI（K400 dict. + GPT3 verbs + BLIP verbs，16帧）：**UCF 78.2 / HMDB 52.2 / K600 Top1 71.5 / Top5 92.5**，相对原始 CLIP 提升约 8–14 个百分点。
  - 在 K600 Top1 上，MAXI 超越全监督 ViFi-CLIP（71.2）；在 HMDB51 上与 ViFi-CLIP（51.3）持平并略优。
- **大域偏移基准**（Charades/MiT/MiniSSv2/UAV）：MAXI 在 MiT（Top1 22.91/Top5 46.38）和 MiniSSv2（mAP 6.37）上超越全监督 ViFi-CLIP。
- **少样本结果**：以 MAXI 预训练权重为初始化，在 HMDB（2-shot: 58.0 vs ViFi-CLIP 55.8）、UCF（2-shot: 86.8 vs 84.0）、SSv2（2-shot: 7.1 vs 6.6）等极低数据量场景下均优于全监督预训练的 ViFi-CLIP。
- **关键超参**：温度 $\sigma = 0.02$，学习率 $5 \times 10^{-6}$（cosine decay），batch size 256，训练 10 epoch；文本袋过滤比例 $p = 90\%$，bag size = 16。

## 相关工作脉络
- **CLIP / ALIGN**：大规模图文对比预训练奠定零样本基础，但动词理解弱，本文在其基础上做视频适配。
- **Action-CLIP / XCLIP / A5 / ViFi-CLIP**：均依赖 K400 全标注标签微调 CLIP 用于零样本动作识别；ViFi-CLIP 证明帧级处理+特征池化即可取得优效，本文沿用其架构范式但消除标注依赖。
- **ER-ZSAR / JigsawNet**：利用人工爬取并校正的动作描述进行零样本识别，需大量人工标注工作；MAXI 完全无需人工标注。
- **MIL-NCE（Miech et al.）**：多实例对比学习框架，本文在其基础上增加了共享最优匹配文本的额外约束。
- **BLIP / GPT-3**：分别作为视觉描述生成器和语言知识扩展源，在本文中被创造性地联合用于构建无监督文本信号。

## 局限性与未来方向
- 动作词典的质量对性能有影响（虽然对噪声词典有一定鲁棒性），完全未知的词典场景下效果未验证。
- 文本袋的构建依赖 GPT-3 和 BLIP 两个外部模型，推理/训练流水线复杂度高，实际部署成本较大。
- 实验主要集中在人类动作识别场景，未涉及更广泛的视频理解任务（如视频检索、视频描述等）。
- 未系统探索在更小规模无标签视频数据上的表现（当前使用 K400 约 24 万视频）。

## 研究启发与可借鉴点
1. **"语言知识蒸馏"范式**：利用预训练 VL 模型的匹配结果作为软监督信号，再结合 LLM 的知识扩展能力构建噪声文本袋，是一种低成本的自监督信号构建策略，可迁移到其他视频理解任务。
2. **MIL-NCE 的损失改进**：引入共享匹配文本的辅助约束来增强语义一致性，思路简洁有效，可推广到图像-文本检索等弱监督场景。
3. **动词注入的启示**：消融实验明确揭示了 VL 模型动词表征不足的问题，后续工作可在 prompt 设计或特征对齐中专门强化动词/动作语义。
4. **权重空间集成**：以固定比例线性插值原始权重与微调权重（0.2），在保持泛化能力的同时提升特定任务性能，可作为微调的稳定策略参考。
5. **无监督视频微调 Pipeline**：CLIP 匹配 → LLM 扩展 → 视觉描述生成 → MIL 微调 的四步流水线设计具有高度模块化，可替换各组件（如使用更强的 LLM 或 captioner）进行迭代改进。

## 关键术语表
**Vision-Language (VL) Model**：同时学习视觉和语言表示并在两者间建立对齐的大规模预训练模型，如 CLIP。
**Multiple Instance Learning (MIL)**：一种弱监督学习框架，假设一个"包"（bag）中包含至少一个正例实例，适用于标签不明确或噪声标签的场景。
**Text Bag**：为每个视频构建的包含多条潜在相关文本的集合，作为 MIL 学习中的正例候选集。
**MIL-NCE Loss**：结合多实例学习与噪声对比估计的损失函数，鼓励视频与其文本袋中至少一个文本对齐。
**Zero-Shot Action Recognition**：在训练阶段未接触的目标动作类别上直接进行识别评估的任务设置。
**Weight-Space Ensembling**：通过在权重空间线性插值原始模型和微调后模型的参数来融合泛化能力和任务适应性的集成策略。
**Kinetics-400 (K400)**：包含约 24 万视频、400 类动作标签的大规模人体动作识别数据集。

## 可复现要素
- **数据集**：K400（微调，公开）、UCF101/HMDB51/K600/MiniSSv2/Charades/UAV/MiT（评测，公开）
- **代码**：论文声明代码将在接受后开源（https://github.com/wlin-at/MAXI）
- **关键超参**：$\sigma = 0.02$，lr $= 5 \times 10^{-6}$，batch size $= 256$，epochs $= 10$，p $= 90\%$，bag size $= 16$，frames $= 16$（zero-shot）/ $32$（few-shot），weight ensembling ratio $= 0.2$
- **模型**：CLIP ViT-B/16，BLIP（captioning），GPT-3（text expansion）
