---
title: "I-can-t-believe-there-s-no-images-Learning-Visual-Tasks-Usin"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_I_Cant_Believe_Theres_No_Images_Learning_Visual_Tasks_Using_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:16:23"
field: "多模态表示学习与跨模态迁移"
keywords: ["跨模态迁移", "零样本学习", "对比学习", "视觉-语言", "模态间隙", "纯文本训练", "图像描述"]
innovations: ["提出CLOSE框架实现仅用文本训练的零样本跨模态视觉任务迁移", "系统分析并缓解对比模型的模态间隙（Modality Gap）", "利用LLM合成数据和风格化文本训练视觉任务模型"]
benchmarks: ["COCO Captioning", "SNLI-VE", "VQA 2.0 / VQA-E", "Visual News"]
---

# 论文速读：I-can-t-believe-there-s-no-images-Learning-Visual-Tasks-Using-Only-Language-Supervision

## 一句话总结
本文提出 **CLOSE**（Cross modaL transfer On Semantic Embeddings）方法，通过在对比训练的视觉-语言模型共享语义空间中训练纯文本模型，仅用语言监督即可在图像编码测试时实现零样本跨模态迁移，在图文描述、视觉蕴含、视觉问答等任务上达到接近双模态训练的效果，并在多个子领域刷新了仅用文本训练的最新结果。

## 研究问题与动机
1. **核心问题**：能否仅用文本数据训练模型，然后将其迁移到视觉任务（Zero-shot Cross-modal Transfer）？即在零图像训练数据的情况下，让模型在图像输入时仍能完成视觉任务。
2. **现有方法不足**：(a) 全零样本方法（如直接用 CLIP 做 classification）难以处理复杂的下游任务如 captioning 和 VQA；(b) 双模态训练需要大量昂贵的图像标注数据；(c) 之前的跨模态迁移方法多聚焦分类，且部分需要少量目标域数据（few-shot），不适用于严格的 zero-shot 场景。
3. **理论依据**：对比学习模型（如 CLIP）的文本和图像编码器在训练时被优化为将匹配对的嵌入拉近，理论上存在共享的语义空间，使得文本处理技能可迁移至图像嵌入。

## 核心贡献（创新点）
1. **提出 CLOSE 框架**：基于冻结的对比模型文本/图像编码器加一个可微调的文本生成模型（T5），通过替换编码器实现零样本跨模态迁移；与之前工作的本质区别在于首次系统地在四种 V&L 任务上证明纯文本训练可达到接近双模态训练的效果。
2. **系统性分析并缓解 Modality Gap**：发现对比模型中图像与文本向量存在系统性偏移（平均余弦相似度仅 0.26 vs 无关对 0.35），并提出添加 Gaussian 噪声作为适配器来弥合此 gap；比先前工作（如 Song et al. 的向量替换方法）更系统地解决了此问题。
3. **突破性地使用合成/风格化文本训练**：利用大语言模型（GPT-J、GPT-3）生成合成 caption 训练 Caption 模型，并用网络评论、书籍、同人小说等非图像数据训练风格化 Caption，无需任何图像标注；此前的风格化 caption 方法均需大量图像-文本配对数据。
4. **全面的迁移敏感性与适配器分析**：证明 CLOSE 对文本向量恒定偏移不敏感（即使偏移量巨大也能工作），并探索了线性适配器与结构化协方差噪声等更复杂的适配策略，揭示了辅助数据质量对性能的影响。

## 方法详解
**CLOSE 整体架构**：
- **编码器**：冻结的 CLIP（ViT-L/14）文本编码器 $E_T$ 和图像编码器 $E_I$，两者均输出单位长度归一化向量。
- **文本输入处理**：输入文本经 $E_T$ 编码为单位向量后，通过一个线性层（linear projection）映射为 $n$ 个（实验中 $n=4$）与语言模型 embedding 维度相同的向量。
- **额外文本输入**：如 VQA 的问题、VE 的假设句，通过 T5 的语言模型 embedding 层编码，与图像/文本向量拼接后送入 T5 解码器生成输出。
- **训练与测试的差异**：训练时仅使用文本编码器输入；测试时替换为图像编码器输入，实现 cross-modal transfer。

**Modality Gap 与 Gaussian Noise 适配器**：
- 现象：对比损失只要求匹配对相对随机对更近，不保证绝对距离接近，导致图像和文本向量在语义空间中分属不同聚类（图 3a）。
- 解决方案：在训练时对文本向量添加从高斯分布采样的噪声 $\epsilon \sim \mathcal{N}(0, w^2I)$，其中 $w$ 为超参数（默认 0.08），添加后重新归一化。
- 双重作用：(1) 使文本向量分布扩散并与图像向量重叠；(2) 模拟图像向量中包含的额外视觉细节（光照、背景等）引入的不确定性，增强模型鲁棒性。

**各任务的具体设置**：
- **Captioning**：输入文本 = 训练集 caption，输出 = 目标 caption；多 caption 设置下用不同 caption 作输入/输出。
- **Visual Entailment**：输入文本 = 前提句（替代图像），输出 = 类别标签（entailment/contradiction/neutral）。
- **VQA**：输入文本 = 场景描述句，额外输入 = 问题，输出 = 答案。使用 VQA-E 数据集避免 caption 与问题不对齐问题。
- **Visual News**：输入文本 = caption，额外输入 = 新闻文章，输出 = 目标 caption。

## 实验与结果
**数据集**：COCO Captioning（Karpathy split）、SNLI-VE、VQA 2.0、VQA-E、Visual News；辅助数据包括 CC3M。

**基线对比**：
- Captioning：超越 ESPER Style（78.2 CIDEr）17 分（CLOSE 文本仅 95.4 CIDEr），远超 MAGIC（49.3）和 Socratic Models（44.5）。
- Visual Entailment：超越 CLIP Cls.（66.6%）9 分（CLOSE 75.9%）。
- Visual News：**超过此前含图像的最佳结果**（50.5 → 80.8 CIDEr），提升 30+ 分。
- VQA-E：CLOSE 文本仅模型达 62.9%（调噪声后 64.3%），接近图像训练模型的 67.9%。

**关键数值**（Table 1）：
| 模型 | Caption (Single) | Caption (Mult.) | VE | VQA-E | VN |
|---|---|---|---|---|---|
| CLOSE w/o Noise | 16.4 | 68.7 | 68.2 | 59.8 | 32.1 |
| **CLOSE (Ours)** | **80.5** | **95.3** | **75.9** | **62.9** | **80.8** |
| CLOSE w/Tuned Noise | 95.4 | 98.4 | 75.9 | 64.3 | 80.8 |
| CLOSE w/Images | 113.2 | 113.2 | 77.7 | 67.9 | 105.7 |

**LLM 合成数据训练**（Table 2）：GPT-J Unigram 方法达到 78.9 CIDEr，超过 MAGIC（49.3 CIDEr）。

**适配器分析**（Table 4）：使用 COCO restval 数据训练的 Structured Covariance 适配器可将 Caption 提升至 106.5 CIDEr（+11.1 over baseline）。

**不同模型变体**（Table 5）：使用更强的对比模型（OpenCLIP、EVA-CLIP）可进一步提升性能，EVA-CLIP + CLOSE 在 VQA-E 达 66.6%，接近图像训练模型的 67.9%。

## 相关工作脉络
1. **CLIP/ALIGN 等对比 V&L 模型**（Radford et al. 2021; Jia et al. 2021）：CLOSE 直接利用其共享嵌入空间，但不同于它们主要用于 zero-shot classification 或特征提取器，CLOSE 在共享空间上微调整个生成模型。
2. **Song et al. [57]（ACL 2022）**：同样使用向量替换思路训练 Visual Entailment，但**不使用噪声等 modality gap 缓解策略**，性能显著低于 CLOSE。
3. **Yu et al. [78]（ESPER Style, 2022）**：用 RL 训练模型生成 CLIP 认为与图像接近的文本，同时用文本学习 caption 风格；与 CLOSE 的区别是 ESPER 间接利用 CLIP 打分而非直接在共享空间迁移。
4. **Nukrai et al. [48] 和 Wei et al. [35]（ICLR 2023, 同期工作）**：分别提出用 Gaussian 噪声和投影的纯文本 caption 方法；CLOSE 更全面地分析 modality gap、涵盖更多任务并展示 LLM 合成数据应用。
5. **CROMA（Liang et al. 2021）**：使用模态不变特征空间实现跨模态迁移，但局限于分类任务且为 few-shot；CLOSE 在 zero-shot 下处理更复杂的生成任务。
6. **风格化 Captioning 方法**（Tan et al. [60], Shuster et al. [56] 等）：均需图像-文本配对数据或对抗训练；CLOSE 完全不需要图像数据即可学习风格化 caption。

## 局限性与未来方向
1. **与图像训练模型仍有差距**：Captioning 单 caption CIDEr 95.4 vs 113.2（约 18 分差距），VQA-E 62.9 vs 67.9（约 5 分差距），说明纯文本迁移存在上限。
2. **Modality Gap 未完全消除**：Gaussian noise 是启发式方案，并非精确对齐两种模态的嵌入空间。
3. **VQA 在 VQA 2.0 上差距较大**：因 caption 与问题间存在不对齐，部分问题在文本中无法回答。
4. **适配器依赖高质量配对数据**：Structured Covariance 适配器的效果高度依赖辅助数据质量和来源（COCO 有效，CC3M 效果差）。
5. **未来方向**（作者提及）：可扩展至视频、点云、音频等其他模态；可用于 3D 场景描述、视频摘要、图表/传感器数据等低资源视觉模态任务。

## 研究启发与可借鉴点
1. **Modality Gap 的噪声正则化思路可迁移**：在任意使用对比学习共享空间的跨模态迁移场景中，Gaussian noise 注入是一种简单有效的 domain shift 缓解策略，可推广至视频-文本、音频-文本等任务。
2. **LLM 合成数据替代人工标注**：用 GPT-J/GPT-3 生成合成训练数据训练视觉任务模型，仅需少量 prompt 设计成本，为低资源视觉任务提供了新的数据获取路径。
3. **冻结预训练编码器 + 轻量微调的设计范式**：冻结 CLIP 编码器、仅微调 T5 和线性层（220M 可训练参数）的方式计算高效且保持了对比学习的语义结构，适合资源受限场景。
4. **敏感性分析的价值**：对恒定偏移的鲁棒性分析表明，即使模态间存在较大绝对差距，只要相对分布关系保持，迁移即可有效——这为理解跨模态迁移机制提供了新视角。
5. **风格化生成的零样本范式**：从非图像文本源（评论、小说、LLM 生成）学习风格并迁移到图像 caption，展示了"text-as-training-data"范式的灵活性。

## 关键术语表
**CLOSE（Cross modaL transfer On Semantic Embeddings）**：本文提出的零样本跨模态迁移框架，通过替换对比模型的文本/图像编码器实现从纯文本训练到图像测试的迁移。
**Modality Gap（模态间隙）**：对比学习中图像和文本向量在共享嵌入空间中存在的系统性位置偏移现象，即使匹配对也被训练为距离较远。
**Contrastive Learning（对比学习）**：通过拉近匹配对（如图像与其 caption）的嵌入、推远非匹配对的嵌入来学习跨模态共享语义空间的方法。
**Visual Entailment（视觉蕴含）**：判断图像前提与文本假设之间的语义关系（蕴含/矛盾/中立）的细粒度图像理解任务。
**Gaussian Noise Adapter**：向文本嵌入添加缩放高斯噪声的适配器方法，用于弥合模态间隙并增强模型对域 shift 的鲁棒性。
**Structured Covariance Noise**：利用辅助配对数据学习噪声的均值和协方差矩阵，以更好地模拟真实文本-图像偏移的先进适配器。
**VQA-E**：经过人工验证的 VQA 子集，确保 caption 中包含问题的可答信息，用于公平评估纯文本训练的 VQA 模型。
**Zero-shot Cross-modal Transfer**：在目标模态无任何训练数据的情况下，将从源模态学到的技能直接迁移到新模态的任务设定。

## 可复现要素
- **数据集**：COCO Captioning（Karpathy split）、SNLI、SNLI-VE、VQA 2.0、VQA-E、Visual News；辅助数据 CC3M（公开）；LLM 合成数据由作者生成
- **代码**：论文声明已开源（代码链接见 footnote 1）
- **模型权重**：使用标准 CLIP ViT-L/14 和 T5-base（均公开可用），CLOSE 模型权重随代码开源
- **关键超参**：噪声水平 $w=0.08$（默认，对所有任务统一）；线性层输出 4 个向量；CLIP 编码器冻结；T5 语言模型微调
- **基线模型**：CLIP ViT-L/14、OpenCLIP ViT-L/14、EVA-CLIP、T5-small/base/large、GPT-J（6B）、OpenAI Curie
