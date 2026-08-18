---
title: "I-can-t-believe-there-s-no-images-Learning-Visual-Tasks-Usin"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_I_Cant_Believe_Theres_No_Images_Learning_Visual_Tasks_Using_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:09:55"
field: "多模态学习"
keywords: ["cross-modal transfer", "zero-shot vision", "contrastive learning", "modality gap", "text-only training", "image captioning"]
innovations: ["提出CLOSE方法实现纯文本训练到图像推理的零样本跨模态迁移", "发现高斯噪声可有效弥合对比模型的模态间隙并显著提升性能", "证明文本向量绝对位置不敏感而相对结构足以支撑跨模态推理"]
benchmarks: ["COCO Captioning", "SNLI-VE", "VQA 2.0", "VQA-E", "Visual News Captioning"]
---

# 论文速读：I can't believe there's no images! Learning Visual Tasks Using Only Language Supervision

## 一句话总结
本文提出了 CLOSE（Cross modaL transfer On Semantic Embeddings）方法，利用对比学习模型（如 CLIP）的文本/图像联合嵌入空间，**仅用文本数据训练模型**，即可在测试时将图像向量替换为文本向量完成视觉任务，实现了零样本跨模态迁移。该方法在图像标注、视觉蕴含、视觉问答和视觉新闻_captioning_ 四个任务上取得了接近全模态训练模型的性能，并在前两个任务上刷新了纯文本训练的 SOTA。

## 研究问题与动机
- **核心问题**：是否可以从纯文本数据中学习高阶语义技能（如解析问题、比较语义、生成描述），并将其零样本迁移到视觉任务上？
- **现有方法不足**：视觉训练数据（带标注的图像-文本对）收集成本高昂；纯零样本方法（如 MAGIC、Socratic Models）无法学习任务特定的风格或输出格式细节。
- **技术挑战**：对比学习模型中存在显著的"模态间隙"（modality gap）——文本向量和图像向量在绝对空间中存在系统性偏移，即使语义匹配对的余弦相似度也可能较低（如 CoCO 仅 0.26）。

## 核心贡献（创新点）
- **提出 CLOSE 框架**：基于冻结的对比编码器进行文本→图像向量替换训练，无需任何图像训练数据即可在视觉任务上推理，与已有工作的本质区别在于完全绕过了视觉训练阶段。
- **揭示高斯噪声的关键作用**：通过在文本嵌入中添加缩放的高斯噪声（hyperparameter _w_ = 0.08），有效弥合模态间隙并增强模型对向量分布偏移的鲁棒性，相比 Song et al. [57] 无噪声方法在视觉蕴含上提升 9 个点（66.6→75.9）。
- **系统性验证跨任务可迁移性**：在 captioning、视觉蕴含、VQA、视觉新闻四个异构任务上一致验证了文本到图像的零样本迁移有效性，并展示了对风格化 captioning 的泛化应用。
- **深入分析模态间隙本质**：通过平移敏感性实验证明 CLOSE 对文本向量的绝对位置不敏感，主要依赖向量间的相对结构关系；同时研究了基于辅助数据（CoCO/CC3M）学习的适配器（线性映射与结构化协方差噪声）对性能的提升潜力。

## 方法详解
- **编码器**：冻结预训练的 contrastive 模型（如 CLIP ViT-L/14）的文本编码器 _f_t_ 和图像编码器 _f_i_，确保预训练学到的语义对应关系不被破坏。
- **输入处理**：文本向量经 L2 归一化后，通过一个可学习的线性层（实验中使用 4 个投影向量）展开为与语言模型 embedding 维度相同的多个向量。
- **语言模型**：使用 T5_base（220M 可训练参数），将投影后的向量与额外文本输入（如问题、假设句）的 token embedding 拼接，构成完整输入序列。
- **生成式训练**：所有任务统一采用自回归生成范式——captioning 生成描述文本，VQA 生成自由形式答案，视觉蕴含生成类别标签（entailment/contradiction/neutral）。
- **高斯噪声适配器**：训练时对文本向量添加 _N(0, w²I)_ 噪声后再重新归一化，噪声尺度 _w_ = 0.08 为默认值，该操作模拟了图像向量相对于文本向量的额外细粒度信息（如光照、背景），迫使模型学习更鲁棒的特征表示。
- **关键公式逻辑**：训练时损失函数为标准语言建模负对数似然，以适配后的文本向量为条件；推理时直接将同一图像用 _f_i_ 编码并替换文本向量，模型无需任何修改即可执行跨模态推理。

## 实验与结果
- **数据集**：CoCO Captioning（Karpathy split）、SNLI-VE、VQA 2.0 / VQA-E、Visual News（VN-CC）。
- **基线**：与 prior text-only 方法 ESPER Style [78]（78.2 CIDEr）、CLIP Cls. [57]（66.6 VE acc）、TAP-C [57]（38.7 VQA）对比；以全模态训练的 CLOSE 作为上界参考。
- **主要结果**：
  - Captioning：CLOSE 多caption设置达 95.3 CIDEr，单caption调优噪声后达 **95.4 CIDEr**，较 ESPER Style 提升 **17.2 点**；全模态上界 113.2。
  - 视觉蕴含：CLOSE 达 **75.9% 准确率**，超越 prior 方法 9.3 点；上界 77.7%。
  - VQA-E：CLOSE 达 61.9%，仅比上界 65.4% 低 3.5 点；VQA 2.0 达 59.6%。
  - 视觉新闻：CLOSE 达 **80.8 CIDEr**，超越此前含图像训练的 best result 50.5 超 **30 点**。
- **额外实验**：使用 GPT-J / OpenAI Curie 生成的合成 caption 训练，最佳达 78.9 CIDEr（CIDEr），超越 MAGIC 零样本方法。

## 相关工作脉络
- **CLIP / ALIGN 等对比模型** [51, 25]：本文利用其联合嵌入空间作为跨模态迁移的基础设施，而非仅用于 zero-shot classification。
- **ESPER Style** [78]：使用 RL 让模型生成 CLIP 评分高的文本，属于 indirect text-only 方法；CLOSE 直接以文本向量作为输入进行端到端训练，更高效。
- **Song et al. (CLIP Few-shot)** [57]：同样使用向量替换思路训练视觉蕴含，但未处理模态间隙；CLOSE 引入高斯噪声后性能显著提升。
- **CROMA** [38]：模态不变特征空间但限于 few-shot 分类；CLOSE 聚焦 zero-shot 生成式任务。
- **Socratic Models / MAGIC** [80, 61]：纯零样本 LLM+CLIP 组合；CLOSE 通过轻量文本微调弥补了零样本方法在风格和格式上的不足。
- **Domain Generalization** [16, 33, 84]：领域泛化中的对抗训练/MixStyle 等技术；本文证明对比嵌入本身即含域不变结构，无需额外对抗损失即可应对极端跨模态偏移。

## 局限性与未来方向
- 模态间隙虽经噪声缓解但未完全消除， captioning 等生成任务与全模态上界仍有约 15-20 CIDEr 的差距。
- 噪声尺度 _w_ 的选取依赖经验，缺乏理论指导；调优需要图像/文本验证集。
- 当前仅验证了文本→图像的单向迁移，反向迁移及其他模态组合的系统性研究尚未展开。
- 对 VQA 2.0 中 caption 与问题对齐不佳的场景仍存在性能衰减，说明纯文本输入会丢失图像细粒度信息。
- 未来可探索更强的 contrastive 模型（如 EVA-CLIP）、结构化适配器，以及扩展至视频/3D/图表等低资源模态。

## 研究启发与可借鉴点
- **高斯噪声弥合模态间隙**：这一简洁有效的策略可迁移到其他跨模态迁移场景（如音频→文本、点云→图像），作为通用的 domain shift 正则化手段。
- **语义相对位置优于绝对位置**：平移敏感性分析表明，保持向量间相对结构不变即可支撑跨模态推理，这为设计轻量级适配器提供了理论依据。
- **大模型合成数据替代标注数据**：本文用 GPT-J/Curie 生成合成 caption 并成功训练 captioning 模型，证明 LLM 可作为低成本的数据放大器，值得在更多视觉-语言任务中探索。
- **风格化 captioning 的零样本学习**：仅需风格示例文本（无需图像-文本对）即可训练风格化生成模型，这一范式可推广至人物视角、情感化描述等多样应用场景。
- **统一生成式框架**：将 captioning、VQA、VE 等多个任务统一为自回归文本生成，简化了多任务系统的设计复杂度，适合后续构建通用视觉-语言代理。

## 关键术语表
- **CLOSE（Cross modaL transfer On Semantic Embeddings）**：本文提出的零样本跨模态迁移方法，通过在文本向量训练、图像向量推理的方式实现任务迁移。
- **Modality Gap（模态间隙）**：对比学习模型中文本向量和图像向量在绝对嵌入空间中存在的系统性偏移，即使匹配对的余弦相似度也可能较低。
- **Contrastive Model（对比学习模型）**：通过 InfoNCE 等对比损失将匹配的图文对拉近、不匹配对推远的多模态模型，如 CLIP、ALIGN。
- **Visual Entailment（视觉蕴含）**：判断图像与文本假设之间的语义关系（蕴含/矛盾/中立）的细粒度视觉理解任务。
- **Visual News Captioning（视觉新闻标注）**：结合新闻文章上下文为新闻图片生成描述的任务，要求提及文章中的人物、地点和事件。
- **T5（Text-to-Text Transfer Transformer）**：Google 提出的统一 seq2seq 架构语言模型，本文使用 base 版本作为任务生成器。
- **CIDEr**：基于共识的图像标注评估指标，通过 TF-IDF 加权 n-gram 匹配衡量生成文本与参考文本的相似度。
- **Adapter（适配器）**：在训练时对输入向量施加的变换模块（如噪声注入、线性映射），用于缩小源模态与目标模态的分布差异。

## 可复现要素
- **数据集**：CoCO Captioning、SNLI-VE、VQA 2.0、VQA-E、VN-CC（均为公开数据集）
- **代码**：论文声明开源（https://github.com/allenai/close，论文标注为 footnote 1）
- **模型权重**：使用预训练的 CLIP ViT-L/14 和 T5_base（开源可用）
- **关键超参**：噪声尺度 _w_ = 0.08（默认），投影向量数 = 4，所有任务共享固定超参；调优噪声实验中使用验证集搜索
- **LLM 生成数据**：GPT-J（6B，开源）、OpenAI Curie（API）
