---
title: "T-oo-Large-Data-Reduction-for-Vision-Language-Pre-Training"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:32:38"
field: "多模态预训练与数据效率"
keywords: ["Vision-Language Pre-Training", "数据压缩", "多模态学习", "数据集蒸馏", "数据去噪", "Image-Text Alignment"]
innovations: ["基于codebook的编码器-解码器captioner实现多模态数据的高效压缩与caption精炼", "发现并验证纯生成caption会导致contrastive model collapse，拼接方案可有效避免", "系统性揭示VLP大规模数据中的严重不对齐与冗余问题，证明25%高质量数据可媲美全量训练"]
benchmarks: ["MSCOCO", "Flickr30K", "MSRVTT", "VQA", "NLVR²", "RefCOCO+", "COCO Captioning", "ImageNet Zero-shot"]
---

# 论文速读：T oo Large; Data Reduction for Vision-Language Pre-Training

## 一句话总结
本文提出 TL;DR，一种高效的多模态数据集压缩算法，通过基于 codebook 的编码器-解码器选择代表性样本并生成补充 caption，将大规模视觉-语言预训练（VLP）数据压缩至 10%–25%，在七项下游任务上达到与全量数据相当或更优的性能。

## 研究问题与动机
1. **严重图像-文本不对齐**：大规模 VLP 数据集中大量样本的 caption 与图像语义不匹配，低 ITM 分数样本损害多模态表征学习。
2. **数据冗余度高**：现有数据盲目扩张带来巨大存储与计算开销（如 CoCa 在 2048 TPUv4 上预训练约需 5 天），且高质量数据收集过滤成本极高。
3. **"数据越多越好"的信念存疑**：实验表明，丢弃 CC3M 中 50% 最低 ITM 分数的样本后，BLIP 在 COCO 检索任务上性能反而略有提升。
4. **降低 VLP 研究门槛**：高质量多模态数据集获取困难，阻碍了大量研究者的参与。

## 核心贡献（创新点）
1. **首个面向大规模多模态 VLP 数据的高效压缩方法**：将 CC3M 从 2.82M 压缩至 0.67M（约 24%）、YFCC15M 从 15M 压缩至 2.5M（约 16.7%），同时保持甚至超越全量数据的下游性能。
2. **提出基于 codebook 的编码器-解码器 captioner（两阶段流程）**：第一阶段用语言建模损失和对称 commitment 损失训练 codebook-based captioner；第二阶段用所学 codebook 聚类筛选样本，再用 decoder 生成补充 caption 修正不对齐问题。
3. **发现纯生成 caption 会导致模型坍塌，而原始+生成 caption 拼接可有效避免**：指出 captioning collapse 和 one-to-many 问题是导致纯生成方案失效的原因，拼接操作不增加额外计算开销。
4. **系统性地揭示了 VLP 数据的不对齐与冗余问题**：通过 ITM 分数分布分析，为社区重新审视数据效率而非盲目追求数据规模提供了实验证据。

## 方法详解
**整体流程分两个阶段：**

**Stage 1 — Codebook-based Captioner 训练：**
- 视觉编码器采用 ViT-B/16，codebook 包含 K=3000 个可学习嵌入向量，每个图像 token 通过最近邻查找对应 codebook 向量，实现图像特征的量化。
- Codebook 初始化为数据集中出现频率最高的 K 个关键词/词组的 text embedding（NLTK 提取），引入有意义的语义先验。
- 文本解码器采用 BertLMHead Model，训练损失包括：① 自回归语言建模损失（最大化文本似然）；② 对称 commitment 损失（约束 codebook 学习）。
- 先在大规模嘈杂源数据上预训练，再在 COCO、VisualGenome 等小规模数据集上微调。

**Stage 2 — 数据压缩：**
- **样本选择**：将图像特征经 codebook 量化为长度 L=196 的 index vector，将图像空间映射至语义空间，加速聚类；使用 K-Means（Faiss 加速）将数据分为 N=3000 个簇；从每个簇中均匀采样 M% 的数据点（默认 M=25）。
- **Caption 精炼**：对选中样本，将原始 caption $T_o$ 与解码器生成的 caption $T_g$ 拼接为 $T = T_o + T_g$，既修正不对齐又保留原始 caption 独特性。纯 $T_g$ 替换原 caption 会导致 contrastive loss 陷入 model collapse，故不采用。

**关键实验发现（消融）：**
- 均匀采样 vs. 梯度-based / hard-sample / 大距离策略效果相近，说明聚类本身是关键而非具体选取策略。
- Codebook 初始化策略中，关键词初始化优于 Xavier 初始化（TR@1 +0.8%，IR@1 +0.7%）。
- 以 codebook 聚类优于直接使用 Image Embedding 或 Text Embedding。

## 实验与结果
**数据集**：CC3M（2.82M，已清洗）、CC12M（10.8M）、YFCC15M（15M，原始噪声数据）、LAION40M(128)（40M，128×128 分辨率）。

**基线模型**：CLIP、ViLT、BLIP（三种主流架构）。

**下游任务（七项）**：COCO/Flickr30K 图像-文本检索（fine-tuning + zero-shot）、MSRVTT 视频检索、VQA、NLVR²、RefCOCO+、COCO Captioning、ImageNet 零样本分类。

**主要结果（BLIP + CC3M，25% 压缩率）：**

| 任务 | TL;DR-CC3M | 全量 CC3M | 提升/差异 |
|------|-----------|-----------|----------|
| COCO ITR R@1 | 72.8 | 70.9 | **+1.9%** |
| COCO RTR R@1 | 54.8 | 54.3 | +0.5% |
| Flickr30K ITR R@1 | 75.7 | 74.8 | +0.9% |
| Flickr30K RTR R@1 | 95.3 | 94.1 | +1.2% |
| VQA test-dev | 73.1 | 71.5 | **+1.6%** |
| NLVR² test-P | 78.0 | 76.2 | **+1.8%** |
| RefCOCO+ val | 75.1 | 72.4 | **+2.7%** |
| COCO Caption B@4 | 37.6 | 36.8 | +0.8% |
| COCO Caption CIDEr | 123.8 | 121.6 | +2.2% |
| ImageNet zero-shot | 62.0 | 62.5 | -0.5% |

**跨数据集最强结果：**
- **LAION40M(128)**：8M（20%）数据，VQA test-dev **76.3**（+1.8 over 全量 40M 的 74.5），CIDEr **120.9**（+3.5）。
- **YFCC15M**：2.5M（16.7%）数据，NLVR² test-P **75.3**（+1.1 over 全量 15M 的 74.2），RefCOCO+ val **72.6**（+2.0）。

**关键结论**：仅用 10%–25% 高质量压缩数据即可在多数多模态任务上匹配或超越全量数据训练；图像分类任务因视觉多样性降低略有下降（约 -0.5%~-1.5%）。

## 相关工作脉络
1. **Dataset Distillation（如 MMT、D2L）**：生成合成样本压缩数据，但仅适用于小规模低分辨率数据（如 CIFAR），且在 ImageNet-1K 上仅达 33.8% vs. 真实数据 80%+；本文方法面向大规模多模态数据，无需类标签。
2. **Data Pruning（如 [41]）**：选取"困难样本"（hard sample），压缩率约 20%–30%，但仅做选择不做质量修正；本文同样可独立使用 uniform sampling，证明聚类步骤比选取策略更重要。
3. **Neural Data Server（NDS）**：需访问下游数据并借助检索辅助训练，属 task-specific；本文无需任何下游数据，task-agnostic。
4. **CiT（Hu Xu et al., 2023）**：通过动态训练数据调度提升效率；本文是在训练前离线压缩数据，二者互补。
5. **Neural Discrete Representation Learning（VQ-VAE）**：本文 codebook 灵感来源，但将其扩展至多模态 captioning 任务并用于数据压缩。

## 局限性与未来方向
1. **最高压缩比需手动设定**：当前 K（簇数）、M（采样比例）均为人工调参，尚未实现端到端自动优化。
2. **视觉多样性下降导致分类性能微降**：ImageNet 零样本分类在 CLIP/BLIP 上略有下降（-0.5%~-1.5%），因压缩减少图像外观多样性，对单模态分类不利。
3. **文本到图像生成方案效果仍有差距**：使用 Stable Diffusion 等生成图像的训练效果仍显著低于真实数据（TR@1 52.4% vs. 58.3%）。
4. **未充分探索视频-语言等扩展场景**：仅简单验证了 MSRVTT 视频检索，其他多模态领域有待研究。
5. **LAION40M 分辨率限制**：实验中使用了 128×128 低分辨率图像以控制计算成本，更高分辨率下效果可能更好。

## 研究启发与可借鉴点
1. **Codebook 语义映射思路可迁移**：将图像编码为离散 code 再聚类的方法，可推广至视频-语言、音频-语言等多模态领域的数据压缩。
2. **"聚类+均匀采样"策略简洁有效**：消融实验表明聚类步骤本身即最关键，无需复杂选取策略，降低实现难度与调参负担。
3. **Caption 拼接避免 model collapse 的发现**：纯生成 caption 导致 contrastive loss 坍塌的机制分析，对设计多模态数据增强/生成策略具有重要参考价值。
4. **ITM 分数分布作为数据质量诊断工具**：文中可视化展示了 CC3M 大量低 ITM 分数样本，这一分析框架可广泛用于评估其他多模态数据集质量。
5. **与团队方向的结合机会**：若团队关注多模态数据质量筛选或预训练效率优化，TL;DR 的 codebook 机制和拼接 refinement 策略可直接复用或改进。

## 关键术语表
**TL;DR**：本文提出的 Vision-Language 数据压缩算法名称（Take Long; Drop Redundancy），核心思想是从大规模 VLP 数据中选取代表性样本并精炼 caption。
**Vision-Language Pre-Training (VLP)**：在大规模图像-文本对上进行的预训练范式，旨在学习跨模态联合表征，典型模型包括 CLIP、BLIP、ViLT。
**Codebook-based Captioner**：由视觉编码器、可学习 codebook 和文本解码器组成的模块，将图像特征量化为离散 code 后生成 caption。
**Image-Text Matching (ITM) Score**：衡量图像与文本匹配程度的分数，低分数样本代表不对齐的噪声数据。
**Symmetric Commitment Loss**：专为 codebook 设计的损失函数，约束 encoder 输出向 codebook 嵌入 commit，同时保持 codebook 可用。
**Model Collapse**：对比学习训练中，模型因数据同质化而退化为输出固定/相似结果的失败模式。
**Captioning Collapse**：图像描述模型对多样图像生成固定或高度相似 caption 的现象，限制生成多样性。

## 可复现要素
- **数据集**：CC3M、CC12M、YFCC15M、LAION40M 均为公开数据集，论文未提供自行压缩后的数据。
- **代码/权重**：论文未明确提及代码开源情况（ICCV 2023）。
- **关键超参**：codebook 大小 K=3000；簇数 N=3000；采样比例 M=25%（默认）；图像 token 数 L=196（ViT-B/16）；学习率 warm-up 至 3e-4，线性衰减率 0.85；batch size 1260；预训练 20 epochs。
