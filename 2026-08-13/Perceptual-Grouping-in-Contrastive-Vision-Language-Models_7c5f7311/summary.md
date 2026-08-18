---
title: "Perceptual-Grouping-in-Contrastive-Vision-Language-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ranasinghe_Perceptual_Grouping_in_Contrastive_Vision-Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:26:56"
field: "视觉语言表示学习"
keywords: ["vision-language models", "perceptual grouping", "zero-shot segmentation", "contrastive learning", "spatial localization", "robustness"]
innovations: ["通过max pooling替代CLS聚合使VLM获得空间定位能力", "联合自监督视觉预训练与句子级文本预训练实现弱监督知觉分组", "知觉分组自然缓解虚假关联，提升counterfactual鲁棒性"]
benchmarks: ["ImageNet", "PASCAL VOC", "ADE20K", "COCO", "Waterbirds"]
---

# 论文速读：Perceptual-Grouping-in-Contrastive-Vision-Language-Models

## 一句话总结
本文提出 **CLIPpy**，通过对对比式视觉语言模型（CLIP）进行最小化修改（最大池化替换CLS聚合、引入自监督视觉预训练、句子级文本预训练、token子采样），使模型同时学会语义与空间定位，实现**知觉分组**（perceptual grouping），在无监督分割与零样本语义分割上达到SOTA，并显著提升对虚假关联的鲁棒性。

---

## 研究问题与动机

1. **核心问题**：当前基于对比学习的视觉-语言模型（CLIP/ALIGN）虽能学习通用语义表示，但**几乎无法定位物体在图像中的空间位置**，无法将视觉上相关的部分分组在一起，容易将背景与前景混淆。
2. **现有方法不足**：
   - 部分工作（如 GroupViT、OVS）依赖定制架构或仅关注固定数量的前景物体，忽略背景类；
   - 密集标注驱动的方法（如 MaskCLIP）丧失泛化性与可扩展性；
   - 无监督自监督方法（如 DINO）虽有分组能力，但未与语言模态显式对齐。
3. **鲁棒性问题**：由于缺乏空间理解，模型易学习标签与无关背景之间的**虚假关联**（spurious correlations），在 counterfactual 数据上表现急剧下降。
4. **目标**：在保留弱监督可扩展性的前提下，让模型学会"在哪里"（where）的几何语义信息。

---

## 核心贡献（创新点）

1. **系统揭示对比式VLM的空间定位缺陷**：通过可视化与定量分析证明 CLIP/ALIGN 无法区分前景与背景，预测结果遍布整图。
2. **极简修改实现知觉分组**：仅替换聚合方式（Max pooling）、调整预训练策略与训练时 token 子采样，无需任何分割标注或任务特定微调，即获得强空间感知能力。
3. **无监督底部分组达 SOTA**：在 VOC、ADE20K、COCO、Cityscapes 等多个数据集上，JS（Jaccard Similarity）指标超越此前所有无分割标注训练的方法。
4. **零样本语义分割显著提升**：VOC mIoU 从 CLIP 的 16.4% 提升至 50.8%（+33.3%），COCO 从 8.7% 提升至 25.5%（+16.8%），与定制架构方法（GroupViT、OVS）相比全面领先。
5. **空间感知带来因果鲁棒性**：在 Waterbirds 基准上，CLIPpy 的 domain gap 仅为 2.0%，远低于 CLIP 的 32.1%，与监督训练的最优方法相当。

---

## 方法详解

**整体架构**：与 CLIP 一致，图像与文本分别通过 ViT-B/16 与 T5-base 编码器嵌入，使用对比损失对齐。

**关键设计改动**（绿色标注，见 Fig. 2）：

1. **空间聚合方式：Max Pooling 替代 CLS Token**
   - CLIP 使用 `[CLS]` token 聚合全图信息，导致空间结构被破坏；
   - CLIPpy 采用**全局最大池化**（global max pooling）沿空间维度聚合，使梯度集中于最具判别性的空间位置，从而"聚焦"于物体区域；
   - 假设：max pooling 迫使模型在不同图像的共同物体位置进行保守的最小空间更新，同时满足跨模态对齐。

2. **预训练初始化策略**
   - **图像编码器**：使用 **DINO**（自监督蒸馏）初始化，而非 ImageNet 监督预训练。自监督方法保留细粒度空间表示，有利于分组；监督预训练分离图像级语义，反而损害分组能力。
   - **文本编码器**：使用 **Sentence-T5**（基于 SNLI 的对比式句子嵌入预训练），与对比学习目标对齐。

3. **Token 子采样（Token Sub-Sampling, TSS）**
   - 训练时使用 2 倍分辨率图像，并**随机 drop 80% 视觉 token**，以降低计算开销；
   - 提升推理时高分辨率分割的鲁棒性，同时具有正则化效果。

4. **推理模式**
   - **分类**：空间聚合后单一 token 与文本 prompt 对比；
   - **底部分组**（bottom-up）：对预聚合特征做 PCA 取前 8 主成分作为聚类中心，进行谱聚类；
   - **顶部分组**（top-down）：逐空间位置与语言 token 计算相似度，生成零样本分割图。

**损失函数**：标准对比损失（image-to-text + text-to-image 双向交叉熵），温度系数为 τ。

---

## 实验与结果

**训练数据集**：
- **CC-12M**（1200万图文对，公开）
- **HQITP-134M**（1.34亿图文对，因版权不公开）

**评估基准**：
- 分类：ImageNet、ImageNet-v2
- 无监督分割（JS）：PASCAL VOC、ADE20K、COCO、Cityscapes
- 零样本语义分割（mIoU）：VOC、COCO、COCO(obj)、ADE20K
- 鲁棒性：Waterbirds（counterfactual 背景干扰）

**关键结果**：

| 指标 | CLIP† (CC-12M) | CLIPpy (CC-12M) | 提升 |
|------|---------------|-----------------|------|
| VOC mIoU | 17.5% | **50.8%** | +33.3% |
| VOC JS | 37.3% | **47.5%** | +10.2% |
| COCO mIoU | 7.8% | **23.8%** | +16.0% |
| ADE20K mIoU | 5.0% | **13.1%** | +8.1% |
| ImageNet Acc | 46.0% | 45.3% | -0.7%（基本持平） |

- 在 HQITP-134M 上 VOC mIoU 达到 **52.2%**（+34.1% vs CLIP† 18.1%）。
- **底部分组**：VOC JS 达 54.6%，超越所有先前无监督方法；Cityscapes JS 达 22.3%。
- **鲁棒性（Waterbirds）**：CLIP domain gap = -32.1%，CLIPpy domain gap = **-2.0%**，与最优监督方法（4%-8%）相当。

**消融实验**：
- Max pooling 对分组贡献最大：改回 Avg pooling 后 VOC mIoU 从 50.8% 降至 11.6%；改回 CLS 后降至 4.0%。
- DINO 预训练 vs ImageNet 监督预训练：后者 ImageNet acc 更高（53.3% vs 42.3%），但 VOC mIoU 仅 22.5%，证明自监督初始化对空间分组至关重要。
- Token 子采样带来小幅稳定提升。

---

## 相关工作脉络

1. **CLIP / ALIGN [85, 50]**：本文基线。证明其在弱监督下学习泛化语义表示，但缺乏空间定位能力，易混淆前景与背景。
2. **GroupViT [113]**：定制 ViT 架构，通过离散化注意力 mask 实现分组，但仅关注固定数量前景物体，忽略背景类；本文在相同训练数据上全面超越。
3. **OVS [114]**：使用类似自监督预训练策略，但本文通过聚合方式与训练细节的联合优化，获得更强分组能力。
4. **DINO [12]**：自监督视觉 Transformer，具备底部分组能力，但无语言对齐；本文结合 DINO 初始化与对比式 VLM 训练，同时获得语言条件分组。
5. **MaskCLIP [126] / FILIP [115]**：依赖任务特定微调或密集标注；本文完全无需分割标注与微调。
6. **Waterbirds [91] / 鲁棒表示学习 [65, 76]**：本文证明知觉分组可自然缓解虚假关联，为鲁棒表征学习提供新思路。

---

## 局限性与未来方向

1. **复杂场景性能下降**：在视觉杂乱或标签基数大的数据集（如 ADE20K）上，分割性能相对 VOC 下降明显。
2. **训练数据偏差**：CC-12M 等网络抓取数据存在 inherent bias，可能迁移至模型。
3. **未来方向**：结合更大规模开放数据集（如 LAION-400M、Coyo-700M）与进阶自监督方法，有望进一步提升复杂场景下的分组与分割性能。

---

## 研究启发与可借鉴点

1. **简单设计往往最有效**：仅将 CLS 聚合替换为 Max pooling，即带来 VOC mIoU +33.3% 的提升，说明聚合策略对空间信息的保留具有决定性影响。
2. **预训练目标与下游任务特性需匹配**：自监督预训练（DINO）保留细粒度空间结构，适合需要感知的任务；监督预训练偏向图像级语义，反而损害分组能力。这一原则可迁移至其他需要空间理解的任务。
3. **空间感知与鲁棒性正相关**：知觉分组使模型聚焦于物体本身而非背景，自然抑制虚假关联。这一洞察可为 fairness 与 domain robustness 研究提供新方向。
4. **Token 子采样作为正则化**：训练时随机 drop 大量 token 并配合高分辨率输入，可同时提升计算效率与模型鲁棒性，适用于大规模 VLM 训练。

---

## 关键术语表

- **Perceptual Grouping（知觉分组）**：模型将视觉上相关的像素/区域组合成语义一致区域的能力，包括自下而上（纯视觉）与自上而下（语言引导）两种方式。
- **Zero-shot Semantic Segmentation（零样本语义分割）**：无需像素级标注，仅通过语言描述对 unseen 类别进行分割。
- **Jaccard Similarity（JS）**：无监督分割评估指标，计算预测掩码与真实掩码的 IoU 在所有实例上的平均值，不依赖类别标签。
- **Spurious Correlations（虚假关联）**：模型错误地将标签与无关背景或纹理关联的学习现象，导致在 counterfactual 数据上性能骤降。
- **Max Pooling Aggregation**：沿空间维度取最大值进行特征聚合，相比 CLS 或 Avg pooling 更能保留最强响应区域的定位信息。
- **DINO**：Self-distillation with no labels 的自监督视觉预训练方法，学习具有良好空间结构的 patch 级表示。
- **Sentence-T5**：基于 T5 编码器、在 SNLI 上进行对比式句子嵌入预训练的文本模型，适合跨模态对齐任务。
- **Token Sub-Sampling（TSS）**：训练时随机丢弃 80% 视觉 token，以降低高分辨率训练的算力开销并增强鲁棒性。

---

## 可复现要素

- **数据集**：CC-12M（公开，可从官方下载）；HQITP-134M（因版权问题**不公开**）。
- **代码**：基于 OpenAI CLIP 源码修改，改动极小，已在论文中详细说明；完整代码未单独开源，但实现难度低。
- **预训练权重**：DINO ViT-B/16（公开）、Sentence-T5 Base（HuggingFace 公开）。
- **关键超参**：图像分辨率 224×224（训练）、TSS 随机 drop 80% token、训练 32 GPU × 4 机、ViT-B/16 + T5-base 架构、对比温度 τ 未明确给出（论文未提及具体值）。

---
