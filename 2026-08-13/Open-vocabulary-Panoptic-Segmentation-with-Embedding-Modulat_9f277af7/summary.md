---
title: "Open-vocabulary-Panoptic-Segmentation-with-Embedding-Modulat"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Open-vocabulary_Panoptic_Segmentation_with_Embedding_Modulation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:26:25"
field: "开放词汇视觉感知"
keywords: ["开放词汇分割", "全景分割", "CLIP", "Embedding Modulation", "视觉-语言对齐", "Mask2Former", "跨数据集泛化"]
innovations: ["Embedding Modulation 动态融合 query 与 CLIP 嵌入并去偏 logit", "Spatial Adapter + Mask Pooling 高效提取 CLIP 视觉特征", "Mask Filtering 用 IoU head 替代分类分数进行 proposal 过滤"]
benchmarks: ["COCO Panoptic", "ADE20K", "Cityscapes", "PascalContext"]
---

# 论文速读：Open-vocabulary-Panoptic-Segmentation-with-Embedding-Modulation

## 一句话总结
本文提出了 **OPSNet**，一种高效且通用的开放词汇全景分割框架，通过精心设计的 **Embedding Modulation** 模块实现分割模型可学习查询嵌入与 CLIP 预训练视觉-语言嵌入之间的信息交换与增强，在 COCO、ADE20K、Cityscapes 等多个数据集上实现了开放词汇与封闭词汇设置下的 SOTA 性能，且仅需少量额外训练数据。

---

## 研究问题与动机

1. **封闭词汇方法的局限**：传统分割方法（如 Mask2Former）仅在预定义类别集内工作，遇到新类别对象（如"printer""kangaroo"）时完全失效。
2. **现有开放词汇方法的问题**：利用 CLIP 的开放词汇分割方法通常需要大量额外数据从头训练文本-图像对齐，或者在训练域（如 COCO）上性能显著下降，存在**泛化能力与训练域性能难以兼顾**的矛盾。
3. **CLIP 视觉特征利用效率低下**：既有方法（如 [47, 12]）需要将每个 proposal 单独送入 CLIP 图像编码器提取特征，计算开销极大；且裁剪后的 mask 区域缺乏上下文信息。
4. **全景分割的开放性挑战**：现有工作（如 MaskCLIP [13]）是首个尝试开放词汇全景分割的方法，但在 COCO 上的表现远不满意，且在处理重复/重叠 mask 时缺乏有效机制。

---

## 核心贡献（创新点）

1. **提出 OPSNet 框架**：首个面向开放词汇全景分割的高效通用框架，同时兼顾开放词汇泛化与封闭词汇精度。
   - *区别于已有工作*：与 OpenSeg/Dense-CLIP 等仅处理语义分割不同，OPSNet 通过 Mask Filtering 解决 instance-level 区分问题，完成真正的 panoptic 分割。

2. **Embedding Modulation 模块**：动态融合查询嵌入（query embeddings）与 CLIP 嵌入（CLIP embeddings），通过 domain similarity 系数控制融合比例，并用 concept similarity 对 logit 去偏。
   - *区别于已有工作*：不同于简单的加权平均或固定比例融合，该方法根据目标域与训练域的距离自适应调整信息流。

3. **Spatial Adapter + Mask Pooling**：用 re-parameterized 1×1 卷积替代 CLIP 的 attention-pooling 层以保留空间分辨率，并通过 class-agnostic masks 一次性池化提取 CLIP 视觉嵌入。
   - *区别于已有工作*：避免了 [47, 12] 对每个 proposal 单独过 CLIP 编码器的高昂计算代价。

4. **Mask Filtering 机制**：引入 IoU head 预测 mask 与 ground truth 的 IoU，用 IoU 分数替代依赖分类分数的 proposal 过滤。
   - *区别于已有工作*：解决了 CLIP 嵌入不适合背景过滤的问题，使 IoU 估计天然泛化到未见类别。

5. **Decoupled Supervision 范式**：利用图像级标签扩展训练概念（如 ImageNet），同时用 mask 布局自约束监督分割质量。
   - *区别于已有工作*：不同于 Detic/OpenSeg 的方法（需大 batch size 或对多标签不兼容），本文的损失函数设计简洁且无需额外内存开销。

---

## 方法详解

### 整体流程
输入图像 → 分割模型（Mask2Former）预测 class-agnostic masks + 查询嵌入 + IoU 分数 → CLIP 图像编码器经 Spatial Adapter 提取视觉特征 → 交叉注意力增强查询嵌入 + Mask Pooling 得到 CLIP 嵌入 → **Embedding Modulation** 融合生成最终嵌入 → Mask Filtering 过滤低质量 proposal → CLIP 文本编码器提取类别文本嵌入 → 余弦相似度匹配分类。

### 关键设计

**（1）Spatial Adapter**
- CLIP 图像编码器的 attention-pooling 层会降低空间分辨率，将其 re-parameterize 为 1×1 卷积投影到语言空间，保持特征图分辨率不变。

**（2）Embedding Modulation**
- **Embedding Fusion（EF）**：
  - 计算训练类别文本嵌入与预测概念集文本嵌入的余弦相似度矩阵 $H^{M \times N}$。
  - 计算 domain similarity：$s = \frac{1}{M}\sum_i \max_j(H_{i,j})$，反映预测域与训练域的距离。
  - 融合公式：$E_m = E_q + \alpha \cdot (1 - s) \cdot E_c$，其中 $\alpha=10$ 默认值。
  - 原理：当目标域与训练域相似（$s$ 大）时多依赖 query embeddings（在域内类别更准）；差异大时（$s$ 小）多依赖 CLIP embeddings（对 novel 类别更强）。

- **Logits Debiasing（LD）**：
  - $\hat{z}_i = z_i / (\max_j(H_{i,j}))^\beta$，$\beta=0.5$ 默认。
  - 用类别相似度对 logit 去偏，缓解对已见类别（seen categories）的偏向。

**（3）Mask Filtering**
- 在 segmentation model 更新 query 后添加一个 IoU head（单线性层），回归预测 mask 与 GT 的 IoU。
- 用 $L_2$ loss 训练，未匹配或重复 proposal 回归到 0。
- 推理时按 IoU 分数排序过滤，不再依赖分类 score 做 background 过滤。

**（4）Decoupled Supervision（OPSNet+）**
- 引入 ImageNet-Val（50K 图像，1K 类别，每类 50 张）的图像级多标签标注。
- **匹配损失（Classification Sup）**：
  $$\mathcal{L}_{match} = 1 - \frac{1}{c}\sum_{j=1}^{c}\max_i(\delta(S_{i,j})) \cdot \mathbb{1}_{j \in \mathbb{R}^c}$$
  鼓励每个图像级标签至少匹配一个有效 proposal 的嵌入。
- **求和损失（Mask Sup）**：
  $$\mathcal{L}_{sum} = ||1 - \sum_{k=1}^{K}(\sigma(M_{k,i,j}))||_2$$
  约束所有预测 mask 的并集填满图像且不重叠。
- 最终 $\mathcal{L}_{total} = \mathcal{L}_{seg} + 1.0 \cdot \mathcal{L}_{match} + 0.4 \cdot \mathcal{L}_{sum}$。

---

## 实验与结果

### 数据集与评估设置
- **训练集**：COCO Panoptic（封闭词汇）、COCO + ImageNet-Val（开放词汇扩展）
- **测试集**：COCO（训练域）、ADE20K（跨域泛化）、Cityscapes（跨域泛化）、PascalContext（语义分割 mIoU）
- **评估指标**：PQ（Panoptic Quality）、PQ_th（things）、PQ_st（stuff）、SQ（Segmentation Quality）、RQ（Recognition Quality）、mIoU

### 核心结果

| 方法 | Backbone | COCO PQ | ADE20K PQ | Cityscapes PQ | PascalContext mIoU |
|------|----------|---------|-----------|---------------|-------------------|
| Mask2Former [8] | Swin-L† | 54.6 | — | — | — |
| MaskCLIP-Full [13] | ResNet-50 | 30.9 | 15.1 | — | — |
| **OPSNet** | ResNet-50 | **52.4** | **17.7** | **37.8** | 50.2 |
| **OPSNet** | Swin-L† | **57.9** | **19.0** | **41.5** | — |
| **OPSNet+** | Swin-L† | **64.8** | **25.4** | — | **57.5** |
| OpenSeg [15] | Efficient-B7 | 38.1 | 24.8 | — | 45.9 |

- **封闭词汇**：OPSNet（Swin-L†）COCO PQ = 57.9，超越 Mask2Former（54.6）约 **+3.3**，与 SOTA 方法竞争。
- **跨域泛化**：RESNET-50 版 OPSNet 在 ADE20K 上 PQ=17.7，相对 MaskCLIP-Full（15.1）提升 **+2.6**；Cityscapes 上 PQ=37.8，为当时最优。
- **扩展训练概念**：加入 ImageNet 数据后 OPSNet+（Swin-L†）COCO PQ 提升至 **64.8**，ADE20K PQ 达 **25.4**，PascalContext mIoU 达 **57.5**。

### 消融要点
- Embedding Modulation vs 简单集成：固定比例集成在不同数据集上表现各异，而 Modulation 自适应平衡，ADE20K 上 EF+LD 达 17.7 vs 最佳集成 17.4。
- Mask Filtering：仅用 IoU 过滤在 CLIP embeddings 下将 ADE20K PQ 从 10.7 提升至 17.7。
- Decoupled Supervision：Cls Sup + Mask Sup 分别带来 +0.5 和 +0.8 的 ADE20K PQ 提升。

---

## 相关工作脉络

1. **Mask2Former [8]**：统一图像分割的基础模型，采用可学习 query 预测 binary masks，本文以此为基础扩展至开放词汇。
2. **ViLD [16] / Detic [55] / Grounding-DINO [25]**：利用 CLIP/ALIGN 进行开放词汇检测，侧重检测而非分割，且需要大规模辅助数据。
3. **OpenSeg [15]**：首个开放词汇语义分割方法，通过 caption 数据训练 mask-text 对齐，但无法处理重复/重叠 mask，不进行 instance 区分。
4. **MaskCLIP [13]**：唯一的开放词汇全景分割前作，从 CLIP 图像编码器聚合特征，但在 COCO 训练域上性能较差（PQ=30.9）。
5. **Dense-CLIP [54] / SimBase [47]**：用 CLIP 文本嵌入做分类器，但视觉特征提取效率低（每个 proposal 单独过 CLIP）。
6. **CLS Seg（Class-Agnostic Segmentation）**：移除分类头只做实例定位，本文在此基础上增加了开放词汇分类能力。

---

## 局限性与未来方向

1. **CLIP 特征分辨率限制**：即使使用 Spatial Adapter，CLIP 的视觉特征仍基于 patch 级表示，对细粒度边缘分割可能不够精确。
2. **对超大概念集（如 ImageNet-21K）的推理效率**：余弦相似度计算开销随概念数量线性增长，推理时全量概念匹配的计算负担未充分讨论。
3. **跨域泛化仍存在边界**：Fig. 3 定性结果显示对极罕见类别（如 Berkey 数据集中的"penguin"）预测仍有噪声，背景误检问题未完全消除。
4. **Hierarchical prediction 的量化评估不足**：Fig. 5 展示了分层类别预测的定性效果，但缺乏系统性定量分析。
5. **未来方向**：可探索更高效的 CLIP 特征复用策略、将 Hierarchical Concept Set 纳入训练损失、结合视频/3D 场景的时序开放词汇分割。

---

## 研究启发与可借鉴点

1. **Domain Similarity 自适应融合策略**：用训练集与目标集类别间的 cosine similarity 统计量（$s$）动态调节不同嵌入来源的权重，这一思想可迁移到其他跨域视觉-语言任务（如开放词汇检测、VQA）。
2. **IoU Head 替代分类分数做 Proposal 过滤**：将 mask 质量估计与类别无关的设计，天然适用于开放词汇场景，可推广至任何 query-based 分割框架。
3. **Decoupled Supervision 范式**：图像级标签监督分类 + 布局自监督分割的解耦思路，可在其他需要扩展类别但不具备像素级标注的任务中复用。
4. **CLIP 视觉特征的一次性提取**：Spatial Adapter + Mask Pooling 的"只过一次编码器"设计，显著降低推理开销，为后续多模态分割工作提供了高效特征提取范式。
5. **Logits Debiasing 思路**：用概念相似度对分类 logit 进行归一化去偏，与长尾识别中的 Logit Adjustment [34] 思路一脉相承，可扩展至开放词汇场景中的 seen/unseen 类别平衡问题。

---

## 关键术语表

**Open-vocabulary Segmentation**：允许模型识别和分割训练集中未出现的类别的图像分割任务。

**Panoptic Segmentation**：统一语义分割（stuff）与实例分割（things）的任务，为每个像素分配类别并区分同一类别的独立实例。

**CLIP（Contrastive Language-Image Pre-training）**：OpenAI 提出的视觉-语言预训练模型，通过对比学习对齐图像和文本嵌入空间。

**Embedding Modulation**：本文核心模块，通过 domain similarity 控制 query embeddings 与 CLIP embeddings 的动态融合，并对分类 logit 进行去偏。

**Spatial Adapter**：将 CLIP 图像编码器中的 attention-pooling 线性层 re-parameterize 为 1×1 卷积，以保留空间分辨率。

**Mask Pooling**：利用 class-agnostic binary masks 对 CLIP 视觉特征图进行池化，一次性提取每个 proposal 的 CLIP 嵌入。

**Decoupled Supervision**：利用图像级多标签分类数据进行分类监督，同时用 mask 布局求和约束进行分割自监督的联合训练范式。

**Mask Filtering**：通过 IoU head 预测 mask 质量并据此过滤无效 proposal 的机制，替代传统依赖分类分数的 background 过滤。

---

## 可复现要素

- **数据集**：COCO Panoptic（公开）、ADE20K（公开）、Cityscapes（公开）、PascalContext（公开）、ImageNet-Val re-labeled（作者使用）
- **代码**：项目页面 https://opsnet-page.github.io，论文声明开源
- **权重**：基于 Mask2Former 和 ResNet-50 / Swin-L CLIP，均公开可用
- **关键超参**：温度系数 $\tau=0.01$、融合系数 $\alpha=10$、去偏系数 $\beta=0.5$、训练 50 epochs（AdamW，LR=1e-4）、OPSNet+ 微调 80K iterations
- **模型基础**：Mask2Former [8] + CLIP ResNet-50 / Swin-L

---
