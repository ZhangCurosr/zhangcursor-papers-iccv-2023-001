---
title: "Exploring-Open-Vocabulary-Semantic-Segmentation-from-CLIP-Vi"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:14:08"
field: "开放词汇语义分割"
keywords: ["open-vocabulary segmentation", "zero-shot semantic segmentation", "CLIP distillation", "masked autoencoder", "visual-language model"]
innovations: ["仅从CLIP视觉编码器蒸馏训练零样本分割模型，无需像素级或文本标注", "提出多尺度特征蒸馏损失与segment matching损失", "在1.3M图像上训练达到与20M图文对方法相当的性能，计算效率提升9倍"]
benchmarks: ["PASCAL VOC 2012", "PASCAL Context", "COCO", "Conceptual Caption"]
---

# 论文速读：Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only

## 一句话总结
本文提出 ZeroSeg，一种仅通过蒸馏预训练 CLIP 视觉编码器知识、无需任何像素级标注或文本标注的零样本开放词汇语义分割方法，在 ImageNet-1k（1.3M 图像）上训练即可达到与训练数据量大 15-20 倍的对标方法相当甚至更优的分割性能。

## 研究问题与动机
- **核心问题**：现有语义分割方法依赖昂贵的人pixel级标注或大规模图文对训练，难以扩展到开放词汇场景；直接将 CLIP 等视觉-语言模型用于像素级分割效果不佳，因其仅在图像级别训练。
- **现有方法不足**：GroupViT、MaskCLIP 等零样本分割方法需 20M-26M 图文对训练，且对子词/复合词（如"ground""bedclothes"）敏感；MaskCLIP+ 仍需伪掩码生成和适配步骤。
- **动机**：探索仅凭 CLIP 视觉编码器（作为教师）蒸馏知识，能否在不依赖任何文本或像素标注的情况下训练出高效的零样本语义分割模型。

## 核心贡献（创新点）
1. **提出 ZeroSeg 框架**：首次实现仅从 CLIP 视觉编码器蒸馏即可训练开放词汇零样本语义分割模型，无需像素级或文本标注。
2. **设计多尺度特征蒸馏损失**：通过将图像划分为 2×2、3×3、4×4 网格并输入 CLIP 提取多尺度特征，使全局表示捕获多分辨率语义信息。
3. **提出 segment matching loss**：将每个 segment token 与其最对齐的多尺度局部区域特征匹配，避免 segment 间语义混淆。
4. **高效率训练**：仅用 ImageNet-1k（1.3M 图像）训练，GPU 耗时仅 84 小时（约为 GroupViT 的 1/9）， Yet 在 VOC/Context/COCO 上达到竞争力的零样本结果。
5. **开放词汇优势**：在 1000 类开放词汇人 studies 中，ZeroSeg 获得 68% 偏好票，显著优于 GroupViT（32%），且在子词/复合词类别上平均提升 18%。

## 方法详解
**整体架构**：基于 ViT-Base（12 层 encoder）+ 两个 Transformer decoder（重建头 8 层、分割头 5 层）+ 两组 grouping stages（32 和 8 个 learnable group tokens）。

**多尺度特征提取**：将输入图像划分为 1×1（全图）、2×2、3×3、4×4 网格，每个子图 resize 到 224×224 后输入预训练 CLIP-L 视觉编码器，得到多尺度特征集合 {v₁, v₂, ..., vₙ}。

**多尺度特征蒸馏损失**：对 segment tokens 经 Transformer + avg pooling + MLP 得到全局表示 z，计算 L₁ 距离：
$$\mathcal{L}_{distill} = \sum_{k=1}^{n} \| \mathbf{z} - \mathbf{v}_k \|_1$$
强制 z 捕获多尺度区域语义。

**Segment matching loss**：对每个 segment token gᵢ 匹配最近的局部区域特征 vⱼ（不含全图特征 v₁）：
$$\mathcal{L}_{match} = \sum_{i=1}^{m} \min_{j} \| \mathbf{g}_i - \mathbf{v}_j \|_1$$
确保每个 segment token 聚焦于物体中心化的语义区域。

**总损失**：$\mathcal{L} = \mathcal{L}_{recon}$（MAE 重建 MSE）+ $\mathcal{L}_{distill}$ + $\mathcal{L}_{match}$。训练时 mask ratio=60%，batch size=4096，学习率 1.5e-4，AdamW，80 epochs（前 20 epoch warmup）。

## 实验与结果
**数据集**：PASCAL VOC 2012（20 类）、PASCAL Context（59 类）、COCO（80 类）；开放词汇评估用 ImageNet 1000 类 + Conceptual Caption 200 张。

**主要结果（zero-shot，mIoU）**：

| 模型 | 训练数据 | VOC | Context | COCO |
|------|----------|-----|---------|------|
| GroupViT | CC12M+YFCC (26M) | 52.3 | 22.4 | 24.3 |
| MaskCLIP | LAION-20M (20M) | — | 17.7 | 11.8 |
| ZeroSeg | IN-1k (1.3M) | **40.8** | **20.4** | **20.2** |
| ZeroSeg | CC3M+COCO (3.4M) | 37.3 | 19.7 | 17.8 |

- ZeroSeg（IN-1k）超越 DINO（39.1）+1.7、MoCo（34.3）+6.5，超过 MaskCLIP（+2.7 Context, +8.4 COCO）且训练数据仅为其 1/15。
- 人机研究：ZeroSeg 在开放词汇 1000 类上获 68% 偏好票（vs GroupViT 32%）。
- 子词/复合词类别平均 mIoU：ZeroSeg 27.85 vs GroupViT 9.78（+18.07）。
- 计算效率：84 GPU 小时 vs GroupViT 768 小时（1/9）。

## 相关工作脉络
1. **GroupViT**（Xu et al., CVPR'22）：从文本监督学习分割，需 26M 图文对；ZeroSeg 仅用 CLIP 视觉编码器蒸馏，无需文本。
2. **MaskCLIP/MaskCLIP+**（Dong et al.）：利用 mask 生成器+伪掩码；ZeroSeg 完全无标注。
3. **SegCLIP**（Luo et al.）：需 CC3M+COCO 图文对；ZeroSeg 用更少数据达可比结果。
4. **DINO/MoCo/DeiT**：自监督分类预训练+微调；ZeroSeg 为零样本无需微调。
5. **MAE**（He et al.）：掩码自编码器；ZeroSeg 借其架构提升训练效率。
6. **DenseCLIP**（Zhou et al.）：从 CLIP 提取密集标签；仍需适配，ZeroSeg 端到端蒸馏。

## 局限性与未来方向
- **分割边界较粗**：相比像素级监督方法，boundary 精度不足，限制实际部署。
- **依赖 CLIP 视觉编码器质量**：教师模型能力决定上限；可探索更强基础模型（如 GPT-4V）作为蒸馏源。
- **未用文本信息**：虽避免文本偏见，但也损失了 CLIP 文本编码器的语义对齐优势。
- **未来方向**：结合像素级微调、引入更强 VL 教师、扩展至视频/3D 分割任务。

## 研究启发与可借鉴点
1. **蒸馏-only 思路**：将预训练模型当作教师、设计匹配损失蒸馏到轻量架构，避免大规模标注；可迁移到其他密集预测任务（深度估计、法线估计）。
2. **多尺度网格特征提取**：将单图划分多网格输入教师模型提取多尺度表示，简单有效；适合任何仅有图像级监督的场景。
3. **Segment matching loss**：用最近邻 L₁ 匹配强制 token-region 对齐，避免 grouping 过程中的语义泄漏；可推广到任意 token-based segmentation 架构。
4. **无文本训练的开放词汇分割**：证明不依赖文本也能获得良好开放词汇泛化，尤其对子词/复合词更鲁棒；可重新审视文本标注的必要性。
5. **MAE + 分割头联合训练**：重建任务辅助分割表示学习，60% mask ratio 为像素级任务的最优折中（vs MAE 原文 75%）。

## 关键术语表
- **ZeroSeg**：本文提出的零样本开放词汇语义分割模型，仅从 CLIP 视觉编码器蒸馏训练。
- **Segment token**：可学习的分组 token，聚合语义相似的区域特征，对应一个 disjoint 图像区域。
- **Multi-scale feature distillation**：将图像划分为多网格输入 CLIP 提取多尺度特征，蒸馏到全局表示 z 的过程。
- **Segment matching loss**：强制每个 segment token 与最近局部区域特征对齐的 L₁ 匹配损失。
- **Open-vocabulary segmentation**：使用不受限词汇表（如 1000 类）进行语义分割，不限于训练时见的类别。
- **Masked autoencoder (MAE)**：通过掩码重建学习视觉表示的自监督方法，ZeroSeg 借其架构提升效率。
- **GroupViT**：从文本监督学习分割的基线方法，需 26M 图文对训练。
- **mIoU**：mean Intersection over Union，语义分割常用评估指标，衡量预测 mask 与 ground truth 的重合度。

## 可复现要素
- **数据集**：ImageNet-1k（训练）、PASCAL VOC 2012/PASCAL Context/COCO（评估）；均公开。
- **代码**：https://github.com/facebookresearch/ZeroSeg（开源）。
- **权重**：论文提供预训练权重。
- **关键超参**：ViT-Base，mask ratio=60%，batch size=4096，lr=1.5e-4，80 epochs（前 20 warmup），AdamW，图像 resize 到 224×224，中心裁剪；推理时短边 resize 到 448，阈值 VOC=0.95/Context=0.05/COCO=0.35。
