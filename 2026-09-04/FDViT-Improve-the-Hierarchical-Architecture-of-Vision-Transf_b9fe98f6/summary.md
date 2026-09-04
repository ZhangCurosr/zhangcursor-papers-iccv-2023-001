---
title: "FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_FDViT_Improve_the_Hierarchical_Architecture_of_Vision_Transformer_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:38:27"
---

# 论文速读：FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf

## 一句话总结
本文提出FDViT，通过引入支持非整数步长的灵活下采样层（FD layer）与掩码自编码器（MAE）辅助训练策略，实现了视觉Transformer空间维度的平滑缩减，在大幅降低计算开销的同时提升了分类、检测与语义分割性能。

## 研究问题与动机
- **核心问题**：纯ViT在深层网络中因MSA的低通滤波效应导致patch相似度急剧升高（冗余计算严重），传统层级ViT（如PiT）为缓解该问题采用步长为2的池化或卷积进行下采样，但空间维度直接减半造成早期阶段信息丢失过多（overcompensation），损害最终精度。
- **现有方法不足1**：整数步长下采样固定丢弃约50%的数据量（$R_d=0.5$），而在网络浅层patch相似性尚处于可控范围（图1虚线所示），激进压缩不利于特征表达。
- **现有方法不足2**：若仅通过扩大通道数来补偿信息丢失，会同步显著增加FLOPs，难以在效率与精度之间取得平衡。
- **动机**：设计一种可连续、平滑控制输出尺寸的下采样机制，在不显著增加计算负担的前提下保留更多早期信息，并解决新增下采样层带来的训练不稳定问题。

## 核心贡献（创新点）
- **提出非整数步长灵活下采样层（FD layer）**：打破传统卷积步长必须为整数的限制，允许输出任意预设空间尺寸，将早期下采样的数据丢失率从0.5降至约0.29，本质区别在于无需可学习偏移量或多次插值，以参数自由的方式直接实现平滑降采样。
- **引入MAE辅助的中间层训练范式**：将FD layer作为编码器、配合轻量解码器与随机掩码重建原始输入，使重建任务难度与下游分类任务对齐，推理时该模块完全剥离；与标准MAE预训练的本质区别在于其作用对象是网络中间特征而非整图预训练，仅用于正则化与特征质量提升。
- **构建基于平滑降采样的全阶段层级ViT架构**：以4个FD layer替代传统2个硬性下采样层，重新设计多阶段空间压缩曲线，在ImageNet上实现81.5% Top-1精度（较ViT-S提升1.7pp）且FLOPs降低39%，并在检测与分割任务上验证了骨干网络的泛化优势。
- **建立数据丢失率（$R_d$）量化分析框架**：形式化定义层级下采样过程中的信息保留比例，为ViT架构设计提供了直观的效率-精度权衡度量工具。

## 方法详解
- **FD layer计算机制**：给定输入$Z_{in} \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in}}$，设定目标输出尺寸$H_{out}$，反推非整数步长$\hat{S}_h = (H_{in} - K_h + 2P_h)/(H_{out} - 1)$。对于非整数坐标处的值，通过四个相邻辅助点聚合得到，支持max pooling、average pooling与bilinear interpolation，实验表明bilinear效果最佳。
- **尺寸与计算量平衡设计**：设空间压缩因子$\alpha$与通道扩展因子$\beta$，则数据丢失率优化为$R_d' = 1 - \beta/\alpha^2$。为保持FLOPs大致不变同时降低信息损失，取$\alpha = \beta = \sqrt{2}$（实现时取整为1.4），使$R_d' \approx 0.29$。
- **MAE辅助训练模块**：输入图像经二值掩码$M_r$（掩码比例$r=0.2$）处理后送入FD encoder得到$I_{mid}$，再经解码器$D(\cdot)$重建。重建损失为$\mathcal{L}_{recon}^{M_r} = \frac{1}{n}\sum (I_{out}^{M_r} - I_{in})^2$。该模块仅在训练期参与梯度回传，推理时完全移除。
- **联合优化目标**：总损失为分类损失与加权重建损失之和：$\mathcal{L} = \mathcal{L}_c + \frac{\theta}{S} \sum_{S} \mathcal{L}_{recon}^{M_r^S}$，其中$\theta=0.1$为权衡系数，$S$为FD layer数量。
- **整体网络结构**：以FDViT-S为例，包含Patch Embedding阶段与4个Transformer Block阶段，各阶段后接FD layer实现平滑降采样（空间尺寸依次为27→19→14→10→7），参数量21.5M，FLOPs 2.8G。

## 实验与结果
- **数据集与对比基线**：主干实验在ImageNet-1k（1.28M训练图/1000类）上进行，对比对象涵盖层级ViT（PiT、PVT、HVT、PoolFormer）、局部增强ViT（Swin、Twins、LIT）、非层级ViT（ViT、T2T-ViT、SAViT）及经典CNN（ResNet、RegNet）。
- **分类结果**：FDViT-Ti取得73.7% Top-1，较DeiT-Ti高1.5%且FLOPs
