---
title: "DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Kang_DDColor_Towards_Photo-Realistic_Image_Colorization_via_Dual_Decoders_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:40:07"
field: "图像上色与恢复"
keywords: ["图像自动上色", "双解码器", "查询型 Transformer", "颜色溢出", "多尺度特征"]
innovations: ["端到端双解码器架构协同像素解码器与查询型颜色解码器", "零初始化颜色查询通过多尺度交叉注意力学习语义感知表示", "颜色丰富度损失与多尺度特征融合缓解颜色溢出"]
benchmarks: ["ImageNet val5k", "ImageNet val50k", "COCO-Stuff", "ADE20K"]
---

# 论文速读：DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual

## 一句话总结
本文提出了一种端到端的双解码器图像自动上色方法 DDColor，通过像素解码器恢复空间结构、颜色解码器学习语义感知的颜色查询，有效解决了传统方法在复杂场景中的颜色溢出和语义不一致问题，在 ImageNet、COCO-Stuff 和 ADE20K 数据集上均取得最佳性能。

## 研究问题与动机
- **高度不适定性**：图像上色需要从灰度图恢复两个缺失的颜色通道（AB 通道），同一物体可能存在多种合理颜色（多模态不确定性）。
- **CNN 方法缺陷**：早期 CNN 方法（如 CIC、InstColor）因缺乏全面的语义理解，常产生不正确或低饱和度的颜色结果。
- **GAN 方法局限**：基于 GAN 先验的方法（如 BigColor、Wu et al.）受限于预训练 GAN 的表示空间，难以处理复杂结构和语义，易产生红色伪影。
- **Transformer 方法不足**：ColTran 采用独立子网导致误差累积；CT2、ColorFormer 仅在单尺度特征上操作，引发可见的颜色溢出，且依赖手工设计的颜色掩码或语义-颜色映射先验，泛化性差。

## 核心贡献（创新点）
1. **端到端双解码器架构**：提出像素解码器和颜色解码器协同工作，像素解码器恢复空间分辨率，颜色解码器学习语义感知颜色表示，两者共同建立颜色与多尺度语义间的关联。
2. **基于查询的颜色解码器**：首次将 DETR 式查询型 Transformer 引入图像上色任务，通过可学习的颜色 embeddings 直接从视觉特征学习颜色表示，无需手工设计的数据集先验。
3. **多尺度特征引导颜色查询优化**：利用像素解码器产生的 1/16、1/8、1/4 三尺度特征作为颜色解码器的输入，通过交叉注意力建立语义与颜色的关联，显著缓解颜色溢出效应。
4. **新型颜色丰富度损失**：引入基于颜色饱和度统计的颜色丰富度损失（colorfulness loss），鼓励模型生成更鲜艳、符合人类审美的上色结果。

## 方法详解
**整体框架**：采用编码器-解码器结构，给定灰度输入 $x_L \in \mathbb{R}^{H \times W \times 1}$，预测两个缺失的颜色通道 $\hat{y}_{AB} \in \mathbb{R}^{H \times W \times 2}$（CIELAB 色彩空间）。

**编码器**：使用 ConvNeXt 作为 backbone，输出 4 个不同分辨率的特征图（$H/4 \times W/4$、$H/8 \times W/8$、$H/16 \times W/16$、$H/32 \times W/32$）。

**像素解码器**：由 4 个阶段组成，每个阶段包含 PixelShuffle 上采样层和 shortcut 连接层。PixelShuffle 将低分辨率特征图 $(h/p, w/p, cp^2)$ 重排为高分辨率 $(h, w, c)$。最终输出与输入同分辨率的图像嵌入 $E_i \in \mathbb{R}^{\tilde{C} \times H \times W}$。

**颜色解码器**：由 M 组颜色解码器块（CDB）堆叠而成，每组包含 3 个 CDB，共 3M 个 CDB。每个 CDB 接收多尺度视觉特征和颜色查询：
- 初始化 K 个零向量作为颜色查询 $\mathcal{Z}_0 \in \mathbb{R}^{K \times C}$
- 交叉注意力：$Q$ 来自颜色查询，$K, V$ 来自图像特征，建立颜色与语义的关联
- 自注意力 + MLP 前馈网络：进一步精炼颜色查询
- 多尺度特征以循环方式喂入各组，依次使用 1/16、1/8、1/4 尺度的特征

**融合模块**：通过点积聚合图像嵌入 $E_i$ 和颜色嵌入 $E_c$，得到 $\hat{\mathcal{F}} = E_c \cdot E_i$，再经 1×1 卷积生成最终颜色通道 $\hat{y}_{AB}$。

**损失函数**：
$$\mathcal{L}_\theta = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{per}\mathcal{L}_{per} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{col}\mathcal{L}_{col}$$
- $\mathcal{L}_{pix}$：L1 像素损失
- $\mathcal{L}_{per}$：VGG16 感知损失
- $\mathcal{L}_{adv}$：PatchGAN 对抗损失
- $\mathcal{L}_{col}$：颜色丰富度损失 $= 1 - [\sigma_{rgyb}(\hat{y}) + 0.3 \cdot \mu_{rgyb}(\hat{y})] / 100$

## 实验与结果
**数据集**：ImageNet（1.3M 训练/50k 测试）、COCO-Stuff（5k 验证）、ADE20K（2k 验证）

**评估指标**：FID（越低越好）、颜色分数 CF（越高越好）、$\Delta$CF（生成图与 GT 的颜色分数差异，越低越自然）、PSNR

**主要结果（ImageNet val5k）**：
- DDColor-large 取得最低 FID：0.96（Best），超越 ColorFormer（1.71）和 CT2（4.95）
- $\Delta$CF 仅 0.44，表明颜色与自然度最佳
- DDColor-tiny（55M 参数）即取得 1.23 的 FID，优于多数更大模型

**泛化能力**：在 COCO-Stuff 上 FID 为 5.18（Best），在 ADE20K 上 FID 为 8.21（Best），未微调即超越所有基线。

**消融实验**：
- 移除颜色解码器后 FID 从 3.92 升至 4.01
- 移除颜色丰富度损失后 FID 升至 5.93
- 单尺度特征（1/4）FID 为 4.44，三尺度融合降至 3.92
- 交叉注意力后接自注意力的架构最优（FID 3.92）
- 颜色查询数 100 时效果最佳（FID 3.92）

**用户研究**：20 名受试者对 50 张图像的偏好评估显示，DDColor 获得最广泛的偏好。

## 相关工作脉络
1. **CIC [49]**：最早 DNN 上色方法之一，预测每像素颜色分布，需额外训练，难以捕捉复杂语义。DDColor 通过端到端查询学习避免此问题。
2. **InstColor [39]**：引入目标检测框作为前景-背景先验。DDColor 无需此类手工先验，直接通过查询学习语义关联。
3. **BigColor [13] / Wu et al. [46]**：利用预训练 GAN 先验增强颜色生动性。DDColor 通过 colorfulness loss 实现类似目标，但无需依赖外部 GAN。
4. **ColTran [24]**：将 Transformer 引入上色，但分三个阶段独立训练，误差累积严重。DDColor 端到端联合训练避免累积误差。
5. **CT2 [45]**：需预计算数据集颜色概率分布作为亮度选择模块。DDColor 摒弃手工先验，完全数据驱动。
6. **ColorFormer [47]**：使用预构建语义-颜色记忆网络。DDColor 的查询机制更灵活，无需构建静态记忆库。

## 局限性与未来方向
- **透明/半透明物体处理困难**：如图 9 所示，对玻璃、水滴等透明物体仍可能产生视觉伪影，需引入额外语义监督。
- **缺乏用户可控性**：当前方法为全自动，不支持文本提示、颜色涂鸦等用户输入。结合多模态提示将是未来方向。
- **颜色查询数量固定**：当前使用固定 100 个查询，对于极端复杂场景可能不够，可探索动态查询机制。

## 研究启发与可借鉴点
1. **双解码器架构的可迁移性**：像素解码器（负责空间结构）+ 查询解码器（负责语义内容）的设计思路可推广至图像修复、超分辨率等任务。
2. **零初始化查询策略**：颜色查询初始化为零向量，通过交叉注意力逐步学习语义感知表示，这一初始化技巧对避免早期训练不稳定有借鉴价值。
3. **交叉注意力优先于自注意力**：消融实验表明先交叉后自注意力的顺序最优，这对查询型 Transformer 的设计顺序提供了重要参考。
4. **多尺度特征融合策略**：颜色解码器以循环方式依次接收 1/16、1/8、1/4 特征，这一轻量高效的多尺度利用方式可借鉴于其他密集预测任务。
5. **颜色丰富度损失的简洁性**：基于 rgby 通道统计的 $\mathcal{L}_{col}$ 形式简单但效果显著，提醒我们在生成任务中可探索更多基于色彩理论的轻量损失函数。

## 关键术语表
**CIELAB 色彩空间**：一种感知均匀的色彩空间，包含亮度通道 L 和两个颜色对立通道 A、B，常用于图像上色任务。
**PixelShuffle**：亚像素卷积上采样方法，将低分辨率特征图的重排操作转换为高分辨率输出，避免去卷积的棋盘效应。
**颜色查询（Color Query）**：可学习的嵌入向量序列，通过注意力机制从图像特征中提取语义感知的颜色表示，类似于 DETR 中的对象查询。
**颜色溢出（Color Bleeding）**：上色结果中颜色跨越语义边界扩散的现象，常见于复杂场景，本文通过多尺度交叉注意力缓解。
**$\Delta$CF**：生成图像与 ground truth 的颜色丰富度分数之差，用于衡量上色结果的自然程度，越低表示越接近真实照片。
**PatchGAN 判别器**：针对局部图像 patches 进行真假判别的全卷积判别器，常用于生成任务的对抗训练。

## 可复现要素
- **数据集**：ImageNet（公开）、COCO-Stuff（公开）、ADE20K（公开）
- **代码**：https://github.com/piddnad/DDColor（已开源）
- **权重**：模型权重已公开
- **骨干网络**：ConvNeXt-L
- **优化器**：AdamW，$\beta_1=0.9, \beta_2=0.99$，weight decay=0.01
- **初始学习率**：$1e^{-4}$
- **损失权重**：$\lambda_{pix}=0.1, \lambda_{per}=5.0, \lambda_{adv}=1.0, \lambda_{col}=0.5$
- **训练迭代**：400,000 次，batch size=16，每 80,000 次及之后每 40,000 次学习率减半
- **输入分辨率**：256×256
- **硬件**：4× Tesla V100 GPU
- **颜色查询数 K**：100
- **解码器组数 M**：3
