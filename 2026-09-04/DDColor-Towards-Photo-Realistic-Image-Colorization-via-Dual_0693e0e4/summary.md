---
title: "DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Kang_DDColor_Towards_Photo-Realistic_Image_Colorization_via_Dual_Decoders_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:13:19"
field: "图像着色与修复"
keywords: ["图像着色", "双解码器", "查询式Transformer", "颜色溢出", "CIELAB"]
innovations: ["端到端双解码器架构：像素解码器恢复空间分辨率，颜色解码器学习语义感知颜色查询", "查询式颜色解码器：通过多尺度视觉特征学习颜色查询，无需手工先验", "色彩丰富度损失：直接优化生成图像的视觉鲜艳度和自然度"]
benchmarks: ["ImageNet val5k", "ImageNet val50k", "COCO-Stuff", "ADE20K"]
---

# 论文速读：DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual

## 一句话总结
DDColor提出了一种端到端的双解码器网络用于图像自动着色，通过像素解码器和基于查询的颜色解码器协同工作，避免了手工设计的先验，显著缓解了颜色溢出效应，在ImageNet、COCO-Stuff和ADE20K数据集上均达到了SOTA性能。

## 研究问题与动机
- 图像着色是高度不适定且具有多模态不确定性的经典CV任务，单个灰度图像可能对应多种合理的颜色分布。
- 早期CNN方法缺乏对图像语义的综合理解，导致颜色错误或饱和度不足（如CIC、InstColor、DeOldify）。
- GAN先验方法表示空间受限，难以处理复杂结构和语义，产生不自然的伪影（如BigColor）。
- Transformer方法要么独立训练子网导致误差累积（ColTran），要么仅在单尺度特征图上进行颜色注意力操作，在复杂场景中产生明显的颜色溢出效应（CT2、ColorFormer），且依赖繁琐的手工先验。

## 核心贡献（创新点）
- 提出端到端双解码器网络架构，像素解码器恢复空间分辨率，颜色解码器学习语义感知颜色查询，两者协同生成自然生动的着色结果。
- 设计新颖的查询式颜色解码器，从多尺度视觉特征中学习颜色查询，无需依赖手工设计的先验（如颜色掩码或语义-颜色映射）。
- 引入简单的色彩丰富度损失（colorfulness loss），显著提升生成结果的色彩鲜艳度，使其更符合人类审美。
- 通过交叉注意力建立颜色与多尺度语义表示之间的关联，有效缓解颜色溢出效应，在复杂场景和小目标着色上表现优异。
- 在ImageNet及两个零样本迁移数据集（COCO-Stuff、ADE20K）上均取得最优定量和定性结果，验证了方法的泛化能力。

## 方法详解
- **整体框架**：采用编码器-解码器结构，输入灰度图像 $x_L \in \mathbb{R}^{H \times W \times 1}$，预测缺失的两个颜色通道 $\hat{y}_{AB} \in \mathbb{R}^{H \times W \times 2}$（CIELAB颜色空间）。
- **编码器**：使用ConvNeXt作为骨干网络，提取4个中间特征图，分辨率分别为 $\frac{H}{4} \times \frac{W}{4}$、$\frac{H}{8} \times \frac{W}{8}$、$\frac{H}{16} \times \frac{W}{16}$ 和 $\frac{H}{32} \times \frac{W}{32}$。
- **像素解码器**：使用PixelShuffle作为上采样层，通过跳跃连接整合编码器特征，逐步恢复空间分辨率，最终输出与输入同分辨率的图像嵌入 $E_i \in \mathbb{R}^{\tilde{C} \times H \times W}$。
- **颜色解码器**：由 $3M$ 个颜色解码器块（CDB）堆叠而成，每个块包含：
  - **交叉注意力**：将可学习的颜色嵌入 $\mathcal{Z}_0$ 与多尺度视觉特征进行交互：
    $$\mathcal{Z}_l' = \text{softmax}(Q_l K_l^T) V_l + \mathcal{Z}_{l-1}$$
  - **自注意力+前馈网络**：对颜色嵌入进行进一步变换：
    $$\mathcal{Z}_l'' = \text{MSA}(\text{LN}(\mathcal{Z}_l')) + \mathcal{Z}_l'$$
    $$\mathcal{Z}_l''' = \text{MLP}(\text{LN}(\mathcal{Z}_l'')) + \mathcal{Z}_l''$$
    $$\mathcal{Z}_l = \text{LN}(\mathcal{Z}_l''')$$
  - 颜色查询初始化为零，首先通过交叉注意力获取语义信息，再进行自注意力交互。
- **多尺度特征**：选取像素解码器输出的1/16、1/8、1/4三个尺度的特征，以轮询方式输入颜色解码器，增强对语义边界的识别能力。
- **融合模块**：通过点积聚合像素解码器和颜色解码器的输出：
  $$\hat{\mathcal{F}} = E_c \cdot E_i$$
  再经过 $1 \times 1$ 卷积生成最终的AB颜色通道 $\hat{y}_{AB}$。
- **损失函数**：
  - 像素损失 $\mathcal{L}_{pix}$：L1距离
  - 感知损失 $\mathcal{L}_{per}$：基于预训练VGG16的特征差异
  - 对抗损失 $\mathcal{L}_{adv}$：PatchGAN判别器
  - 色彩丰富度损失（新增）：
    $$\mathcal{L}_{col} = 1 - [\sigma_{rgyb}(\hat{y}) + 0.3 \cdot \mu_{rgyb}(\hat{y})] / 100$$
  - 总损失：$\mathcal{L}_{\theta} = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{per}\mathcal{L}_{per} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{col}\mathcal{L}_{col}$

## 实验与结果
- **数据集**：ImageNet（训练1.3M/测试50k）、COCO-Stuff（5k零样本）、ADE20K（2k零样本）
- **评估指标**：FID（↓越低越好）、CF（↑越高越好）、$\Delta$CF（↓越低越好，衡量与真实颜色的偏差）、PSNR
- **主要结果**（ImageNet val5k）：
  - DDColor-large：FID=**3.92**（SOTA）、CF=38.26、$\Delta$CF=**0.05**（SOTA）、PSNR=23.85
  - 相比ColorFormer（FID=4.91）提升显著，参数量仅44.8M vs 44.8M
- **COCO-Stuff**：FID=5.18（SOTA）、$\Delta$CF=0.24（SOTA）
- **ADE20K**：FID=8.21（SOTA）、$\Delta$CF=0.24（SOTA）
- **消融实验**：
  - 颜色解码器+色彩丰富度损失：FID从6.04降至3.92
  - 多尺度特征：相比单尺度（1/16），FID从5.09降至3.92
  - 颜色解码器架构：cross-attn + self-attn最优
  - 颜色查询数量：100个查询时达到最佳性能
- **用户研究**：在50张测试图像上，20名用户对DDColor的偏好度最高

## 相关工作脉络
- **CIC (Zhang et al.)**：首个DNN像素级颜色分布预测方法，但缺乏语义理解导致颜色饱和度过低；DDColor通过语义感知查询解决此问题。
- **ColTran**：首个Transformer着色方法，但独立训练子网导致误差累积；DDColor通过端到端双解码器避免此问题。
- **CT2 (Weng et al.)**：将着色视为分类任务，依赖预计算的概率分布先验；DDColor通过可学习颜色查询消除手工先验依赖。
- **ColorFormer (Xiaozhong et al.)**：使用混合注意力+语义-颜色记忆网络；DDColor无需预构建先验，通过多尺度特征直接学习颜色查询。
- **BigColor (Geonung et al.)**：利用预训练GAN的颜色先验；DDColor不依赖GAN先验，泛化性更强。
- **DETR (Carion et al.)**：首个将查询式Transformer引入视觉任务的检测框架；DDColor首次将其应用于图像着色任务。

## 局限性与未来方向
- **透明/半透明物体处理困难**：如图9所示，网络在处理透明或半透明物体时可能产生视觉伪影，需要额外的语义监督来理解复杂场景。
- **缺乏用户控制**：与大多数自动着色方法一样，无法接受用户输入（如文本提示、颜色涂鸦）进行引导着色。
- **未来方向**：引入多模态用户输入（文本提示、颜色标注）以提升可控性；结合更强的语义理解模块以改善透明物体着色效果。

## 研究启发与可借鉴点
- **查询式Transformer的应用拓展**：将DETR-style的查询机制迁移至像素级生成任务（着色），为其他图像修复/增强任务提供了新思路。
- **多尺度特征融合策略**：轮询方式将多尺度特征输入解码器，平衡了计算复杂度与表征能力，可借鉴至超分、分割等任务。
- **色彩丰富度损失设计**：基于RGB-YCbCr颜色空间的统计特性设计损失函数，直接优化视觉美感，该思路可迁移至风格迁移、图像增强等任务。
- **双解码器架构**：像素解码器负责空间重建，颜色解码器负责语义-颜色关联，这种分工设计值得在其他多任务学习场景中探索。
- **零样本泛化验证**：在COCO-Stuff和ADE20K上无需微调即达到SOTA，展示了模型强大的跨域泛化能力，可作为评估指标参考。

## 关键术语表
- **Image Colorization**：将灰度图像自动转换为彩色图像的任务，具有高度不适定性和多模态不确定性。
- **Query-based Transformer**：基于可学习查询向量的Transformer架构，通过注意力机制与输入特征交互以提取目标信息（源自DETR）。
- **Color Bleeding**：颜色溢出效应，指颜色错误地扩散到不相关的区域，常见于复杂语义场景的着色结果中。
- **CIELAB Color Space**：CIELAB颜色空间，将图像分为亮度（L）和色度（AB）两个独立通道，便于分别处理明暗和颜色信息。
- **PixelShuffle**：亚像素卷积上采样层，通过重排列低分辨率特征图的通道维度来恢复空间分辨率。
- **Colorfulness Score (CF)**：衡量图像色彩鲜艳度的指标，基于RGB-YCbCr颜色空间中像素云的均值和标准差计算。
- **FID (Fréchet Inception Distance)**：衡量生成图像与真实图像分布相似度的指标，值越低表示生成质量越好。
- **$\Delta$CF**：生成图像与真实图像之间色彩丰富度分数的差异，用于评估着色结果的真实性与自然度。

## 可复现要素
- **数据集**：ImageNet（公开）、COCO-Stuff（公开）、ADE20K（公开）
- **代码**：已开源，https://github.com/piddnad/DDColor
- **模型权重**：已公开
- **关键超参**：
  - Backbone：ConvNeXt-L
  - 颜色查询数量 $K$：100
  - 颜色解码器组数 $M$：3（共9个CDB）
  - 优化器：AdamW ($\beta_1=0.9, \beta_2=0.99$, weight decay=0.01)
  - 初始学习率：$1e^{-4}$
  - 损失权重：$\lambda_{pix}=0.1, \lambda_{per}=5.0, \lambda_{adv}=1.0, \lambda_{col}=0.5$
  - 训练迭代数：400k
  - 批次大小：16
  - 输入分辨率：256×256
  - 硬件：4×Tesla V100 GPU
