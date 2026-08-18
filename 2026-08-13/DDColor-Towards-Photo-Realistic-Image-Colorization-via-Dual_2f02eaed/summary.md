---
title: "DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Kang_DDColor_Towards_Photo-Realistic_Image_Colorization_via_Dual_Decoders_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:32:41"
field: "图像着色与颜色恢复"
keywords: ["图像着色", "双解码器", "Transformer", "颜色查询", "端到端学习"]
innovations: ["提出端到端双解码器架构，像素解码器恢复空间结构，颜色解码器学习语义感知颜色查询", "首次将查询式Transformer应用于自动图像着色，消除对手工设计前验的依赖", "引入颜色丰富度损失提升生成图像的色彩生动性与自然度"]
benchmarks: ["ImageNet", "COCO-Stuff", "ADE20K"]
---

# 论文速读：DDColor-Towards-Photo-Realistic-Image-Colorization-via-Dual

## 一句话总结
提出了一种端到端的双解码器方法DDColor，通过像素解码器恢复空间结构、基于查询的颜色解码器学习语义感知颜色查询，协同完成高质量图像着色，显著减少颜色溢出效应并提升颜色丰富度。

## 研究问题与动机
1. 图像着色高度不适定，存在多模态不确定性（如物体可能有多种合理颜色）。
2. 传统CNN方法缺乏对图像语义的综合理解，导致着色结果颜色不正确或饱和度不足。
3. Transformer方法虽有效果提升，但依赖手工设计的前验（如颜色掩码、语义-颜色映射），泛化能力受限，且在单尺度特征图上操作易产生明显的颜色溢出。
4. GAN先验方法受限于表示空间，难以处理复杂结构和语义，产生不自然的伪影。

## 核心贡献（创新点）
1. 提出端到端双解码器架构（像素解码器+颜色解码器），像素解码器通过多尺度特征恢复空间结构，颜色解码器学习语义感知的颜色查询，两者协同工作。
2. 设计基于查询的颜色解码器，从无到有学习颜色嵌入，消除了对手工设计前验（如颜色分类或预构建语义-颜色对）的依赖。
3. 引入颜色丰富度损失（colorfulness loss），基于LAB颜色空间中像素云的标准差和均值，引导模型生成更生动自然的颜色。
4. 首次在自动图像着色任务中应用查询式Transformer（query-based transformer），证明其能有效捕捉语义-颜色关联并减少颜色溢出。

## 方法详解
**整体架构**：采用编码器-解码器框架，输入灰度图 $x_L \in \mathbb{R}^{H \times W \times 1}$，预测缺失的两个颜色通道 $\hat{y}_{AB} \in \mathbb{R}^{H \times W \times 2}$（CIELAB颜色空间）。

**编码器**：使用ConvNeXt-L作为主干网络，输出4个中间特征图（分辨率分别为 $\frac{H}{4}\times\frac{W}{4}$、$\frac{H}{8}\times\frac{W}{8}$、$\frac{H}{16}\times\frac{W}{16}$、$\frac{H}{32}\times\frac{W}{32}$）。

**像素解码器**：由4个上采样阶段组成，采用PixelShuffle进行上采样，每个阶段通过跳跃连接融合对应编码器阶段的特征，最终输出与输入相同空间分辨率的图像嵌入 $E_i$。

**颜色解码器**：
- 初始化 $K$ 个零向量作为颜色查询 $\mathcal{Z}_0 \in \mathbb{R}^{K \times C}$
- 颜色解码器块（CDB）包含：跨注意力（cross-attention）→ 多头自注意力（MSA）→ 前馈网络（MLP）
- 跨注意力公式：$\mathcal{Z}_l' = softmax(Q_l K_l^T)V_l + \mathcal{Z}_{l-1}$，其中 $Q$ 来自颜色查询，$K、V$ 来自图像特征
- 采用三尺度特征（$1/16、1/8、1/4$），以轮询方式输入，每组包含3个CDB，共M组（M=3），总计3M个CDB

**融合模块**：通过点积聚合两个解码器输出 $\hat{\mathcal{F}} = E_c \cdot E_i$，再经1×1卷积生成AB颜色通道 $\hat{y}_{AB}$，最后与输入 $x_L$ 拼接得到最终结果。

**损失函数**：
$$\mathcal{L}_\theta = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{per}\mathcal{L}_{per} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{col}\mathcal{L}_{col}$$
- $\mathcal{L}_{pix}$：L1像素损失
- $\mathcal{L}_{per}$：VGG16感知损失
- $\mathcal{L}_{adv}$：PatchGAN对抗损失
- $\mathcal{L}_{col}$：颜色丰富度损失 $1 - [\sigma_{rgyb}(\hat{y}) + 0.3 \cdot \mu_{rgyb}(\hat{y})] / 100$

## 实验与结果
**数据集**：ImageNet（1.3M训练，5k/50k验证）、COCO-Stuff（5k验证）、ADE20K（2k验证，无微调测试）

**评估指标**：FID（↓）、CF（↑）、∆CF（↓）、PSNR（↑）

**主要结果**（ImageNet val5k）：
- DDColor-large（227.9M参数）：FID=3.92（最优），CF=38.26，∆CF=0.05（最优），PSNR=23.85
- DDColor-tiny（55.0M参数）：FID=4.38，CF=37.66，∆CF=0.55
- 相比ColorFormer（44.8M）：FID降低0.99，∆CF降低0.16

**泛化能力**：在COCO-Stuff和ADE20K上均取得最低FID，无需微调。

**消融实验**：
- 去掉颜色解码器：FID从3.92降至4.01
- 去掉颜色丰富度损失：∆CF从0.05升至0.21
- 单尺度特征：FID最高5.09（1/16尺度），多尺度最优3.92
- 颜色查询数最优为100个

## 相关工作脉络
1. **CNN方法（CIC、InstColor）**：预测像素级颜色分布或引入分割掩码引导，但缺乏综合语义理解，结果不饱和或不正确。
2. **GAN先验方法（Wu et al.、BigColor）**：利用预训练GAN的颜色分布先验生成生动颜色，但表示空间受限，复杂结构下产生伪影。
3. **ColTran**：三阶段独立子网（粗着色→上采样），累积误差导致不自然结果。
4. **CT2、ColorFormer**：Transformer方法但依赖手工设计前验（颜色掩码、语义-颜色映射），且单尺度操作导致颜色溢出。
5. **定位差异**：DDColor无需预构建前验，端到端学习颜色查询，多尺度特征有效缓解颜色溢出，泛化性更强。

## 局限性与未来方向
1. 对透明/半透明物体着色仍存在失败案例，需要额外语义监督。
2. 缺乏用户交互控制（如文本提示、颜色涂鸦），未来可引入更多用户输入。
3. 当前方法为全自动，无法实现个性化颜色指定。

## 研究启发与可借鉴点
1. **查询式Transformer应用于着色任务**：首次将DETR-style查询机制用于图像着色，启示可将此范式迁移至图像修复、超分辨率等生成任务。
2. **∆CF新评估指标**：提出生成图像与真实图像颜色丰富度差异，更准确反映自然度，可作为后续研究的参考指标。
3. **多尺度特征在解码器中的协同使用**：像素解码器输出多尺度特征供给颜色解码器，这种"结构恢复+语义引导"的解耦设计具有通用性。
4. **颜色丰富度损失**：基于LAB颜色空间的简单统计量设计损失，有效提升视觉质量，计算代价低，可借鉴到其他色彩相关任务。
5. **零初始化颜色查询**：从零开始学习颜色嵌入，而非预设颜色分类或分布，这一设计避免了手工前验的限制。

## 关键术语表
**CIELAB颜色空间**：包含亮度L和色度AB通道的光感知均匀颜色空间，常用于图像处理。
**Query-based Transformer**：借鉴DETR的检测器架构，使用可学习查询向量通过注意力机制提取特征。
**Color Bleeding（颜色溢出）**：颜色不恰当地扩散到不应该着色的区域，常见于复杂边缘或纹理边界。
**PixelShuffle**：亚像素卷积上采样操作，将低分辨率特征的通道重排为空间维度，避免人工设计的插值。
**Colorfulness Score（CF）**：衡量图像颜色丰富程度的指标，基于LAB颜色空间中像素分布的统计量。
**∆CF**：本文提出的新指标，衡量生成图像与真实图像颜色丰富度之间的差异。
**Cross-Attention**：注意力机制的一种，Query来自一个序列，Key/Value来自另一个序列，用于建立跨模态关联。

## 可复现要素
- **数据集**：ImageNet公开；COCO-Stuff和ADE20K公开
- **代码**：已开源，地址https://github.com/piddnad/DDColor
- **权重**：已开源
- **关键超参**：M=3（组数），K=100（颜色查询数），$\lambda_{pix}=0.1$，$\lambda_{per}=5.0$，$\lambda_{adv}=1.0$，$\lambda_{col}=0.5$，学习率$1e^{-4}$，训练400k迭代，batch size=16，输入分辨率256×256
- **骨干网络**：ConvNeXt-L
- **优化器**：AdamW（$\beta_1=0.9, \beta_2=0.99$，weight decay=0.01）
