---
title: "FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_FDViT_Improve_the_Hierarchical_Architecture_of_Vision_Transformer_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:38:41"
field: "高效视觉Transformer架构设计"
keywords: ["Vision Transformer", "Hierarchical Architecture", "Flexible Downsampling", "Masked Auto-encoder", "Efficient Computer Vision"]
innovations: ["提出非整数步长的柔性下采样层（FD Layer）以平滑压缩空间维度", "引入掩码自编码器训练策略对齐辅助任务难度并促进特征学习", "在保持FLOPs基本不变的前提下降低早期信息丢失率并提升分类与下游性能"]
benchmarks: ["ImageNet-1k", "MSCOCO 2017", "ADE20K"]
---

# 论文速读：FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf

## 一句话总结
论文提出 FDViT，通过引入非整数步长的柔性下采样层（FD Layer）与掩码自编码器训练策略，在层级化 Vision Transformer 中实现空间维度的平滑压缩，在显著降低计算开销的同时缓解早期阶段的信息过度丢失，在 ImageNet 上以更低 FLOPs 取得更优的分类与下游任务性能。

## 研究问题与动机
- ViT 等纯 Transformer 架构在全局自注意力作用下，随着网络加深 patch 间余弦相似度持续上升，产生大量冗余计算，难以部署于资源受限设备。
- 现有层级化 ViT（如 PiT、PVT 等）直接引入 stride=2 的池化或卷积下采样，导致早期层信息丢失率高达 50%~75%，属于“过度补偿”，损害了浅层表征质量。
- 单纯增加通道数虽可降低丢失率，但会同步推高 FLOPs，无法在计算效率与信息保留之间取得平衡。
- 需要一种能按任意预设尺寸平滑缩小特征图空间维度、且保持 FLOPs 大致不变的下采样机制。

## 核心贡献（创新点）
- **提出 FD Layer（Flexible Downsampling Layer）**：利用非整数步长卷积直接生成任意预设尺寸的特征图，替代传统固定 stride=2 的下采样，平滑控制空间压缩比例。
- **设计掩码自编码器训练策略**：将 FD Layer 作为编码器、轻量解码器重建原始输入，并引入 MAE 风格的随机掩码对齐重建任务难度，避免辅助任务过早收敛干扰主分类任务。
- **构建完整的 FDViT 层级架构**：将 FD Layer 与掩码自编码器嵌入 ViT 骨干，提供 Ti/S/B 三种规模，在 ImageNet 分类、COCO 检测与 ADE20K 分割上均取得优于基线的精度/效率权衡。

## 方法详解
- **FD Layer 设计**：给定输入尺寸 $H_{in}$ 与目标输出 $H_{out}$，推导非整数步长 $\hat{S}_h = (H_{in} - K_h + 2P_h)/(H_{out}-1)$。对于卷积核覆盖的非整数坐标，通过周围 4 个辅助点聚合值得到像素值，论文实验表明双线性插值效果最佳。
- **信息丢失率控制**：定义 $R_d' = 1 - \beta/\alpha^2$，其中 $\alpha$ 为空间压缩比、$\beta$ 为通道扩张比。令 $\alpha = \beta = \sqrt{2}$ 时，$R_d' \approx 0.29$，远低于传统 stride=2 卷积的 $R_d=0.5$，且在调整通道数后可保持 FLOPs 基本一致。
- **掩码自编码器训练**：FD Layer 输出送入解码器 $D$（Conv1 → BN → ReLU → FDConv2），重建原始输入。施加二进制掩码 $M_r$（掩码率 $r=0.2$）后计算 $\mathcal{L}_{recon}^{M_r}$，仅在前向训练时生效，推理阶段不参与。
- **联合优化目标**：$\mathcal{L} = \mathcal{L}_c + \frac{\theta}{S}\sum_S \mathcal{L}_{recon}^{M_r^S}$，其中 $\mathcal{L}_c$ 为交叉熵分类损失，$\theta=0.1$，$S$ 为 FD Layer 数量。整体端到端反向传播。

## 实验与结果
- **ImageNet-1k 分类**：FDViT-S 取得 81.5% top-1 准确率，较 ViT-S 提升 1.7 个百分点，同时 FLOPs 从 4.6G 降至 2.8G（减少 39%）；FDViT-Ti 达 73.7%，FLOPs 仅 0.6G，较 DeiT-Ti 降低一半以上；FDViT-B 达 82.4%（FLOPs 11.9G）。
- **对象检测（MSCOCO 2017）**：以 FDViT-S 为骨干配合 Deformable DETR，mAP 达 39.9%，较 ViT-S 提升 3.0%，较 PiT-S 提升 0.5%，参数量更少。
- **语义分割（ADE20K）**：FDViT-S 作为 UperNet 骨干取得 44.0 mIoU，较 PiT-S 提升 1.4。
- **消融结论**：使用 4 个 FD Layer 效果最优；双线性插值聚合函数最佳；$\alpha=\beta=1.4$（即 $\sqrt{2}$ 取整）为最优超参；FD Layer 与掩码自编码器结合在多种层级 ViT 基线上均稳定提升。

## 相关工作脉络
- **ViT [15]**：纯 Transformer 图像分类开端，无下采样机制，序列长度恒定导致计算开销大；FDViT 在此基础上引入层级化空间压缩。
- **层级化 ViT（PiT [28]、PVT [45]、HVT [38]、PoolFormer [58]）**：普遍采用 stride=2 卷积/池化，早期层丢失 50% 数据；FDViT 以非整数步长实现更平滑的尺寸过渡。
- **局部增强 ViT（Swin [34]、Twins [9]、LIT [37]）**：通过限制注意力范围降低复杂度，与 FDViT 的空间压缩策略正交，理论上可结合。
- **掩码自编码器 MAE [21]**：原用于大规模预训练；本文将其思想迁移至训练期辅助 FD Layer 表征学习，通过掩码率调节任务难度。
- **RoIAlign [23] / Deformable Conv [11]**：均利用非整数坐标，但前者需多次插值+池化，后者依赖可学习偏移；FD Layer 参数自由且直接控制输出尺寸，更高效。

## 局限性与未来方向
- FD Layer 的性能依赖辅助点聚合方式（论文选定双线性插值），未探索动态或可学习的聚合策略。
- 掩码自编码器仅用于训练阶段，增加了训练侧复杂度，且未讨论在端侧推理时的实际加速收益。
- 超参数 $\alpha, \beta$ 固定为 $\sqrt{2}$，未进行自动化搜索或按任务自适应调整。
- 动机中提及面向资源受限设备，但实验主要集中于标准 benchmark，缺乏真实移动端/边缘设备的部署耗时与功耗评估。
- 未来可探索与局部注意力机制的结合、硬件感知搜索 $\alpha/\beta$ 配比，以及向视频、医学影像等序列/高分辨率任务扩展。

## 研究启发与可借鉴点
- **非整数步长卷积作为轻量下采样模块**：无需新增可学习参数即可实现任意尺寸特征图输出，可平滑替换其他层级 ViT 中的 stride-2 层，降低信息损失。
- **通过掩码调节辅助任务难度**：将 MAE 风格的掩码策略用于训练期辅助模块（如下采样层、特征金字塔），可有效防止辅助损失过早收敛，值得迁移至多任务/预训练微调流程。
- **以量化指标驱动架构设计**：论文用 patch 余弦相似度曲线与数据丢失率公式明确刻画问题，论证严谨，可作为后续 Transformer 效率优化的参考范式。
- **FLOPs 守恒下的 $(\alpha, \beta)$ 联合设计**：在压缩空间维度的同时反向调节通道维度以维持计算预算，为效率-精度权衡提供简洁的代数约束思路。

## 关键术语表
- **FD Layer（Flexible Downsampling Layer）**：基于非整数步长的卷积下采样模块，可按预设尺寸平滑输出特征图，避免传统 stride=2 造成的信息骤降。
- **Data Loss Ratio ($R_d$)**：下采样前后特征图总维度之比所反映的信息丢失比例，传统步长2卷积为 0.5，FD Layer 可降至约 0.29。
- **Gathering Function**：在 FD Layer 中用于从四个相邻整数坐标点聚合非整数位置像素值的操作，论文优选双线性插值。
- **Masked Auto-encoder Training**：对 FD Layer 输入施加随机掩码后通过编码器-解码器重建，用以平衡重建任务难度并提升特征信息量。
- **Hierarchical Vision Transformer**：沿网络深度逐步降低空间分辨率、提升通道数的 Transformer 骨干架构，典型代表包括 PiT、PVT 等。
- **Patch Similarity / Redundancy**：用余弦相似度衡量的 ViT 各层 token 重复程度，随层数加深显著上升，是触发层级化下采样的核心动机。

## 可复现要素
- **数据集**：ImageNet-1k（公开）、MSCOCO 2017（公开）、ADE20K（公开）。
- **代码/权重**：论文未提及开源代码仓库或预训练权重下载链接。
- **关键超参**：$\alpha = \beta = \sqrt{2} \approx 1.4$（取整后）；掩码率 $r = 0.2$；权衡系数 $\theta = 0.1$；优化器 AdamP（lr=0.001，wd=0.05，momentum=0.9）；batch size=1024；训练 300 epoch，余弦学习率衰减；数据增强与 DeiT 一致。
- **实现细节**：下采样聚合函数选用双线性插值；FD Layer 推理阶段无额外开销；掩码与解码器仅在训练前向参与。
