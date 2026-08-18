---
title: "FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_FDViT_Improve_the_Hierarchical_Architecture_of_Vision_Transformer_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:05:12"
field: "视觉Transformer高效架构"
keywords: ["Vision Transformer", "Hierarchical Architecture", "Flexible Downsampling", "Masked Auto-encoder", "Model Efficiency", "Image Classification"]
innovations: ["提出非整数stride的灵活下采样层(FD layer)以平滑降低特征图空间尺寸，避免传统stride=2造成的早期信息大量丢失", "引入掩码自编码器训练策略辅助FD layer学习信息丰富的输出，对齐重建任务与分类任务难度"]
benchmarks: ["ImageNet-1k", "MSCOCO 2017", "ADE20K"]
---

# 论文速读：FDViT-Improve-the-Hierarchical-Architecture-of-Vision-Transf

## 一句话总结
本文提出FDViT，通过引入**灵活下采样层（FD layer，支持非整数stride）**和**掩码自编码器训练策略**，改进Vision Transformer的层次化架构，在降低冗余计算的同时保留更多信息，实现更优的精度-效率权衡。

## 研究问题与动机
1. **ViT后层冗余严重**：自注意力机制类似低通滤波，随着网络加深patch间余弦相似度持续上升（最终层超80%），导致大量冗余计算。
2. **现有层次化架构过度补偿**：传统下采样（max-pooling或stride=2卷积）将空间维度减半，数据损失率达50%~75%，在网络早期阶段丢失过多信息，损害性能。
3. **精度与效率难以兼顾**：仅靠增加通道数来缓解信息丢失会同步增加计算量，缺乏一种能平滑控制空间维度缩减的方法。

## 核心贡献（创新点）
1. **灵活下采样层（FD layer）**：突破整数stride限制，可通过非整数stride将特征图空间尺寸平滑缩减至任意预设大小，而非简单地减半。
2. **掩码自编码器辅助训练策略**：将FD layer视为encoder，decoder还原输入，通过masked auto-encoder对齐重建任务难度与分类任务，帮助FD layer学习信息丰富的输出。
3. **整体架构升级**：在FDViT中引入多层（4-5个）FD layer替代传统下采样，以逐步平滑降低patch相似度，同时保持计算开销相近。

## 方法详解
**1. 灵活下采样层（FD layer）**
- 标准卷积输出尺寸公式：$H_{out} = \frac{H_{in} - K_h + 2P_h}{S_h} + 1$，其中$S_h$为整数stride。
- FD layer**放松stride为整数限制**，给定目标输出尺寸$H_{out}$，推导非整数stride：
  $$\hat{S}_h = \frac{H_{in} - K_h + 2P_h}{H_{out} - 1}$$
- 对非整数坐标处的值，通过周围4个辅助点聚合（max pooling / average pooling / **bilinear interpolation**）获取，其中bilinear效果最佳。
- 设空间压缩比$\alpha$、通道扩展比$\beta$，数据损失率变为：$R_d' = 1 - \frac{\beta}{\alpha^2}$。论文选取$\alpha = \beta = \sqrt{2} \approx 1.414$，使$R_d' \approx 0.29$，远低于传统stride=2的$R_d = 0.5$。

**2. 掩码自编码器训练**
- 简单MSE重建损失收敛太快且目标不稳定，效果有限。
- 借鉴MAE思路：对输入施加随机mask $M_r$（mask ratio $r=0.2$），使重建任务难度与分类任务对齐。
- 重建损失：$\mathcal{L}_{recon}^{M_r} = \frac{1}{n}\sum(I_{out}^{M_r} - I_{in})^2$。
- 总损失：$\mathcal{L} = \mathcal{L}_c + \frac{\theta}{S}\sum_S \mathcal{L}_{recon}^{M_r^S}$，其中$\theta=0.1$，$S$为FD layer数量。**mask和decoder仅在训练时使用，推理时不参与**。

**3. 整体架构**
- FDViT-S/B/Ti共5个stage，Stage1为Patch Embedding，Stage2-5交替使用FD layer和Transformer Block。
- 相比基线PiT的2次下采样，FDViT使用4-5次平滑下采样，逐步降低patch相似度（见图1）。

## 实验与结果
**数据集与基线**
- ImageNet-1k（主实验）、MSCOCO 2017（检测）、ADE20K（语义分割）。
- 对比：PiT、HVT、PVT、PoolFormer、Swin、Twins、TPViT等层次化ViT；ResNet、RegNet等CNN；ViT、DeiT等非层次化ViT。

**主要结果（ImageNet-1k）**
| 模型 | Params (M) | FLOPs (G) | Top-1 (%) |
|------|-----------|-----------|-----------|
| ViT-S (DeiT-S) | 22.1 | 4.6 | 79.8 |
| **FDViT-S (Ours)** | **21.5** | **2.8** | **81.5** |
| ViT-B (DeiT-B) | 86.6 | 17.6 | 81.8 |
| **FDViT-B (Ours)** | **67.8** | **11.9** | **82.4** |

- **FDViT-S**：81.5% top-1，较ViT-S **提升1.7个百分点**，FLOPs **减少39%**。
- **FDViT-Ti**：73.7%，较DeiT-Ti提升1.5%，FLOPs减少超50%。
- **FDViT-B**：82.4%，超越ViT-B 0.6个百分点，FLOPs减少约32%。

**下游任务**
- MSCOCO检测：FDViT-S backbone mAP=39.9%，较PiT-S +0.5%，参数少1.9M。
- ADE20K分割：FDViT-S mIoU=44.0，较PiT-S +1.4。

**消融实验关键结论**
- FD layer数量：4层最优（平滑降低相似度，避免过早/过晚降采样）。
- 聚合方式：bilinear > max pooling > average pooling。
- $\alpha/\beta$：1.4/1.4（≈$\sqrt{2}$）最优。
- FD layer + MAE训练联合使用时效果最佳（表6：+0.7% vs 仅FD +0.4% vs 仅MAE +0.3%）。

## 相关工作脉络
1. **ViT（Dosovitskiy et al.）**：纯transformer用于图像分类的开创性工作，固定patch数导致后层冗余，本文改进其层次化版本。
2. **PiT（Heo et al., ICCV 2021）**：最早引入pooling下采样的层次化ViT，但stride=2造成早期信息大量丢失，本文以其为baseline进行改进。
3. **PVT（Wang et al., ICCV 2021）**：金字塔结构ViT，使用stride=2卷积下采样，与本文的核心差距在于下采样方式的灵活性。
4. **HVT / TPViT**：同样采用传统下采样策略，本文的FD layer可替换其下采样模块获得提升（表6验证通用性）。
5. **Swin / Twins**：局部增强型ViT，与本文方向正交——前者约束attention范围，后者减少patch数量，可结合使用。
6. **MAE（He et al., CVPR 2022）**：掩码自编码器预训练范式，本文借鉴其mask策略解决重建任务与分类任务难度不匹配的问题。

## 局限性与未来方向
1. **FD layer的计算开销**：非整数stride涉及双线性插值，推理时的硬件友好性有待验证（尤其移动端部署）。
2. **仅验证了分类/检测/分割**：未在更密集的下游任务（如实例分割、视频理解）上全面评估。
3. **下游扩展依赖预训练**：本文FDViT从随机初始化训练，未探索与大规模自监督预训练（如MAE、DINO）的结合，潜在提升空间未充分释放。
4. **超参数$\alpha/\beta$固定**：不同任务/分辨率可能需要自适应调整，缺乏自动化搜索机制。

## 研究启发与可借鉴点
1. **非整数stride下采样设计**：通过放松stride整数约束实现特征图尺寸的平滑过渡，这一思路可迁移到其他视觉 backbone（如ConvNeXt、Swin-V2）的层次化设计中。
2. **MAE式训练辅助轻量模块**：将重建任务与主任务难度对齐的思路，可用于训练其他"难以直接优化"的中间模块（如压缩层、路由网络）。
3. **数据损失率$R_d$作为分析工具**：用定量指标（而非仅凭经验）评估下采样层的信息保留程度，可作为架构设计的通用诊断工具。
4. **下游通用性验证**：在COCO和ADE20K上验证 backbone 的有效性，为后续迁移到目标检测/分割任务提供了完整参考模板。

## 关键术语表
**FD layer（Flexible Downsampling Layer）**：支持非整数stride的下采样层，可将特征图空间尺寸平滑缩减至任意预设大小，避免传统stride=2造成的早期信息大量丢失。
**非整数stride（Non-integer stride）**：放松标准卷积stride必须为整数的限制，通过双线性插值等聚合相邻辅助点信息来生成目标位置的特征值。
**数据损失率（Data Loss Ratio, $R_d$）**：下采样前后特征图总维度之比，用于量化下采样过程丢失的信息量；传统stride=2卷积的$R_d=0.5$。
**掩码自编码器（Masked Auto-encoder, MAE）**：借鉴自MAE的辅助训练策略，对输入施加随机mask后进行重建，帮助FD layer学习信息丰富的输出表示。
**层次化Vision Transformer（Hierarchical ViT）**：通过多阶段下采样逐步减小特征图空间尺寸、增大通道数的ViT变体，以适配多种视觉任务。
**patch相似度冗余**：ViT中随网络加深，self-attention使各patch特征趋于一致（余弦相似度上升），导致计算冗余。

## 可复现要素
- **数据集**：ImageNet-1k（公开）、MSCOCO 2017（公开）、ADE20K（公开）。
- **代码/权重**：论文未提及开源代码，权重未提供下载链接。
- **关键超参**：$\alpha=\beta=\sqrt{2}$（取整后约1.4）、mask ratio $r=0.2$、$\theta=0.1$、lr=0.001、batch_size=1024、训练300 epoch、AdamP优化器（weight decay=0.05，momentum=0.9）。
