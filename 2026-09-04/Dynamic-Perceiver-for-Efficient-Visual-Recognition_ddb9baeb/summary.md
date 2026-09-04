---
title: "Dynamic-Perceiver-for-Efficient-Visual-Recognition"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:22:10"
field: "高效视觉识别"
keywords: ["Dynamic Early Exiting", "Vision Recognition", "Perceiver", "Dual-branch Architecture", "Efficient Inference"]
innovations: ["双分支解耦架构分离特征提取与早期分类", "对称双向交叉注意力实现渐进式信息融合", "仅分类分支部署早退分类器避免性能退化"]
benchmarks: ["ImageNet", "Something-Something V1", "COCO"]
---

# 论文速读：Dynamic-Perceiver-for-Efficient-Visual-Recognition

## 一句话总结
论文提出 **Dynamic Perceiver (Dyn-Perceiver)**，通过双分支架构显式解耦特征提取与早期分类任务，将多个分类器仅置于分类分支中，在保持后期出口性能的同时实现动态早退推理，显著提升多种视觉骨干网络的推理效率。

## 研究问题与动机
- **早退网络的性能退化问题**：现有动态早退网络（如 MSDNet、RANet）将分类器直接构建在中间特征上，迫使低层特征编码高层语义并保持线性可分，严重损害最终出口性能。
- **特征提取与分类任务的耦合**：传统方法中特征提取与早期分类紧密交织，浅层网络需同时完成特征学习和分类适配，设计次优。
- **Perceiver 架构的计算成本挑战**：Perceiver 虽引入 latent code 解耦输入与任务，但在视觉识别中因图像 token 数量庞大，计算开销过高。
- **动态计算资源的灵活分配需求**：静态模型对简单样本浪费计算，需要自适应不同输入复杂度的动态推理机制。

## 核心贡献（创新点）
1. **双分支解耦架构**：提出特征分支与分类分支分离的设计，特征提取与早期分类不再共享底层参数，与 MSDNet/RANet 等方法在中间特征上直接加分类器的设计本质不同。
2. **对称双向交叉注意力机制**：设计 X2Z（特征→latent code）和 Z2X（latent code→特征）交叉注意力，实现两分支间渐进式信息融合，区别于 Perceiver 单向蒸馏输入的架构。
3. **仅分类分支部署早退分类器**：所有早期分类器仅置于分类分支末尾，避免干扰特征提取过程，实验证明甚至能提升最终出口精度（+0.7%）。
4. **通用且简洁的框架设计**：可在任意视觉骨干（ResNet、RegNet、MobileNet）上构建，无需如 MS-DNet 般手工设计复杂结构，并成功扩展至动作识别与目标检测任务。
5. **Forward Knowledge Transfer (FKT) 模块**：提出早退分类器到深层分类器的知识传递机制，在端到端训练下直接提升各出口性能，无需预训练-微调范式。

## 方法详解
**整体架构**：包含 4 个 stage 的双分支结构：
- **特征分支（Feature Branch）**：采用 CNN（ResNet/RegNet/MobileNet-v3）从低层到高层提取图像特征 $\mathbf{X}_0 \sim \mathbf{X}_4$。
- **分类分支（Classification Branch）**：输入可训练的 latent code $\mathbf{Z}_0$（随机初始化），通过 self-attention 处理分类任务。

**核心组件**：
1. **X2Z Cross Attention**（阶段起始）：latent code 作为 query，图像特征 $\mathbf{X}_{i-1}$ 作为 key/value。为降低计算量，先用 depth-wise convolution (DWC) 增强局部特征，再池化至 $7 \times 7$，并引入相对位置偏置 (RPB)。
2. **Token Mixer**（阶段间）：用两个线性层减少 token 数、扩展通道数，使 latent code 与图像特征对齐。
3. **Z2X Cross Attention**（阶段末尾）：特征 $\tilde{\mathbf{X}}_i$ 作为 query，latent code $\mathbf{Z}_i$ 作为 key/value，将语义信息反馈回特征分支。
4. **分类器部署**：仅在分类分支最后两个 stage 后放置早期分类器；最终分类器融合两分支输出。

**关键公式**：
- 分类分支 stage $i$：$\mathbf{Z}_i = \psi_i(f_i^{\text{att}}(g_i(\mathbf{Z}_{i-1}, \mathbf{X}_{i-1})))$
- 特征分支 stage $i$：$\mathbf{X}_i = h_i(f_i^{\text{conv}}(\mathbf{X}_{i-1}), \mathbf{Z}_i)$

**训练策略**：
- **自蒸馏损失**：$\mathcal{L}_k = \alpha \mathcal{L}_k^{\text{CE}} + (1-\alpha)\mathcal{L}_k^{\text{KD}}$（$\alpha=0.5$），用最终分类器软标签指导早退分类器训练。
- **FKT 模块**：早期分类器输出经线性层后拼接至下一阶段 pooled latent code，形成分类器间的 shortcut。
- **动态推理**：基于分类置信度（Softmax 最大值）判断是否提前终止，"easy"样本在浅层退出。

## 实验与结果
**数据集**：ImageNet（1000类，1.2M训练图）、Something-Something V1（98k视频）、COCO（80类，118k训练图）。

**图像分类（ImageNet）**：
- **ResNet系列**：Dyn-Perceiver 显著优于层跳过（Conv-AIG、SkipNet）、通道跳过（BAS-ResNet）、空间动态网络（DynConv、LASNet）等基线。
- **RegNet系列**：在 400M-3.2G FLOPs 范围持续提升；相比 RegNet-Y-4GF 减少 **4.8×** 计算且保持相同精度；相比 Swin-Transformer 和 DAT 分别减少 **1.8×** 和 **1.4×** 计算。
- **MobileNet-v3系列**：在 0.2-0.8 GFLOPs 预算下，同等性能下比 MobileNet-v3 少 **~1.2-1.4×** 计算。
- **对比早退网络**：优于 MSDNet、RANet、GFNet、DVT、CF-ViT 等 SOTA 方法。

**硬件效率验证**：
- 在 TX2（移动端）、Intel i5-8265U（CPU）、A100（GPU）上测试，理论效率有效转化为实际加速。
- MobileNet-v3-based Dyn-Perceiver 比 Mobile-Former 运行更快且精度相当（双分支并行 + 常规激活函数硬件友好）。

**动作识别（Something-Something V1）**：
- 基于 TSM 框架，与 TSM、TRN、ECO、AdaFuse 对比，在精度-效率权衡上表现优异。

**目标检测（COCO）**：
- 基于 RegNet-Y-1.6GF 的 Dyn-Perceiver 在 RetinaNet 中达到 **mAP 40.2**，超越 RegNet-X-3.2GF（39.0）和 RegNet-Y-3.2GF*（39.3）， computation 减少 **43%**。

## 相关工作脉络
- **Early-exiting 网络（MSDNet、RANet）**：在中间特征直接加分类器，存在性能退化问题；本文通过双分支解耦从根本上解决此缺陷。
- **Perceiver/Perceiver IO**：使用 latent code 查询输入信息的通用架构；本文引入特征分支降低计算量，并设计对称交叉注意力与动态早退机制，从静态模型扩展为动态高效框架。
- **Mobile-Former**：探索卷积-注意力交互的高效静态网络；本文是专为动态早退设计的通用框架，且双分支并行执行更利于硬件加速。
- **动态网络（SkipNet、DynConv、LASNet）**：通过层/通道/空间跳过实现动态计算；本文通过早退分类器置信度判断实现 depth 维度动态计算，且可单模型适应不同计算预算。
- **Vision Transformer 变体（DeiT、T2T-ViT、Swin、DAT）**：静态架构；本文的 Dyn-Perceiver 可作为 backbone 与其竞争甚至在相同 FLOPs 下更优。

## 局限性与未来方向
- **计算图复杂度增加**：双分支与交叉注意力引入额外参数与计算，在极低 FLOPs 约束下可能不如轻量级 CNN 直接设计。
- **早退阈值的离线设定**：当前需在验证集上搜索最优阈值，尚未实现完全在线自适应。
- **仅验证了主流视觉任务**：虽扩展至检测与动作识别，但未涉及分割、生成等其他视觉任务。
- **latent code 初始化与长度选择**：论文实验了 {128, 192, 256}，缺乏系统性的敏感性分析。
- **未来方向**：探索更高效的交叉注意力变体、在线自适应阈值学习、扩展至 3D 视觉与多模态任务。

## 研究启发与可借鉴点
1. **解耦设计范式**：将特征提取与任务分类分离的双分支思路可迁移至其他动态网络场景（如检测、分割），避免早退对主干性能的干扰。
2. **对称信息流动**：X2Z 与 Z2X 的双向交叉注意力设计保证特征分支不因早退分类器而受损，此思想可用于设计其他多任务学习的特征共享机制。
3. **FKT 知识传递机制**：分类器间的 shortcut 连接与自蒸馏结合，在端到端训练中直接优化多层级分类器，可推广至 multi-exit 其他架构。
4. **硬件友好的并行执行**：前两个 stage 双分支独立计算，后两个 stage 串行以获取早退预测，这种调度策略值得在其他动态推理框架中借鉴。
5. **通用框架而非专用设计**：无需针对特定 backbone 手工调参，直接套用于 ResNet/RegNet/MobileNet 均取得提升，体现了设计的简洁性与泛化性。

## 关键术语表
**Dynamic Early Exiting（动态早退）**：根据样本难度在较浅层提前输出预测，避免对"easy"样本执行深层计算。
**Latent Code（潜在码）**：可训练的低维 token 序列，用于编码语义信息并直接服务于分类任务，源自 Perceiver 架构。
**Cross Attention（交叉注意力）**：两个不同序列间的注意力机制，此处用于 feature branch 与 classification branch 的信息交换。
**Token Mixer（Token 混合器）**：通过线性层调整 latent code 的 token 数量与通道数，使其与图像特征对齐。
**Forward Knowledge Transfer (FKT)**：将早期分类器输出通过线性层传递至后续阶段，作为分类器间的知识 shortcut。
**Self-distillation（自蒸馏）**：用最终分类器的软标签指导早退分类器训练，损失函数为 CE 与 KL 散度的加权组合。
**FLOPs（浮点运算数）**：衡量模型计算复杂度的标准指标，本文用于评估推理效率。
**Confidence Threshold（置信度阈值）**：判断样本是否为"easy"的阈值，超过则提前退出，需在验证集上优化设定。

## 可复现要素
- **数据集**：ImageNet（公开）、Something-Something V1（公开）、COCO（公开）。
- **代码**：已开源，地址 https://www.github.com/LeapLabTHU/Dynamic_Perceiver。
- **权重**：论文未明确提及是否公开权重，代码仓库可能包含。
- **关键超参**：latent code 初始 token 数 L ∈ {128, 192, 256}；自蒸馏权重 α = 0.5；cross-attention head 数每 stage 为 $2^{i-1}$（stage 1-4 分别为 1, 2, 4, 8），cross-attention 均为 1 head。
