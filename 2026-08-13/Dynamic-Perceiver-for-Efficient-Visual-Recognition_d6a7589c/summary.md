---
title: "Dynamic-Perceiver-for-Efficient-Visual-Recognition"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:36:58"
field: "高效视觉识别"
keywords: ["early-exiting", "dynamic inference", "visual recognition", "dual-branch architecture", "cross-attention", "latent code", "efficient deep learning"]
innovations: ["双分支解耦架构：特征提取与早期分类显式分离，避免低层特征被迫线性可分", "对称双向交叉注意力（X2Z+Z2X）：分类分支与特征分支持续信息交互，早期出口反而提升最后出口精度", "端到端FKT+自蒸馏训练策略：无需pretrain-finetune，分类器间捷径与last-classifier蒸馏协同优化多出口"]
benchmarks: ["ImageNet-1K", "Something-Something V1", "COCO Object Detection"]
---

# 论文速读：Dynamic-Perceiver-for-Efficient-Visual-Recognition

## 一句话总结
论文提出 **Dynamic Perceiver (Dyn-Perceiver)**，通过双分支架构将**特征提取**与**早期分类**任务解耦，利用可训练的隐式代码（latent code）承载语义信息，使早期分类器仅部署在分类分支中，在显著降低推理计算量的同时保持甚至提升模型精度。

## 研究问题与动机
- **现有 Early-exiting 方法的性能瓶颈**：传统多退出网络在中间特征上直接构建线性分类器，迫使低层特征必须包含高层语义并能线性可分，严重损害了深层出口（last exit）的性能（MS-DNet [24] 等已有工作已观察到该问题）。
- **特征提取与分类任务纠缠**：深层模型从低层到高层渐进提取特征，但早期分类器介入会干扰这一过程，现有方法未能从根本上解耦两者。
- **Perceiver 架构的计算开销问题**：Perceiver [29] 的隐式代码直接查询原始像素输入，在视觉任务中因图像 token 数量庞大而导致计算成本过高。
- **现有动态网络的通用性不足**：MS-DNet、RANet 等方法需要精心设计的专用结构，难以直接迁移到下游任务（如目标检测）。

## 核心贡献（创新点）
1. **双分支解耦架构**：首次通过显式双分支（特征分支 + 分类分支）解耦特征提取与早期分类，分类分支中的隐式代码承担语义聚合，避免了对低层特征线性可分性的要求。
2. **对称双向交叉注意力机制（X2Z + Z2X）**：特征分支到分类分支（X2Z）和分类分支到特征分支（Z2X）的双向交互，既丰富了隐式代码的语义，又反过来增强了特征表示，且早期出口反而提升了最后一层的精度（+0.7%）。
3. **向前知识传递（FKT）模块与自蒸馏训练**：提出无需 pretrain-finetune 的端到端训练策略，FKT 作为分类器间的"快捷方式"，结合 last classifier 对早期出口的 KL 蒸馏损失，有效缓解多出口间的干扰。
4. **通用框架与下游验证**：Dyn-Perceiver 可构建于任意主流视觉骨干（ResNet、RegNet-Y、MobileNet-v3），首次在动态 early-exiting 网络中成功验证于 **COCO 目标检测**任务，mAP 提升 0.9% 同时减少 43% 计算量。

## 方法详解
- **整体架构**：由 4 个 stage 组成的双分支结构。特征分支（CNN，如 RegNet-Y）从低层到高层逐步提取图像特征 $X_0 \sim X_4$；分类分支以可训练的隐式代码 $Z_0$ 为输入，经自注意力与 token mixer 处理得到 $Z_1 \sim Z_4$。
- **X2Z 交叉注意力**：在每个 stage 开始时，分类分支通过 cross-attention 从特征分支查询信息（$Z_{i-1}$ 为 query，$X_{i-1}$ 为 key/value），为提升效率先对 $X_{i-1}$ 做 depth-wise convolution（DWC）并 pooling 到 $7 \times 7$，同时引入相对位置偏置（RPB）。
- **Token Mixer**：借鉴视觉骨干惯例，用两个线性层在每两个 stage 之间缩减隐式代码的 token 数并扩展通道数，实现对齐。
- **Z2X 交叉注意力**：每个 stage 末尾，将分类分支的语义信息回馈到特征分支（$\tilde{X}_i$ 为 query，$Z_i$ 为 key/value），形成对称双向交互。
- **分类器部署**：早期分类器仅置于分类分支的最后两个 stage，最后一层融合双分支输出。
- **FKT 模块**（图6）：早期分类器输出经线性层后与下一 stage 的隐式代码拼接再输入后续分类器，作为"分类器间捷径"。
- **训练策略**：总损失 $\mathcal{L} = \sum_{k=1}^{K} \mathcal{L}_k$，其中前 $K-1$ 个出口的 $\mathcal{L}_k = 0.5\,\mathcal{L}_k^{\text{CE}} + 0.5\,\mathcal{L}_k^{\text{KD}}$（KD 为 last classifier 的软标签蒸馏），最后一出口仅用 CE 损失。
- **动态推理**：若某早期分类器的 softmax 最大概率超过阈值，则提前终止推理（跳过深层计算），"简单"样本在前几 stage 即被正确分类。

## 实验与结果
- **ImageNet 分类**：
  - **ResNet 系列**：Dyn-Perceiver 显著优于 Conv-AIG、SkipNet、BAS-ResNet 等层/通道/空间跳过的动态网络，且单一模型可通过调整阈值适应不同计算预算。
  - **RegNet-Y 系列（400M–3.2G FLOPs）**：与同精度静态模型相比，计算量减少 **1.9–4.8×**；相较 Swin-Transformer 和 DAT 分别减少 1.8× 和 1.4×。
  - **对比 Early-exiting 竞品**：优于 MS-DNet、RANet、GFNet、DVT、CF-ViT 等，一致性领先。
  - **MobileNet-v3 系列**：在 0.2–0.8 GFLOPs 预算下，与 MobileNet-v3 同等精度时计算量减少约 **1.2–1.4×**；优于 DeiT、PVT、MobileViT、Efficient-Former 等。
- **实际硬件加速**：在 TX2 移动设备、Intel i5 CPU、A100 GPU 上均验证了理论效率到实际延迟的转化，MobileNet-v3 版 Dyn-Perceiver 比 Mobile-Former 实际更快。
- **Action Recognition（Something-Something V1）**：在 TSM 框架中替换为 ResNet-based Dyn-Perceiver，精度-效率权衡优于 TSM、TRN、ECO、AdaFuse。
- **目标检测（COCO）**：基于 RetinaNet + RegNet-Y-1.6GF，mAP 达 **40.2**（较 RegNet-Y-1.6GF* 的 38.5 提升 **+1.7**），骨干 FLOPs 更少，是首个在检测任务上验证的动态 early-exiting 方法。
- **对比 Perceiver**（Tab.1）：Dyn-Perceiver（MobileNet-v3-1.25×）仅需 **0.55G FLOPs** 达到 79.0% Top-1，而 Perceiver IO 需 407G FLOPs 仅 79.0%，减少三个数量级。

## 相关工作脉络
- **MS-DNet [24]**：多尺度密集网络的 early-exiting 先驱，但早期分类器仍构建于中间特征，存在性能退化问题；Dyn-Perceiver 通过双分支从根本上解决了这一缺陷。
- **RANet [69]**：分辨率自适应网络，同样在特征图上加 early exit；Dyn-Perceiver 不用多尺度密集连接，框架更简洁通用。
- **Perceiver [29] / Perceiver IO [28]**：引入隐式代码查询输入的通用架构；本文取其"隐式代码"思想但引入特征分支+双向交叉注意力，将计算从 400G+ FLOPs 降至亚 GFLOPs 级别，并加入动态 early-exiting 机制。
- **Mobile-Former [7]**：卷积-注意力混合架构；本文与之不同在于：① 面向动态 early-exiting 而非静态高效网络；② 双分支可并行计算，而 Mobile-Former 是串行的。
- **TSM [34]**：视频动作识别框架；本文将其 CNN 骨干替换为 Dyn-Perceiver，验证了方法在时序任务上的可扩展性。
- **RetinaNet [36]**：两阶段目标检测框架的基线；本文证明 Dyn-Perceiver 可无缝作为检测 backbone，且不会削弱 FPN 特征金字塔质量。

## 局限性与未来方向
- **阈值依赖**：动态推理需预先在验证集上设定置信度阈值，不同硬件/场景下可能需要重新调参（论文 Section 4.1 提及 via grid search）。
- **仅最后两 stage 设出口**：消融实验表明第一、二 stage 的早期分类器收益有限，说明隐式代码在浅层阶段语义积累不足，未来可探索更深的信息整合机制。
- **未探索更大规模预训练**：本文主要验证 ImageNet 单阶段训练，未见与 ImageNet-21k 或大规模自监督预训练结合的实验。
- **目标检测中未使用 early-exiting**：COCO 实验仅为验证框架通用性，未将动态推理应用于检测本身（可能是由于检测任务本身的复杂性）。
- **仅验证了 CNN 骨干**：分类分支内部使用标准 Transformer block，但特征分支仅用了 CNN；未来可探索全 ViT 或 hybrid 版本的特征分支。

## 研究启发与可借鉴点
- **双分支解耦思想可迁移至其他动态推理任务**：将特征提取与决策任务分离的思路，可推广到目标检测、分割、视频理解等 dense prediction 任务，有望解决这些任务中 early-exiting 长期存在的性能退化问题。
- **对称交叉注意力（X2Z + Z2X）是一种高效的跨模态/跨分支信息融合范式**：比单向 attention 更能避免信息瓶颈，可借鉴到多模态融合、远程传感器数据整合等场景。
- **FKT 作为端到端的分类器间捷径**：无需 pretrain-finetune 即可同时优化浅层和深层分类器，这一设计可复用到多标签分类、层级分类等其他多出口设置。
- **与 Token Mixer 结合的下采样策略**：通过线性层在 channel 和 token 维度交替操作来压缩隐式代码，避免了降采样带来的信息损失，可作为其他 latent-variable 模型的通用组件。
- **自蒸馏（self-distillation）配合 FKT 的组合**：论文消融显示两者配合效果最佳，这对多出口网络的训练策略设计具有直接参考价值。

## 关键术语表
- **Early-exiting（早期退出）**：在深度网络中间层设置分类器，置信度高的样本提前输出，跳过深层计算以节省推理开销。
- **Latent Code（隐式代码）**：一组可训练的向量（tokens），通过 attention 机制从输入中聚合语义信息，用于下游分类任务。
- **X2Z Cross-Attention（特征到隐式代码交叉注意力）**：分类分支的隐式代码作为 query，特征分支的图像特征作为 key/value，实现信息从特征流向分类分支。
- **Z2X Cross-Attention（隐式代码到特征交叉注意力）**：特征分支的中间特征作为 query，隐式代码作为 key/value，将语义信息回馈到特征分支。
- **Token Mixer**：通过线性层对隐式代码进行 token 数下采样和通道数上采样的模块，对齐不同 stage 的特征维度。
- **FKT（Forward Knowledge Transfer）**：将早期分类器的输出经过线性层后传递到后续分类器，作为分类器之间的快捷连接，提升端到端训练效果。
- **Self-distillation（自蒸馏）**：用最后一个（最深）分类器的软标签作为 teacher，通过 KL 散度指导早期分类器的训练。
- **Symmetric Cross-attention（对称交叉注意力）**：X2Z 和 Z2X 双向交互机制，使两个分支能够持续交换信息，区别于 Perceiver 的单向 attention。

## 可复现要素
- **数据集**：ImageNet（公开）、Something-Something V1（公开）、COCO（公开）。
- **代码**：论文明确声明开源，链接为 https://www.github.com/LeapLabTHU/Dynamic_Perceiver。
- **权重**：论文未明确说明权重是否公开，代码仓库可进一步确认。
- **关键超参**：隐式代码初始 token 数 $L \in \{128, 192, 256\}$；各 stage self-attention head 数为 $2^{i-1}$（$i=1,2,3,4$）；cross-attention head 数固定为 1；蒸馏系数 $\alpha = 0.5$；X2Z 中特征 pooling 至 $7 \times 7$；输入图像尺寸 $224 \times 224$（ImageNet）、$1280 \times 800$（COCO）。
