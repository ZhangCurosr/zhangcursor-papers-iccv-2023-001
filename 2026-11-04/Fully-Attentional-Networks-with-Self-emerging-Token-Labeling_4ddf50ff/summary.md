---
title: "Fully-Attentional-Networks-with-Self-emerging-Token-Labeling"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fully_Attentional_Networks_with_Self-emerging_Token_Labeling_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:58:51"
field: "视觉Transformer鲁棒预训练"
keywords: ["Vision Transformer", "Token Labeling", "Self-emerging Token Labeling", "Robustness", "FAN", "Knowledge Distillation", "Image Classification"]
innovations: ["首次用ViT自身替代CNN作为token labeler生成语义token标签", "提出clean teacher/noisy student的增广解耦策略提升鲁棒性", "引入Gumbel-Softmax自适应过滤低置信度token标签"]
benchmarks: ["ImageNet-1K", "ImageNet-C", "ImageNet-A", "ImageNet-R", "Cityscapes", "COCO"]
---

# 论文速读：Fully-Attentional-Networks-with-Self-emerging-Token-Labeling

## 一句话总结
本文提出 Self-emerging Token Labeling (STL) 框架，让 Vision Transformer 自身充当 token labeler 生成高质量的 token 级语义标签，替代传统依赖 CNN 的方法，显著提升 FAN 系列模型的 ImageNet 预训练鲁棒性，在 IN-A/IN-R 上创下了新的 SOTA。

## 研究问题与动机
- **核心问题**：现有 token labeling 方法依赖预训练的 CNN（如 NFNet）作为 token labeler，能否让 ViT 模型自主生成高质量的 token 级标签？
- **动机1**：ViT 的分类任务中，目标分割能力会自然涌现（如 DINO 所示），但全监督场景下缺乏系统利用；FAN 模型展现出优秀的视觉分组能力，可被用来生成有意义的 token 标签。
- **动机2**：ViT 虽在零样本鲁棒性上优于 CNN，但其 pretraining 的 token-labeling 训练范式探索不足，现有方法（LV-ViT、ReLabel 等）均依赖外部 CNN teacher，难以保证 token 标签与 student 模型的表征一致性。
- **动机3**：现有强数据增强（RandAug、MixUp、CutMix 等）会破坏 token 标签的准确性，如何设计增广策略以保留高质量 token 信息是关键挑战。

## 核心贡献（创新点）
- **提出 STL 框架**：首次用 ViT（FAN）自身作为 token labeler，无需外部 CNN 即可生成语义有意义的 token 级标签，与 LV-ViT（依赖 CNN token-labeler）形成本质区别。
- **两阶段自监督 Token 标签生成机制**：第一阶段同时对 class token 和全局平均池化的 patch tokens 施加 class label 监督，使 patch token 自然具备定位能力；第二阶段联合 class label 与 token label 训练 student 模型，比仅用 image-level label 的传统 FAN 训练方式更丰富。
- **引入 Gumbel-Softmax 自适应过滤低置信度 token 标签**：基于 token 标签置信度分数（最大类别概率），自动保留正确 foreground token 标签、消除错误标注，从而提升 student 模型的鲁棒性。
- **提出 "clean teacher + noisy student" 数据增广策略**：FAN-TL 仅使用 spatial-only 增广（flip/rotate/shear/translation），而 student 使用完整增广（含 RandAug/CutOut/MixUp/CutMix），使 token 标签在噪声输入下仍保持干净准确，显著提升了 IN-A/IN-R 上的鲁棒性。
- **在多个下游任务验证泛化性**：STL 在图像分类（IN-1K/IN-C/IN-A/IN-R）、语义分割（Cityscapes/Cityscapes-C）和目标检测（COCO）上均显著优于原始 FAN 及多种 SOTA 方法，且不增加额外训练数据。

## 方法详解
- **整体框架**：STL 为两阶段 teacher-student 训练范式。
  - **Stage 1（训练 FAN-TL）**：在标准 FAN 分类训练目标（仅对 class token 施加 cross-entropy）基础上，额外对 patch tokens 做全局平均池化后，同时赋予 class label 监督：
    $$\mathcal{L} = \mathcal{H}(\mathbf{T}_{cls}, \mathbf{Y}_{cls}) + \alpha \cdot \mathcal{H}\left(\frac{1}{N}\sum_{i=1}^{N}\mathbf{T}_{p_i}, \mathbf{Y}_{cls}\right)$$
    其中 $\alpha=1$，使 patch token 输出具备语义定位能力（黄色 foreground / 蓝色 background 分割效果）。
  - **Stage 2（训练 Student）**：将 FAN-TL 的 patch token 输出作为 token label，与学生模型的 patch tokens 进行匹配：
    $$\mathcal{L} = \mathcal{H}(\mathbf{T}_{cls}, \mathbf{Y}_{cls}) + \beta \cdot \frac{1}{N}\sum_{i=1}^{N}\mathcal{H}(\mathbf{T}_{p_i}, \hat{\mathcal{F}}(\hat{\mathbf{I}}_{p_i}))$$
    其中 $\beta=1$，$\hat{\mathcal{F}}$ 为经 Gumbel-Softmax 处理后的 hard label。
- **Gumbel-Softmax 标签选择**：对 FAN-TL 输出的 softmax 概率 $\pi_i$ 施加 Gumbel-Softmax：
  $$\mathbf{y}_i = \frac{e^{(\log(\pi_i) + \mathcal{G}_i)/\tau}}{\sum_j e^{(\log(\pi_j) + \mathcal{G}_j)/\tau}}$$
  高置信度 token（foreground 且分数 0.7~0.9）标签基本不变，低置信度（0~0.2）易被重新采样，从而实现自适应过滤。
- **数据增广策略**：FAN-TL 仅用 spatial-only 增广，student 用完整增广，形成 "clean teacher, noisy student" 设计，防止强增广污染 token 标签质量。
- **硬标签转换**：将 soft token labels 转为 one-hot hard labels 训练 student，参考 FixMatch 的 entropy minimization 思想，促使 student 输出高置信度预测。
- **Architecture**：FAN-TL 与学生模型均采用 FAN-Hybrid 架构（含 channel attention block），参数量可同构或异质（heterogeneous）。

## 实验与结果
- **数据集**：ImageNet-1K（训练）、ImageNet-C（mCE）、ImageNet-A、ImageNet-R；下游：Cityscapes / Cityscapes-C（语义分割）、COCO（目标检测）。
- **主要分类结果**（Table 2）：STL 在所有 FAN 规模上均优于原始 FAN；FAN-L-Hybrid (77.3M) 在 IN-1K 达 84.7%，IN-C mCE 42.5，Retention rate 81.2%；超越 LV-ViT（CNN token-labeler）显著。
- **OOD 鲁棒性**（Table 3）：STL (FAN-L-Hybrid) 在 IN-A 达 **46.1%**、IN-R 达 **56.6%**，创当时 SOTA，较原 FAN-L-Hybrid（IN-A 41.8%，IN-R 53.2%）分别提升 +4.3% 和 +3.4%。
- **语义分割**（Table 4）：FAN-L-Hybrid + STL 在 Cityscapes mIoU 82.8（+0.5 vs 原 FAN）、City-C mIoU 69.2（+0.5），Retention 83.6%。
- **目标检测**（Table 5）：FAN-L-Hybrid + STL 在 COCO mAP 54.1，与原始 FAN 持平，优于 Swin-B（51.9）和 ConvNeXt-B（51.9）。
- **异质 Token-Labeler 实验**（Table 8）：用更小的 FAN-TL-B-Hybrid (50.9M) 指导 FAN-L-Hybrid student，IN-1K 84.8%、IN-C mCE 42.1，略优于同构设置，证明 STL 对 teacher 规模的鲁棒性。
- **对比 CNN token-labeler**（Table 9）：STL (FAN-TL-L-Hybrid, 77.3M) 优于 NFNet-F6 (400M+) 作为 teacher 的 LV-ViT，说明 self-emerging token labeling 比外部大 CNN 更有效。
- **消融**：$\beta \in \{0.5, 1.0, 2.0\}$ 对结果影响小（Table 10），Gumbel-Softmax 比 Softmax 在鲁棒性上更优（Table 7），spatial-only 增广是获得高质量 token labels 的关键（Table 6）。

## 相关工作脉络
- **LV-ViT [3]**（NeurIPS 2021）：首个提出 token labeling 的 ViT 工作，使用预训练 CNN（如 ResNet/NFNet）作为 token-labeler；本文与其本质区别在于用 ViT 自身替代 CNN 作为 token labeler，保证 teacher-student 表征一致性，且在鲁棒性上全面超越。
- **ReLabel [4]**：将单标签扩展为多标签，但仍基于 image-level 监督，无法提供 patch-level 密集标注；本文的 token labeling 是逐 patch 的 dense 监督，信息更丰富。
- **DINO [20]**：自监督 ViT 中涌现出对象定位能力；本文受此启发，但在 **全监督** 设定下通过 class token + 全局平均池化 patch token 双重监督实现类似涌现，方法更简单且可直接用于分类预训练。
- **GroupViT [21]**：纯文本监督下涌现语义分割；本文利用 FAN 的 channel attention 和视觉分组能力，在全监督条件下生成 token 标签，定位精度更高。
- **BEiT [18]**：通过离散 VAE tokenizer 产生 visual tokens；这些 tokens 无显式语义，而本文的 token labels 具有明确的类别语义，可直接用于监督学习。
- **知识蒸馏 [2, 16]**：Token labeling 可视为 hard distillation 的密集形式；本文统一 teacher/student 为同构 ViT，避免了跨架构蒸馏带来的信息损失。

## 局限性与未来方向
- **标注依赖 class label**：FAN-TL 的 token 标签生成依赖 image-level class label，无法在无监督/弱监督场景直接推广。
- **Token 标签质量上限受限于 FAN 的视觉分组能力**：对于背景复杂、目标不显著或类别高度相似的样本，token 标签可能存在系统性误标（如图 2(b) 中 human/lake 区域被误标为 target object）。
- **异质 teacher 实验仅在小→大方向验证**：用大 model 指导小 model 的效果未在论文中系统评估。
- **计算开销**：两阶段训练增加了 FAN-TL 的训练成本（虽然可复用或离线），对大规模部署可能不友好。
- **未来方向**：可探索 self-supervised / semi-supervised 版本的 STL；将 Gumbel-Softmax 的置信度阈值自适应化；扩展到视频、3D 点云等多模态任务。

## 研究启发与可借鉴点
- **"Clean teacher, noisy student" 的增广解耦策略**值得借鉴：在蒸馏/token labeling 场景中，teacher 与 student 使用不同强度的数据增广可提升 student 对噪声的鲁棒性，这一思想可迁移到其他 self-training 或半监督场景。
- **Gumbel-Softmax 用于软标签过滤**是一种轻量且可微的标签质量控制手段，可在其他需要 confidence-aware 伪标签筛选的任务（如自训练、半监督分割）中复用。
- **同构 teacher-student 的表征一致性优势**：用同类模型做 token labeler 可避免跨架构知识迁移中的表征错位问题，这一原则可推广到其他 distillation-based 预训练方法。
- **利用涌现的视觉分组能力**：FAN 的 channel attention 带来的分组特性是高质量 token label 的关键，提示我们在设计 ViT 变体时，应关注隐式的空间/语义分组能力，并加以利用。
- **下游任务迁移评估**：本文不仅评估分类鲁棒性，还验证了分割和检测的 transferability，这种多维度的下游评估方法可作为团队后续工作的标准流程。

## 关键术语表
- **Fully Attentional Network (FAN)**：一种在 ViT 基础上新增 channel attention block 的 Vision Transformer 变体，通过聚合跨通道信息提升表征质量和鲁棒性。
- **Self-emerging Token Labeling (STL)**：本文提出的两阶段训练框架，让 ViT 自身生成语义有意义的 patch token 级标签，用于指导 student 模型的预训练。
- **FAN Token-Labeler (FAN-TL)**：在 Stage 1 中训练的 FAN 模型，其 patch token 输出经池化后附带 class label 监督，能够生成类似语义分割的 foreground/background token 标签。
- **Token Label**：分配给每个 image patch 的类别标签，提供位置敏感的稠密监督信号，区别于传统 image-level class label。
- **Retention Rate**：鲁棒准确率与干净准确率之比（$R = \text{Robust Acc.} / \text{Clean Acc.}$），衡量模型在面对分布偏移时的性能保持能力。
- **Gumbel-Softmax**：一种可微分的 softmax 重参数化技术，通过引入 Gumbel 噪声使离散选择过程可微，此处用于自适应过滤低置信度 token 标签。
- **LV-ViT**：All Tokens Matter (Jiang et al., NeurIPS 2021)，首个将 token labeling 引入 ViT 预训练的工作，使用预训练 CNN 作为 token labeler。
- **Channel Attention Block**：FAN 中新增的模块，通过 attention 机制聚合跨通道信息，与 self-attention 互补，增强特征的判别力。

## 可复现要素
- **数据集**：ImageNet-1K（公开）、ImageNet-C（公开）、ImageNet-A（公开）、ImageNet-R（公开）、Cityscapes（公开）、COCO（公开）。
- **代码开源**：论文未明确声明代码开源仓库，但基于 PyTorch、timm、MMSegmentation 实现。
- **关键超参**：learning rate 4e-3（分类）、batch size 2048、训练 350 epochs；$\alpha=1, \beta=1$；label smoothing 0.9；cosine scheduler，每 30 epoch decay 0.1；Gumbel-Softmax temperature $\tau$ 未明确给出（论文未提及具体值）。
- **硬件**：8× NVIDIA Tesla V100。
- **模型架构**：FAN-Hybrid（T/S/B/L 四档规模）。
