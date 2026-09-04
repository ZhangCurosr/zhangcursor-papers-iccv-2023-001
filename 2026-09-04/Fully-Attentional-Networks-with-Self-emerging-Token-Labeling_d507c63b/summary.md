---
title: "Fully-Attentional-Networks-with-Self-emerging-Token-Labeling"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fully_Attentional_Networks_with_Self-emerging_Token_Labeling_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:14:50"
field: "视觉Transformer鲁棒性"
keywords: ["Vision Transformer", "Token Labeling", "Robustness", "Self-emerging", "Fully Attentional Network", "Out-of-distribution"]
innovations: ["提出STL框架用ViT自身生成token标签替代CNN labeler", "干净教师-噪声学生数据增强策略提升OOD鲁棒性", "Gumbel-Softmax自适应筛选高置信度foreground token标签"]
benchmarks: ["ImageNet-1K", "ImageNet-C", "ImageNet-A", "ImageNet-R", "Cityscapes", "COCO"]
---

# 论文速读：Fully-Attentional-Networks-with-Self-emerging-Token-Labeling

## 一句话总结
本文提出自涌现Token标注（STL）框架，利用Vision Transformer自身作为token labeler生成语义有意义的patch token标签，以两阶段方式改进FAN模型的预训练；无需额外数据，STL训练的FAN-L-Hybrid模型在ImageNet-A和ImageNet-R上分别创下新SOTA（46.1%和56.6%），同时显著提升了模型鲁棒性。

## 研究问题与动机
- **核心问题**：能否让ViT模型自我生成有意义的token标签，并用自产知识改进预训练，而非依赖外部CNN教师？
- **现有方法不足**：当前token labeling方法（如LV-ViT）依赖预训练CNN（如NFNet）作为token labeler，无法充分利用ViT自身的视觉分组与自注意力表征能力；且强数据增强会破坏token标签质量。
- **FAN的优势**：FAN具有优秀的零样本鲁棒性和自涌现视觉分组能力，可生成高质量token标签，适合用于自监督token标注场景。
- **动机延伸**：验证Transformer架构作为token labeler的有效性，探索"干净教师-噪声学生"训练范式对OOD鲁棒性的增益。

## 核心贡献（创新点）
1. **提出自涌现Token标注（STL）框架**：首次用Transformer-based模型（FAN-TL）替代CNN作为token labeler，实现自产高质量token标签。与依赖外部CNN标注的方法本质不同，统一了师生架构。
2. **设计两阶段训练与token筛选机制**：第一阶段用类别token和全局平均池化token联合监督训练FAN-TL；第二阶段引入Gumbel-Softmax自适应筛选高置信度foreground token标签，剔除错误标注。这与传统hard distillation直接全量使用token标签的做法不同。
3. **提出"干净教师-噪声学生"数据增强策略**：FAN-TL仅用空间变换增强生成干净token标签，学生模型使用全量增强；这一设计使token标签提供抗噪信息，显著提升OOD鲁棒性，区别于传统蒸馏中师生使用相同增强的设定。
4. **系统性验证与SOTA结果**：在ImageNet-A（46.1%）、ImageNet-R（56.6%）创下无额外数据的新SOTA，并在语义分割（Cityscapes mIoU提升1.7%）和目标检测（COCO mAP）上验证了可迁移性。

## 方法详解
- **整体框架**：两阶段训练，FAN-TL（教师）→ STL学生模型，师生均采用FAN-Hybrid架构。
- **Stage 1：训练FAN-TL**
  - 输入图像I，经FAN编码得到class token $T_{cls}$ 和patch tokens $[T_{p_1}, ..., T_{p_N}]$。
  - 对patch tokens做全局平均池化 $\frac{1}{N}\sum_{i=1}^{N} T_{p_i}$，与$T_{cls}$共享同一类别标签$Y_{cls}$。
  - 损失函数：$\mathcal{L} = \mathcal{H}(T_{cls}, Y_{cls}) + \alpha \cdot \mathcal{H}(\frac{1}{N}\sum T_{p_i}, Y_{cls})$，其中$\alpha=1$。
  - 数据增强：仅使用空间变换（flip, rotate, shear, translation），禁用RandAug、CutOut、MixUp、CutMix。
- **Stage 2：训练学生模型**
  - FAN-TL输出的patch token logits作为token标签$\mathcal{F}(I_{p_i})$分配给学生的对应patch token。
  - 引入Gumbel-Softmax进行token标签筛选：$\mathbf{y}_i = \frac{e^{(\log(\pi_i) + \mathcal{G}_i)/\tau}}{\sum_j e^{(\log(\pi_j) + \mathcal{G}_j)/\tau}}$，高置信度标签保留，低置信度标签被抑制，实现foreground token去噪。
  - Softmax输出转为one-hot hard label（借鉴self-training中的熵最小化效应）。
  - 损失函数：$\mathcal{L} = \mathcal{H}(T_{cls}, Y_{cls}) + \beta \cdot \frac{1}{N}\sum_{i=1}^{N} \mathcal{H}(T_{p_i}, \hat{\mathcal{F}}(\hat{I}_{p_i}))$，其中$\beta=1$。
  - 数据增强：学生模型使用完整增强（含RandAug、CutOut、MixUp、CutMix），形成"干净教师-噪声学生"设计。

## 实验与结果
- **数据集**：ImageNet-1K（训练）、ImageNet-C/mCE、ImageNet-A、ImageNet-R（鲁棒性评估）；Cityscapes/City-C（语义分割）、COCO（目标检测）。
- **关键结果（ImageNet-1K）**：
  - FAN-L-Hybrid → STL(FAN-L-Hybrid)：Top-1 84.3% → 84.7%，mCE 43.0 → 42.1。
  - STL(FAN-B-Hybrid)：Top-1 84.5%，mCE 43.6。
  - 超越所有CNN基线及LV-ViT（CNN token labeler）。
- **OOD鲁棒性（无额外数据，新SOTA）**：
  - ImageNet-A：46.1%（STL FAN-L-Hybrid，77.3M参数），超越Swin-B（46.6%但参数88M）、ConvNeXt-B（48.3%参89M）。
  - ImageNet-R：56.6%，创纪录。
  - ImageNet-C mCE：最低42.1%，优于FAN原版（43.0）和LV-ViT-M（50.5）。
- **Retention Rate（Robust/Clean）**：STL模型在各级别均达78.5%~83.6%，显著高于对照。
- **下游迁移**：
  - 语义分割（Cityscapes）：STL(FAN-L-Hybrid) mIoU 82.8 vs 原版82.3，City-C提升1.7%。
  - 目标检测（COCO）：STL(FAN-L-Hybrid) mAP 54.1，略优于原版。
- **Ablation结论**：空间增强对FAN-TL至关重要；Gumbel-Softmax较Softmax提升鲁棒性；异构图token labeler（小Labeler+大学生）效果相当甚至更优；loss权重β在0.5~2.0范围内模型不敏感。

## 相关工作脉络
- **Token Labeling（LV-ViT, NeurIPS 2021）**：使用预训练CNN（NFNet）为patch生成token标签；本文用ViT自身替代CNN labeler，证明自涌现token标注更优。
- **Vision Transformers（ViT, ICLR 2020）**：基础ViT架构；本文在其FAN变体（引入channel attention）上扩展token labeling能力。
- **Knowledge Distillation（DeiT, ICML 2021）**：image-level硬标签蒸馏；本文稠密化token-level蒸馏，利用更细粒度局部信息。
- **Self-supervised ViT（DINO, ICCV 2021）**：自监督下object segmentation涌现；本文采用全监督方式，利用FAN的视觉分组能力生成token标签。
- **GroupViT（CVPR 2022）**：文本监督下的语义分割涌现；与本文共同揭示ViT的分组与定位能力，但训练范式不同。
- **FAN（ICML 2022）**：本文基础架构，引入channel attention提升鲁棒性；STL进一步在其上叠加token labeling训练框架。

## 局限性与未来方向
- **Token label质量仍不完美**：即使经Gumbel-Softmax筛选，仍存在误分类background/foreground token，缺乏patch级ground truth验证。
- **依赖FAN架构特性**：方法建立在FAN出色的视觉分组能力之上，未验证在其他ViT变体（如Swin、DeiT）上的通用性。
- **两阶段训练成本**：需先训练FAN-TL再生成token标签，增加训练流程复杂度；虽可用小labeler+大学生降低成本，但未系统分析极限压缩情况。
- **下游任务提升不均衡**：目标检测提升幅度（~0.2 mAP）远小于语义分割（+1.7 mIoU），对非密集预测任务的迁移潜力待探索。
- **未来方向**：探索更通用的ViT backbones上的STL适配；结合自监督预训练（如MAE）与token labeling；研究更高效的token筛选与去噪机制。

## 研究启发与可借鉴点
- **"干净教师-噪声学生"训练范式**：对需要生成中间监督信号（如token标签、伪标签）的自蒸馏场景，教师侧避免强数据增强可有效提升标签质量，值得迁移至其他自训练/半监督任务。
- **Gumbel-Softmax用于token置信度筛选**：无需额外网络或人工阈值，以可微方式实现hard label选择性传递，可推广至其他token-level伪标签过滤场景。
- **ViT自产出表征的再利用**：证明ViT内部特征可自生成高质量局部监督信号，启发后续研究挖掘ViT自身各层表征的多粒度利用价值。
- **异构图师生规模搭配**：小labeler+大学生的组合仍保持甚至提升性能，为计算受限场景提供低成本替代方案。
- **稠密监督对下游鲁棒性的迁移增益**：不仅提升分类clean accuracy，更显著改善segmentation/detection在corrupted数据上的表现，提示稠密预训练信号的长期价值。

## 关键术语表
- **STL（Self-emerging Token Labeling）**：自涌现Token标注，利用ViT自身生成patch级别语义标签的训练框架。
- **FAN（Fully Attentional Network）**：全注意力网络，在ViT基础上引入channel attention块，以高鲁棒性著称的ViT骨干系列。
- **FAN-TL（FAN Token-Labeler）**：基于FAN架构的token标注器，负责为每个patch token生成语义标签。
- **Token Labeling**：将类别标签分配给图像每个patch token的稠密监督方法，等价于hard knowledge distillation的稠密化形式。
- **Gumbel-Softmax**：用于可微采样离散变量的技巧，本文用于高置信度token标签筛选与one-hot hard label生成。
- **Retention Rate**：鲁棒准确率与干净准确率之比，衡量模型分布外退化程度的指标。
- **ImageNet-A / ImageNet-R**：评估OOD鲁棒性的数据集，分别含自然对抗样本和艺术渲染图像。
- **Cityscapes-C**：Cityscapes语义分割数据集的腐蚀版本，用于评估下游任务鲁棒性。

## 可复现要素
- **数据集**：ImageNet-1K（公开）、ImageNet-C/A/R（公开）、Cityscapes（公开）、COCO（公开）。
- **代码**：基于PyTorch、timm、MMSegmentation构建，论文未提供独立开源仓库声明，需自行实现或联系作者获取。
- **关键超参**：优化器AdamW，lr=4e-3，batch=2048，epochs=350；$\alpha=\beta=1$；label smoothing=0.9； cosine scheduler decay=0.1/30epochs；Gumbel-Softmax温度τ未明确给出；空间增强仅含flip/rotate/shear/translation。
- **硬件**：8× NVIDIA Tesla V100。
- **模型权重**：论文未声明开源，基线FAN模型需参考原FAN论文获取。
