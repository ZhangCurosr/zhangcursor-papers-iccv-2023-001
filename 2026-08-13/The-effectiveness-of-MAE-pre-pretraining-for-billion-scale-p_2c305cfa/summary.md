---
title: "The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Singh_The_Effectiveness_of_MAE_Pre-Pretraining_for_Billion-Scale_Pretraining_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:34:02"
field: "视觉基础模型预训练"
keywords: ["MAE", "预预训练", "弱监督预训练", "视觉基础模型", "大规模预训练", "自监督学习"]
innovations: ["提出MAE预预训练框架，在弱监督预训练前增加自监督初始化阶段，无需额外数据或超参调优", "首次揭示MAE不仅随模型规模扩展，也随训练数据集规模扩展", "发现MAE初始化可突破WSP在目标检测任务上的模型规模扩展瓶颈"]
benchmarks: ["ImageNet-1k", "iNaturalist-18", "COCO", "LVIS", "Kinetics-400", "Food-101", "INv2", "IN-ReaL", "ObjectNet", "Something-Something-v2"]
---

# 论文速读：The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p

## 一句话总结
本文在标准弱监督预训练（WSP）前增加一个 MAE 自监督预预训练阶段（MAE→WSP），首次揭示 MAE 不仅随模型规模扩展，也随训练数据规模扩展；在 IG-3B（30 亿图像）上，MAE 预预训练在 10 类视觉任务上持续优于纯 WSP，并在 iNaturalist-18（91.3%）、1-shot ImageNet-1k（62.1%）、Food-101 零样本（96.2%）上刷新 SOTA。

## 研究问题与动机
1. **大规模视觉基础模型普遍采用弱监督预训练**：直接使用互联网海量带噪声标签的数据（如 Instagram hashtags）进行十亿级图像预训练，但忽略了初始化质量对收敛与下游性能的影响。
2. **自监督（MAE）与弱监督通常独立使用**：两者分别被证明有效，但缺乏在十亿级数据规模下系统探索其组合效果的实证研究。
3. **MAE 的扩展性此前仅被验证于模型规模**：He et al. [33] 证明 MAE 随模型参数量增长而扩展，但未探索其是否随训练数据量增长而扩展。
4. **模型初始化对 web-scale 预训练的影响被低估**：即使使用数十亿级别弱监督数据，初始化的重要性仍未被充分研究。

## 核心贡献（创新点）
1. **提出 MAE 预预训练框架**：先用 MAE 自监督初始化编码器，再用弱监督交叉熵进行标准预训练，无需额外数据或超参调优——本质区别在于将"中间微调"思路反转，用于预训练前的初始化。
2. **揭示 MAE 随数据集规模扩展**：在 IG-3B（30 亿图像）上验证 MAE 不仅随模型规模扩展，也随训练数据量扩展——这是对 [33] 结论的重要补充。
3. **系统评估十亿级预预训练在 10 类任务上的效果**：覆盖图像分类、视频识别、目标检测、少样本/零样本迁移、鲁棒性，证明预预训练在所有模型尺寸（86M-6.5B）和数据规模下均稳定提升——本质区别在于首次提供大规模多维度实证。
4. **发现 MAE 初始化可突破 WSP 在目标检测上的扩展瓶颈**：纯 WSP 扩大模型规模无法提升检测性能，而加入 MAE 预预训练后检测性能随模型规模持续扩展——本质区别在于揭示了不同预训练任务对下游任务可扩展性的差异化影响。

## 方法详解
- **MAE 预预训练**：随机遮盖图像 75% 的 patch，训练 ViT 编码器仅处理未遮盖的 25%，再通过轻量解码器重建缺失 patch 的像素值；目标 patch 像素值按均值-标准差归一化；训练 1 epoch，沿用 [33] 超参数无需调整。
- **弱监督预训练（WSP）**：从 Instagram 图像 captions 提取 hashtags，映射至 WordNet synsets 得到多标签分类标签；使用交叉熵损失训练编码器；默认从头随机初始化训练 1 epoch。
- **MAE→WSP 两阶段流程**：① 用 IG-3B 图像（无标签）训练 MAE 1 epoch；② 取出 MAE 编码器权重，继续使用 IG-3B（含 hashtag 标签）以交叉熵损失预训练 1 epoch；全程复用同一数据集，无需超参搜索。
- **零样本对齐**：冻结图像编码器，使用 LiT 方法从零样本图像-文本对训练文本编码器（XLM-R Large），采用 CLIP loss 对齐图文嵌入。

## 实验与结果
- **预训练数据集**：Instagram-3B（IG-3B），28K 类，30 亿唯一图像（重采样 50 亿），由 SWAG pipeline 生成。
- **评估任务与数据集**（Table 1）：ImageNet-1k、iNaturalist-18、INv2、IN-ReaL、ObjectNet、Food-101、COCO、LVIS、Kinetics-400、SSv2。
- **模型规模**：ViT-B (86M)、ViT-L (307M)、ViT-H (632M)、ViT-2B、ViT-6.5B。
- **最强结果**：
  - iNaturalist-18 fine-tune：**ViT-2B MAE→WSP 达 91.3%**（+4.5% vs. MAE [33]）
  - 1-shot ImageNet-1k：**62.1%**（VPT 协议，创 SOTA）
  - Food-101 零样本：**96.2%**（ViT-2B + LiT，SOTA）
  - IN1k fine-tune：ViT-2B 达 89.7%，ViT-H 达 89.3%
  - LVIS APbox：ViT-2B 达 51.8，COCO APbox：ViT-2B 达 58.0
  - K400：ViT-L 达 86.8%（仅用图像预训练，超过多数视频预训练方法）
- **关键对比**：MAE→WSP 始终优于纯 WSP（SWAG）和纯 MAE，且 2B 模型超过 6.5B WSP 模型。

## 相关工作脉络
1. **MAE [33]**：自监督 masked autoencoder，证明扩展性随模型规模；本文补充证明其也随数据集规模扩展。
2. **SWAG [70]**：十亿级弱监督预训练基线（ViT-H + IG-3B）；MAE→WSP 使用相同数据/架构但增加预预训练阶段，全面超越。
3. **DINOv2 [56]**：自监督大模型，冻结表征能力强；MAE→WSP 在 IN1k linear probing（88.1%）上超越 DINOv2（86.5%）。
4. **Scale-ViT [84]**：JFT-3B 上的大规模监督预训练；MAE→WSP 略低于 Scale-ViT 的 IN1k 精度，但在鲁棒性（IN-ReaL、ObjectNet）和细粒度任务上反超。
5. **CoCa [81] / Florence [82]**：结合图像-文本预训练；MAE→WSP 在零样本 Food-101 上达到 96.2%，与这些方法相当或更优。
6. **LiT [85]**：零样本图文对齐方法，本文沿用以赋予 MAE→WSP 零样本能力。

## 局限性与未来方向
1. **预训练数据集依赖性**：IN1k 精度落后于使用 JFT-3B+ALIGN 的方法（如 CoCa），说明预训练数据质量/内容对零样本和分类性能有显著影响。
2. **仅评估了 ViT 架构**：方法在 Vision Transformer 上验证，未探索 CNN 或其他架构的适用性。
3. **预预训练时长固定为 1 epoch**：虽证明 0.1 epoch 即可见效，但未系统研究更长预预训练阶段的效果边界。
4. **检测性能仍依赖额外数据**：在 COCO 上最高 APbox（58.0）落后于使用 Objects365/FLOD-9M 等方法，说明预预训练不能完全替代专门检测数据。
5. **未探索其他自监督方法**：仅使用 MAE 作为预预训练方法，其他自监督技术（如 DINO、iBOT）的效果未评估。

## 研究启发与可借鉴点
1. **模型初始化的重要性被低估**：即使是十亿级弱监督预训练，预初始化也能带来稳定收益；团队在大规模自监督/弱监督训练中应重视初始化策略。
2. **MAE 可作为通用预训练前置模块**：只需图像数据、无需额外标注，可复用开源 MAE 权重快速初始化，大幅简化训练流程。
3. **检测任务的扩展瓶颈可通过预训练策略突破**：纯 WSP 无法随模型规模提升检测性能，加入 MAE 预预训练后恢复扩展性——提示不同下游任务对预训练任务的敏感度存在差异。
4. **FLOPs 效率提升策略**：相同计算预算下 MAE→WSP 比纯 WSP 效率高约 2×，适合算力受限的大规模训练场景。
5. **跨尺度迁移的可行性**：在 IN21k（14M 图像）上也验证了预预训练有效性，说明方法不局限于十亿级数据，小数据场景同样可用。

## 关键术语表
**MAE（Masked Autoencoder）**：自监督预训练方法，随机遮盖图像 75% patch，训练模型重建缺失部分，利用非对称编解码设计提升效率。
**WSP（Weakly Supervised Pretraining）**：利用互联网图像的噪声标签（如 hashtags）进行多标签分类预训练，介于自监督与全监督之间。
**MAE→WSP**：先 MAE 自监督预预训练，再 WSP 弱监督预训练的两阶段框架，本文提出的核心方法。
**LiT（Locked Image Text tuning）**：冻结图像编码器、仅训练文本编码器进行图文对齐的零样本迁移方法。
**VPT（Visual Prompt Tuning）**：冻结 ViT 主干、仅训练视觉 prompt token 的高效微调协议，在少样本设置下表现优于线性分类器和 Adapter。
**IG-3B（Instagram-3B）**：本文使用的十亿级预训练数据集，28K 类、30 亿唯一 Instagram 图像及其 hashtag 标签。
**Linear Probing**：冻结预训练编码器，仅训练线性分类器评估特征质量的标准化评估协议。

## 可复现要素
- **预训练数据集**：IG-3B（由 SWAG pipeline 生成，基于 Instagram 数据）；论文未提供公开链接，但说明可复用 [70] 数据。
- **评估数据集**：全部为公开数据集（ImageNet-1k、iNaturalist-18、COCO、LVIS、Kinetics-400、SSv2、Food-101、INv2、IN-ReaL、ObjectNet）。
- **代码/权重**：论文未提及开源代码或权重；Meta AI 团队通常开源，需进一步确认。
- **关键超参**：MAE 遮盖比例 75%，预预训练 1 epoch，沿用 [33] 超参数；WSP 训练 1 epoch，沿用 [70] 超参数；ViT-B/L patch=16，ViT-H/2B/6.5B patch=14；分辨率 224×224（fine-tune 用 518×518）。
