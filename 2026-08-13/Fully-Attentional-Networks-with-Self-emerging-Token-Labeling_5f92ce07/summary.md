---
title: "Fully-Attentional-Networks-with-Self-emerging-Token-Labeling"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fully_Attentional_Networks_with_Self-emerging_Token_Labeling_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:07:40"
field: "视觉Transformer鲁棒性训练"
keywords: ["Vision Transformer", "Token Labeling", "Robustness", "Self-emerging Supervision", "Knowledge Distillation", "Out-of-distribution Generalization"]
innovations: ["首次证明ViT可作为有效token-labeler替代CNN教师", "Gumbel-Softmax自适应token筛选机制保留高置信度foreground标签", "Clean teacher-noisy student增强隔离设计提升分布外鲁棒性"]
benchmarks: ["ImageNet-1K", "ImageNet-C", "ImageNet-A", "ImageNet-R", "Cityscapes", "COCO"]
---

# 论文速读：Fully-Attentional-Networks-with-Self-emerging-Token-Labeling

## 一句话总结
论文提出**自涌现Token标注（STL）**框架，利用FAN ViT自身生成高质量语义token标签替代传统CNN token-labeler，在ImageNet-A（46.1%）和ImageNet-R（56.6%）上刷新SOTA，同时77.3M参数模型在ImageNet-1K达到84.8% Top-1准确率。

## 研究问题与动机
1. **ViT能否自我生成有意义的token标签？** 现有token labeling工作均依赖预训练CNN（如NFNet）作为token-labeler，缺乏对Transformer自产知识能力的探索。
2. **如何用自产知识替代外部教师改进预训练？** 解决"鸡生蛋"困境——无显式监督下如何训练出能生成准确token标签的FAN-TL。
3. **强数据增强破坏token标签质量。** Mixup、CutMix、RandAug等增强会干扰patch-level标注，需设计clean teacher-noisy student分离策略。
4. **token标签存在噪声。** 即使空间增强下，部分foreground tokens仍被误分类，需机制筛选高置信度标签。

## 核心贡献（创新点）
1. **ViT作为token-labeler的首次验证**：提出FAN-TL两阶段训练范式，通过同时监督class token和global average-pooled patch tokens，使FAN自产出语义明确的token标签；与LV-ViT本质区别在于teacher和student同构且均为ViT，而非CNN教师。
2. **Gumbel-Softmax自适应token筛选**：基于token label confidence score（最大类概率）识别错误foreground tokens并用Gumbel-Softmax替换低置信度标签；区别于传统hard/soft label选择，该方法无需额外标注即可实现token级质量过滤。
3. **Clean Teacher-Noisy Student增强隔离设计**：FAN-TL仅用空间增强（flip/rotate/shear/translation）生成干净标签，student使用全增强（含RandAug/CutOut/Mixup/CutMix）；与已有工作本质区别在于利用augmented discrepancy提升鲁棒性而非一致性蒸馏。
4. **多基准SOTA刷新**：77.3M参数模型在ImageNet-A（46.1%）、ImageNet-R（56.6%）创纪录，IN-C mCE降至42.1，超越原FAN-L-Hybrid分别达+4.3%/+3.4%/−0.9 mCE。

## 方法详解
**两阶段训练框架：**

**Stage 1 — FAN-TL训练（Token Labeler）：**
- 输入图像I，patch tokens序列 $[\mathbf{T}_{p_1}, ..., \mathbf{T}_{p_N}]$，class token $\mathbf{T}_{cls}$
- 对patch tokens做global average-pooling后与class token共享同一class label $\mathbf{Y}_{cls}$ 监督
- 损失函数：$\mathcal{L} = \mathcal{H}(\mathbf{T}_{cls}, \mathbf{Y}_{cls}) + \alpha \cdot \mathcal{H}(\frac{1}{N}\sum_{i=1}^{N}\mathbf{T}_{p_i}, \mathbf{Y}_{cls})$，其中 $\alpha=1$
- 输出每个patch token的类别概率分布作为token label

**Stage 2 — Student训练（含STL）：**
- FAN-TL为student的patch tokens提供token labels $\mathcal{F}(\mathbf{I}_{p_i})$
- 损失函数：$\mathcal{L} = \mathcal{H}(\mathbf{T}_{cls}, \mathbf{Y}_{cls}) + \beta \cdot \frac{1}{N}\sum_{i=1}^{N}\mathcal{H}(\mathbf{T}_{p_i}, \hat{\mathcal{F}}(\hat{\mathbf{I}}_{p_i}))$，其中 $\beta=1$
- $\hat{\mathcal{F}}(\cdot)$ 为Gumbel-Softmax处理后转one-hot的token label
- Gumbel-Softmax公式：$\mathbf{y}_i = \frac{e^{(\log(\pi_i)+\mathcal{G}_i)/\tau}}{\sum_j e^{(\log(\pi_j)+\mathcal{G}_j)/\tau}}$，低置信度token标签被随机化替换

**关键设计细节：**
- FAN-TL在Stage 2也仅接收spatial-only增强输入
- Student接收完整增强（含RandAug/CutOut/MixUp/CutMix）
- Softmax输出转为hard label（one-hot）以强化entropy minimization自训练效果

## 实验与结果
**数据集：** ImageNet-1K（分类）、ImageNet-C/A/R（鲁棒性）、Cityscapes/City-C（分割）、COCO（检测）

**主要结果（Table 2-5）：**
| 模型 | IN-1K Top-1 | IN-C mCE↓ | IN-A | IN-R |
|------|-------------|-----------|------|------|
| FAN-L-Hybrid | 84.3% | 43.0 | 41.8 | 53.2 |
| **STL (FAN-L-Hybrid)** | **84.7%** | **42.5** | **46.1%** | **56.6%** |
| Swin-B | 83.5% | 54.4 | 35.8 | 46.6 |
| ConvNeXt-B | 83.8% | 46.8 | 36.7 | 51.3 |

- **最强结果：** STL(FAN-L-Hybrid) 77.3M参数，IN-1K 84.7% / IN-C mCE 42.1（heterogeneous TL-B-Hybrid）/ IN-A 46.1% / IN-R 56.6%
- **提升幅度：** 相对原FAN-L-Hybrid，IN-A +4.3pp、IN-R +3.4pp、mCE −0.9
- **分割迁移：** Cityscapes mIoU 82.8（+0.5）、City-C mIoU 69.2（+0.5，retention 83.6%）
- **检测迁移：** COCO mAP 54.1（FAN-L，持平；FAN-B +0.4pp）
- **heterogeneous训练（Table 8）：** FAN-L-Hybrid + FAN-TL-B-Hybrid 达到最优 84.8% IN-1K / 42.1 mCE

## 相关工作脉络
1. **Token Labeling (Jiang et al., NeurIPS 2021)**：首次提出token labeling，使用预训练CNN（NFNet-F6，400M参数）作为token-labeler；本文区别在于用同构ViT替代CNN，且参数量仅77M。
2. **Fully Attentional Network (Zhou et al., ICML 2022)**：本文基础架构，引入channel attention block；本文在其上叠加STL进一步提升鲁棒性。
3. **Knowledge Distillation (Hinton et al.)**：token labeling本质为dense版hard KD；本文区别为teacher和student同构、自产label而非外部教师。
4. **ReLabel (Yun et al., CVPR 2021)**：图像级multi-label标注；本文扩展到patch-level dense标注。
5. **BEiT (Bao et al., 2021)**：使用离散VAE tokenizer生成视觉token；本文token label具有显式语义类别含义，不同于codebook的无监督discrete token。
6. **DINO (Caron et al., ICCV 2021)**：自监督ViT涌现对象分割；本文采用fully supervised方式利用FAN的visual grouping能力生成token label。

## 局限性与未来方向
1. **两阶段训练复杂度增加**：需分别训练FAN-TL和student，训练成本高于单阶段方法。
2. **下游任务提升不均衡**：目标检测提升有限（FAN-L持平），作者归因于dense prediction与regression任务的差异。
3. **Gumbel-Softmax超参τ未充分调优**：实验仅展示默认设置，temperature调度策略值得探索。
4. **heterogeneous TL规模探索有限**：大student配小TL可降本但TL能力上限受限，需进一步平衡。
5. **仅验证FAN架构**：方法迁移性在Swin/其他ViT变体上未系统评估。

## 研究启发与可借鉴点
1. **Clean Teacher-Noisy Student范式可迁移**：对任何依赖辅助监督信号（如pseudo label、feature distillation）的方法，教师-学生对数据增强的"清洁度"差异均可作为提升鲁棒性的正则手段。
2. **Confidence-based token筛选机制**：Gumbel-Softmax+hard label转换的自训练策略，可推广至其他dense prediction预训练（如MAE预训练中的masked token质量过滤）。
3. **同构ViT替代CNN作为knowledge source**：打破"ViT学生需CNN教师"的范式，为后续研究纯Transformer蒸馏路线提供实证依据。
4. **Visual grouping能力用于监督信号生成**：FAN的channel attention驱动的object gestalt捕捉能力，可启发其他具备类似 grouping 特性的架构（如GroupViT、SAM）探索自监督token生成。
5. **Retention Rate指标的应用**：将鲁棒精度与clean精度的比值作为公平比较不同容量模型的指标，值得在鲁棒性研究中推广。

## 关键术语表
- **FAN (Fully Attentional Network)**：在ViT基础上引入channel attention block的ViT变体家族，通过聚合跨通道信息提升表征能力和分布外鲁棒性。
- **Token Labeling**：为ViT的每个patch token分配类别标签，实现dense supervision的预训练策略，区别于传统的image-level single-label训练。
- **Gumbel-Softmax**：可微分的 categorical reparameterization技巧，通过对softmax输出施加Gumbel噪声实现低置信度token标签的随机化替换。
- **Self-emerging Token Labeling (STL)**：本文提出的两阶段框架，ViT自产token标签替代外部CNN教师，实现"鸡生蛋"问题的闭环训练。
- **Retention Rate**：鲁棒准确率与clean准确率的比值（$R = \text{Robust Acc.} / \text{Clean Acc.}$），用于公平比较不同参数量模型的鲁棒性保持能力。
- **mCE (mean Corruption Error)**：ImageNet-C上各类 corruption 误差的平均值，越低表示对分布偏移的鲁棒性越强。
- **Foreground Tokens**：与图像level标签一致的patch tokens，通常对应目标物体区域；本文重点保障其标签准确性。
- **Heterogeneous Token-Labeler**：token-labeler与student模型规模不同的配置（如大student配小TL），可降低成本并保持性能。

## 可复现要素
- **代码**：基于PyTorch + timm + MMSegmentation，作者未明确提供开源链接，代码实现依赖公开库
- **数据集**：ImageNet-1K（公开）、ImageNet-C/A/R（公开）、Cityscapes（公开）、COCO（公开）
- **关键超参**：α=1, β=1, learning rate=4e-3, batch size=2048, 350 epochs, cosine scheduler decay 0.1/30epochs, label smoothing=0.9
- **硬件**：8× NVIDIA Tesla V100
- **训练细节**：FAN-TL Stage 2输入仅spatial增强；Student输入全增强；token label转one-hot hard label
