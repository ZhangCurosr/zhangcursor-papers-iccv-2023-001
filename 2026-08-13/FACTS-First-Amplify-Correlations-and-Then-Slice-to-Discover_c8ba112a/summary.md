---
title: "FACTS-First-Amplify-Correlations-and-Then-Slice-to-Discover"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yenamandra_FACTS_First_Amplify_Correlations_and_Then_Slice_to_Discover_Bias_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:05:17"
---

# 论文速读：FACTS-First-Amplify-Correlations-and-Then-Slice-to-Discover

## 一句话总结
本文提出FACTS（First Amplify Correlations and Then Slice），一种两阶段算法：先通过强正则化ERM放大模型对虚假相关性的依赖，再在偏差放大后的特征空间中进行混合模型聚类，自动发现语义连贯的偏差冲突切片，无需额外标注且能处理多偏差场景。

## 研究问题与动机
- **核心问题**：计算机视觉数据集中常存在任务无关属性（如背景）与任务标签之间的虚假相关性，导致模型学"捷径"，在偏差冲突切片上表现差。
- **现有方法不足**：
  - 手动标注方案无法扩展到大尺度的未知潜在属性；
  - Domino等方法只能诊断每类单一偏差冲突切片，且对严重相关性偏置泛化差；
  - FD方法同样受限于每类仅能识别单一失败模式；
  - 基于外部预训练模型（如CLIP）的方法未控制预训练模型自身偏差，且可能遗漏被模型"记住但未理解"的少数群体样本。

## 核心贡献（创新点）
- **提出两阶段算法FACTS**：先AmCo（放大相关性）再CoSi（相关性感知切片），无需任何额外标注即可自动发现偏差冲突切片。
- **强正则化ERM放大偏差**：通过大weight decay限制模型容量，强制学习简单的偏差对齐假设，最大化类内偏差对齐/冲突样本的特征分离度。
- **混合模型在偏差放大特征空间聚类**：联合建模预测分布先验（相关性先验）和CLIP嵌入先验（语义连贯性），逐类拟合高斯混合模型发现多偏差切片。
- **广泛的多样化评测设置**：首次系统评测多偏差冲突切片（NICO++）、无偏差冲突切片（CelebA）等挑战性设置，Precision@10提升最高达35个百分点。

## 方法详解
**整体框架**：两阶段流程，如图2所示。

**Stage 1 – AmCo（放大相关性）**：
- 训练目标：$\min_W \mathbb{E}_{(x,y)\in D} \mathcal{L}_{CE}(h_W(x), y) + \lambda \|W\|_2^2$，其中$\lambda$取较大值，配合类别平衡采样防止预测坍缩。
- 超参选择策略： Sweep $\lambda \in [10^{-3}, 10^{-2}, 10^{-1}, 1.0, 2.0]$，在每个$\lambda$取训练准确率峰值的checkpoint，计算$\sigma_{\text{AmCo}}(\lambda) = \frac{1}{|\mathcal{N}|}\sum_c \text{Var}_{(x,y)\in D_c}[\text{softmax}(h_W(x))[y]]$，选最大化$\sigma_{\text{AmCo}}$的$\lambda^*$。
- 直觉：高$\sigma_{\text{AmCo}}$表明类内偏差对齐与偏差冲突样本的预测置信度方差大，即特征空间分离度高。

**Stage 2 – CoSi（相关性感知切片）**：
- 对每个类别$c$独立拟合混合模型，避免跨类污染。
- 两个先验：
  - **相关性先验**：将$h_{\text{AmCo}}(x)$的输出logits作为分布$\mathcal{N}(\mu_p^{(j)}, \Sigma_p^{(j)})$，鼓励预测相似的样本聚类。
  - **连贯性先验**：使用CLIP embedding $z_i = g(x_i)$建模为$\mathcal{N}(\mu_c, \Sigma_c)$，确保切片对应语义连贯概念，并支持自动caption。
- 对数似然：$l(\phi) = \sum_i \log \sum_j [P(S^{(j)}=1) \cdot P(Z=z_i|S^{(j)}=1) \cdot P(B=h_b(x_i)|S^{(j)}=1)^\alpha]$，用EM优化100步或直到log-likelihood变化<10⁻⁷。
- 每类设置$\hat{k}=36$个切片，按验证准确率升序排列，输出最差前k个切片给从业者。

## 实验与结果
**数据集**：
- Waterbirds（4795样本，2类，2个BC切片，相关性95%）
- CelebA（162770样本，2类，1个BC切片，相关性98%）
- NICO++⁷⁵/⁹⁰/⁹⁵（多类，每类5个BC切片，相关性75%/90%/95%）

**基线**：Domino [24]、Failure Directions (FD) [15]

**主要结果（Precision@10）**：
| 方法 | Waterbirds | CelebA | NICO++⁷⁵ | NICO++⁹⁰ | NICO++⁹⁵ |
|------|-----------|--------|----------|----------|----------|
| FD   | 0.9 | 0.7 | 0.19 | 0.19 | 0.19 |
| Domino | 1.0 | 0.9 | 0.24 | 0.25 | 0.27 |
| **FACTS** | **1.0** | **0.9** | **0.56** | **0.60** | **0.62** |

- NICO++⁹⁵上相对Domino绝对提升+35个百分点。
- Avg-AP（AmCo检索偏冲突样本）：FACTS达0.31，Lin. probe 0.25，最优早停策略为max train acc（0.31）。
- 消融表明：丢弃logits先验或CLIP先验均显著降低Precision@10；仅用预测label比用完整logits差0.24。

## 相关工作脉络
- **Domino [24]**：跨模态混合建模发现错误模式，但不做偏差放大，且使用软类别分配+预测label先验，无法泛化至多偏差场景；FACTS在偏差放大特征空间+完整logits建模。
- **Failure Directions (FD) [15]**：SVM蒸馏失败方向，每类仅能召回单一错误模式；FACTS可同时发现多个偏差冲突切片。
- **BAM [14]**：引入辅助变量平方损失放大偏差，但目标是直接提升worst-group accuracy而非切片发现；FACTS专注于切片发现与可解释性呈现。
- **JTT [12] / LfF [13]**：偏差缓解方法，先检测偏差再upweight；FACTS可作为上游的无监督偏差切片发现模块。
- **DrML [25] / Explaining Visual Biases [26]**：利用CLIP等外部模型，但未控制预训练模型自身偏差；FACTS纯数据驱动且不依赖外部模型做切片发现（仅用CLIP做连贯性先验）。

## 局限性与未来方向
- 仅针对相关性偏差，未覆盖标签噪声、歧义样本等其他偏差类型；
- 假设每个样本最多含一个虚假属性，现实中可能违反；
- 实验局限于ResNet50 + SGD，未验证于其他架构/优化器；
- 未来可探索多虚假属性共存的场景、其他偏差类型，以及更大规模真实数据集验证。

## 研究启发与可借鉴点
- **偏差放大策略可迁移**：通过强weight decay+early stop于训练准确率峰值来强制模型学简单假设，思路简洁有效，可借鉴到其它需要"暴露模型短板"的场景。
- **方差指标$\sigma_{\text{AmCo}}$作为自动化早停准则**：利用类内预测置信度方差最大化来选择超参，避免了依赖验证集性能的过拟合风险，值得推广。
- **混合模型双先验设计**：相关性先验（模型logits）+连贯性先验（CLIP embedding）的组合兼顾了偏差敏感性与语义可解释性，可在其它切片/分组发现任务中复用。
- **切片的排序+呈现机制**：按验证准确率升序排列切片，便于从业者快速定位最差子群体，这种"可操作化输出"设计值得参考。

## 关键术语表
- **Bias-conflicting slice**：数据子集，其中虚假相关性不成立（如天空背景上出现鸡）。
- **AmCo（Amplify Correlations）**：第一阶段，通过强正则化ERM放大模型对虚假属性的依赖。
- **CoSi（Correlation-aware Slicing）**：第二阶段，在偏差放大特征空间拟合混合模型发现切片。
- **$\sigma_{\text{AmCo}}$**：类内预测置信度方差，用于自动化选择最优weight decay。
- **Precision@k**：评估切片发现质量的核心指标，衡量Top-k样本中属于真实BC切片的比例。
- **Avg-AP**：对各类别分别计算AP后平均，评估AmCo阶段检索偏冲突样本的能力。
- **Spurious attribute**：与任务标签虚假相关的任务无关属性（如背景）。
- **Shortcuts**：模型利用虚假相关性走"捷径"而非学习真正判别特征的现象。

## 可复现要素
- **数据集**：Waterbirds、CelebA公开；NICO++公开（论文提供了${}^{75/90/95}$三种相关性强度设定）。
- **代码**：开源，见 https://github.com/yvsriram/FACTS。
- **关键超参**：weight decay sweep范围$[10^{-3}, 10^{-2}, 10^{-1}, 1.0, 2.0]$；学习率$10^{-5}$（AmCo模型）；$\hat{k}=36$；EM迭代上限100步或log-likelihood变化<10⁻⁷；CLIP coherence prior权重$\alpha$论文未明确给出具体值。
- **模型架构**：ResNet50，ImageNet预训练初始化，SGD优化器，momentum 0.9，batch size 64。
