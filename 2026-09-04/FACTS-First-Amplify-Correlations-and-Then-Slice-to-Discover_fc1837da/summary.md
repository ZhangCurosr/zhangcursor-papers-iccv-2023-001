---
title: "FACTS-First-Amplify-Correlations-and-Then-Slice-to-Discover"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yenamandra_FACTS_First_Amplify_Correlations_and_Then_Slice_to_Discover_Bias_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:14:19"
field: "计算机视觉公平性与偏差分析"
keywords: ["correlation bias", "slice discovery", "spurious correlation", "bias amplification", "mixture modeling", "fairness"]
innovations: ["提出 AmCo 偏差放大阶段通过强正则化 ERM 最大化偏差对齐与冲突样本分离", "设计 CoSi 相关性感知切片策略联合使用 logits 分布与 CLIP 一致性先验"]
benchmarks: ["Waterbirds", "CelebA", "NICO++"]
---

# 论文速读：FACTS: First Amplify Correlations and Then Slice to Discover Bias

## 一句话总结
论文提出 FACTS 算法，通过两阶段策略自动发现数据集中的相关偏差切片：首先通过强正则化训练放大模型的偏差依赖（AmCo），随后在放大的特征空间中进行聚类切片（CoSi），从而定位语义连贯的偏差冲突子群体。

## 研究问题与动机
- **虚假相关导致的偏差问题**：CV 数据集常存在任务无关属性（如背景）与标签的虚假相关，ERM 模型会学习"捷径"并在偏差冲突切片上表现不佳。
- **手动标注不可扩展**：现有方法依赖人工标注潜在虚假属性，但难以扩展到大数据集，且某些属性可能是隐式的或不可解释的。
- **既有自动化方法存在局限**：FD 方法每类只能发现一个偏差切片；Domino 在严重相关偏差下泛化性差；使用 CLIP 的方法无法控制外部预训练模型自身的偏差。
- **实用评估场景尚未被覆盖**：先前工作未考虑每类多个少数群体或某类无少数群体的实际场景。

## 核心贡献（创新点）
- **AmCo 偏差放大阶段**：通过大权重衰减强制模型学习简单的偏差对齐假设，最大化同类内偏差对齐与偏差冲突样本的分离度，与 ERM 直接利用多特征的学习形成对比。
- **CoSi 相关性感知切片**：在偏差放大特征空间拟合每类混合模型，结合预测 logits 的相关性先验与 CLIP 嵌入的一致性先验，比 Domino 的软类别约束更精确。
- **无需额外标注的全自动方法**：相比需要人工验证特征的 Wong 等方法，FACTS 无需任何额外属性标注即可发现语义连贯的偏差切片。
- **泛化到挑战性评估设置**：支持每类多个偏差冲突切片及某类无少数群体的场景，显著优于 FD（仅支持每类一个切片）和 Domino。

## 方法详解

**两阶段架构**：

**阶段一：AmCo（Amplify Correlations）**
- 通过强正则化 ERM 训练模型：$\arg\min_W \mathbb{E}_{(x,y)\in\mathcal{D}} \mathcal{L}_{CE}(h_W(x), y) + \lambda\|W\|_2$
- 使用类平衡采样防止模式坍缩
- 超参选择策略：对 $\lambda \in [10^{-3}, 10^{-2}, 10^{-1}, 1.0, 2.0]$ 搜索，在训练准确率峰值点计算类内预测方差 $\sigma_{\text{AmCo}}$，选择使其最大化的 $\lambda^*$
- 高 $\sigma_{\text{AmCo}}$ 表示偏差对齐与偏差冲突样本分离良好

**阶段二：CoSi（Correlation-aware Slicing）**
- 每类独立拟合高斯混合模型（GMM）
- 相关性先验：切片内模型预测 logits $B$ 建模为 $\mathcal{N}(\mu_p^{(j)}, \Sigma_p^{(j)})$
- 一致性先验：切片内 CLIP 嵌入 $z_i$ 建模为 $\mathcal{N}(\mu_c^{(j)}, \Sigma_c^{(j)})$
- 对数似然最大化：$l(\phi) = \sum_i \log\sum_j [P(S^{(j)}=1) \cdot P(Z=z_i|S^{(j)}=1) \cdot P(B=h_b(x_i)|S^{(j)}=1)^\alpha]$
- 通过 EM 算法拟合，$\hat{k}=36$ 切片数，$\Sigma_p$ 用全协方差、$\Sigma_c$ 对角协方差
- 推理时按验证准确率排序切片，提交最差 top-k 给人工干预

## 实验与结果

**数据集与设置**：
- Waterbirds（鸟类水/陆背景，2 偏差冲突切片）
- CelebA（金发/非金发分类，1 偏差冲突切片）
- NICO++（6 类图像在不同语境，设置 75%/90%/95% 相关强度）

**主要结果**（Precision@10）：

| 数据集 | FD | Domino | FACTS |
|--------|-----|--------|-------|
| Waterbirds | 0.9 | 1.0 | 1.0 |
| CelebA | 0.7 | 0.9 | 0.9 |
| NICO++⁷⁵ | 0.19 | 0.24 | **0.56** |
| NICO++⁹⁰ | 0.19 | 0.25 | **0.60** |
| NICO++⁹⁵ | 0.19 | 0.27 | **0.62** |

- 在挑战性 NICO++ 设置上相较 prior work 提升高达 **+35% precision points**
- 在 Slice Ranking AP 指标上 FACTS 达 0.97 vs Domino 0.92

**消融结论**：
- AmCo 阶段：偏差放大策略较 ERM 提升 Avg-AP +0.17；最大训练准确率作为停止准则有效（Oracle 仅 0.36）
- CoSi 阶段：同时使用 logits 分布 + CLIP 嵌入效果最佳（0.62 vs 仅 CLIP 0.38 或仅标签 0.26）
- 偏差放大验证：GT-Acc-gap 与 Avg-AP Pearson 相关系数 0.85

## 相关工作脉络

- **Failure Directions (FD)**：通过 SVM 学习每类错误方向，仅支持单偏差切片发现，FACTS 扩展至多切片场景
- **Domino**：基于跨模态嵌入的误差感知聚类，但未进行偏差放大、使用软类别约束，FACTS 通过特征空间放大增强分离
- **DrML**：使用 CLIP 文本嵌入探测视觉错误，但可能引入外部模型自身偏差，FACTS 用 CLIP 仅作为一致性先验
- **BAM/LfF/JTT**：偏差缓解方法侧重重加权，FACTS 专注于切片发现本身
- **AGRO/MAPLE**： adversarial 切片/双层级重新加权，FACTS 不依赖对抗训练

## 局限性与未来方向

- **仅聚焦相关偏差**：未处理标注错误、歧义样本等其他类型偏差
- **简化假设**：假设样本最多含一个虚假属性，实际数据可能更复杂
- **架构限制**：仅在 ResNet50 + SGD 设置下验证，未扩展到 Transformer 等新架构
- **切片数量固定**：$\hat{k}=36$ 为经验设置，自适应切片数选择未探索
- **外部 CLIP 偏差未完全消除**：一致性先验仍依赖预训练模型质量

## 研究启发与可借鉴点

- **偏差放大策略可迁移**：大权重衰减 + 类平衡采样的组合思想可用于其他偏差分析任务
- **混合模型 + 跨模态先验的设计**：将模型 logits 分布与语义嵌入联合建模，对多模态偏差发现具有借鉴价值
- **方差指标 $\sigma_{\text{AmCo}}$ 作为自监督信号**：无需标注即可度量偏差放大程度，可推广到其他偏差检测场景
- **切片排序辅助人工审核**：按验证准确率排序切片再呈现 Top-k，为 human-in-the-loop 偏差干预提供实用范式

## 关键术语表

**Correlation Bias**：数据集中任务无关属性与标签之间的虚假统计关联
**Spurious Attribute**：与任务标签强相关但非因果的辅助属性（如背景）
**Bias-conflicting Slice**：虚假相关性不成立的子数据集切片
**AmCo (Amplify Correlations)**：通过强正则化放大模型对虚假属性的依赖
**CoSi (Correlation-aware Slicing)**：在偏差放大特征空间中进行聚类切片的策略
**Precision@k**：评估切片发现质量的指标，衡量检索前 k 个样本中真阳性比例
**Avg-AP**：平均平均精度，衡量偏差冲突样本排序质量
**Cross-modal Embedding Prior**：使用 CLIP 跨模态嵌入作为切片语义一致性正则

## 可复现要素

- **数据集**：Waterbirds、CelebA、NICO++（均有公开来源）
- **代码**：已开源（https://github.com/yvsriram/FACTS）
- **关键超参**：weight decay $\lambda \in [10^{-3}, 2.0]$；学习率 $10^{-5}$（AmCo 阶段）；$\hat{k}=36$；EM 迭代 100 步或收敛阈值 $10^{-7}$
- **模型**：ResNet50，ImageNet 预训练初始化，SGD + momentum 0.9，batch size 64
