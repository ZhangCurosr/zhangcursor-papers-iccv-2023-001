---
title: "Improving-Generalization-of-Adversarial-Training-via-Robust"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Improving_Generalization_of_Adversarial_Training_via_Robust_Critical_Fine-Tuning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:56:12"
---

# 论文速读：Improving-Generalization-of-Adversarial-Training-via-Robust

## 一句话总结
本文提出**鲁棒性关键微调（RiFT）**，通过模块鲁棒临界度（MRC）识别对抗训练模型的“非鲁棒关键模块”，仅对该模块进行标准数据微调并与原始权重线性插值，在不损失对抗鲁棒性的前提下，显著提升了模型的分布内泛化与分布外（OOD）鲁棒性。

## 研究问题与动机
- 对抗训练（AT）能有效提升对抗鲁棒性，但普遍以牺牲分布内泛化与OOD鲁棒性为代价，存在固有的鲁棒性-泛化性权衡（trade-off）。
- 现有工作多聚焦于AT训练阶段改进（如实例重加权、引入无标签数据、重构损失函数），缺乏从训练后模型冗余容量角度出发的轻量化优化思路。
- 深度神经网络存在拟合冗余容量，但针对“对抗鲁棒性”的模块级冗余尚未被系统量化与利用。
- 核心科学问题：AT模型是否具备鲁棒性冗余容量？如何精准定位并激活该冗余以提升泛化与OOD表现，同时保持对抗鲁棒性？

## 核心贡献（创新点）
- **提出MRC度量**：首次量化评估AT模型各模块在最坏情况权重扰动下的对抗损失上升上限，验证了对抗训练模型存在显著的鲁棒性冗余容量。
- **设计RiFT三步框架**：MRC识别非鲁棒关键模块 → 仅微调该模块 → 与原始AT权重线性插值，实现泛化与鲁棒性的协同提升，且无需修改原有AT训练流程。
- **提供反直觉实证洞察**：打破“最优标准分类器与鲁棒分类器特征本质不同”的论断，证明微调可同步提升两者；同时证实AT模型微调不会恶化OOD鲁棒性。
- **强兼容性验证**：RiFT可无缝接入TRADES、MART、AWP、SCORE等主流AT方法，进一步放大各项指标收益。

## 方法详解
- **MRC定义与松弛计算**：给定扰动缩放因子 $\epsilon$ 与约束集 $\mathcal{C}_\theta = \{\Delta\theta \mid \|\Delta\theta\|_p \leq \epsilon\|\theta^{(i)}\|_p\}$，MRC衡量仅扰动第 $i$ 个模块时对抗损失的最大增量（式2）。为降低联合优化成本，采用松弛策略：先用PGD固定对抗样本，再对权重扰动执行梯度上升迭代，并在每步将扰动投影回约束球内（Algorithm 1）。
- **非鲁棒关键模块定位**：计算所有模块MRC值，选取最低者 $\tilde{\theta}$ 作为非鲁棒关键模块；其参数变动对对抗鲁棒性影响最小，适合作为微调目标。
- **模块级微调**：冻结除 $\tilde{\theta}$ 外的全部参数，在标准数据集 $\mathcal{D}_{std}$ 上用SGD+动量微调10 epoch，目标函数加入 $\ell_2$ 权重衰减（式8），学习率设为0.001，5 epoch后衰减10倍。
- **插值寻优**：构造 $\theta_\alpha = (1-\alpha)\theta_{AT} + \alpha\theta_{FT}$，在验证集上搜索使泛化提升最大且对抗鲁棒性下降 $\leq 0.1\%$ 的 $\alpha^*$。实证表明 $\alpha^*$ 通常位于 $0.6\sim0.9$，插值过程兼具轻量级权重集成（weight-ensemble）效应。

## 实验与结果
- **实验设置**：骨干网络 ResNet18、ResNet34、WRN34-10；训练集 CIF
