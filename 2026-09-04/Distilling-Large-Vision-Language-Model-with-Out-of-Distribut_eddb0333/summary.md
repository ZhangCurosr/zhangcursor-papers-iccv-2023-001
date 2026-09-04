---
title: "Distilling-Large-Vision-Language-Model-with-Out-of-Distribut"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:21:36"
---

# 论文速读：Distilling-Large-Vision-Language-Model-with-Out-of-Distribut

## 一句话总结
本文针对大视觉-语言模型（VLM）难以部署至资源受限设备的问题，研究利用中小规模数据集将其蒸馏至轻量学生模型，首次系统聚焦**开放词汇分布外（OOD）泛化**能力，提出视觉表征相对结构对齐、V-L对齐拓扑保留及LLM语言语义增强三大策略，显著提升了学生模型的零样本与少样本OOD分类性能。

## 研究问题与动机
- **部署瓶颈**：CLIP等大规模VLM参数量庞大、推理耗时高，无法满足移动端、IoT设备及实时机器人控制的部署需求，亟需通过知识蒸馏压缩。
- **OOD泛化被忽视**：现有VLM蒸馏工作多关注下游封闭集任务精度提升，**缺乏对开放词汇/分布外概念泛化能力的蒸馏机制设计**，导致蒸馏后学生模型在未见类别上性能骤降。
- **直接特征匹配失效**：仅用对比分类损失（$\mathcal{L}_{cls}$）对齐学生视觉特征与教师文本特征，会导致学生视觉空间严重过拟合训练标签；而直接MSE匹配教师高维视觉特征在实践中极难收敛，且绝对距离最小化并非保留空间拓扑的最优路径。
- **V-L对齐结构未受保护**：传统蒸馏仅关注最终分类概率匹配，忽略了教师图像特征与各类别语言特征之间的相对对齐顺序，该结构的破坏直接削弱学生向OOD概念的外推能力。

## 核心贡献（创新点）
1. **提出一套可量化的表征一致性度量体系**：设计 $\mathcal{M}_{rel}$、$\mathcal{
