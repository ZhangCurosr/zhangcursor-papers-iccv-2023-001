---
title: "FedPerfix-Towards-Partial-Model-Personalization-of-Vision-Tr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_FedPerfix_Towards_Partial_Model_Personalization_of_Vision_Transformers_in_Federated_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:14:45"
field: "联邦学习"
keywords: ["个性化联邦学习", "Vision Transformer", "部分模型个性化", "Prefix-tuning", "插件", "数据异构性"]
innovations: ["提出FedPerfix，将Prefix插件与并行适配器结合实现ViT自注意力层个性化", "揭示prefix学习相当于全局与局部注意力的混合，与APFL建立理论联系", "系统评估ViT各层敏感性，为个性化位置选择提供实证依据"]
benchmarks: ["CIFAR-100", "OrganAM-NIST", "Office-Home"]
---

# 论文速读：FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning

## 一句话总结
论文针对个性化联邦学习（PFL）中视觉Transformer（ViT）的部分模型个性化问题，通过经验研究揭示ViT的自注意力层和分类头对数据分布最敏感，并提出FedPerfix方法，利用Prefix插件结合并行适配器在自注意力层实现高效个性化，在多个数据集上达到SOTA性能。

## 研究问题与动机
- 个性化联邦学习旨在应对数据异构性，但现有部分模型个性化方法主要基于CNN，缺乏对ViT架构的系统探索。
- ViT在集中式训练中表现优异，但在PFL环境中的应用研究有限，需探索其在非独立同分布（non-IID）数据下的个性化策略。
- 现有部分模型个性化方法对于“在哪里个性化”和“如何个性化”ViT尚未有充分研究，直接沿用CNN方法可能导致次优性能。
- 完全本地更新自注意力层会阻碍全局知识共享，需要一种既能保留全局信息又能捕捉客户端特定知识的平衡机制。

## 核心贡献（创新点）
- 系统评估ViT各层对数据分布的敏感性，为部分模型个性化提供理论依据和实证支持。
- 提出FedPerfix方法，将Prefix插件与并行适配器结合，实现自注意力层的高效个性化。
- 将Prefix自适应视为全局与局部注意力的混合，揭示与APFL方法的内在联系。
- 在多个数据集和不同联邦学习设置下验证方法的有效性和资源效率。

## 方法详解
- 问题形式化：客户端模型参数分为全局参数 $u$ 和局部参数 $v_i$，通过联邦平均进行全局聚合。
- 经验研究：在CIFAR-100上测试各层独立更新或组合更新的性能，确定自注意力层和分类头为最敏感层（见表1）。
- Vanilla Prefix-tuning基线：将可学习prefix附加到自注意力层的key和value矩阵，本地更新，原始自注意力层全局聚合。
- FedPerfix核心设计：引入并行适配器生成prefix，公式为 $P_k, P_v = \mathrm{Tanh}(Z^{l-1} W_{down}) W_{up}$，使用标量 $s$ 控制适配器效率。
- 自注意力层输出修改为 $\mathrm{head}_i = \mathrm{Attention}(Z^{l-1} W_q^{(i)}, Z^{l-1}[sP_k, W_k^{(i)}], Z^{l-1}[sP_v, W_v^{(i)}])$。
- 方法灵感来自迁移学习中的插件思想，将聚合后的ViT视为预训练模型，prefix作为下游特定知识捕获模块。

## 实验与结果
- 数据集：CIFAR-100（标签偏斜）、OrganAM-NIST（医疗图像）、Office-Home（概念偏斜）。
- 基线方法：FedAvg、Local、APFL、Per-FedAVG、FedBN、FedRep、FedBABU、Vanilla Attention等。
- 主要结果：FedPerfix在CIFAR-100上达到48.10%准确率，提升约3.7个百分点（相对FedRep）；OrganAM-NIST达93.17%，Office-Home达24.38%，均为SOTA。
- 资源分析：FedPerfix存储参数24.42M（116%），计算量66.58M FLOPs（101%），通信参数20.66M（98%），与基线相当。
- 客户级性能：FedPerfix平均性能增益超10%，几乎保证所有客户端性能提升（图3）。
- 鲁棒性：在不同通信轮次、客户端参与率和数据集划分下保持优越性能（表4）。

## 相关工作脉络
- FedRep、FedBN、FedBABU等CNN-based部分个性化方法：本文借鉴其“敏感层本地更新”思想，但针对ViT架构重新评估。
- APFL（自适应个性化联邦学习）：本文揭示FedPerfix与APFL的数学等价性（混合系数形式），但仅混合敏感层而非全模型。
- Prefix-tuning（Li & Liang, 2021）：将自然语言处理中的插件技术迁移至计算机视觉联邦学习场景。
- 参数高效微调（PEFT）方法如LoRA、Adapter：本文强调插件在联邦学习中的适用性，尤其是ViT的自注意力层适配。
- 视觉Transformer在联邦学习中的应用：Qu et al. (2022)发现ViT可加速收敛，本文进一步探索其个性化潜力。

## 局限性与未来方向
- 实验主要集中在图像分类任务，未探索目标检测、分割等更复杂视觉任务。
- 未深入分析prefix规模、适配器维度等超参数对性能的影响。
- 仅考虑客户端参与率为12.5%-25%的场景，极端低参与率下的表现有待验证。
- 未讨论prefix初始化策略的替代方案（如预训练权重初始化）。
- 可拓展至多模态联邦学习或跨域个性化场景。

## 研究启发与可借鉴点
- **层敏感性评估框架**：系统评估模型各层对数据分布的敏感性，为个性化位置选择提供通用方法论。
- **插件思想的联邦迁移**：将迁移学习中的高效微调插件（prefix、adapter）与联邦学习结合，实现知识传递与个性化平衡。
- **并行适配器稳定初始化**：用缩放-激活-升维的并行适配器替代随机初始化，提升插件训练的稳定性。
- **全局-局部混合视角**：将prefix学习解释为全局与局部注意力的自适应混合，为其他个性化方法提供理论洞察。
- **资源效率权衡**：在保持通信开销基本不变的前提下，显著提升性能，为资源受限联邦场景提供实用方案。

## 关键术语表
- **个性化联邦学习（PFL）**：目标为各客户端训练个性化模型而非单一全局模型的联邦学习范式。
- **部分模型个性化**：仅更新模型参数子集（本地参数），其余参数通过服务器聚合的全局参数。
- **Prefix-tuning**：在模型特定层前插入可学习向量，仅训练这些向量而冻结主干参数。
- **并行适配器**：与主网络并行的轻量级模块，通常包含降维、激活、升维操作，用于稳定特征转换。
- **自注意力层**：Transformer核心组件，计算序列内各位置的注意力权重以捕获长距离依赖。
- **数据异构性**：不同客户端数据分布差异，包括标签偏斜（label skew）和概念偏斜（concept skew）。

## 可复现要素
- 数据集：CIFAR-100、OrganAM-NIST、Office-Home；代码已开源（https://github.com/imguangyu/FedPerfix）。
- 模型架构：ViT-Small，patch size 16，输入尺寸224×224。
- 超参数：通信轮数T=50，本地训练10 epoch，学习率0.01，优化器SGD，batch size 64。
- 数据划分：CIFAR-100用Dirichlet α=0.1分给64客户端，OrganAM-NIST用α=0.5，Office-Home用α=1.0。
- 基线实现：参考原文实验设置复现，使用PyTorch和TIMM库。
