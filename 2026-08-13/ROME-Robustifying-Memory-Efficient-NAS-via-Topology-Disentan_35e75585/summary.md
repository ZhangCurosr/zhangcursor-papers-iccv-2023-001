---
title: "ROME-Robustifying-Memory-Efficient-NAS-via-Topology-Disentan"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_ROME_Robustifying_Memory-Efficient_NAS_via_Topology_Disentanglement_and_Gradient_Accumulation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:27:50"
---

# 论文速读：ROME-Robustifying-Memory-Efficient-NAS-via-Topology-Disentan

## 一句话总结
本文针对单路径可微分 NAS（如 GDAS）在降低显存的同时出现的严重性能坍塌问题，提出 ROME 算法，通过拓扑解耦与梯度累积策略，在保持极低显存开销与搜索成本的前提下，实现稳定、高效的架构搜索，并在 15 个基准上达到 SOTA。

## 研究问题与动机
- 单路径可微分 NAS（Single-path DARTS，如 GDAS、SNAS）虽通过每次仅激活一条子路径大幅降低显存与计算成本，但作者发现其同样存在严重的性能坍塌（Performance Collapse），即搜索出的网络中无参操作（尤其是 skip connection）比例过高，导致最终模型精度劣化。
- 现有坍塌分析多集中于全路径 DARTS，单路径方法中的该问题长期未被深入探究，缺乏针对性机制。
- 坍塌根源一：搜索期与评估期拓扑不一致。超网中节点连接所有前驱边，而最终网络要求每个中间节点入度严格为 2，造成结构偏差。
- 坍塌根源二：采样不充分导致的梯度方差过大。每步仅采样 1 条边和 1 个操作，使候选操作权重训练不公平，且架构权重 α 的梯度估计方差过高，阻碍收敛。

## 核心贡献（创新点）
1. **首次系统揭示单路径可微分 NAS 的性能坍塌现象**。与既往工作聚焦全路径 DARTS 不同，本文通过复现 GDAS 并统计无参操作比例，明确指出单路径方法同样受困于坍塌，填补了该方向的认知空白。
2. **提出拓扑解耦（Topology Disentanglement）实现搜索与评估一致**。通过独立引入拓扑参数 β 控制边选择、架构参数 α 控制操作选择，并结合 Gumbel-Top2 重parameterization，是首个在单路径可微分 NAS 中达成搜索-评估拓扑一致性的方法。
3. **设计梯度累积策略稳健化双层优化**。提出两种累积技术：对 K 个子模型累加梯度以公平训练操作权重 θ；对 K 次采样的架构权重梯度取平均，将方差从 σ² 降至 σ²/K，显著提升收敛稳定性。
4. **在 15 个基准上实现 SOTA 且保持低显存**。相比 GDAS 和 PC-DARTS 分别节省 26% 和 38% 显存，在标准 DARTS 空间下仅需 0.3 GPU days 即可在 CIFAR-10 达到 97.52% 准确率，具强鲁棒性与实用性。

## 方法详解
- **拓扑解耦与双参数建模**：将架构采样 $z$ 分解为两步，引入二值变量 $B_{i,j} \in \{0,1\}$ 表示边 $e_{i,j}$ 是否选中，$A_{i,j}^o \in \{0,1\}$ 表示边上操作 o 是否选中。强制约束 $\sum_{i<j} B_{i,j} = 2, \forall j$，确保搜索期节点入度与评估期完全一致。
- **操作采样（Gumbel-Softmax）**：基于归一化架构权重 $\tilde{\alpha}_{i,j}^o = \exp(\alpha_{i,j}^o) / \sum_{o'} \exp(\alpha_{i,j}^{o'})$ 与 Gumbel(0,1) 噪声采样，经温度 τ 退火得到连续松弛近似 $\tilde{A}_{i,j}^o$，再通过 argmax 得到 one-hot 操作选择，保证操作搜索可微。
- **边采样（Gumbel-Top2 重parameterization）**：ROME-v2 提出 Gumbel-Top2，对每个节点的前驱边按概率 $\tilde{\beta}_{i,j}$ 抽取 Top-2 条边。论文给出理论证明，该策略等价于不带替换的概率单纯形采样，且具备完整的梯度回传路径。
- **梯度累积双层更新（Algorithm 1）**：每次迭代划分两个不重叠数据集 $D_s$（验证）与 $D_t$（训练）。先采样 K 个架构在 $D_s$ 上累积更新 α、β（$\alpha \leftarrow \alpha - \frac{1}{K}\sum_{k=1}^K \nabla_\alpha L_{val}$）；再采样 K 个架构在 $D_t$ 上累加更新 θ（$\theta \leftarrow \theta - \sum_{k=1}^K \nabla_\theta L_{train}$）。多采样平均有效抑制单次随机采样带来的估计偏差。

## 实验与结果
- **数据集与搜索空间**：S0（标准 DARTS 空间，排除 none 操作以满足入度约束，用于 CIFAR-10/100 与 ImageNet）、S1-S4（R-DARTS 提出的四个子空间，用于 CIFAR-10/100 与 SVHN），共计 15 个评测基准。
- **坍塌诊断与鲁棒性**：在 NAS-Bench-1Shot1 中加入 skip connection 后，GDAS 搜索模型几乎全被无参操作占据（平均误差 96.52%），ROME 有效避免该现象并达到 97.42%，验证了解耦机制的有效性。
- **S0 空间性能**：ROME-v2 最佳模型在 CIFAR-10 达 97.52% 准确率（2.48% error，3.6M params），搜索成本仅 0.3 GPU days；CIFAR-100 达 17.71% error；ImageNet 直接搜索达 75.5% top-1。
- **显存与效率对比**：ROME 峰值显存仅 2.3 GB，优于 GDAS (3.1 GB)、PC-DARTS (3.7/5.7 GB)。搜索速度比 R-DARTS 快 5×，比 SDARTS-ADV 快 4×。
- **消融实验**：采样数 K 在 7 时性能饱和（97.42%±0.07%）；拓扑解耦（TD）与梯度累积（GA）均独立贡献提升，组合后效果最优。

## 相关工作脉络
- **DARTS / GDAS / PC-DARTS**：可微分 NAS 的代表性工作。本文针对单路径变体的坍塌缺陷进行修补，而传统 DARTS 因显存限制难以直接扩展至大图数据集，单路径方法虽高效却存在稳定性盲区。
- **R-DARTS / DARTS- / SDARTS-ADV**：通过 Hessian 特征值正则或扰动正则缓解全路径坍塌。本文指出单路径存在同类问题，且无需依赖额外正则先验或复杂指标，仅凭结构设计与累积策略即可根治。
- **DOTS**：同样解耦操作与拓扑，但需预先将操作分为“无参/有参”两组并多阶段训练，依赖强先验且需逐数据集调参。ROME 端到端单阶段完成，无需额外超参校准。
- **SNAS / FairNAS / ProxylessNAS**：单路径/记忆高效 NAS 工作。SNAS 未解决坍塌且 supernet
