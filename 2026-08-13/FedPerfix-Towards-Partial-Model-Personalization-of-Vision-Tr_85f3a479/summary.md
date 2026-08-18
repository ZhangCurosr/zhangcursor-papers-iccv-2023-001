---
title: "FedPerfix-Towards-Partial-Model-Personalization-of-Vision-Tr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_FedPerfix_Towards_Partial_Model_Personalization_of_Vision_Transformers_in_Federated_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:06:35"
field: "个性化联邦学习"
keywords: ["Federated Learning", "Personalized FL", "Vision Transformer", "Partial Model Personalization", "Prefix-Tuning", "Parameter-Efficient Fine-tuning"]
innovations: ["实证定位 ViT 中 Self-attention 和分类头为最敏感层", "引入带并行适配器的 Prefix 插件实现稳定个性化", "揭示 Prefix 与全局-本地注意力混合系数的理论联系"]
benchmarks: ["CIFAR-100", "OrganAM-NIST", "Office-Home"]
---

# 论文速读：FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning

## 一句话总结
本文针对 Vision Transformer（ViT）在个性化联邦学习中的部分模型个性化问题，提出 FedPerfix 方法：通过实证研究定位 ViT 中最敏感层（Self-attention 和分类头），并引入带并行适配器的 Prefix 插件实现全局知识到客户端特有条件的高效迁移。

## 研究问题与动机
1. **现有方法架构偏差**：部分模型个性化（Partial Model Personalization）工作几乎全部面向 CNN，缺乏对 ViT 架构的系统性研究。
2. **"在哪里个性化"未知**：ViT 各层对数据分布的敏感度不同，需实证确定哪些层应本地保留、哪些可全局聚合。
3. **完全本地化的知识隔离问题**：若直接将敏感层完全本地更新，会阻断全局通用知识的传递；需设计能桥接全局与本地知识的机制。
4. **ViT 在 PFL 中的潜力未被挖掘**：已有研究显示 ViT 在联邦学习中收敛更快、全局模型更优，但个性化场景下如何适配尚属空白。

## 核心贡献（创新点）
1. **层敏感性实证研究**：系统量化 ViT 各层（Patch Embedding、Position Embedding、LayerNorm、Self-attention、MLP、Classification Head）对数据分布的敏感度，首次明确 Self-attention 和 Classification Head 为最敏感层。
2. **FedPerfix 插件式个性化框架**：借鉴 NLP 领域 Prefix-Tuning，将可学习 Prefix 拼接至 Self-attention 的 K/V 矩阵，Prefix 本地训练捕获客户端特有知识，原始 Self-attention 层仍全局聚合。
3. **并行适配器稳定初始化**：针对 Vanilla Prefix-tuning 对随机初始化敏感的问题，引入并行适配器（Scale-down → Tanh → Scale-up）生成 Prefix，显著提升训练稳定性。
4. **揭示 FedPerfix 与 APFL 的理论联系**：推导表明 Prefix 学习的效果等价于全局注意力与个性化注意力的混合系数，为理解 PFL 提供统一视角。
5. **多数据集 SOTA 且资源高效**：在 CIFAR-100、OrganAM-NIST、Office-Home 上全面超越 FedAvg、APFL、FedRep、FedBABU 等基线，通信开销降低约 2%。

## 方法详解
**层敏感性实验设计**：在 CIFAR-100 上分别将各层类型设为本地更新（不聚合），以 Top-1 Accuracy 均值和标准差评估敏感度；结果如 Table 1 所示，Self-attention stand-alone 得分 42.95，仅次于 Classification Head（44.42）。

**FedPerfix 核心公式**：
- Prefix 生成（并行适配器）：$P_k, P_v = \text{Tanh}(Z^{l-1} W_{down}) W_{up}$，其中 $W_{down}$、$W_{up}$ 为降维/升维投影，$Z^{l-1}$ 为上一层输出。
- Self-attention 输出：$\text{head}_i = \text{Attention}(Z^{l-1} W_q^{(i)},\ Z^{l-1}[sP_k, W_k^{(i)}],\ Z^{l-1}[sP_v, W_v^{(i)}])$，$s$ 为控制适配器效率的超参数。
- 混合注意力解释：公式(6)显示 Prefix 使 attention 输出等效为全局聚合注意力与个性化 Prefix 注意力的加权混合。

**训练流程**：每轮客户端接收全局参数 $u^{(t-1)}$，插入本地 Prefix $v_i^{(t-1)}$，在本地数据上训练；仅上传全局 Self-attention 参数参与服务器聚合。

## 实验与结果
**数据集与设置**：
- CIFAR-100：64 clients，Dirichlet $\alpha=0.1$，参与率 $r=12.5\%$（label skew）
- OrganAM-NIST：64 clients，$\alpha=0.5$，相同参与率
- Office-Home：16 clients，$\alpha=1.0$，参与率 $r=25\%$（concept skew）
- 模型：ViT-Small（21.03M 参数），50 轮通信，每轮 10 epoch，SGD lr=0.01

**主要结果（Table 2）**：
| 数据集 | FedPerfix | 次优方法 | 提升幅度 |
|---|---|---|---|
| CIFAR-100 | **48.10 ± 7.76** | APFL 44.88 | +3.22 |
| OrganAMNIST | **93.17 ± 3.51** | FedBABU 92.63 | +0.54 |
| Office-Home | **24.38 ± 8.47** | APFL 24.23 | +0.15 |

**关键结论**：
- FedPerfix 在三个数据集上均取得最优，且标准差最低（尤其 OrganAM-NIST 仅 ±3.51）。
- 相比 FedAvg，CIFAR-100 上绝对提升 **+24.81**，相对提升约 **106%**。
- 客户端级分析（Fig.3）显示 FedPerfix 几乎保证所有客户端获得 ≥10% 性能增益，上限达 30%。
- 资源分析（Table 2）：存储 +16%（24.42M vs 21.03M）、计算 +1%（FLOPs 持平）、通信 -2%（仅传全局层参数）。

**消融实验**：
- 不同 backbone：ViT 全面优于 ResNet50（Table 3），证实 ViT 在 PFL 中的优势。
- 不同 FL 设置（Table 4）：在 N=128、r=6.25% 极端设置下仍保持最优。
- Prefix 初始化对比（Table 6）：FedPerfix（48.10）> Prefix-Z（47.37）> Prefix-R（46.98），验证并行适配器必要性。
- 插件类型对比：Adapters（47.99）接近 FedPerfix，Prompts（44.19）效果最差。

## 相关工作脉络
1. **FedRep / FedBABU / FedBN（CNN 部分个性化）**：分别个性化分类头、冻结分类头+微调、本地 BN；本文将这些思想迁移至 ViT 的 Self-attention 层，并引入插件机制。
2. **APFL（全模型个性化）**：通过自适应混合系数融合全局/本地模型；本文发现 FedPerfix 在 Self-attention 层实现了等效的混合机制，但以更低资源开销实现。
3. **Per-FedAVG（元学习个性化）**：在 ViT 上受限明显（CIFAR-100 仅 33.86），因小样本元训练对大模型不友好，凸显部分个性化的优势。
4. **Prefix-Tuning（Li & Liang, 2021）**：NLP 领域的连续 prompt 方法；本文首次将其引入联邦学习的 ViT 个性化，并解决初始化不稳定问题。
5. **Qu et al. [32]（ViT 在 FL 中的应用）**：首个系统研究 ViT 在标准 FL 中表现的论文，发现 ViT 收敛更快；本文将其推进至 PFL 场景。
6. **FedPerfix vs. 插件式 FL（Sun et al. [35]）**：之前工作用插件解决通信约束；本文聚焦"哪些层敏感"+"如何稳定个性化"，方法论更精细化。

## 局限性与未来方向
1. **任务单一**：仅在图像分类上验证，未扩展到目标检测、语义分割等 ViT 优势任务。
2. **超参数依赖**：Prefix 长度、适配器降维比例、混合系数 $s$ 等需手动调优，缺乏自适应性分析。
3. **极端非 IID 鲁棒性待验证**：虽测试了 N=128/r=6.25%，但更大规模（数百客户端）下的表现未评估。
4. **未探索混合架构**：仅用 ViT-Small，未比较 ViT-B/ ViT-L 等更大模型的缩放规律（Table 5 显示大模型收益递减）。

## 研究启发与可借鉴点
1. **层敏感性实证范式**：本文"逐层本地更新 → 衡量敏感度 → 定位个性化目标"的方法论可直接迁移至其他架构（如 Swin、MaE）的个性化研究。
2. **并行适配器设计**：Scale-down → Activation → Scale-up 的稳定初始化模式可复用于其他 PEFT 方法（如 LoRA、Adapter）在联邦学习中的部署。
3. **全局-本地混合的数学统一视角**：公式(6)揭示了 Prefix 本质是动态混合系数，这一视角可指导未来设计更高效的混合策略（如可学习的 $\lambda(x)$）。
4. **资源三维评估框架**：存储/计算/通信的联合分析为方法对比提供了标准化基准，建议在后续工作中沿用。
5. **插件式 PFL 的新可能**：FedPerfix 证明了插件机制在 PFL 中的有效性，可探索 Adapters、LoRA、Visual Prompt 等其他插件在 ViT 联邦学习中的组合策略。

## 关键术语表
**Partial Model Personalization**：仅对模型子集参数进行本地个性化更新，其余参数全局共享的联邦学习策略。
**Prefix-Tuning**：在 Transformer 的 Key/Value 矩阵前拼接可学习前缀向量，以极少量参数实现任务适配。
**Parallel Attention / Adapter**：与原始 Self-attention 并行的小型神经网络，用于生成稳定的 Prefix 初始值。
**Data Heterogeneity（非 IID）**：各客户端数据分布不一致的现象，分为 Label Skew（类别分布差异）和 Concept Skew（特征-标签映射差异）。
**Plugin**：插入预训练模型的轻量级可学习参数模块，训练时冻结原模型权重。
**Mixture Coefficient（混合系数）**：全局模型与个性化模型之间权重分配的比例因子。

## 可复现要素
- **数据集**：CIFAR-100（公开）、OrganAM-NIST（公开，Nature Scientific Data）、Office-Home（公开）
- **代码**：https://github.com/imguangyu/FedPerfix（已开源）
- **模型**：ViT-Small（timm 库），Patch Size=16，Image Size=224
- **优化器**：SGD，learning rate=0.01，batch size=64
- **训练配置**：50 轮通信，每轮 10 个 local epoch，4×Nvidia A5000 GPU
- **联邦设置**：Dirichlet 分布划分数据（α=0.1/0.5/1.0），参与率 12.5%/25%
