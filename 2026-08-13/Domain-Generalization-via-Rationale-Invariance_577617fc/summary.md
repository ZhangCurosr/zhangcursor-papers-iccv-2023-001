---
title: "Domain-Generalization-via-Rationale-Invariance"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Domain_Generalization_via_Rationale_Invariance_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:04:04"
field: "域泛化与分布外泛化"
keywords: ["域泛化", "理由不变性", "决策归因", "正则化", "分布外泛化"]
innovations: ["提出理由矩阵表示决策过程并施加不变性约束", "动量更新机制实现高效的类别理由均值建模", "简单即插即用正则化在多个基准上稳定超越ERM"]
benchmarks: ["DomainBed", "Wilds"]
---

# 论文速读：Domain-Generalization-via-Rationale-Invariance

## 一句话总结
本文提出一种基于"理由不变性"（Rationale Invariance）的域泛化方法，通过将分类器层的元素级贡献建模为理由矩阵，并要求同类别样本的理由矩阵保持一致，从而提升模型在未知域上的泛化能力。该方法仅需在标准 ERM 训练流程上增加数行代码，即在 DomainBed 和 Wilds 基准上实现稳定提升。

## 研究问题与动机
- **核心问题**：深度神经网络在训练域与测试域存在分布偏移时，泛化性能显著下降；现有域泛化方法效果不稳定，部分甚至不如简单 ERM 基线。
- **现有方法局限**：特征不变性正则化（如 MMD、MIRO）忽略分类器权重的影响，可能放大无关特征；logit 不变性正则化（如 SD）仅关注粗粒度输出值，缺乏对决策过程的细粒度刻画。
- **关键洞察**：若模型决策依赖于跨域不变的线索，则同一类别样本的"决策依据"（理由）应趋于一致。
- **设计目标**：从分类器层决策过程出发，提出一种简单有效的理由不变性正则化方案。

## 核心贡献（创新点）
1. **提出"理由"概念用于域泛化**：将每类样本的最终 logit 拆解为特征与分类器权重的元素级乘积之和，并将所有类别的贡献组合为理由矩阵 $\mathbf{R} \in \mathbb{R}^{D \times K}$，实现对决策过程的细粒度表征。
2. **设计理由不变性正则化**：通过确保同类别样本的理由矩阵与其动量更新后的类别均值相近，约束模型依赖跨域不变的决策依据，该正则项仅需数行代码即可实现。
3. **系统性实验验证与对比分析**：在 DomainBed（5 个数据集）和 Wilds（4 个数据集）基准上进行严格评估，证明所提方法在所有数据集上稳定优于 ERM 基线，且在 PACS、TerraInc、Camelyon17 等数据集上达到 SOTA 或接近 SOTA 水平。

## 方法详解
- **理由矩阵构造**：给定特征提取器 $f$ 得到特征 $\mathbf{z} \in \mathbb{R}^D$，分类器权重 $\mathbf{W} \in \mathbb{R}^{D \times K}$，logit $\mathbf{o} = \mathbf{W}^\top \mathbf{z} \in \mathbb{R}^K$。理由矩阵定义为：
  $$\mathbf{R}_{\{j,k\}} = W_{\{j,k\}} \cdot z_j$$
  即第 $j$ 个特征维度对第 $k$ 类 logit 的贡献，收集所有 $D \times K$ 个元素组成 $\mathbf{R}$。
- **理由不变性损失**：
  $$\mathcal{L}_{inv} = \frac{1}{N_b} \sum_k \sum_{\{n|y_n=k\}} \|\mathbf{R}_n - \overline{\mathbf{R}}_k\|^2$$
  其中 $\overline{\mathbf{R}}_k$ 为第 $k$ 类样本的理由矩阵均值。
- **动量更新机制**：为避免全量样本平均的计算开销，采用指数移动平均在线更新类别均值：
  $$\overline{\mathbf{R}}_k^t = (1-m) \cdot \overline{\mathbf{R}}_k^{t-1} + m \cdot \mathrm{mean}(\{\mathbf{R}_n | y_n = k\})$$
  动量超参数 $m \in [0.0001, 0.1]$，初始化时 $\overline{\mathbf{R}}_k$ 取首个 batch 计算的均值。
- **总体损失**：
  $$\mathcal{L}_{all} = \mathcal{L}_{cla} + \alpha \cdot \mathcal{L}_{inv}$$
  其中 $\mathcal{L}_{cla}$ 为标准交叉熵分类损失，$\alpha \in [0.001, 0.1]$ 为正则化权重。训练流程与标准 ERM 完全兼容，仅需额外维护 $\overline{\mathbf{R}}_k$ 并计算 $\mathcal{L}_{inv}$。

## 实验与结果
- **DomainBed 基准**（ResNet18 骨干，60 次试验均值）：
  - PACS：82.8%（最优），较 ERM（79.8%）提升 3.0%，较 SD（81.9%）提升 0.9%
  - VLCS：75.9%（第二，SD 为 75.5%，两者持平）
  - OfficeHome：63.3%（最优），较 ERM（60.6%）提升 2.7%
  - TerraInc：43.7%（最优），较 ERM（38.8%）提升 4.9%
  - DomainNet：36.0%（最优），较 ERM（35.3%）提升 0.7%
  - 平均准确率：60.3%（最优），较第二（SD 59.7%）提升 0.6%，较 ERM 提升约 2%
  - Top5 领先数：4 个数据集进入前五，评分（Score）4（并列最优）
- **Wilds 基准**（不同骨干网络，多次试验）：
  - iWildCam：70.9% 平均准确率，Macro F1 30.7%
  - Camelyon17：90.6% 平均准确率，较 ERM（80.1%）提升超 10%
  - RxRx1：30.0% 平均准确率
  - FMoW：36.1% 最坏域准确率（最优），平均准确率 55.9%（最优）
  - 在全部 4 个 Wilds 数据集上均优于 ERM，2 个数据集取得最优
- **消融实验**：
  - 理由不变性（R）> logit 不变性（O）> 特征不变性（Z）
  - 动量更新（Mt.）优于固定值（m=0）、当前 batch 均值（m=1）和零矩阵（R=0）

## 相关工作脉络
- **特征不变性方法（如 MIRO、MMD、Fishr）**：直接约束中间特征表示的跨域一致性，但忽略分类器权重的缩放效应，可能导致无关特征被放大；本文理由矩阵同时考虑特征与权重，提供更准确的决策归因。
- **Logit 不变性方法（如 SD）**：仅约束最终 logit 输出的一致性，缺乏对内部贡献元素的细粒度控制，无法避免"小贡献被放大以匹配均值"的问题；本文理由矩阵可精确捕捉每个元素级贡献。
- **对抗训练与域混淆方法（如 DANN、CDANN）**：通过对抗手段消除域判别信息，属于隐式不变性学习；本文方法显式约束决策理由的一致性，机制更直接且实现更简单。
- **梯度正则化方法（如 Fish、Gradient Matching）**：通过匹配不同域的梯度方向实现不变性；本文从预测层面出发，无需额外梯度操作，计算开销更低。
- **元学习与数据增强方法（如 MLDG、MixStyle、Mixup）**：依赖元优化或数据混合策略学习可迁移特征；本文作为即插即用的正则化模块，可与多数现有方法结合使用。

## 局限性与未来方向
- 理由矩阵构造依赖最后全连接层输出 logits，无法直接推广至回归任务（如最后一层为卷积操作）或连续标签场景（如 Poverty map 估计）。
- 类别均值更新策略假设类别数量有限且固定，不适用于开放集或无限类别场景。
- 论文未探索将理由不变性与其他 DG 方法（如对抗训练、元学习）结合的潜力。
- 未来方向包括：推广至回归/连续标签任务、结合其他正则化策略、探索预训练理由矩阵的初始化方式。

## 研究启发与可借鉴点
- **细粒度决策归因思路**：将 logit 拆解为元素级贡献再施加正则，这一"自底向上"的视角可迁移至可解释性分析或其他需细粒度控制的场景（如安全关键决策）。
- **动量更新的均值建模**：采用指数移动平均维护类别统计量而非全量平均，计算高效且适合在线训练，这一技巧可应用于其他需要动态统计量的正则化方法。
- **简单有效的正则化设计**：仅增加一个损失项和少量代码即可获得稳定提升，证明在 DG 领域"简单可靠"的方法仍有很大价值，值得在更多基准上验证。
- **与预训练模型结合**：消融实验暗示从预训练模型初始化理由矩阵可能有益，为后续研究提供了可探索的方向。

## 关键术语表
- **域泛化（Domain Generalization, DG）**：利用多个源域数据训练模型，使其在未见目标域上保持良好泛化性能的任务设定。
- **理由矩阵（Rationale Matrix）**：由特征向量与分类器权重的元素级乘积组成的矩阵，表征每个特征维度对各类别决策的贡献。
- **理由不变性（Rationale Invariance）**：要求同类别样本的理由矩阵与其类别均值相近的约束，确保决策依据跨域一致。
- **动量更新（Momentum Update）**：采用指数移动平均在线维护类别统计量（如均值）的机制，避免全量数据重新计算。
- **Empirical Risk Minimization（ERM）**：经验风险最小化，即标准深度学习训练目标（最小化训练集上的损失）。
- **DomainBed**：用于域泛化方法系统评估的标准化基准平台，包含多个具有严格控制协议的 benchmark。
- **Wilds**：真实世界分布偏移基准平台，涵盖遥感、医学影像、细胞成像等多模态数据。

## 可复现要素
- **数据集**：DomainBed（PACS、VLCS、OfficeHome、TerraInc、DomainNet）和 Wilds（iWildCam、Camelyon17、RxRx1、FMoW），均为公开数据集。
- **代码**：已开源，GitHub 地址为 https://github.com/liangchen527/RIDG。
- **骨干网络**：DomainBed 使用 ImageNet 预训练 ResNet18，Wilds 使用 ResNet50（iWildCam、RxRx1）和 DenseNet121（Camelyon17、FMoW）。
- **关键超参**：动量 $m \in [0.0001, 0.1]$，正则权重 $\alpha \in [0.001, 0.1]$，其他设置遵循 DomainBed/Wilds 官方默认配置。
- **评估协议**：DomainBed 采用 leave-one-out 策略，每个目标域 60 次随机种子试验；Wilds 使用官方 leaderboard 设定。
