---
title: "Towards-Attack-tolerant-Federated-Learning-via-Critical-Para"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:34:38"
field: "联邦学习安全与鲁棒性"
keywords: ["Federated Learning", "Poisoning Attack Defense", "Critical Parameter Analysis", "Non-IID", "Byzantine Robustness", "Model Aggregation"]
innovations: ["提出基于关键参数重要性(top/bottom-k)的跨客户端相似度度量，突破传统欧氏距离在非IID场景的局限", "设计结合全局模型锚点的正常度打分机制以抵抗Sybil攻击", "通过逆Sigmoid加权聚合实现攻击容忍的柔性防御聚合策略"]
benchmarks: ["CIFAR-10", "SVHN", "TinyImageNet"]
---

# 论文速读：Towards-Attack-tolerant-Federated-Learning-via-Critical-Para

## 一句话总结
本文提出了 FedCPA（Federated Learning with Critical Parameter Analysis），一种面向非 IID 数据分布下对抗投毒攻击的联邦学习防御聚合方法，通过关键参数重要性分析衡量本地模型相似度，实现对恶意客户端的过滤与降权。

## 研究问题与动机
1. **联邦学习易受投毒攻击**：恶意客户端可发送虚假更新进行无差别攻击（降低整体性能）或有目标攻击（注入后门），而中心化聚合的简单平均操作缺乏鲁棒性。
2. **现有防御方法在非 IID 场景下失效**：Krum、FoolsGold、ResidualBase 等方法基于欧氏空间距离判断异常，但在数据分布高度异构时，良性更新本身差异较大，导致恶意与良性更新难以区分。
3. **缺乏基于参数重要性的跨客户端相似度度量**：已有工作忽略了不同参数对优化贡献的不均衡性，以及良性客户端间存在的关键参数一致性模式。
4. **Sybil 攻击下的防御脆弱性**：仅依赖客户端间相似度可能因多数客户端恰好为恶意而高估异常模型的正常性，需引入上一轮全局模型作为参考锚点。

## 核心贡献（创新点）
1. **实证发现良性本地模型的关键参数具有稳定性**：良性客户端经一轮训练后，top-k 和 bottom-k 重要性参数的排名变化显著小于中等重要性参数；而投毒攻击会扰动关键参数的排名顺序。与已有工作仅关注参数值或范数不同，本文从参数重要性视角建立了新的异常检测依据。
2. **提出基于关键参数集的相似性度量指标**：结合 Jaccard 相似度和 Spearman 秩相关系数，同时考察 top-k 和 bottom-k 参数集，比传统欧氏距离度量更适配非 IID 场景。区别于 FoolsGold 仅依赖 Euclidean 空间的协同行为检测，该方法捕捉的是参数重要性的结构性一致。
3. **设计了 attack-tolerant 加权聚合机制**：将正常度分数通过缩放+逆 Sigmoid 映射为聚合权重，既抑制恶意更新影响又避免对低相似度良性客户端的过度惩罚；相比 Median/Trimmed Mean 等粗粒度聚合，该权重分配更为精细。
4. **系统性验证了 FedCPA 在多数据集、多攻击类型、多非 IID 程度下的优越性**：在 CIFAR-10 和 TinyImageNet 上将目标攻击成功率分别降低约 3 倍和 2 倍，且在各类实验中稳定优于8个基线方法。

## 方法详解
FedCPA 包含三个核心步骤：

**Step 1 — 参数重要性计算**：对每个客户端 $i$ 的本地更新 $\Delta_i^t = \theta_i^t - \phi^t$，按公式 $p_i[n] = |\Delta_i[n] \cdot \theta_i^t[n]|$ 计算每个参数的"重要性"，综合了更新幅度和权重贡献两个信号。随后对重要性排序，提取 top-k 和 bottom-k 参数索引集 $\Theta_i^{\text{top}}$ 和 $\Theta_i^{\text{bottom}}$。

**Step 2 — 模型相似性与正常度打分**：两模型间相似性定义为：
$$\mathrm{sim}(\theta_i, \theta_j) = J(\Theta_i^{\text{top}}, \Theta_j^{\text{top}}) + J(\Theta_i^{\text{bottom}}, \Theta_j^{\text{bottom}}) + \rho(r_i(\Theta_{i\cap j}^{\text{top}}), r_j(\Theta_{i\cap j}^{\text{top}})) + \rho(r_i(\Theta_{i\cap j}^{\text{bottom}}), r_j(\Theta_{i\cap j}^{\text{bottom}}))$$
其中 $J$ 为 Jaccard 相似度，$\rho$ 为归一化 Spearman 秩相关。客户端 $i$ 的正常度为：
$$\mathcal{N}(\theta_i^t) = \mathrm{sim}(\theta_i^t, \phi^{t-1}) + \frac{1}{|\mathcal{C}^t|}\sum_{j \in \mathcal{C}^t} \mathrm{sim}(\theta_i^t, \theta_j^t)$$
引入上一轮全局模型 $\phi^{t-1}$ 以抵御 Sybil 攻击。

**Step 3 — 攻击容忍聚合**：将 $\mathcal{N}$ Min-Max 缩放至 $[0,1]$ 得 $\tilde{\mathcal{N}}$，再通过逆 Sigmoid 转换为权重：
$$\lambda_i^t = \mathrm{Clip}_{0\sim1}\left(\ln\frac{\tilde{\mathcal{N}}(\theta_i^t)}{1-\tilde{\mathcal{N}}(\theta_i^t)} + 0.5\right)$$
最终聚合：$\phi^{t+1} \leftarrow \phi^t + \frac{1}{\sum_i \mathbf{1}(\lambda_i^t>0)} \sum_i \lambda_i^t \cdot \Delta_i^t$，权重为0的更新被完全过滤。

## 实验与结果
**数据集**：CIFAR-10（10类）、SVHN（数字识别）、TinyImageNet（200类），均以 Dirichlet$(N, \beta=0.5)$ 模拟非 IID，$N=20$ 客户端。

**基线**：No Defense、Median、Trimmed Mean、Multi Krum、FoolsGold、Norm Bound、RFA、ResidualBase（共8个）。

**攻击类型**：目标攻击（backdoor注入）、标签翻转（label flipping）、高斯噪声（Gaussian noise）。

**关键结果**：
- **目标攻击（$\gamma_p=0.5$）**：FedCPA 在 CIFAR-10 上达到 ACC=77.8%、ASR=21.9%（对比无防御 ASR=51.4%），TinyImageNet 上 ACC=97.2%、ASR=4.8%（对比无防御 ASR=74.6%），**ASR 降低约3倍和2倍**。
- **目标攻击（$\gamma_p=0.8$）**：FedCPA 在 CIFAR-10 上 ACC=72.3%、ASR=12.5%，TinyImageNet 上 ACC=97.1%、ASR=4.8%，均最优。
- **标签翻转攻击（$\gamma_p=0.8$）**：FedCPA 在三个数据集上 ACC 分别为 74.9%、93.2%、36.8%，均优于所有基线。
- **高斯噪声攻击（$\gamma_p=0.8$）**：FedCPA 在 CIFAR-10（ACC=74.8%）、SVHN（ACC=93.6%）、TinyImageNet（ACC=36.1%）均表现最佳或接近最佳。
- **综合排名（Table 4）**：FedCPA 平均排名 1.8，显著优于 ResidualBase（3.4）和 RFA（3.3）。
- **消融实验**：去除 bottom-k 组件导致性能下降最大（目标攻击 ASR 从 12.5% 升至 36.8%），验证了关键假设——投毒攻击通过"唤醒"低重要性参数造成过拟合。

## 相关工作脉络
1. **Krum / Multi-Krum**（Blanchard et al., NeurIPS 2017）：基于欧氏距离的 Byzantine 鲁棒聚合，FedCPA 指出其在非 IID 下失效，改用参数重要性结构替代欧氏距离。
2. **FoolsGold**（Fung et al., RAID 2020）：通过检测客户端间更新相似性来识别协同目标攻击，FedCPA 认为其 Euclidean 相似度度量不够细粒度，且对 Sybil 攻击敏感，因此引入关键参数集相似度并结合全局模型锚点。
3. **ResidualBase**（Fu et al., 2019）：基于残差的参数置信度聚合，在非 IID 下仍有局限，FedCPA 强调其仅考虑参数值分布而忽略参数重要性模式的差异。
4. **RFA（Robust Federated Aggregation）**（Pillutla et al., 2022）：几何中值聚合方法，FedCPA 在保持高准确率的同时能更有效地降低目标攻击成功率。
5. **Median / Trimmed Mean**（Xie et al., 2018; Yin et al., 2018）：维度-wise 离群值鲁棒统计，FedCPA 指出这些方法忽略参数间的结构化关联，在异构数据下效果有限。
6. **Norm Bound**（Sun et al., 2019）：基于更新范数截断的防御，FedCPA 的实验显示其在目标攻击下性能明显弱于自身方法。

## 局限性与未来方向
1. **超参数 k 的敏感性**：k 过小导致证据不足无法区分恶意更新，过大则易受数据异构性干扰（表6显示 k∈[1%, 2%] 为较优区间，但需人工调参）。
2. **未讨论隐私保护**：FedCPA 依赖服务端对本地模型进行相似度分析，但未结合差分隐私或安全聚合等隐私增强技术。
3. **非 IID 程度的极端场景未充分验证**：实验仅在 β=0.5 下为主，对极低 β（极端异构）下的表现未深入分析。
4. **攻击者视角的局限性**：假设攻击者无全局视图，但实际中高级攻击者可能获取一定统计信息进行针对性攻击。
5. **仅针对中心化联邦设置**：未探索去中心化/对等网络场景下的适用性。

## 研究启发与可借鉴点
1. **"参数重要性×更新幅度"复合度量**（Eq.4）可作为通用的参数敏感性评估工具，迁移至模型剪枝、可解释性分析、早停策略等领域。
2. **top/bottom-k 双重关键参数集思路**：良性模型在"最重要"和"最不重要"两端参数上趋于一致，这一模式可启发其他分布式学习中的一致性检测设计。
3. **实验设计借鉴**：跨多数据集（CIFAR-10/SVHN/TinyImageNet）、多攻击类型（目标/标签翻转/高斯噪声）、多污染率（0.5/0.8/1.0）、多非 IID 程度（β 变量）的系统性评测框架值得参考。
4. **结合全局模型锚点的防御设计**：将上一轮全局模型纳入正常度计算以抵抗 Sybil 攻击的思路，可推广至其他需要抗共谋的分布式协议。
5. **与团队方向结合机会**：若团队从事联邦学习安全/隐私方向，可将 FedCPA 的关键参数分析与差分隐私结合，或在异质推荐（FedAttack 相关）场景中验证该方法的有效性。

## 关键术语表
**Federated Learning (FL)**：联邦学习，允许多个参与方在不共享本地数据的前提下协同训练共享模型的分布式学习框架。
**Model Poisoning Attack**：模型投毒攻击，恶意客户端通过发送虚假更新破坏全局模型训练的过程，分为无目标攻击和有目标攻击（后门注入）。
**Critical Parameter**：关键参数，通过 $p_i[n] = |\Delta_i[n] \cdot \theta_i^t[n]|$ 衡量的对模型优化和预测贡献最大的参数。
**Normality Score**：正常度分数，衡量某客户端本地模型与其他模型（含全局模型）的相似程度，分数越低越可能为恶意。
**Attack-Tolerant Aggregation**：攻击容忍聚合，在聚合过程中通过权重调整抑制恶意更新影响而非直接丢弃的策略。
**Non-IID**：非独立同分布，指各客户端本地数据来自不同分布，是联邦学习的典型挑战场景。
**Jaccard Similarity**：Jaccard 相似度，两个集合交集大小与并集大小的比值，用于衡量 top/bottom-k 参数集的重叠程度。
**Spearman Correlation**：Spearman 秩相关系数，衡量两个参数重要性排序的一致性，用于评估参数重要性模式的相似性。

## 可复现要素
- **数据集**：CIFAR-10（公开）、SVHN（公开）、TinyImageNet（公开），均通过 Dirichlet 分布模拟非 IID。
- **代码**：已开源，地址 https://github.com/Sungwon-Han/FEDCPA。
- **模型架构**：ResNet18。
- **关键超参**：通信轮数 100、本地 epoch 数 1、学习率 0.01、动量 0.9、weight decay 1e-5、batch size 64、top/bottom-k 比例 $k=0.01$（1%）、默认客户端数 $N=20$、每轮参与客户端比例 1/2、Dirichlet $\beta=0.5$。
- **攻击设置**：目标攻击 backdoor 尺寸 5×5 位于右下角；高斯噪声标准差 0.05；恶意客户端占比默认 20%。
