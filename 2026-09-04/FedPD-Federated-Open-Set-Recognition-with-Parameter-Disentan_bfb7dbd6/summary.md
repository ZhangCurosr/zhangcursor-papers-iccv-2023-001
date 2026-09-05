---
title: "FedPD-Federated-Open-Set-Recognition-with-Parameter-Disentan"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:56:15"
field: "联邦学习与开放集识别交叉"
keywords: ["federated learning", "open set recognition", "parameter disentanglement", "optimal transport", "federated open-set recognition", "model aggregation"]
innovations: ["首次提出联邦开放集识别（FedOSR）问题，填补 FL 在开放集场景下的空白", "局部参数解耦（LPD）：基于彩票假设用任务重要性将 OSR 模型解耦为闭集/开集子网，阻断组间干扰", "全局分而治之聚合（GDCA）：将子网划分为特有/共享三部分并以最优传输逐层对齐后聚合"]
benchmarks: ["HDR-FL", "CIFAR-10"]
---

# 论文速读：FedPD-Federated-Open-Set-Recognition-with-Parameter-Disentanglement

## 一句话总结
本文首次提出了**联邦开放集识别（FedOSR）**问题，并设计了参数解耦引导的联邦算法 **FedPD**，通过客户端局部参数解耦（LPD）与服务端"分而治之"聚合（GDCA），有效解决了跨客户端闭集/开集知识相互干扰和参数不对齐两大核心挑战。

## 研究问题与动机
- **现有联邦学习均局限于闭集设定**：训练与测试类别完全一致，无法识别未见过的未知类别，难以部署于临床诊断、自动驾驶等高风险场景。
- **直接将现有开放集识别（OSR）方法迁移至联邦环境面临两大挑战**：① 跨客户端的**组间干扰**——某客户端负责闭集分类的参数可能被其他客户端的开集相关参数污染；② 跨客户端的**组内不一致性**——由于神经网络参数的置换不变性与数据异构性，简单聚合会导致模型退化。
- **FedAvg 直接聚合多 OSR 模型会导致模型坍塌**（Fig. 1），已对齐的参数分布差异使无效参数被反复传输，拖慢收敛。

## 核心贡献（创新点）
- **首次形式化联邦开放集识别（FedOSR）问题**，填补了 FL 在开放集场景下的空白，与现有闭集 FL 方法及中心化 OSR 方法形成明确区分。
- **提出局部参数解耦（LPD）策略**：基于彩票假设，利用任务相关重要性（闭集损失梯度×参数值、开集损失梯度×参数值）将客户端 OSR 模型解耦为闭集子网和开集子网，从源头阻断组间干扰——这与 FedMA 等仅解决闭集参数对齐的方法本质不同。
- **提出全局分而治之聚合（GDCA）策略**：将两个子网进一步划分为闭集特有、开集特性和共享三部分非重叠参数，再逐层以最优传输（Optimal Transport）对齐对应部分后聚合，区别于 FedAvg 直接平均和 FedMA 贝叶斯非参数匹配，专门应对 OSR 模型中复杂的参数组成。

## 方法详解
**整体流程**（客户端→服务端循环）：

1. **客户端本地训练**：以生成类方法 **PROSER** 为基线，同时优化闭集交叉熵损失 $\mathcal{L}_{close}$ 和开集损失 $\mathcal{L}_{open}$：
   $$\mathcal{L}_{cls} = \mathcal{L}_{close} + \lambda \cdot \mathcal{L}_{open}$$

2. **局部参数解耦（LPD）**：基于彩票假设，用一阶泰勒展开近似参数重要性：
   $$\mathcal{T}_{close}^k(i) = |\nabla\mathcal{L}_{close}(\omega_i) \times \omega_i|, \quad \mathcal{T}_{open}^k(i) = |\nabla\mathcal{L}_{open}(\omega_i) \times \omega_i|$$
   选取重要性最高的 Top-K（掩码比设为 **0.5**）生成闭集掩码 $\mathcal{M}_{close}^k$ 和开集掩码 $\mathcal{M}_{open}^k$，得到两个子网。

3. **全局分而治之聚合（GDCA）**：
   - **分（Divide）**：将子网划分为三个非重叠部分：
     $$\mathcal{P}_{close} = \mathcal{M}_{close} \odot \overline{\mathcal{M}_{open}}, \quad \mathcal{P}_{open} = \mathcal{M}_{open} \odot \overline{\mathcal{M}_{close}}, \quad \mathcal{P}_{share} = \mathcal{M}_{open} \odot \mathcal{M}_{close}$$
   - **治（Conquer）**：以某一客户端（如第 2 个）为参考，对所有客户端的对应部分逐层计算最优传输映射 $\boldsymbol{T}^{(\ell)}$（channel-wise 分布匹配），并将前一层的传输矩阵进行后乘对齐输入通道顺序，最终对齐后按式 (16) 平均得到全局模型：
     $$G = \frac{1}{K}\left(\sum_k \widetilde{S}_{close}^k + \sum_k \widetilde{S}_{open}^k + \sum_k \widetilde{S}_{share}^k\right)$$

## 实验与结果
- **数据集**：HDR-FL（非 IID，含 MNIST/SVHN/USPS/SynthDigits/MNIST-M 五个手写数字域，已知 6 类/未知 4 类）和 CIFAR-10（IID，分别测试 known=4/6/8 三种设置）。
- **评估指标**：闭集 ACC + 开集 AUROC（AUC）。
- **对比基线**：SoftMax、OpenMax、RPL、PROSER、ARPL、DIAS、SSB（均以 FedAvg 聚合）。
- **HDR-FL 结果**：FedPD 取得 **平均闭集 ACC 90.96%**、**平均开集 AUC 81.36%**，较次优方法 DIAS 分别提升 **+2.76% / +2.72%**；生成类方法 SSB 因参数未对齐出现严重模型坍塌，平均 ACC 仅 82.33%，差距达 8.17%。
- **CIFAR-10 结果**：在 known=4/6/8 三种设置下，FedPD 的 Open-set AUC 分别为 **71.50% / 85.07% / 69.12%**，均为最优，证明对不同开放度具有稳定性。
- **消融**：去掉 LPD（无分割）ACC/AUC 降至 85.61%/76.19%；去掉 GDCA 对齐（w/o Conquer）ACC 降至 87.76%、AUC 降至 77.02%；掩码比 0.5 为最优。

## 相关工作脉络
- **FedAvg [23]**：经典联邦平均，本文基线聚合方式，但未考虑 OSR 场景的参数不对齐问题。
- **FedMA [37]**：通过贝叶斯非参数方法逐层匹配权重，适用于闭集识别，不适用于 OSR 复杂参数构成。
- **FedBN [20]**：仅聚合 BN 层以外参数以保留个性化，不涉及参数对齐或解耦。
- **PROSER [43]**：生成类 OSR 方法的代表，本文本地训练采用其作为基线网络。
- **DIAS [26]**：难度感知模拟器，闭集场景下表现优异，本文 HDR-FL 上的最强基线。
- **Dice [34]**：利用稀疏化进行 OOD 检测，揭示了闭集/开集能力与参数子集的关系，是本文 LPD 设计的理论依据之一。

## 局限性与未来方向
- **拓扑固定**：当前以第 2 个客户端为参考对齐目标，未探索动态选参考或自适应对齐策略。
- **掩码比需手工调参**：实验显示 0.5 最优，但不同数据集/场景可能需要调整，缺乏自适应机制。
- **仅验证了 CNN/WideResNet 两类架构**：更深层或 Transformer 架构下的适用性未验证。
- **通信开销增加**：除模型参数外还需上传掩码 $\mathcal{M}_{close}, \mathcal{M}_{open}$，额外带宽成本未量化分析。
- **未知类数为预设固定值**：实际场景中未知类数量和类别可能变化，泛化能力有待考察。

## 研究启发与可借鉴点
- **参数解耦思路可迁移**：利用任务重要性（梯度×权重）将模型拆解为功能子网的设计，可应用于其他需要多目标优化的联邦场景（如联邦多任务学习、联邦领域自适应）。
- **"分而治之+最优传输对齐"的聚合范式**：将模型参数按功能分区后分别对齐再聚合的策略，可推广至联邦迁移学习、联邦元学习等跨域联邦设定。
- **掩码可视化验证假设**：通过可视化不同客户端子网参数的空间分布来直观验证参数不对齐现象，这种方法论值得借鉴。
- **与生成类 OSR 的松耦合设计**：本地训练任选用何种 OSR 方法（本文用 PROSER），全局聚合方法独立设计，具有良好的模块通用性。

## 关键术语表
- **Federated Open Set Recognition (FedOSR)**：在联邦学习框架下同时实现已知类别的分类与未知类别的检测的新型任务。
- **Local Parameter Disentanglement (LPD)**：基于彩票假设，利用任务损失梯度与参数值的乘积作为重要性指标，将客户端模型解耦为闭集子网和开集子网的策略。
- **Global Divide-and-Conquer Aggregation (GDCA)**：将解耦后的子网进一步划分为闭集特有、开集特性和共享三部分，分别通过最优传输对齐后聚合的全局聚合策略。
- **Lottery Ticket Hypothesis**：神经网络中存在稀疏的可训练子网，仅少量参数对任务泛化起关键作用。
- **Optimal Transport (OT)**：通过寻找最小代价的"运输"方案实现两个分布对齐的数学方法，本文用于逐层对齐客户端间的参数分布。
- **Open Set Recognition (OSR)**：分类模型在测试时不仅能识别已知类别，还能将未知类别样本判为"未知"。
- **Parameter Misalignment**：由于神经网络的置换不变性和数据异构性，不同客户端对应位置参数的语义不一致问题。

## 可复现要素
- **数据集**：HDR-FL（MNIST/SVHN/USPS/SynthDigits/MNIST-M，公开）；CIFAR-10（公开）。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：掩码比 0.5；HDR-FL 使用 SGD（闭集 lr=1e-2，开集 lr=1e-4），batch=32，100 epochs；CIFAR-10 使用 WideResNet + Adam（闭集 lr=1e-1，开集 lr=1e-3），batch=128，E=5 通信一次，总 T=250 epochs；所有实验在 NVIDIA 2080Ti 上完成。
