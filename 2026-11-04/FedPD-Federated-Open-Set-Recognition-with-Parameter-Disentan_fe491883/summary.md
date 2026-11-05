---
title: "FedPD-Federated-Open-Set-Recognition-with-Parameter-Disentan"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:42:20"
field: "联邦学习与开放集识别"
keywords: ["Federated Learning", "Open Set Recognition", "Parameter Disentanglement", "Optimal Transport", "Federated Open Set Recognition"]
innovations: ["首次提出联邦开放集识别(FedOSR)问题并系统研究", "基于任务重要性度量的本地参数解耦(LPD)策略", "分治+最优传输对齐的全局聚合(GDCA)机制"]
benchmarks: ["HDR-FL", "CIFAR-10"]
---

# 论文速读：FedPD: Federated Open Set Recognition with Parameter Disentanglement

## 一句话总结
本文首次提出联邦开放集识别（FedOSR）问题，针对联邦场景下已知类别分类与未知类别检测协同学习的两大挑战，设计了参数解耦引导的联邦学习算法FedPD——通过本地参数解耦（LPD）将模型拆分为闭集与开集子网络，再通过全局分治聚合（GDCA）结合最优传输对齐并聚合，有效缓解跨客户端参数错位导致的模型崩溃问题。

## 研究问题与动机
- **联邦学习现有限于闭集假设**：现有FL方法（如FedAvg、FedProx等）均在训练与测试类别完全一致的闭集设定下评估，实际部署（如临床诊断、自动驾驶）中测试时可能出现未知类别，直接应用存在巨大风险。
- **现有开源OSR方法无法直接迁移至联邦场景**：主流OSR方法（OpenMax、ARPL、PROSER等）均为集中式设计，若简单通过FedAvg聚合各客户端OSR模型，会因参数错位（permutation invariance）和数据异质性导致已知类参数被污染、未知类参数被干扰，引发模型崩溃。
- **跨客户端集合间干扰（inter-set interference）**：客户端A的闭集相关参数可能被其他客户端的开集相关参数污染，反之亦然，导致未知样本被错误分类为已知类。
- **跨客户端集合内不一致性（intra-set inconsistency）**：即使聚合的是同类参数，神经网络排列不变性与数据异质性仍使各客户端对应参数分布错位，直接平均造成模型发散与收敛缓慢。

## 核心贡献（创新点）
1. **首次形式化并研究联邦开放集识别（FedOSR）问题**，填补联邦学习与开放集识别交叉领域的空白，区别于以往仅在集中式设定下工作的OSR方法。
2. **提出本地参数解耦（LPD）策略**，基于彩票猜想（Lottery Ticket Hypothesis）利用任务相关性重要性度量将本地OSR模型解耦为闭集子网络与开集子网络，从源头隔离两类知识的参数依赖，与FedMA等闭集参数对齐方法本质不同。
3. **设计全局分治聚合（GDCA）机制**，将两个子网络进一步划分为闭集专属、开集专属和共享三部分组成非重叠部分，通过最优传输（Optimal Transport, OT）逐层对齐后聚合，解决传统FedAvg/FedMA在OSR复杂参数结构下失效的问题。
4. **在HDR-FL（Non-IID）与CIFAR-10（IID）双基准上验证有效性**，在多项指标上超越现有SOTA OSR方法（以FedAvg聚合为基线），并给出参数错位可视化分析揭示问题根源。

## 方法详解
**整体流程**：每个联邦轮次，客户端接收全局模型后使用已知类别数据联合训练闭集损失$\mathcal{L}_{close}$与开集损失$\mathcal{L}_{open}$（总损失$\mathcal{L}_{cls} = \mathcal{L}_{close} + \lambda \cdot \mathcal{L}_{open}$，采用生成式OSR基线PROSER），然后通过LPD解耦出两个子网络掩码上传服务器；服务器将子网络划分为三类参数后以最优传输对齐并加权平均得到新全局模型。

**本地参数解耦（LPD）**：
- 基于彩票猜想，参数$h_i$对某任务的重要性通过损失变化量近似为 $\Omega^t(h_i) = |\nabla \mathcal{L}_t(h_i) \times h_i|$。
- 分别计算闭集重要性$\mathcal{T}_{close}^k(i) = |\nabla \mathcal{L}_{close}(\omega_i) \times \omega_i|$与开集重要性$\mathcal{T}_{open}^k(i) = |\nabla \mathcal{L}_{open}(\omega_i) \times \omega_i|$。
- 按阈值$\delta_{close}/\delta_{open}$（实验中掩码比例设为0.5）生成闭集子网络掩码$\mathcal{M}_{close}^k$与开集子网络掩码$\mathcal{M}_{open}^k$。

**全局分治聚合（GDCA）**：
- **划分**：利用集合运算得到三个非重叠部分
  - 闭集专属：$\mathcal{P}_{close} = \mathcal{M}_{close} \odot \overline{\mathcal{M}_{open}}$
  - 开集专属：$\mathcal{P}_{open} = \mathcal{M}_{open} \odot \overline{\mathcal{M}_{close}}$
  - 共享：$\mathcal{P}_{share} = \mathcal{M}_{open} \odot \mathcal{M}_{close}$
- **对齐**：以某一客户端（如客户端2）为参考，对每类参数通过最优传输逐层计算运输矩阵$T^{(\ell)}$并按$\widetilde{W}_A^{(\ell)} = {T^{(\ell)}}^\top \widehat{W}_A^{(\ell,\ell-1)}$对齐，不改变预测结果仅重排通道顺序。
- **聚合**：$G = \frac{1}{K}\left(\sum_k \widetilde{S}_{close}^k + \sum_k \widetilde{S}_{open}^k + \sum_k \widetilde{S}_{share}^k\right)$

## 实验与结果
**数据集与设置**：
- **HDR-FL**（Non-IID）：MNIST/SVHN/USPS/SynthDigits/MNIST-M五数据集作为5个客户端，10类中取6类已知、4类未知。
- **CIFAR-10**（IID）：10类按4:6/6:4/8:2已知:未知比例划分，5客户端。
- 评估指标：闭集Accuracy（ACC）与开集AUROC（AUC）。
- 本地训练基线为PROSER，聚合对比方法均以FedAvg实现；FedPD本地同样使用PROSER。

**主要结果**：
- **HDR-FL**（Table 1）：FedPD取得平均闭集ACC **90.96%**、平均开集AUC **81.36%**，超越次优DIAS分别+2.76% ACC / +2.72% AUC；对生成式基线SSB实现巨大增益（SSB平均ACC仅82.33%，存在严重模型崩溃）。
- **CIFAR-10**（Table 2）：在已知类比例为4/6/8三种设定下分别取得开集AUC **71.50% / 85.07% / 69.12%**，均显著优于所有对比方法，证明对 openness 变化的鲁棒性。
- **消融**（Table 3）：去掉划分（w/o Divide）与去掉最优传输对齐（w/o Conquer）均造成性能下降；仅用梯度不乘权重的解耦方式（Grad）逊于本文Grad×Weight方案（Table 4）。掩码比例0.5为最优（Fig. 5）。

## 相关工作脉络
- **FedAvg / FedProx / FedBN**：经典联邦聚合与正则化方法，均面向闭集设定，未处理参数错位与OSR复杂目标，本文将其作为FedOSR场景下的朴素基线进行对比。
- **FedMA [37]**：通过贝叶斯方法逐层匹配权重解决参数错位，但在闭集任务上设计，无法直接处理OSR模型中闭集/开集参数混叠的复杂结构，本文指出其在FedOSR上反而劣于FedAvg。
- **OpenMax [2] / RPL [4] / ARPL [3] / PROSER [43] / DIAS [26] / SSB [35]**：集中式OSR SOTA方法，本文将其适配为FedAvg基线（各客户端独立训练OSR模型后平均），以此凸显直接联邦聚合在OSR上的缺陷，并验证FedPD的增益。
- **Lottery Ticket Hypothesis [7, 11, 36]**：本文参数解耦的理论基础，将"子网重要性"思想从单一任务扩展至闭集/开集双任务场景。
- **OPT / 最优传输在模型对齐中的应用**：本文引入OT进行逐层通道级对齐，区别于FedMA的层状匹配，更适配OSR模型中跨客户端分布错位的具体形态。

## 局限性与未来方向
- **掩码比例需手动调参**：LPD中阈值$\delta_{close}/\delta_{open}$（掩码比例0.5）依赖实验观察确定，未给出自动化的自适应选择机制。
- **通信开销增加**：除模型权重外还需上传子网络掩码（尽管为二进制可压缩），GDCA中的逐层最优传输也带来额外服务器端计算成本，论文未详细讨论带宽与延迟影响。
- **仅验证CNN/WideResNet架构**：未在Transformer等新兴 backbone 上验证泛化性。
- **开集损失依赖生成式基线（PROSER）**：若更换判别式或原型式OSR损失，参数解耦与对齐策略是否需要调整尚待研究。
- **未讨论恶意客户端/隐私攻击**：联邦开放集场景下引入未知类检测后可能面临的新型安全威胁未被分析。

## 研究启发与可借鉴点
1. **"任务重要性×参数值"的重要性度量形式**（$|\nabla \mathcal{L} \times \omega|$）兼顾梯度方向与参数幅度，比单纯用梯度或权重大小更稳定，可迁移至其他多任务/多目标联邦场景的参数路由与解耦设计。
2. **先解耦再分治对齐的范式**：将模型按语义分成互斥模块（专属+共享）分别对齐聚合，避免了混合对齐带来的语义冲突，该思路可推广至联邦多任务学习、联邦持续学习等需要区分"通用"与"专用"参数的场景。
3. **以最优传输逐层对齐替代全参数平均**：对OSR这种多目标优化导致参数分布差异更大的任务尤为关键；可在任何存在跨客户端参数错位（permutation mismatch）的联邦变体中引入OT对齐模块。
4. **可视化参数分布揭示问题根源**：Fig. 4与Fig. 6通过掩码分布图直观展示异构数据与梯度发散均会导致参数错位，这一诊断性可视化手段可作为后续工作的标准分析工具。
5. **FedOSR作为一个新问题**：将OSR的"拒识"能力嵌入联邦框架具有明确应用价值（医疗、自动驾驶），本团队可沿此方向探索个性化联邦OSR、动态 openness 设定、或非生成式OSR损失的联邦化。

## 关键术语表
- **Federated Open Set Recognition (FedOSR)**：在联邦学习框架下同时完成已知类别分类与未知类别检测的新问题设定。
- **Local Parameter Disentanglement (LPD)**：基于任务相关重要性度量将本地OSR模型解耦为闭集子网络与开集子网络的客户端侧策略。
- **Global Divide-and-Conquer Aggregation (GDCA)**：服务器端将子网络划分为专属/共享三部分并分别经最优传输对齐后聚合的全局策略。
- **Optimal Transport (OT)**：通过最小化运输代价实现两个参数分布对齐的数学工具，本文用于重排卷积层通道顺序以消除参数错位。
- **Lottery Ticket Hypothesis**：深度网络中存在少量关键参数即足以保持性能的假说，本文据此筛选闭集/开集关键参数构建子网掩码。
- **Open-set AUC (AUROC)**：衡量模型区分已知类与未知类能力的指标，值越接近1表示开集检测性能越好。
- **Closed-set ACC**：仅对已知类别进行分类的准确率，反映闭集分类性能。
- **Parameter Misalignment**：由于神经网络排列不变性与数据异质性导致不同客户端对应参数在空间分布上不对齐的现象。

## 可复现要素
- **数据集**：HDR-FL（MNIST/SVHN/USPS/SynthDigits/MNIST-M公开数据集组合）、CIFAR-10（公开），均可公开获取。
- **代码/权重**：论文未明确提供开源链接（ICCV 2023），正文及附录未给出GitHub地址；建议通过作者邮件或会议补充材料页面查询。
- **关键超参**：掩码比例0.5；HDR-FL用6层CNN，SGD优化器，闭集学习率$10^{-2}$、开集$10^{-4}$，batch=32，epochs=100，每epoch聚合；CIFAR-10用WideResNet，Adam优化器，闭集学习率$10^{-1}$、开集$10^{-3}$，batch=128，每5轮聚合共250轮；λ未在正文明示（论文未提及）。
- **硬件**：NVIDIA 2080Ti，PyTorch实现。
- **基线实现**：所有对比方法均基于PROSER本地训练 + FedAvg聚合，细节见正文Section 5.2。
