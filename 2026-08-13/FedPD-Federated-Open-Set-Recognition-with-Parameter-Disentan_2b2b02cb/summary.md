---
title: "FedPD-Federated-Open-Set-Recognition-with-Parameter-Disentan"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:06:25"
field: "联邦学习与开放集识别"
keywords: ["Federated Learning", "Open Set Recognition", "Parameter Disentanglement", "Optimal Transport", "FedOSR", "Model Aggregation"]
innovations: ["首次提出联邦开放集识别(FedOSR)任务", "基于任务重要性梯度的本地参数解耦(LPD)", "基于最优传输的分治对齐全局聚合(GDCA)"]
benchmarks: ["HDR-FL", "CIFAR-10"]
---

# 论文速读：FedPD-Federated-Open-Set-Recognition-with-Parameter-Disentan

## 一句话总结
本文首次提出联邦开放集识别（FedOSR）任务，并设计参数解耦指导的联邦算法FedPD，通过本地参数解耦（LPD）阻断跨客户端闭集/开集知识干扰，再经全局分治最优传输对齐（GDCA）消除参数错位，显著提升了联邦模型在已知类分类与未知类检测上的联合性能。

## 研究问题与动机
- **闭集假设脱离实际**：现有联邦学习（FL）方法均假设备测集与训练集类别完全一致（closed-set），但临床诊断、自动驾驶等真实场景中测试阶段必然出现未知类别，传统FL模型会强行将其归类为已知类，存在重大安全风险。
- **跨集合知识干扰**：OSR模型的不同参数分别负责闭集分类与开集检测，直接套用FedAvg聚合会导致客户端A的闭集参数被客户端B的开集参数污染，造成已知类准确率与未知类拒绝率双降。
- **集合内参数错位**：即便只聚合同类参数，受神经网络置换不变性与数据异构性影响，不同客户端的同功能参数在通道/神经元维度上分布不一致，简单平均会引发模型崩溃与收敛震荡。
- **现有对齐方法不适用**：FedMA等侧重闭集权重的匹配方法无法处理OSR多目标优化（闭集损失+开集损失）带来的复杂参数构成，缺乏针对联邦OSR的专用聚合机制。

## 核心贡献（创新点）
- **任务定义创新**：首次形式化联邦开放集识别（FedOSR）问题，填补FL在未知类检测场景的研究空白。
- **LPD模块**：基于彩票假设与Taylor展开，提出本地参数解耦策略，用梯度×参数值的乘积评估任务重要性，将本地OSR模型解耦为闭集/开集两个子网络，从源头切断跨集合干扰。
- **GDCA模块**：设计全局分而治之聚合框架，将两子网络进一步划分为闭集专属、开集专属与共享三部分，利用最优传输（OT）逐层对齐对应参数后再平均，彻底解决异构场景下的参数错位问题。
- **实验验证**：在非IID基准HDR-FL与IID基准CIFAR-10上全面优于OpenMax、PROSER、DIAS、SSB等SOTA基线，消融与可视化分析完整支撑了方法有效性。

## 方法详解
- **整体流程**：遵循标准FedAvg循环，每轮服务器下发全局模型 $G_{t-1}$，各客户端 $S^k$ 本地用 $\mathcal{L}_{local}=\mathcal{L}_{close}+\lambda\mathcal{L}_{open}$ 训练E轮，上传本地权重及闭集/开集掩码 $\mathcal{M}_{close}^k, \mathcal{M}_{open}^k$，服务器执行GDCA生成 $G_t$。
- **LPD（本地参数解耦）**：
  - 参数重要性评分：$\mathcal{T}_{close}^k(i)=|\nabla\mathcal{L}_{close}(\omega_i)\times\omega_i|$，$\mathcal{T}_{open}^k(i)=|\nabla\mathcal{L}_{open}(\omega_i)\times\omega_i|$。
  - 掩码生成：对评分排序取Top-K，阈值设为保留比例0.5，生成二值掩码 $\mathcal{M}_{close}^k$ 与 $\mathcal{M}_{open}^k$。
- **GDCA（全局分治对齐聚合）**：
  - **Learn to divide**：划分三组非重叠参数 $\mathcal{P}_{close}=\mathcal{M}_{close}\odot\overline{\mathcal{M}_{open}}$，$\mathcal{P}_{open}=\mathcal{M}_{open}\odot\overline{\mathcal{M}_{close}}$，$\mathcal{P}_{share}=\mathcal{M}_{open}\odot\mathcal{M}_{close}$。
  - **Learn to conquer**：选定某一客户端（如client 2）为对齐目标，对其他客户端的各部分执行通道级最优传输：$\tilde{S}_{part}^k=OT(S^k\odot\mathcal{P}_{part}^k,\ S^2\odot\mathcal{P}_{part}^2)$。OT逐层计算时，通过 post-multiply 传递前一层 transport map 以保持输入通道顺序一致。
  - **聚合**：$G_t=\frac{1}{K}\sum_{k=1}^K(\tilde{S}_{close}^k+\tilde{S}_{open}^k+\tilde{S}_{share}^k)$。

## 实验与结果
- **数据集**：HDR-FL（Non-IID，5个手写数字域，6类已知/4类未知）；CIFAR-10（IID，分别设置已知:未知=4:6、6:4、8:2）。
- **评估指标**：闭集准确率（ACC）与开集AUROC（AUC）。
- **HDR-FL结果**：FedPD平均闭集ACC **90.96%**，平均开集AUC **81.36%**，超越最强基线DIAS（ACC +2.76%，AUC +2.72%）；生成类方法SSB因参数错位严重遭遇模型崩溃，性能差距达8.17%。
- **CIFAR-10结果**：在三种开放度设置下均取得最优开集AUC（**71.50% / 85.07% / 69.12%**），证明方法对不同已知未知比例的稳定性。
- **消融结论**：Grad×Weight比仅用Grad解耦更优；三分治（Divide）比对两子网络直接对齐提升ACC 0.46%、AUC 0.58%；OT对齐（Conquer）较无对齐平均策略提升ACC 3.2%、AUC 4.34%；掩码比例0.5为闭集/开集双指标最优折中点。

## 相关工作脉络
- **FL聚合对齐**：FedAvg/FedProx/FedBN侧重优化与个性化，FedMA引入贝叶斯匹配但仅针对闭集单层权重，无法处理OSR多目标导致的参数耦合；本文GDCA将OT引入联邦参数对齐，专门应对开集模型的复杂结构。
- **集中式OSR**：OpenMax、RPL、PROSER、ARPL、DIAS、SSB均在集中式设定下工作，未考虑数据隐私约束与跨设备参数异构；本文首次将OSR部署于联邦架构并解决由此衍生的干扰与错位问题。
- **彩票假设与子网络提取**：Frankle等提出稀疏子网络可维持性能；本文将其延伸至“按任务重要性划分参数”，用于解耦闭集与开集功能，拓展了彩票假设在联邦多目标场景的应用。
- **最优传输对齐**：OT广泛用于领域自适应与GAN；本文将其改造为层间串联的通道级匹配算子，用于联邦环境下异构客户端的同功能参数对齐，提供了可复用的分布对齐范式。
- **定位差异**：现有工作要么只做闭集联邦分类，要么只做集中式开集检测；本文 bridging 两者，提出FedOSR新问题与新解法，强调“解耦+分治对齐”而非简单损失叠加或FedAvg直传。

## 局限性与未来方向
- **计算与通信开销**：每轮需计算梯度-参数乘积、生成掩码并执行逐层OT对齐，K较大或模型较深时服务器端计算压力显著，论文未讨论轻量化近似或异步策略。
- **已知类集合一致性假设**：当前设定所有客户端共享相同已知类C，未探索客户端已知类异质（heterogeneous known classes）场景下的知识融合。
- **依赖生成式OSR基线**：本地训练采用PROSER等生成式方法，需额外训练生成器/伪未知样本，对资源受限客户端不够友好。
- **未来方向**：可拓展至个性化联邦OSR（pFedOSR）、支持异质已知类的联邦开放集学习、结合模型压缩的轻量OT对齐，以及向医疗影像、自动驾驶等高敏感实时场景落地。

## 研究启发与可借鉴点
- **任务重要性解耦思路可迁移**：梯度×权重乘积评估参数贡献的方法不局限于OSR，可复用于联邦多任务学习、联邦持续学习或灾难性遗忘缓解，实现不同任务/增量知识的参数隔离。
- **分治+OT对齐的通用聚合框架**：将模型参数按功能划分为专属/共享部分再分别对齐，这一设计思想可推广至联邦多模态学习、联邦图神经网络等参数结构复杂的场景。
- **消融设计的严谨性**：w/o Divide、w/o Conquer、Masking Ratio曲线、参数分布可视化等多维验证手段层次分明，实验可复现性强，值得在后续联邦OSR/OOD工作中沿用。
- **与团队方向结合机会**：若团队关注联邦异常检测、联邦OOD检测或跨机构医疗影像协作，LPD的参数隔离机制可直接嫁接；GDCA的OT对齐模块可与FedBN/FedProx等成熟基线组合，探索隐私保护下的开放集联邦学习新范式。

## 关键术语表
- **FedOSR (Federated Open-Set Recognition)**：联邦开放集识别，指在多方协同训练且数据不出本地的前提下，使全局模型既能准确分类已知类，又能可靠拒绝测试阶段出现的未知类。
- **Parameter Disentanglement (参数解耦)**：依据功能或任务相关性将模型权重分离为独立子结构的技术，本文特指将闭集分类参数与开集检测参数分开管理。
- **LPD (Local Parameter Disentanglement)**：本地参数解耦模块，基于闭集/开集损失的梯度与参数值乘积计算重要性得分，生成二值掩码以提取对应子网络。
- **GDCA (Global Divide-and-Conquer Aggregation)**：全局分治聚合模块，将解耦后的参数进一步划分为闭集专属、开集专属与共享三部分，经OT对齐后分别平均融合。
- **Optimal Transport (OT, 最优传输)**：寻找两组分布间最小代价映射的数学工具，本文用于层间对齐不同客户端的同功能参数通道顺序，避免置换错位。
- **Lottery Ticket Hypothesis (彩票假设)**：神经网络中存在一小部分关键参数对任务泛化起决定性作用，本文据此以阈值筛选高重要性参数构建功能子网络。
- **Inter-set Interference (跨集合干扰)**：联邦聚合时不同客户端的闭集参数与开集参数相互混合污染，导致已知类识别与未知类检测性能同时劣化。
- **Intra-set Misalignment (集合内参数错位)**：即使只聚合同类参数，因数据异构与网络置换不变性，各客户端参数分布仍不一致，直接平均会破坏特征表达。

## 可复现要素
- **数据集**：HDR-FL（由MNIST/SVHN/USPS/SynthDigits/MNIST-M拼接的非IID基准）与CIFAR-10，均为公开数据集。
- **代码/权重**：论文未声明开源代码或预训练权重。
- **关键超参**：掩码保留比例0.5；手写数字场景闭集学习率1e-2、开集1e-4、batch=32、local epoch=100；CIFAR-10场景闭集学习率1e-1、开集1e-3、batch=128、local epoch=5、总轮次250；优化器SGD/CIFAR-10用Adam；服务器每轮执行一次FedAvg-style聚合。
