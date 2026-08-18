---
title: "GPFL-Simultaneously-Learning-Global-and-Personalized-Feature"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_GPFL_Simultaneously_Learning_Global_and_Personalized_Feature_Information_for_Personalized_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:08:25"
field: "联邦学习"
keywords: ["个性化联邦学习", "特征提取", "条件计算", "全局类别嵌入", "统计异质性", "对比学习"]
innovations: ["提出GPFL框架同时学习全局和个性化特征，突破现有pFL方法只能关注单一特征的局限", "设计可训练全局类别嵌入GCE从角度和幅度双重引导特征提取，避免FedProto/FedPHP依赖预训练特征提取器的悖论", "创新性地引入条件阀门CoV创建双特征路由，实现端到端的全局-个性化分离式特征学习"]
benchmarks: ["Cifar100", "Tiny-ImageNet", "FMNIST", "AG News", "Amazon Review", "HAR"]
---

# 论文速读：GPFL-Simultaneously-Learning-Global-and-Personalized-Feature

## 一句话总结
论文提出了一种新颖的个性化联邦学习（pFL）方法GPFL，通过在客户端同时学习全局特征和个性化特征来解决现有pFL方法只能关注单一特征目标的不足。该方法引入可训练的全局类别嵌入（GCE）和条件阀门（CoV），在标签偏斜、特征偏移及真实世界设置下相比10个SOTA方法显著提升了有效性、扩展性、公平性、稳定性和隐私保护能力。

## 研究问题与动机
- **核心问题**：现有pFL方法在特征提取视角下存在"顾此失彼"——FedRoD等仅训练特征提取器提取全局特征用于协作目标，但不提取个性化特征；FedPer、FedRep等仅用本地数据训练个性化目标，丢失全局信息，不利于协作学习。
- **FedPHP/FedProto的悖论**：这些方法虽然利用全局特征/原型引导个性化特征提取，但全局特征/原型的质量依赖于特征提取器本身的质量，形成"鸡生蛋蛋生鸡"的循环依赖；当大型骨干网络（如ResNet-18）从头训练时，早期迭代中差的全局特征会误导本地训练。
- **pFL的双重目标困境**：理想pFL需同时实现两个目标——（1）聚合信息用于协作学习；（2）训练合理的个性化模型。但从特征角度，现有方法只能满足其一。
- **实际异构挑战**：真实场景中客户端面临标签偏斜（label skew）、特征偏移（feature shift）及真实世界分布异质性，传统pFL方法难以兼顾泛化能力与个性化性能。

## 核心贡献（创新点）
- **首次同时学习全局与个性化特征**：提出GPFL框架，使每个客户端在同一本地训练中同时学习全局特征信息（用于协作）和个性化特征信息（用于个性化），突破了现有方法只关注单一目标的局限。
- **GCE可训练全局类别嵌入机制**：通过全局类别嵌入层（GCE）引入额外的全局信息，从角度和幅度两个层面引导特征提取，与FedProto等依赖已训练特征提取器生成原型的方法形成本质区别，避免了早期迭代中的误导问题。
- **CoV条件阀门双路由设计**：创新性地将条件计算技术引入联邦学习，通过CoV将原始特征变换为全局特征和个性化特征两条独立路径，每条路径使用不同的仿射变换参数，实现了端到端的分离式特征学习。
- **多维度的系统性验证**：在CV、NLP、IoT三个领域的六个数据集上，涵盖标签偏斜、特征偏移和真实世界三种异构设置，从有效性、扩展性（20-500客户端）、公平性、稳定性和隐私五个维度全面评估，证明了方法的广泛适用性。

## 方法详解
- **模型架构拆分**：将骨干网络拆分为特征提取器ϕ和头ψ（最后的全连接层），与FedRep类似。客户端i共享参数包括特征提取器参数$W^{fe}$、CoV参数$V$、GCE参数$C$，各自保留个性化头参数$W_i^h$。
- **CoV双路由变换**：通过公式$f_i^G = \sigma[(\gamma + \mathbf{1}) \odot f_i + \beta]$和$f_i^P = \sigma[(\gamma_i + \mathbf{1}) \odot f_i + \beta_i]$将原始特征$f_i$变换为全局特征$f_i^G$和个性化特征$f_i^P$，其中$\gamma, \beta$由全局条件输入$\pmb{g}$生成，$\gamma_i, \beta_i$由个性化条件输入$\pmb{p}_i$生成。
- **全局条件输入$\pmb{g}$的构造**：对所有冻结的全局类别嵌入取平均：$\pmb{g} = \frac{\sum_{u \in [U]} \widehat{\text{GCE}}(u; \hat{C})}{U}$，确保所有客户端获得相同的全局引导信号。
- **个性化条件输入$\pmb{p}_i$的构造**：基于客户端i的类别分布统计$\alpha_i^u = \mathbb{E}_{(x_i,y_i)\sim\mathcal{D}_i}\mathbb{I}\{y_i = u\}$，加权计算：$\pmb{p}_i = \frac{\sum_{u \in [U]} \widehat{\text{GCE}}(u; \hat{C}) \cdot \alpha_i^u}{U}$，使CoV能感知本地数据分布。
- **角度层面全局引导损失**：$\mathcal{L}_i^{alg} = -\log\frac{\exp(\sin(f_i^G, \text{GCE}(y_i; C)))}{\sum_{u \in [U]}\exp(\sin(f_i^G, \text{GCE}(u; C)))}$，基于余弦相似度（仅衡量角度），使特征向量靠近对应类别嵌入、远离其他类别嵌入。
- **幅度层面全局引导损失**：$\mathcal{L}_i^{mlg} = ||f_i^G - \widehat{\text{GCE}}(y_i; \hat{C})||_2$，利用冻结的类别嵌入使全局特征保持与全局信息的距离接近，防止个性化训练过程中全局特征偏离过多。
- **个性化任务损失**：$\mathcal{L}_i^P = \ell(\psi(f_i^P; W_i^h), y_i)$，使用交叉熵损失，仅作用于个性化特征路径。
- **总损失函数**：$\mathcal{L}_i = \mathcal{L}_i^P + \mathcal{L}_i^{alg} + \lambda\mathcal{L}_i^{mlg} + \mu||V||_2 + \mu||C||_2$，其中$\lambda$和$\mu$为超参数，后两项为正则化项。
- **隐私保护机制**：客户端不共享$W_i^h$和类别分布系数$\alpha_i^u$，服务器只能通过融合ϕ和CoV的伪特征提取器$\tilde{\phi} = \phi \circ \text{CoV}$获取全局特征，结合GCE构建伪模型，相比FedPer/FedRep仅共享ϕ包含更多全局信息，具有更好的隐私保护能力。

## 实验与结果
- **数据集**：计算机视觉（FMNIST、Cifar100、Tiny-ImageNet）、自然语言处理（AG News、Amazon Review）、物联网（HAR Human Activity Recognition）。
- **基线方法**：FedAvg、FedProx、Per-FedAvg、pFedMe、Ditto、FedPer、FedRep、FedRoD、FedPHP、FedProto共10个SOTA方法。
- **异构设置**：病理型标签偏斜（pathological label skew）、实用型标签偏斜（practical label skew, Dirichlet分布$\beta=0.1/1$）、特征偏移（feature shift）、真实世界（real world）。
- **最强结果**：在Cifar100实用标签偏斜设置下，GPFL达到61.86%准确率，比最优基线Ditto（52.87%）提升**8.99%**；在Tiny-ImageNet实用设置下达到43.70%，比FedProto（26.38%）提升17.32%。
- **NLP任务**：AG News上GPFL达到97.97%，比FedProto（96.34%）提升1.63%，比其他基线提升5.72%。
- **特征偏移设置**：GPFL在Amazon Review上收敛后性能保持稳定，而Ditto、FedRep、FedProto等方法出现明显过拟合下降；FedRoD的BSM损失仅针对标签偏斜设计，在此设置下无优势。
- **扩展性**：在20-500客户端规模下均保持领先；当N=500时多数pFL方法退化为类似FedAvg的性能，而GPFL凭借GCE提供的额外全局信息仍达37.28%，比Ditto（30.24%）高7.04%。
- **公平性**：在真实世界设置下，GPFL的标准差为8.42（系数变异8.98×10⁻²），显著优于FedPer（21.16）、FedRep（21.16）等仅学习单一特征的方法。
- **稳定性**：客户端参与率$\rho \in [0.1, 1]$波动时，GPFL准确率仅从60.98%降至60.04%，而pFedMe从48.36%降至41.71%，FedPHP从50.23%降至44.43%。
- **隐私保护**：DLG攻击下GPFL的PSNR值最小（6.56-6.71 dB），优于所有基线，验证了GCE引入全局信息增强隐私的理论分析。

## 相关工作脉络
- **FedPer/FedRep**：仅共享特征提取器或让每个客户端训练独立特征提取器，只学习个性化特征，丢失协作所需的全局信息；GPFL通过双路径设计同时保留两种特征。
- **FedRoD**：每个客户端拥有独立特征提取器和两个头（全局头+个性化头），但个性化目标的梯度不回传到特征提取器，导致特征提取器仅学习全局特征；GPFL通过CoV实现特征级的双目标训练。
- **FedPHP**：对齐个性化特征提取器和全局特征提取器的输出，但依赖已训练好的全局特征提取器质量；GPFL通过可训练的类别嵌入直接引导特征空间，不依赖预训练特征提取器。
- **FedProto**：将特征向量拉近到对应类别原型，但原型从特征向量中派生，存在循环依赖问题；且仅拉近同一类别、不推开不同类别，导致分类边界相交；GPFL通过对比学习思想（拉近同类、推开异类）解决此问题。
- **Per-FedAvg/pFedMe/Ditto**：基于元学习或正则化的pFL方法，关注模型参数的个性化而非特征层面的分离学习；GPFL从特征表示角度切入，提供正交的设计思路。
- **条件计算技术（SpotTune/BlockDrop/D²NN）**：原本在集中式学习中使用动态路由选择网络层，GPFL首次将其引入联邦学习，通过CoV创建双特征路由，拓展了条件计算的应用场景。

## 局限性与未来方向
- **超参数敏感性**：损失函数中的$\lambda$和$\mu$需要手动调节，论文未系统分析其对不同设置下性能的影响，鲁棒性有待进一步验证。
- **大型骨干网络的训练挑战**：虽然展示了ResNet-18在Tiny-ImageNet上的优势，但当客户端数据量极小（如N=500时每客户端仅几个样本）时，个性化仍然困难，GCE虽能缓解但无法完全解决数据稀缺问题。
- **条件阀门的计算开销**：CoV增加了额外的仿射变换参数和计算步骤，在资源极度受限的边缘设备上可能带来额外的推理延迟。
- **单向全局引导的局限**：当前架构仅从全局到个性化引导，未考虑客户端间个性化特征的正则化或知识蒸馏，可能错过跨客户端个性化知识迁移的机会。
- **固定类别嵌入的假设**：GCE假设所有客户端共享相同的类别空间，对于类别不可见的pFL场景（heterogeneous labels）需要进一步推广。

## 研究启发与可借鉴点
- **双路由特征分离的思想**：CoV设计启发了特征层面的"全局-个性化"分离范式，可迁移到其他多目标学习场景（如多任务学习、域适应），值得探索更复杂的条件网络结构。
- **可训练嵌入引导特征学习**：GCE的可训练类别嵌入机制比FedProto的静态原型更灵活，可在联邦推荐系统、联邦知识图谱等需要语义先验的场景中复用。
- **角度+幅度双重引导的互补性**：消融实验证明两种引导损失的协同效应，这一思路可推广到其他需要同时约束特征方向和大小的对比学习场景。
- **实验设计的全面性**：从有效性、扩展性、公平性、稳定性、隐私五个维度系统评估pFL方法，为团队后续研究的实验设计提供了参考模板。
- **跨领域验证策略**：在同一框架下验证CV、NLP、IoT三个领域，证明方法的一般性，团队在提出新方法时可借鉴此多领域验证策略。

## 关键术语表
**Personalized Federated Learning (pFL)**：个性化联邦学习，在保护数据隐私的同时实现每个客户端的个性化模型训练，兼顾协作学习与个体化目标。
**Conditional Valve (CoV)**：条件阀门模块，基于条件计算技术，通过仿射变换根据条件输入动态生成特征变换参数，创建全局和个性化双特征路由。
**Global Category Embedding (GCE)**：全局类别嵌入层，存储可训练的全局类别嵌入向量，用于从角度和幅度两个层面引导特征提取器学习全局特征。
**Pathological Label Skew**：病理型标签偏斜，极端非独立同分布场景，每个客户端仅拥有少量类别（如2个）且类别互不相交。
**Practical Label Skew**：实用型标签偏斜，通过Dirichlet分布模拟更真实的非独立同分布数据划分，客户端类别重叠但比例不同。
**Feature Shift**：特征偏移设置，不同客户端拥有相同类别但特征分布不同（如不同领域的评论数据），测试方法对特征异质性的适应能力。
**Deep Leakage from Gradients (DLG)**：基于梯度的深度泄漏攻击，通过分析模型更新梯度反推原始训练数据的隐私攻击方法。
**Balanced Softmax (BSM)**：平衡 softmax 损失，FedRoD 使用的针对标签偏斜设计的分类损失，在特征偏移设置下效果不佳。

## 可复现要素
- **数据集**：FMNIST、Cifar100、Tiny-ImageNet、AG News、Amazon Review、HAR 均为公开数据集；论文已提供代码在 supplementary materials。
- **代码开源**：论文声明"we provide the code in the supplementary materials"，代码随论文补充材料提供。
- **关键超参数**：本地学习率$\eta$：4层CNN和3层MLP设为0.005，ResNet-18和fastText设为0.1，HAR-CNN设为0.01；批量大小默认10；本地轮数默认1；总迭代次数2000；客户端加入率$\rho=1$（稳定性实验中变化）；Dirichlet分布参数$\beta=0.1$（CV/NLP默认）或$\beta=1$；损失权重$\lambda$和正则化系数$\mu$论文未明确给出具体数值，需在补充材料中确认。
- **重复试验**：所有方法在每个任务上运行3次，报告均值和标准差。
