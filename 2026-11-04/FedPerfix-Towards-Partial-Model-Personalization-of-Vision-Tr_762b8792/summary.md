---
title: "FedPerfix-Towards-Partial-Model-Personalization-of-Vision-Tr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_FedPerfix_Towards_Partial_Model_Personalization_of_Vision_Transformers_in_Federated_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:42:30"
field: "个性化联邦学习"
keywords: ["Personalized Federated Learning", "Vision Transformer", "Partial Model Personalization", "Prefix-tuning", "Parameter-Efficient Fine-tuning", "Federated Learning"]
innovations: ["系统实证ViT各层对数据分布敏感性，定位self-attention与分类头为最佳个性化候选", "提出FedPerfix方法，通过平行注意力Adapter生成Prefix实现稳定的部分模型个性化", "揭示Prefix-tuning与APFL的注意力混合等价性，以更低资源开销实现SOTA性能"]
benchmarks: ["CIFAR-100", "OrganAM-NIST", "Office-Home"]
---

# 论文速读：FedPerfix-Towards-Partial-Model-Personalization-of-Vision-Tr

## 一句话总结
论文提出FedPerfix方法，通过在Vision Transformer的self-attention层引入带平行注意力设计的Prefix插件，实现部分模型个性化，以较低资源开销在联邦学习场景中平衡全局知识与客户端特定知识。

## 研究问题与动机
1. **现有部分模型个性化方法主要针对CNN**：已有多项工作（FedRep、FedBN、FedBABU等）将部分层保留本地更新，但未系统探索ViT架构下的适用性。
2. **ViT在个性化联邦学习中存在研究空白**：ViT在集中式训练下表现优异，但在PFL中的应用鲜有探索，且简单将CNN方法迁移到ViT可能不够有效。
3. **完全本地更新会阻断全局知识传播**：若将所有敏感层（如self-attention）完全保留本地，将无法利用全局聚合带来的通用知识。
4. **ViT各层对数据分布的敏感性未知**：缺乏系统性实证研究来确定ViT中哪些层最值得个性化。

## 核心贡献（创新点）
1. **ViT层敏感性实证研究**：首次系统评估ViT各类型层（Patch Embedding、Position Embedding、LayerNorm、Self-attention、MLP、Classification Head）对数据分布的敏感性，发现self-attention层和分类头是最敏感部分。
2. **FedPerfix方法**：借鉴迁移学习中Plugin思想，在ViT的self-attention层引入Prefix插件，通过平行注意力设计稳定初始化，使Prefix能捕获客户端特定知识同时保留全局特征。
3. **发现与APFL的结构联系**：证明Prefix-tuning可被理解为全局注意力与个性化注意力的混合系数，为部分个性化提供新的理论视角。
4. **资源高效且SOTA性能**：在CIFAR-100、OrganAM-NIST、Office-Home三个数据集上超越现有PFL方法，同时通信参数量从100%降至98%，存储仅增加16%。

## 方法详解
1. **敏感性实证分析**：在CIFAR-100上分别将各类层保留本地更新（不上传服务器聚合），统计Top-1准确率，发现Self-attention stand-alone得分42.95±8.68，Combined with classification head达44.63±8.67，总体最高（43.79）。

2. **Prefix-tuning基础设计**：在每个self-attention层前追加可学习的Prefix $P_k, P_v$，拼接到Key和Value矩阵后：
$$head_i = Attention(Z^{l-1}W_q^{(i)}, Z^{l-1}[P_k, W_k^{(i)}], Z^{l-1}[P_v, W_v^{(i)}])$$
其中$Z^{l-1}$为上一层输出，Prefix仅客户端本地更新，原始self-attention权重正常参与全局聚合。

3. **平行注意力稳定Prefix**：为避免随机初始化的不稳定性，引入平行Adapter生成Prefix：
$$P_k, P_v = \text{Adapter}(Z^{l-1}) = \text{Tanh}(Z^{l-1}W_{down})W_{up}$$
并通过超参数$s$控制Adapter效率，最终attention输出为：
$$head_i = Attention(Z^{l-1}W_q^{(i)}, Z^{l-1}[sP_k, W_k^{(i)}], Z^{l-1}[sP_v, W_v^{(i)}])$$

4. **联邦学习流程**：每轮客户端接收全局参数$u^{(t-1)}$，将本地Prefix $v_i$代入ViT，本地训练后仅上传全局层参数参与FedAvg聚合（$\alpha_i = m_i/M$），评估时本地模型=全局层+本地Prefix+本地分类头。

## 实验与结果
- **数据集**：CIFAR-100（N=64，Dirichlet $\alpha=0.1$）、OrganAM-NIST（N=64，$\alpha=0.5$）、Office-Home（N=16，4域，$\alpha=1.0$）；模型ViT-Small（patch=16，img=224）。
- **主要结果（Top-1准确率均值±标准差）**：
  - CIFAR-100：FedPerfix **48.10±7.76**（最高），FedRep 44.42±7.80，FedBABU 44.42±7.80，APFL 44.88±10.50，FedAVG 23.29±11.31。
  - OrganAM-NIST：FedPerfix **93.17±3.51**（最高），FedBABU 92.63±3.77，Local 78.26±11.07。
  - Office-Home：FedPerfix **24.38±8.47**（最高），FedRep 24.23±7.02，APFL 24.23±7.02。
- **最强结果与提升**：FedPerfix在三个数据集上均获得最优；相比FedRep在CIFAR-100提升约3.68个百分点，相比Local在极端label skew下提升约13%。
- **资源对比**：FedPerfix存储24.42M（116%），计算66.58M FLOPs（101%），通信13.89M（66%），显著优于APFL/Per-FedAVG（通信100%），与FedAVG相当。
- **Client-wise表现**：FedPerfix平均增益超10%，几乎覆盖所有客户端正增益，上限达30%。

## 相关工作脉络
1. **FedRep [5]**：保留最后分类层及若干block本地更新，其余聚合；本文扩展至ViT并进一步个性化self-attention层而非仅分类头。
2. **FedBN [18]**：本地更新BatchNorm层；论文指出其对ViT的LayerNorm替换效果有限，仅为微小改进。
3. **FedBABU [27]**：冻结分类头并仅在评估前fine-tune一步；本文认为仅个性化分类头不足以捕捉ViT的self-attention敏感性。
4. **APFL [6]**：通过自适应混合系数融合全局与本地模型；本文发现FedPerfix的Prefix机制本质上实现了类似的注意力权重混合，但以更低的资源开销实现。
5. **Prefix-Tuning [20]**：NLP领域插件方法；本文首次将其引入ViT的联邦个性化场景，并改进为带平行Adapter的稳定版本。
6. **ViT in FL [32]**：Qu et al.证明ViT在标准FL中收敛更快；本文将其延伸至PFL并设计针对性个性化策略。

## 局限性与未来方向
1. **仅针对图像分类任务**：实验局限于分类 benchmark，未验证于目标检测、分割等任务。
2. **ViT-Small单一规模**：虽验证了Tiny/Base规模趋势，但未深入探讨极大规模ViT的个性化边界。
3. **仅考虑Label Skew与Concept Skew**：未涉及更复杂的特征分布偏移或隐私攻击场景。
4. **Prefix数量固定**：不同层使用相同数量的Prefix，未探索分层动态配置策略。
5. **未来方向**：可扩展至Video ViT、多模态Transformer；探索与其他PEFT方法（LoRA等）的结合；在真实分布式环境中验证通信效率。

## 研究启发与可借鉴点
1. **层敏感性实证方法论**：可复用的"逐层替换为本地更新→比较准确率"范式，适用于评估任何新架构在PFL中的个性化候选层。
2. **Plugin思想连接PFL与Transfer Learning**：将联邦全局模型视为"预训练模型"、客户端数据视为"下游任务"，通过小参数插件实现个性化，为其他架构提供通用设计思路。
3. **平行Attention稳定初始化**：Adapter+Prefix的组合设计可有效缓解随机初始化带来的训练不稳定，可迁移至其他基于Prefix的PEFT场景。
4. **资源-性能联合分析框架**：同时报告存储/计算/通信三项指标，便于实际部署评估，值得纳入后续工作的标准报告范式。
5. **与APFL的结构等价性发现**：从公式层面揭示不同方法间的内在联系，可为统一PFL框架提供理论基础。

## 关键术语表
**Personalized Federated Learning (PFL)**：在联邦学习中为每个客户端训练个性化模型而非单一全局模型的范式。
**Partial Model Personalization**：仅对模型中部分参数进行本地个性化更新，其余参数正常聚合的部分个性化策略。
**Vision Transformer (ViT)**：将Transformer架构应用于图像识别的模型，通过将图像分块并序列化处理。
**Prefix-tuning**：在Transformer的Key/Value矩阵前追加可学习的前缀向量，仅训练前缀而冻结主干参数的参数高效微调方法。
**Plugin**：插入预训练模型中的小型可学习模块，用于捕获特定下游任务知识而无需更新原模型权重。
**Parallel Attention**：在原始自注意力层旁并联Adapter生成辅助输入，以稳定训练并增强个性化能力的设计。
**Label Skew**：各客户端数据标签分布不均（某些类集中在少数客户端）的非IID数据划分方式。
**Concept Skew**：同一标签在不同客户端对应不同视觉概念（如"猫"在A客户端为 domestic cat，在B客户端为 cartoon cat）的异质性。

## 关键术语表
**Personalized Federated Learning (PFL)**：在联邦学习中为每个客户端训练个性化模型而非单一全局模型的范式。
**Partial Model Personalization**：仅对模型中部分参数进行本地个性化更新，其余参数正常聚合的部分个性化策略。
**Vision Transformer (ViT)**：将Transformer架构应用于图像识别的模型，通过将图像分块并序列化处理。
**Prefix-tuning**：在Transformer的Key/Value矩阵前追加可学习的前缀向量，仅训练前缀而冻结主干参数的参数高效微调方法。
**Plugin**：插入预训练模型中的小型可学习模块，用于捕获特定下游任务知识而无需更新原模型权重。
**Parallel Attention**：在原始自注意力层旁并联Adapter生成辅助输入，以稳定训练并增强个性化能力的设计。
**Label Skew**：各客户端数据标签分布不均（某些类集中在少数客户端）的非IID数据划分方式。
**Concept Skew**：同一标签在不同客户端对应不同视觉概念（如"猫"在A客户端为domestic cat，在B客户端为cartoon cat）的异质性。

## 可复现要素
- **数据集**：CIFAR-100、OrganAM-NIST、Office-Home（均为公开数据集）。
- **代码开源**：是，代码已公开于 https://github.com/imguangyu/FedPerfix。
- **模型权重**：论文未提及预训练权重开源。
- **关键超参**：通信轮数T=50，本地epoch=10；CIFAR-100：N=64，Dirichlet α=0.1，参与率r=0.125；OrganAM-NIST：α=0.5；Office-Home：N=16，α=1.0，r=0.25；优化器SGD，学习率0.01，batch size=64；图像 resized 至224×224；Backbone为ViT-Small（TIMM库，patch=16）。
- **实验环境**：4×Nvidia A5000 GPU，PyTorch实现。
