---
title: "Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Knowledge_Proxy_Intervention_for_Deconfounded_Video_Question_Answering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:11:15"
field: "视觉-语言因果推理"
keywords: ["VideoQA", "causal inference", "front-door adjustment", "dataset bias", "knowledge proxy", "deconfounding", "multi-modal reasoning"]
innovations: ["提出 KPI 框架，通过知识代理变量 Z 实现前门调整，无需观测混淆变量即可去偏", "设计知识空间构建流水线：从训练数据提取共现概念，用 ConceptNet+Atomic 筛选因果边，BERT 编码为可训练嵌入", "模型无关框架，在三种不同架构基线（记忆/图/层次）和五个 benchmark 上一致提升 1.4%-5.3%"]
benchmarks: ["MSVD-QA", "MSRVTT-QA", "TGIF-QA", "NExT-QA", "Causal-VidQA"]
---

# 论文速读：Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question

## 一句话总结
本文提出 Knowledge Proxy Intervention (KPI)，一个模型无关的因果推断框架，通过引入知识代理变量 Z 实现前门调整（front-door adjustment），切断混淆变量 C 导致的后门路径，从而缓解视频问答（VideoQA）中由数据集偏差引起的虚假相关性。

## 研究问题与动机
- **数据集偏差导致虚假相关**：现有 VideoQA 方法直接建模观测概率 $P(A|V, Q, H)$，训练数据中的共现模式（如"guitar"与"man"的高条件概率）会被模型当成因果信号，推理时忽略视频/问题中的真实视觉线索。
- **混淆变量不可观测且不可枚举**：偏差来源于环境、动作、社会惯例等多种因素的组合（如"run indoors"、"jump outdoors"），无法像以往工作那样分解出可枚举的混淆因子。
- **现有因果 VideoQA 方法局限**：IGV [35] 和 EIGV [34] 针对场景偏差（scene bias），仅处理可观测的帧级偏差，对通用数据集偏差（包括因果场景中仍存在的偏差，如图 7 所示）无效，且依赖可观测混淆变量的假设。
- **前门调整理论上可行但需中间变量**：后门调整要求对混淆变量 C 进行完整分解求和（公式 2），不可行；前门调整通过引入中间变量 Z 分解为两部分（公式 3），无需先验知识，但需设计合适的 Z 实现。

## 核心贡献（创新点）
1. **系统性因果图分析 VideoQA 的数据集偏差**：构建包含 V、Q、H、A、C 的因果图，证明混淆变量 C 和后门路径导致虚假因果，并通过 MSVD-QA 上可视化条件概率分布（图 3）量化偏差程度。
   - 与已有工作区别：不同于 IGV/EIGV 仅关注场景级可观测偏差，本文从通用视角刻画整个 VideoQA 管道的混淆效应。

2. **提出 KPI 框架，利用前门调整实现无先验去偏**：在因果图中插入知识代理变量 Z，将干预分解为 $P(Z|Q,H)$ 和 $P(A|do(Z))$ 两部分，通过 NWGM 近似为期望乘积（公式 7），完全不需要对混淆变量 C 进行观测或枚举。
   - 与已有工作区别：区别于所有基于后门调整的工作，首次将前门调整引入 VideoQA 去偏任务。

3. **设计知识空间 Z 的构建流水线与多注意力 EXP 模块**：从视频/问题中提取概念，用 ConceptNet + Atomic 知识图谱筛选因果边（head-relation-tail），BERT 编码为可训练嵌入；提出 channel attention、product attention、multi-head attention 三种 EXP 模块近似条件期望。
   - 与已有工作区别：知识空间由训练数据+外部知识图谱联合构建，而非人工定义或仅依赖数据驱动。

4. **广泛的实验验证框架的泛化性与模型无关性**：在三个不同类别的基线（CoMem 记忆型、HGA 图型、HQGA 层次型）上，于五个基准数据集均取得显著提升；同时验证了对预训练 VLP 方法（VIOLET、JustAsk、MERLOT）的有效性。
   - 与已有工作区别：现有因果 VideoQA 工作通常只验证单一基线，本文系统验证了多种架构的通用性。

## 方法详解
**整体流程（图 4）**：给定视频 V 和问题 Q，先用基线方法得到视频子嵌入 $\hat{v}_o, \hat{v}_a, \hat{v}_m$、问题嵌入 $\hat{q}$、对齐特征 $\hat{h}$；再将四个特征空间（V、Q、H、Z）分别送入 EXP 模块计算期望，最终通过全连接层 + Softmax 输出答案分布 $P(A|do(V,Q,H))$。

**知识空间 Z 的构建（Section 4.1）**：
1. 用 I3D ResNeXt-101 + Faster R-CNN 提取视频概念 $C_v$（动作、物体），用 NLTK 提取问题概念 $C_q$ 和答案概念 $C_a$。
2. 生成相关概念对（head-tail）$(h, t)$，其中 $h \in C_v \cup C_q$，$t \in C_a$。
3. 将所有相关概念对初始化到知识空间 Z。
4. 若 h 和 t 在现有知识图谱中为相邻节点，则保留并扩展为因果概念 $h\text{-}r\text{-}t$，否则删除。
5. 用预训练 BERT 将因果概念转换为可训练的知识嵌入向量。
6. 知识图谱来源：ConceptNet（物理实体关系）+ Atomic（事件/社会交互关系）。

**特征空间的构建**：
- **视频特征空间 V**：对训练集所有运动/外观/边界框特征分别做 k-means 聚类，压缩至 $k_V$ 个向量，形成三个子空间 ${\bf V}_m, {\bf V}_a, {\bf V}_o$。
- **问题特征空间 Q**：微调 BERT 提取问题序列特征后做 average-pool，再用 k-means 压缩至 $k_Q$。
- **对齐特征空间 H**：用基线模型在观测概率下训练后，对所有视频-问题对推理得到对齐向量，k-means 压缩至 $k_H$。

**期望近似——EXP 模块（Section 4.2）**：
以 $\mathbb{E}_{[z|Q,H]}[z] = \sum_z P(z|Q,H) z$ 为例，用注意力机制近似条件分布 $P(z|Q,H)$：
- **Channel Attention**（公式 8）：$a_i = w^T \tanh(W_1 z_i + W_2 Cat(\hat{q}, \hat{h}))$，加权求和得到 $\bar{z}$。
- **Product Attention**（公式 9）：引入键值间的二阶交互，$\bar{z} = \text{softmax}(\frac{Cat(\hat{q},\hat{h})W_1(ZW_2)^T}{\sqrt{d_z}})ZW_3$。
- **Multi-head Attention**（公式 10）：8 头并行，捕捉多样化注意力模式，拼接后线性投影。

其他三个期望（$\mathbb{E}_v[v]$、$\mathbb{E}_{[q|v]}[q]$、$\mathbb{E}_{[h|q,v]}[h]$）也通过类似 EXP 模块计算，其中视频空间需对每种特征子类型独立做 self-attention + average-pool 得到子嵌入后再处理。

**最终预测（公式 7 近似 + 损失函数）**：
$$P(A|do(V,Q,H)) \approx \text{Softmax}[g(\bar{z}, \bar{v}_o, \bar{v}_a, \bar{v}_m, \bar{q}, \bar{h})]$$
训练时最小化交叉熵损失：$\mathcal{L} = -\log P(A^*|do(V,Q,H))$，$A^*$ 为真实答案。

## 实验与结果
**数据集**：MSVD-QA、MSRVTT-QA（描述性）、TGIF-QA（Action/Transition/FrameQA 三子集）、NExT-QA（时序+证据推理）、Causal-VidQA（常识推理）。

**基线方法**：CoMem（记忆型）、HGA（图型）、HQGA（层次型）+ KPI；对比 IGV、EIGV（因果 VideoQA）；进一步验证 VIOLET、JustAsk、MERLOT（VLP 基线）+ KPI。

**主要结果（表 1）**：
| 数据集 | 最优基线 | HQGA+KPI | 提升幅度 |
|---|---|---|---|
| MSVD-QA | HQGA 41.2 | **43.3** | +2.1 |
| MSRVTT-QA | EIGV 39.3 | **40.0** | +0.7 |
| TGIF-QA Action | HQGA 76.9 | **79.3** | +2.4 |
| TGIF-QA Transition | HQGA 85.6 | **88.3** | **+2.7** |
| TGIF-QA FrameQA | HQGA 61.3 | **63.0** | +1.7 |
| NExT-QA | EIGV 53.7 | **55.0** | +1.3 |
| Causal-VidQA | HQGA 52.9 | **56.7** | **+3.8** |

- KPI 在所有五数据集三基线上一致提升 **+1.4%~+5.3%**，最大提升在 CoMem/MSVD-QA 达 **+5.3%**。
- HQGA+KPI 在 NExT-QA（55.0）和 Causal-VidQA（56.7）超越所有已有方法，分别获得 SOTA。
- VLP 基线实验（表 2）：KPI 同样带来 +0.6%~+1.9% 提升，最强在 NExT-QA（VIOLET 54.6→55.9）。

**消融实验关键发现（表 3-4）**：
- EXP 模块：Multi-head > Product > Channel，二阶交互和多头多样性是关键。
- 知识空间：ConceptNet+Atomic 联合最佳；MSVD-QA 用 ConceptNet 更好，NExT-QA 用 Atomic 更好。
- 变量重要性：Z 对预测贡献最大（含去偏信息），H 其次（融合 V+Q）。
- Z 构建：H 比 Q 更重要（含更多信息）。
- 特征空间大小：准确率随 $k_V/k_Q/k_H$ 从 100 增至 1000 先升后平稳，500 附近已接近饱和。

## 相关工作脉络
1. **IGV [35] / EIGV [34]**：因果 VideoQA 的代表性工作，通过因果帧/补帧消除场景偏差；本文与之定位差异在于——两者假设混淆变量可观测（帧级），仅处理 scene bias 这一种偏差类型；本文处理更广义的数据集偏差，且无需混淆变量先验。
2. **Causal Attention for V-L [70] / Deconfounded Image Captioning [69]**：将前门调整应用于图像描述；本文将其迁移到 VideoQA 的多模态复杂推理场景，并设计了面向视频的知识空间构建方法。
3. **CoMem [12]、HGA [26]、HQGA [62]**：代表性 VideoQA 基线（记忆/图/层次结构）；本文不替代它们，而是作为通用插件框架提升其性能，验证了模型无关性。
4. **Counterfactual VQA [39]**：从语言偏差角度做反事实推理；本文从因果关系建模全局数据集偏差，方法论不同（前门调整 vs. 反事实）。
5. **Bias-aware VideoQA [19]（Women also snowboard）**：在图像 captioning 中发现性别偏差；本文首次在 VideoQA 系统性地用因果图分析并解决该问题。

## 局限性与未来方向
- **知识空间覆盖不全**：现有知识图谱（ConceptNet、Atomic）无法覆盖全部预测所需的因果概念，知识空间大小受限于资源开销。
- **EXP 模块表达能力有限**：当前只能捕获一阶 head-tail 关系，复杂的 head-head-tail 或 head-tail-tail 高阶因果结构无法表达。
- **小数据集上偏差更强但效果不稳定**：MSRVTT-QA 数据量最大，KPI 提升最小（+1.4%~2.5%），说明其在更泛化的场景下增益有限；未来需探索更大规模知识图谱和更强 EXP 模块。
- **共享知识空间的双刃剑效应**：MSVD-QA 共享知识空间会引入噪声，而 NExT-QA 共享则互补，说明跨数据集知识迁移需要更精细的设计。

## 研究启发与可借鉴点
1. **前门调整的通用范式**：对于混淆变量不可观测/不可枚举的视觉-语言任务（如图像描述、视频检索），KPI 的前门调整思路可直接迁移，只需重新设计对应的中间变量 Z。
2. **知识图谱+数据驱动的混合知识空间构建**：用外部 KG 过滤数据中共现概念、保留因果边，再 BERT 编码为可训练嵌入——该流程可在其他多模态任务中复用。
3. **EXP 模块的注意力近似方案**：channel/product/multi-head 三种 Attention 的对比揭示了二阶交互对去偏的重要性，为设计更复杂的因果注意机制提供了方向。
4. **特征空间 k-means 降维 + EXP 模块期望近似**：将连续特征空间离散化为有限字典，再用注意力近似条件期望，是一种可复用的"空间采样+期望估计"范式。
5. **跨数据集知识空间的互补性洞察**：简单场景（MSVD）不需要复杂知识，复杂场景（NExT）可受益于跨域知识；这对多数据集联合训练策略设计有启发。

## 关键术语表
**Knowledge Proxy Intervention (KPI)**：本文提出的模型无关因果干预框架，通过知识代理变量 Z 实现前门调整以去除 VideoQA 中的数据集偏差。

**Front-door Adjustment（前门调整）**：因果推断中的一种干预方法，当混淆变量不可观测时，通过引入中间变量 Z 将因果效应分解为两段路径之和，避免对混淆变量直接建模。

**Confounder（混淆变量）**：同时影响自变量（Q、H）和因变量（A）的隐藏因素，导致观测相关性不等于因果性，在 VideoQA 中体现为数据集共现偏差。

**Backdoor Path（后门路径）**：从自变量经混淆变量到达因变量的非因果路径（如 Q←C→A），是需要被阻断或调整的虚假关联通道。

**Knowledge Space Z（知识空间）**：由训练数据中的因果概念对（head-relation-tail）经 BERT 编码形成的可训练嵌入集合，充当前门调整中的中间变量。

**EXP Module**：基于注意力机制的期望近似模块，用于计算 $\mathbb{E}[z|Q,H]$、$\mathbb{E}[v|V]$ 等条件期望，支持 channel/product/multi-head 三种变体。

**Normalized Weighted Geometric Mean (NWGM)**：将外层期望移至特征级别的数学工具，使前门调整的期望乘积形式可被近似为对均值向量的函数计算。

## 可复现要素
- **数据集**：MSVD-QA、MSRVTT-QA、TGIF-QA、NExT-QA、Causal-VidQA——公开可用。
- **代码/权重**：论文未明确声明开源（ICCV 2023 当时未附代码链接）。
- **关键超参**：特征空间大小 $k_V, k_Q, k_H \in [100, 1000]$（实验取 500）；知识图谱使用 ConceptNet 5.5 + Atomic 2020；BERT 用于问题编码和知识嵌入；Multi-head attention 使用 8 个 head。
- **基线复现**：CoMem、HGA、HQGA、IGV、EIGV 均有公开实现或代码可参考。
