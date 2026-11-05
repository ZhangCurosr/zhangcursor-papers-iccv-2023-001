---
title: "Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Knowledge_Proxy_Intervention_for_Deconfounded_Video_Question_Answering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:44:45"
field: "视频问答与多模态推理"
keywords: ["VideoQA", "causal inference", "dataset bias", "front-door adjustment", "knowledge proxy", "deconfounding"]
innovations: ["首次从因果图视角系统分析VideoQA的数据集偏差并提出前门调整解法", "构建知识代理变量Z实现无需预知混杂因素的因果干预", "提出EXP模块用注意力机制近似特征空间条件期望"]
benchmarks: ["MSVD-QA", "MSRVTT-QA", "TGIF-QA", "NExT-QA", "Causal-VidQA"]
---

# 论文速读：Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question

## 一句话总结
本文针对VideoQA任务中的**数据集偏差(dataset bias)**问题，提出**知识代理干预(KPI)框架**，通过构建知识代理变量并利用**前门调整(front-door adjustment)**切断后门路径，在不需预知混杂因素具体内容的情况下消除虚假相关性对答案预测的干扰。

---

## 研究问题与动机
- **数据集偏差导致虚假相关性**：现有VideoQA方法直接基于观测概率 $P(A|V,Q,H)$ 预测答案，容易受到训练数据中共现模式（如"吉他"多与"man"共现）的误导，而非真正理解视频-问题因果。
- **混杂因素不可观测且不可枚举**：混清器C包含复杂的关联概念组合（环境偏差、动作偏差等），传统后门调整要求显式枚举所有混杂因子，在VideoQA中几乎不可行。
- **现有去偏方法存在局限**：已有关联工作（如IGV、EIGV）仅关注场景层面的偏差，且依赖可观测混杂假设，无法解决通用的数据集偏差问题。
- **偏差源于自然规律**：数据集偏差源自Zipf定律和社会惯例（如室内多猫、户外多袋鼠），单纯扩大数据集无法消除。

---

## 核心贡献（创新点）
- **首次从因果图视角系统分析VideoQA的数据集偏差**：证明混杂因子通过后门路径（$\mathcal{Q} \leftarrow \mathcal{C} \rightarrow \mathcal{A}$）引入虚假因果，与仅关注帧级偏差的已有方法形成本质区别。
- **提出模型无关的知识代理干预(KPI)框架**：通过引入中间变量Z实现前门调整，无需预知混杂因素C的具体内容，区别于依赖可观测混杂的后门调整方法。
- **构建可学习的知识空间Z**：利用ConceptNet和Atomic知识图谱过滤共现概念中的因果关系（head-relation-tail结构），将知识图谱转化为可训练嵌入向量，实现因果关系的显式建模。
- **提出期望的近似的EXP模块设计**：通过channel attention、product attention和multi-head attention三种机制近似条件期望，多头注意力效果最优。
- **在5个基准数据集上验证泛化能力**：KPI框架与三种不同类别的基线方法（CoMem、HGA、HQGA）结合后均获得显著提升，证明其模型无关性和广泛适用性。

---

## 方法详解

### 1. 因果图建模
构建VideoQA任务的因果图：$\mathcal{V} \to \mathcal{H} \to \mathcal{A}$，$\mathcal{Q} \to \mathcal{A}$，同时存在后门路径 $\mathcal{Q} \leftarrow \mathcal{C} \rightarrow \mathcal{A}$ 和 $\mathcal{H} \to \mathcal{Q} \leftarrow \mathcal{C} \rightarrow \mathcal{A}$。混杂因子C由视频中同时出现的概念（如"jump;run"）构成。

### 2. 前门调整理论推导
引入知识代理变量Z后，将因果干预分解为：
$$P(\mathcal{A}|do(\mathcal{V},\mathcal{Q},\mathcal{H})) = \sum_{\mathbf{z}} P(\mathbf{z}|\mathcal{Q},\mathcal{H}) \cdot P^*(\mathcal{A})$$
其中 $P^*(\mathcal{A}) = P(\mathcal{A}|\mathbf{v},\mathbf{q},\mathbf{h},\mathbf{z})$。

通过NWGM（Normalized Weighted Geometric Mean）将外层期望移至特征级别：
$$P(\mathcal{A}|do(\mathcal{V},\mathcal{Q},\mathcal{H})) \approx \text{Softmax}[g(\bar{\mathbf{v}}, \bar{\mathbf{q}}, \bar{\mathbf{h}}, \bar{\mathbf{z}})]$$

### 3. 知识空间构建（4步）
1. 提取视频概念（I3D ResNeXt-101 + Faster R-CNN）、问题概念（NLTK关键词提取）、答案概念；
2. 生成共现概念对(head-tail)；
3. 用ConceptNet和Atomic知识图谱过滤：若head和tail在图谱中相邻，则扩展为因果概念(head-relation-tail)，否则移除；
4. 用预训练BERT将因果概念转换为可训练嵌入向量。

### 4. EXP模块（期望近似）
对四个特征空间分别计算期望：
- $\mathbb{E}_{[\mathbf{v}|\mathcal{V}]}[\mathbf{v}]$：视频子嵌入（运动/外观/边界框）经自注意力+平均池化后，在聚类空间中使用EXP模块
- $\mathbb{E}_{[\mathbf{q}|\mathbf{v}]}[\mathbf{q}]$：BERT抽取的问题特征在聚类空间中计算
- $\mathbb{E}_{[\mathbf{h}|\mathbf{q},\mathbf{v}]}[\mathbf{h}]$：基线模型推理得到的对齐特征
- $\mathbb{E}_{[\mathbf{z}|\mathcal{Q},\mathcal{H}]}[\mathbf{z}]$：用Q和H的拼接向量检索知识空间Z中的相关嵌入

注意力机制选项：channel attention → product attention → multi-head attention（最优，8 heads）

### 5. 整体流程与损失
输入视频V和问题Q → 基线方法提取特征 → 四个EXP模块计算期望 → 全连接层+Softmax输出答案分布 → 交叉熵损失训练。

---

## 实验与结果

**数据集**：MSVD-QA、MSRVTT-QA（描述型）、TGIF-QA（Action/Transition/FrameQA三子集）、NExT-QA（时序/证据推理）、Causal-VidQA（常识推理）

**基线方法**：CoMem（记忆）、HGA（图）、HQGA（层次）、IGV/EIGV（因果VideoQA）、VIOLET/JustAsk/MERLOT（视频语言预训练）

**核心结果**（Table 1）：
| 方法 | MSVD-QA | MSRVTT-QA | TGIF-QA(Transition) | NExT-QA | Causal-VidQA |
|------|---------|-----------|---------------------|---------|--------------|
| HQGA | 41.2 | 38.6 | 85.6 | 51.8 | 52.9 |
| HQGA+KPI | **43.3** (+2.1) | **40.0** (+1.4) | **88.3** (+2.7) | **55.0** (+3.2) | **56.7** (+3.8) |
| CoMem+KPI | **40.0** (+5.3) | **37.6** (+2.5) | 80.0 (+3.4) | 52.1 (+3.6) | 51.3 (+3.6) |

**提升幅度**：所有基线在所有数据集上均有显著提升，范围为 **+1.4% ~ +5.3%**；最强提升在MSVD-QA（小数据集，偏差更严重）上达+5.3%。

**消融实验**（Table 3-4）：
- EXP模块：multi-head attention > product attention > channel attention
- 知识空间：ConceptNet+Atomic联合优于单一图谱；MSVD-QA共享知识空间有害，NExT-QA共享有益
- 各变量贡献：Z贡献最大（前门调整核心），H其次（含V和Q信息），Q > V

**预训练模型验证**（Table 2）：KPI同样能提升VIOLET、JustAsk、MERLOT等强基线，增幅在NExT-QA上最大（+1.3%~1.9%）。

---

## 相关工作脉络
- **VideoQA基线方法**：CoMem（记忆网络）、HGA（异构图对齐）、HQGA（条件图层次），本文在此基础上叠加KPI框架而非修改模型结构。
- **因果VideoQA方法**：IGV和EIGV关注场景偏差（causal frame），依赖可观测混杂假设；本文从更普遍的共现偏差出发，无需预知混杂因素。
- **前门调整应用**：Yang et al. [70]将前门调整用于图像描述和Transformer内部去偏；本文首次将其推广至视频问答任务。
- **去混杂视觉任务**：Wang et al. [57]用于无偏视觉识别，Liu et al. [36]用于图像字幕生成；本文首次系统分析VideoQA因果图。
- **知识图谱辅助推理**：ConceptNet（物理实体关系）和Atomic（事件/社交交互关系）结合使用，本文发现两者互补且针对不同数据集有不同效用。

---

## 局限性与未来方向
- **知识空间覆盖不完整**：现有知识图谱无法涵盖答案预测所需的全部因果概念，且知识库过大导致训练/推理资源不可承受。
- **仅捕捉一阶关系**：EXP模块只能捕获head-tail关系，更复杂的head-head-tail或head-tail-tail关系可能同样重要。
- **未来方向**：构建更全面的知识空间、设计更丰富的EXP模块以捕捉高阶关系。

---

## 研究启发与可借鉴点
- **前门调整的因果干预范式**：在混杂因素不可观测时，通过引入中间代理变量实现因果推断的思路可迁移至其他视觉-语言任务（如图像字幕、视觉对话）。
- **知识图谱驱动的代理变量构建**：利用外部知识图谱（ConceptNet/Atomic等）过滤共现并保留因果关系的策略，可作为通用的去偏机制嵌入多种多模态模型。
- **基于聚类词典的特征空间期望近似**：将连续特征空间离散化为k-means聚类中心，再用注意力机制计算条件期望，为期望估计提供了一种实用的工程近似方案。
- **消融设计中的变量贡献分离**：分别验证Z的构造（Z w/o Q/H）和使用（w/o Z）两个维度，为因果模块的可解释性分析提供了清晰范式。

---

## 关键术语表
**Causal Intervention（因果干预）**：通过do算子主动赋值变量以切断特定因果关系，消除混杂因子的影响。
**Front-door Adjustment（前门调整）**：当混杂因子不可观测时，通过引入中介变量Z将因果效应分解为两条可直接估计的路径。
**Knowledge Proxy（知识代理）**：变量Z，作为问题和对齐特征的代理，用于近似阻断后门路径。
**Knowledge Space（知识空间）**：由知识图谱中因果概念构成的可训练嵌入集合，作为前门调整中的中间变量。
**Dataset Bias（数据集偏差）**：训练数据中因共现模式导致的虚假相关性，使模型偏离真正的因果推理。
**Confounder（混杂因子）**：同时影响问题和答案的隐变量，导致后门路径产生虚假关联。
**EXP Module（期望近似模块）**：使用注意力机制（channel/product/multi-head）近似计算特征空间条件期望的网络模块。
**NWGM（Normalized Weighted Geometric Mean）**：将外层期望移至特征级别的技术，允许在嵌入空间直接计算期望。

---

## 可复现要素
- **数据集**：MSVD-QA、MSRVTT-QA、TGIF-QA、NExT-QA、Causal-VidQA（均为公开数据集）
- **代码/权重**：论文未提供开源链接，但基线方法（CoMem、HGA、HQGA）均有官方代码
- **关键超参**：知识空间大小由知识图谱决定；特征空间大小$k_V, k_Q, k_H$在[100, 1000]范围内可调；EXP模块中multi-head attention使用8 heads

---
