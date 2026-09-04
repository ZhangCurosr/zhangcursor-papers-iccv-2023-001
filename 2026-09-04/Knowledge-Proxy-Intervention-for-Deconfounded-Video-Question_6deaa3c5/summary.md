---
title: "Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Knowledge_Proxy_Intervention_for_Deconfounded_Video_Question_Answering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:38:48"
field: "视频问答中的因果推理"
keywords: ["VideoQA", "因果推断", "数据集偏差", "门控调整", "知识代理", "去混淆"]
innovations: ["提出KPI框架，通过知识代理变量Z实现门控调整消除数据偏差", "无需混淆变量先验即可在VideoQA中实施因果干预"]
benchmarks: ["MSVD-QA", "MSRVTT-QA", "TGIF-QA", "NExT-QA", "Causal-VidQA"]
---

# 论文速读：Knowledge-Proxy-Intervention-for-Deconfounded-Video-Question

## 一句话总结
本文针对 VideoQA 中的数据集偏差问题，提出模型无关的知识代理干预（KPI）框架，通过引入知识代理变量 Z 实现门控调整（front-door adjustment），在无需预知混淆变量的前提下切断后门路径，从而消除虚假相关并提升模型因果推理能力。

## 研究问题与动机
- **数据集偏差误导模型**：现有 VideoQA 方法直接从观测概率 P(A|V, Q, H) 预测答案，容易依赖训练数据中的共现统计（如"吉他"常与"man"共现而非"woman"），导致答案预测被虚假相关误导。
- **混淆变量不可观测且不可枚举**：VideoQA 中的混淆变量 C 包含大量并发概念（如环境偏差、动作偏差及其组合），无法像已有工作那样枚举或观测所有混淆因素。
- **后门调整不适用于 VideoQA**：后门调整要求对混淆变量 C 进行全部分割与枚举，在 VideoQA 的复杂数据偏差下几乎不可行。
- **已有因果方法存在局限**：现有因果 VideoQA 工作（如 IGV、EIGV）仅关注帧级偏差，且需满足可观测混淆假设，无法处理更一般的数据集偏差。

## 核心贡献（创新点）
- **首次从一般视角分析 VideoQA 数据偏差**：构建 VideoQA 因果图，证明混淆变量与后门路径导致虚假因果关系，区别于仅关注帧级偏差的已有工作。
- **提出模型无关的 KPI 框架**：引入知识代理变量 Z 实现门控调整，无需任何关于混淆变量的先验知识，区别于现有需要可观测混淆假设的因果方法。
- **设计可训练的知识空间构建方法**：利用 ConceptNet 和 Atomic 知识图谱从视频-问题-答案概念中筛选因果概念并生成知识嵌入向量，为门控调整提供可学习的代理变量。
- **验证 KPI 框架的通用性与有效性**：在 MSVD-QA、MSRVTT-QA、TGIF-QA、NExT-QA、Causal-VidQA 五个基准上，结合三种不同类别的基线方法（CoMem、HGA、HQGA）均取得显著提升。
- **理论推导与近似计算相结合**：将门控调整公式通过 NWGM 转化为期望形式，并通过注意力机制高效近似各特征空间的期望值。

## 方法详解
**知识空间 Z 的构建**：
1. 从视频提取动作与对象概念（I3D ResNeXt-101 + Faster R-CNN），从问题提取关键词/短语（NLTK），从答案提取关键词。
2. 生成头尾相关概念对（head-tail），利用 ConceptNet 和 Atomic 知识图谱筛选相邻节点关系，扩展为因果概念（head-relation-tail）。
3. 使用预训练 BERT 将因果概念转换为可训练的知识嵌入向量。

**四个特征空间的构建**：
- **视频特征空间 V**：包含运动特征（V_m）、外观特征（V_a）、边界框特征（V_o）三个子空间，分别通过 k-means 聚类至 k_V 个向量。
- **问题特征空间 Q**：使用微调 BERT 提取问题特征后平均池化，再通过 k-means 聚类至 k_Q 个向量。
- **对齐特征空间 H**：先用基线模型训练观测概率下的对齐向量，再通过 k-means 聚类至 k_H 个向量。

**EXP 模块与期望近似**：
通过注意力机制近似条件期望 E[z|Q,H][z]，支持三种注意力实现：
- **通道注意力（Channel Attention）**：一阶交互，公式为 α_i = softmax(w^T tanh(W_1 z_i + W_2 Cat(q̂, ĥ)))。
- **乘积注意力（Product Attention）**：二阶交互，引入 key-query 的二次项。
- **多头注意力（Multi-head Attention）**：在最外层引入多头机制，捕获更多样化的注意力模式。

**最终预测公式**：
通过门控调整将因果干预分解为两步：P(A|do(V,Q,H)) = Σ_z P(z|Q,H) · P*(A)，其中 P*(A) = P(A|v,q,h,z)。使用 NWGM 将外层期望移至特征级，最终通过全连接层 g(·) 与 Softmax 得到答案分布。

**损失函数**：
L = -log P(A*|do(V,Q,H))，使用交叉熵损失，以真实答案 A* 为目标。

## 实验与结果
- **数据集**：MSVD-QA、MSRVTT-QA、TGIF-QA（Action/Transition/FrameQA 三子集）、NExT-QA、Causal-VidQA，共五个基准。
- **基线方法**：Memory-based（CoMem、HME）、Graph-based（L-GCN、HGA、B2A）、Hierarchy-based（HCRN、MSPAN、HOSTR、HQGA）、Causal VideoQA（IGV、EIGV），以及视频-语言预训练方法（VIOLET、JustAsk、MERLOT）。
- **主要结果**：
  - **最强提升**：CoMem + KPI 在 MSVD-QA 上从 34.7% 提升至 40.0%（+5.3%）。
  - **HQGA + KPI 在 TGIF-QA Transition 上达到 88.3%**（+2.7%），为单数据集最高分。
  - HQGA + KPI 在 Causal-VidQA 上达到 56.7%（+3.8%）。
  - KPI 框架对更强基线（VIOLET、JustAsk、MERLOT）同样有效，NExT-QA 上提升最大（约 1.3%-1.9%）。
- **消融实验关键发现**：
  - Z 对所有变量贡献最大，移除 Z 后性能大幅下降。
  - 多头注意力 > 乘积注意力 > 通道注意力。
  - ConceptNet + Atomic 联合使用优于单独使用任一知识图谱。
  - 特征空间大小在 500 附近时性能趋于稳定。

## 相关工作脉络
- **IGV [35] / EIGV [34]**：因果 VideoQA 工作，通过因果场景与互补帧来缓解偏差，但仅关注帧级偏差且需满足可观测混淆假设；本文从更一般的因果图出发，无需混淆变量先验。
- **Causal Attention for VLN [70] / Deconfounded Image Captioning [69]**：将门控调整应用于图像描述，但本文首次将其引入视频问答任务，面向描述、证据推理与常识推理三类问题。
- **ConceptNet [52] / Atomic [23]**：知识图谱资源，本文首次系统地将它们用于构建 VideoQA 知识代理空间。
- **CoMem [12] / HGA [26] / HQGA [62]**：代表性 VideoQA 基线方法，本文证明 KPI 框架可无缝集成于不同类别架构。
- **Counterfactual VQA [39]**：基于反事实推理处理语言偏差，侧重语言偏差；本文关注更广泛的数据集偏差，通过门控调整而非反事实干预解决。

## 局限性与未来方向
- **知识空间覆盖不全**：现有知识图谱无法包含答案预测所需的全部因果概念，知识空间规模过大又会带来资源开销。
- **EXP 模块表达能力有限**：当前 EXP 模块仅能捕获一阶 head-tail 关系，无法表达更复杂的 head-head-tail 或 head-tail-tail 关系。
- **未来方向**：探索更合适的知识空间构建方式，设计更具表达力的 EXP 模块，以提升 VideoQA 任务上门控调整的效果。

## 研究启发与可借鉴点
- **门控调整用于无混淆先验场景**：当混淆变量不可观测且不可枚举时，通过引入中间代理变量实现因果干预的思路，可迁移至其他视觉-语言任务（如图像描述、视频检索）的偏差缓解。
- **知识图谱辅助特征空间构建**：利用 ConceptNet/Atomic 等通用知识图谱构建可训练的知识嵌入空间，是一种低成本引入外部先验知识的有效方式。
- **注意力机制近似期望的计算范式**：通过不同复杂度（一阶/二阶/多头）的注意力机制近似条件期望，为因果推断与表示学习的结合提供了可复用的模块设计。
- **跨数据集知识空间共享策略**：消融实验揭示简单任务（MSVD-QA）共享知识空间可能引入噪声，而复杂任务（NExT-QA）可从共享中受益，这一发现对多任务知识迁移有参考意义。

## 关键术语表
- **Confounder（混淆变量）**：同时影响输入与输出的隐藏因素，通过后门路径引入虚假相关。
- **Backdoor Path（后门路径）**：从输入变量经混淆变量到达输出的非因果路径，会导致观测概率估计有偏。
- **Front-door Adjustment（门控调整）**：通过引入中间变量 Z，将因果效应分解为两段可识别路径，无需观测混淆变量。
- **Knowledge Proxy（知识代理）**：中间变量 Z，汇总问题与对齐特征并覆盖答案预测所需知识。
- **do-calculus**：因果推断的形式化语言，用于描述对变量的主动干预操作。
- **NWGM（Normalized Weighted Geometric Mean）**：归一化加权几何平均，用于将外层期望移至特征级别进行近似。
- **EXP Module**：通过注意力机制近似特征空间期望值的模块。
- **Spurious Correlation（虚假相关）**：由数据集偏差导致的统计相关关系，非因果驱动。

## 可复现要素
- **数据集**：MSVD-QA、MSRVTT-QA、TGIF-QA、NExT-QA、Causal-VidQA，均为公开数据集。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：k_V、k_Q、k_H（特征空间聚类大小）在 [100, 1000] 范围测试，500 附近性能稳定；多头注意力头数为 8。
