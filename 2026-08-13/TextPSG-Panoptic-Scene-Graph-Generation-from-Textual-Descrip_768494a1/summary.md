---
title: "TextPSG-Panoptic-Scene-Graph-Generation-from-Textual-Descrip"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_TextPSG_Panoptic_Scene_Graph_Generation_from_Textual_Descriptions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:33:04"
field: "弱监督场景图生成"
keywords: ["Panoptic Scene Graph", "弱监督场景理解", "视觉-语言接地", "开放词汇", "自回归生成", "图像 caption"]
innovations: ["首次提出无位置先验/无显式链接/无预定义概念集的 Caption-to-PSG 任务", "利用实体接地伪标签显式学习段落相似度矩阵以提升定位能力", "将标签预测重构为自回归生成问题并设计 PET 激活预训练 VLM 常识"]
benchmarks: ["Panoptic Scene Graph Dataset", "COCO Caption", "COCO Semantic Segmentation"]
---

# 论文速读：TextPSG: Panoptic Scene Graph Generation from Textual Descriptions

## 一句话总结
本文首次提出从纯文本描述生成全景场景图（Caption-to-PSG）的新问题，在无任何位置先验、显式区域-实体链接和预定义概念集的三重约束下，通过 TextPSG 框架利用网络海量免费图像-caption 数据学习全景结构化场景理解，显著优于基线方法并展现出强 OOD 泛化鲁棒性。

## 研究问题与动机
1. **全监督 PSG 生成依赖密集标注**：现有 PSG 方法（如 Yang et al. [49]）需像素级分割和关系标注，标注成本极高，且当前 PSG 数据集仅覆盖 133 个对象语义和 56 个关系谓词，难以扩展。
2. **弱监督 SGG 存在强先验依赖**：现有弱监督方法（如 SGGNLS [58]）仍依赖预训练的 Region Proposal Network 和预定义对象/关系词表，限制了未见物体的定位能力和概念泛化性。
3. **bbox 形式不够精细**：bbox 包含大量噪声像素（如图 1 中约一半的"girl" bbox 像素属于 wall），且重叠 bbox 难以无歧义地覆盖整个场景；mask-based PSG 可实现更精细的全景理解。
4. **三个约束下的核心挑战**：如何在无位置先验、无显式区域-实体链接、无预定义概念集的条件下，仅从文本描述学习划分（partitioning）、接地（grounding）、对象语义和关系谓词。

## 核心贡献（创新点）
1. **首次提出 Caption-to-PSG 问题**：设定三重约束（无位置先验、无显式链接、无预定义概念集），探索完全从免费图像-caption 数据学习全景结构化场景理解，区别于之前依赖 region proposal 和固定词表的弱监督方法。
2. **提出模块化框架 TextPSG**：包含区域分组器（region grouper）、实体接地器（entity grounder）、段落合并器（segment merger）和标签生成器（label generator），四个模块协同工作完成从文本到 PSG 的端到端生成。
3. **显式学习的段落合并策略**：与 GroupViT [48] 完全隐式学习合并不同，利用实体接地结果作为伪标签，显式学习段落间的相似度矩阵，显著提升定位能力。
4. **自回归生成式标签预测 + PET 技术**：将标签预测从传统分类问题重构为自回归生成问题，利用预训练 VLM（BLIP）的常识知识，并设计 prompt-embedding-based technique（PET）有效引导生成。
5. **开源贡献与副产品价值**：代码、数据和结果已公开；证明 entity grounder 和 segment merger 可增强 text-supervised semantic segmentation（TSSS），在 COCO 上 mIoU 提升 2.15%。

## 方法详解
**整体流程**：训练时输入批次的图像-caption 对，图像经 region grouper 划分为多个图像段落，caption 经 NLP 解析构建 text graph，再由 entity grounder 将语言实体接地到图像段落；接地结果作为伪标签，指导 segment merger 学习相似度矩阵（用于推理时合并）和 label generator 学习对象/关系预测。推理时，region grouper 输出候选段落，经 segment merger 的图割聚类合并后，由 label generator 生成语义标签和关系谓词。

**1) Text Graph 预处理**：使用基于 Stanford CoreNLP 的规则语言解析器和 OpenIE 系统提取 caption 的语言结构，构建 text graph（节点为实体，有向边为实体对间的关系）。

**2) Region Grouper（区域分组器）**：继承 GroupViT 的分层设计。输入图像首先被切分为 N 个不重叠的 patch 作为初始段落 $\{s_i^0\}$，经过 K 个分组层逐层合并：每层有 $H_k$ 个可学习的分组中心 $\{c_i^k\}$，通过 attention 机制将 $H_{k-1}$ 个输入段落合并为 $H_k$ 个更大的段落，即 $\{s_i^k\} = Grp_k(\{c_i^k\}, \{s_i^{k-1}\})$。

**3) Entity Grounder（实体接地器）**：借鉴 FILIP [51] 的细粒度对比学习。对每个分组阶段 k，图像侧通过 MLP $Proj^I$ 将段落投影到特征空间 $\mathcal{F}$ 得到段落嵌入 $\{x_i^k\}$；文本侧将 caption tokenize 后经 Transformer 传播信息，再用 RNN 合并同一实体的 token 得到实体嵌入 $\{y_j\}$。计算段落-实体间 token-wise cosine 相似度，并设置阈值 $\theta$ 过滤低相似度对，得到图像→文本的细粒度相似度 $p^k$ 和文本→图像的 $q^k$。采用双向 fine-grained contrastive loss：
$$\mathcal{L}_{fine}^k = \frac{1}{2}(\mathcal{L}_{fine}^{k, I \to T} + \mathcal{L}_{fine}^{k, T \to I})$$
其中 $\mathcal{L}_{fine}^{k, I \to T} = -\frac{1}{B}\sum_i \frac{\exp(p^{k,i \to i}/\tau)}{\sum_j \exp(p^{k,i \to j}/\tau)}$，$\tau$ 为可学习温度参数。最小化该 loss 后，段落 $s_i^k$ 对应的接地实体 $l_i^k = \arg\max_j \cos[x_i^k, y_j]$ 即作为伪标签。

**4) Segment Merger（段落合并器）**：与 GroupViT 完全隐式合并不同，本文利用接地伪标签显式学习段落相似度矩阵。计算段落对的余弦相似度并线性缩放到 $[0,1]$：$\text{Sim}_k[i,j] = \frac{1}{2}(\cos[x_i^k, x_j^k] + 1)$。用伪标签构建目标矩阵：$\text{Sim}_k^{target}[i,j]=1$ 当且仅当 $l_i^k = l_j^k$ 且两者与对应实体相似度均 $>\theta$，否则为 0。损失函数为：$\mathcal{L}_{sim}^k = \frac{1}{H_k^2}\|\text{Sim}_k - \text{Sim}_k^{target}\|_F^2$。

**5) Label Generator（标签生成器）**：将标签预测重构为自回归生成问题。使用预训练 BLIP [21] 的 decoder。设计 Prompt-embedding-based Technique（PET）：对象预测时用 prompt "a photo of [ENT]"，[ENT] 期望输出伪标签 $b_i^k$；关系预测时用 prompt "a photo of [SUB] and [OBJ], what is their relation [REL]"，[SUB] 和 [OBJ] 由伪标签嵌入，[REL] 期望输出关系谓词。额外设计三个可学习位置嵌入 $\mathbf{f}_{sub}, \mathbf{f}_{obj}, \mathbf{f}_{region}$ 以区分不同区域。使用两个交叉熵损失 $\mathcal{L}_{ent}^k, \mathcal{L}_{rel}^k$ 分别监督 [ENT] 和 [REL] token 的生成。

**6) 推理**：输入图像经 region grouper 得到候选段落后，通过 segment merger 的图割（spectral clustering + graph cut）进行合并；对每个聚类及其配对，label generator 用 PET 生成对象语义和关系谓词。对概念集 $\mathcal{C}_o, \mathcal{C}_r$ 中每个标签计算生成概率，选取概率最高的作为最终预测。语义分割转实例分割通过识别连通分量实现。

## 实验与结果
**数据集**：训练使用 COCO Caption（123,287 张图像，每张 5 个 caption），按 2017 split 使用 118,287 张训练；评估使用 Panoptic Scene Graph 数据集 [49]，合并歧义对象语义后得到 127 个对象语义和 56 个关系谓词。

**评估指标**：Visual Phrase Detection (PhrDet) 和 Scene Graph Detection (SGDet)，使用 No-Graph-Constraint-X Recall@K（NXR@K, %）衡量。

**主要结果（Table 1）**：在严格遵循 Caption-to-PSG 三重约束的条件下，本文方法显著优于所有基线：
- PhrDet N5R100：Ours (mask) = 10.51，Ours (bbox) = 14.37，最优 bbox 基线 SGCLIP = 3.71
- SGDet N5R100：Ours (mask) = 4.18，Ours (bbox) = 5.48，最优 bbox 基线 SGCLIP = 2.70
- 即使与使用预训练检测器的 SGGNLS-o 对比，本文方法依然更优（PhrDet N5R100: 14.37 vs 7.93；SGDet N5R100: 5.48 vs 5.02）

**OOD 鲁棒性（Table 2）**：SGGNLS-c/o 在 OOD 集上 PhrDet N5R100 骤降至 0，而本文方法仅小幅下降（ID: 14.82 → OOD: 11.69，bbox mode），展现出强泛化能力。

**消融实验**：
- Segment Merger（Table 3）：Stage 1（64 segments）+ Graph Cut 效果最佳，验证显式合并优于隐式合并
- Label Generator（Table 4）：Gen w/ PET（BLIP）效果最优（PhrDet N3R100 = 12.64），远优于 Cls+WordNet（8.82）和 Gen w/o PET（2.33），验证了生成式预测和 PET 的关键作用
- TSSS 应用（Table 5）：将 entity grounder 和 segment merger 引入 GroupViT，mIoU 从 24.72 提升至 26.87（+2.15%）

## 相关工作脉络
1. **Bbox-based SGG**：Krishna et al. [18] Visual Genome、Yang et al. [50] Graph R-CNN 等全监督方法依赖密集标注；Zhong et al. [58] SGGNLS 是最相关的弱监督 baselines，但仍依赖 region proposal network 和预定义词表，本文彻底移除了这些先验。
2. **Panoptic Scene Graph**：Yang et al. [49] 首次提出 PSG 概念，但采用全监督训练，本文在其基础上探索完全弱监督的生成路径。
3. **Text-supervised Semantic Segmentation (TSSS)**：GroupViT [48] 从文本监督中学习语义分割，但仅输出语义 mask 不含关系结构；本文将其扩展到全景场景图生成，并改进合并策略为显式学习。
4. **Visual Grounding**：Karpathy & Fei-Fei [15]、Plummer et al. [33] 等方法依赖 region proposals 和预定义词汇；本文在无 proposal、无预定义词汇的约束下完成接地。
5. **VLM-based 生成式方法**：BLIP [21]、CLIP [34] 等预训练视觉语言模型为本文提供了常识知识基础，但本文首次将其用于 PSG 生成任务并通过 PET 进行了适配。

## 局限性与未来方向
1. **语义分割转实例分割的策略不够精细**：当前通过连通分量识别实例，在物体重叠或遮挡时会低估或高估实例数量。
2. **小物体定位困难**：受限于输入分辨率和分组策略，小物体难以准确定位。
3. **关系预测对图像条件依赖不足**：label generator 有时过度依赖对象语义而非实际图像内容，导致关系预测不准确。
4. **Caption 数据粒度限制**：caption 常将相同语义的多个实例合并为复数形式描述（如图 4 所示），限制了模型学习 panoptic（而非仅 semantic）分割的能力。
5. **未来方向**：更精细的分割转换策略、提高输入分辨率、改进图像条件推理机制、构建粒度更细的图像-caption 对数据集。

## 研究启发与可借鉴点
1. **生成式标签预测替代分类**：在无预定义概念集的开放场景下，将标签预测从分类问题重构为自回归生成问题，有效避免了词表遗漏和 WordNet 匹配不精确的问题，可迁移至其他开放词汇场景理解任务。
2. **显式学习合并代替隐式学习**：利用接地伪标签构建相似度矩阵的目标函数，相比纯对比学习驱动的隐式合并，在定位精度上有显著提升，这一思路可用于改进 GroupViT 类方法。
3. **PET（Prompt-embedding Technique）的设计**：通过精心设计的 prompt 模板将伪标签信息注入预训练 VLM decoder，有效激活模型预训练常识，可推广到其他 VLM 微调场景。
4. **细粒度对比接地用于弱监督学习**：融合过滤阈值的双向 fine-grained contrastive loss，在无显式区域-实体对齐标注的情况下实现了可靠的视觉-语言接地，可应用于其他弱监督视觉语言任务。
5. **副产品价值**：entity grounder 和 segment merger 可提升 TSSS 性能，说明本文模块具有跨任务的可迁移性。

## 关键术语表
**Panoptic Scene Graph (PSG)**：一种基于图的全景场景表示，每个对象由 panoptic segmentation mask（而非 bbox）定位并附带语义标签，边表示对象间关系，实现像素级精细的结构化场景理解。
**Caption-to-PSG**：本文提出的新任务，指仅从图像-caption 对出发生成全景场景图，设定无位置先验、无显式区域-实体链接、无预定义概念集三重约束。
**Region Grouper**：继承 GroupViT 分层设计的模块，通过可学习的 grouping centers 和 attention 机制将图像 patch 逐级合并为更大、任意形状的段落。
**Entity Grounder**：基于细粒度对比学习的模块，将 caption 中的语言实体接地到图像段落，产生伪标签用于后续训练。
**Segment Merger**：利用接地伪标签显式学习段落间相似度矩阵的模块，推理时通过图割聚类合并相似段落。
**Label Generator**：基于预训练 BLIP decoder 的模块，将标签预测重构为自回归生成问题，通过 PET 技术引导对象和关系谓词的生成。
**Prompt-embedding-based Technique (PET)**：设计特定 prompt 模板（如 "a photo of [ENT]"），将伪标签信息嵌入 decoder 输入，以激活预训练 VLM 的常识知识。
**No-Graph-Constraint-X Recall@K (NXR@K)**：评估指标，限制每个 subject-object 对最多预测 X 个谓词标签的 Recall@K，适用于非互斥关系谓词的评测。

## 可复现要素
- **数据集**：COCC Caption（118,287 训练图像，公开）；Panoptic Scene Graph 数据集（公开）
- **代码**：项目页面 https://vis-www.cs.umass.edu/TextPSG，论文声明代码/数据/结果已开源
- **权重**：使用预训练的 GroupViT 和 BLIP [21] 进行初始化
- **关键超参**：分组层数 K=2，H₁=64，H₂=8；推理阶段 k_inf=1；阈值 θ 和温度 τ（τ 为可学习参数）；论文未提及具体数值
- **训练**：Tfm^T 和 label generator 在训练过程中冻结
