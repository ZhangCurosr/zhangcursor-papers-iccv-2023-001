---
title: "Unsupervised-Manifold-Linearizing-and-Clustering"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_Unsupervised_Manifold_Linearizing_and_Clustering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:31:48"
field: "无监督表示学习与聚类"
keywords: ["流形线性化", "无监督聚类", "MCR²", "双随机矩阵", "自监督表示学习", "次空间聚类"]
innovations: ["将MCR²目标扩展至无监督场景，联合优化表征与双随机隶属矩阵", "提出基于Sinkhorn投影的高效参数化与单步自监督初始化策略", "在多个大规模数据集上验证流形线性化与聚类的联合学习效果，超越SOTA深度聚类方法"]
benchmarks: ["CIFAR-10", "CIFAR-20", "CIFAR-100", "TinyImageNet-200", "Imb-CIFAR-10", "Imb-CIFAR-100"]
---

# 论文速读：Unsupervised-Manifold-Linearizing-and-Clustering

## 一句话总结
本文提出**流形线性化与聚类（MLC）**方法，在无标签条件下**同时**将数据从低维流形并集映射为线性子空间并集表示，并完成聚类。核心设计是将 MCR² 目标与受次空间聚类启发的**双随机聚类隶属矩阵**相结合，利用自监督初始化实现高效优化。

## 研究问题与动机
1. **非线性流形的几何假设失效**：经典次空间聚类假设数据位于低维线性子空间并集上，但真实数据（如自然图像）往往分布在非线性低维流形上，直接应用次空间聚类方法精度低。
2. **已有流形线性化方法的局限**：手工构造变换（如核映射）不适用于复杂数据；局部邻域近似方法要求极高采样密度；现有深度学习方法被证明易学到平凡解（trivial representations）。
3. **监督 MCR² 无法直接迁移到无监督场景**：MCR² 在有标签时能学习到"类间正交、类内均匀分布"的理想表示，但需要真实隶属关系，实际场景中无标签可用。
4. **鸡生蛋问题**：表征 Z 和隶属矩阵 Π 相互依赖——已知好表征可用次空间聚类求隶属，已知正确隶属可优化 MCR² 学表征，但两者初始都不理想。

## 核心贡献（创新点）
1. **MLC 形式化框架**：将 MCR² 目标同时优化表征 Z 和**双随机点-点隶属矩阵 Γ**，而非传统的点-簇硬隶属矩阵 Π，从根本上解决了组合优化难题。与 MCR² 的本质区别在于：MCR² 依赖真标签，MLC 通过双随机矩阵在无监督下自适应学习隶属关系。
2. **基于 Sinkhorn 投影的高效参数化**：将隶属矩阵参数化为 $\Gamma = P_{\Omega,\eta}(C^\top C)$，其中 $P_{\Omega,\eta}$ 为 Sinkhorn 双随机投影层，使得 O(n²) 约束矩阵可在小批量下高效计算且约束自动满足。
3. **无额外训练的单步自监督初始化**：特征通过总编码率（TCR）自监督学习初始化，隶属矩阵通过**复制已初始化特征头参数**到簇头后直接计算得到，无需额外训练阶段，相比 SCAN 等需 10 次随机初始化的方案更稳定高效。
4. **在多个大规模数据集上验证**：CIFAR-10 达到 86.3%（ACC），CIFAR-20 达 52.2%，CIFAR-100 达 49.4%，TinyImageNet-200 达 33.5%（ACC），全面超越 SCAN、SPICE、NMCE 等 SOTA 深度聚类方法。
5. **证明了正交子空间并集表征的学习能力**：实验显示 MLC 学到的表征类内数值秩低且类间余弦相似度小（正交性），证实了流形被成功线性化。

## 方法详解
**整体架构**：输入图像经 ResNet-18 backbone 提取特征，接特征头映射到 $\mathbb{R}^d$ 再投影到单位球面 $\mathbb{S}^{d-1}$ 得到 $Z$；同时 cluster head 输出 $C$，经 Sinkhorn 投影得到 $\Gamma \in \Omega$。

**MLC 目标函数（公式 4）**：
$$\max_{\theta} \ R(Z;\epsilon) - R_c(Z,\Gamma;\epsilon) \quad \text{s.t.} \ Z = f_\theta(X), \ \Gamma = h_\theta(X) \in \Omega$$
其中 $R(Z;\epsilon) = \log\det(I + \frac{d}{n\epsilon^2}ZZ^\top)$ 度量全部特征的多样性（最大化），$R_c(Z,\Gamma;\epsilon) = \frac{1}{n}\sum_{j=1}^n \log\det(I + \frac{d}{\epsilon^2}Z\text{Diag}(\Gamma_j)Z^\top)$ 度量各类内特征的紧凑性（最小化）。两者之差推动不同簇的子空间正交分离，簇内特征紧凑分布。

**双随机隶属矩阵参数化**：$\Gamma = P_{\Omega,\eta}(C^\top C)$，$C$ 为 cluster head 输出，$P_{\Omega,\eta}$ 为带熵正则化的 Sinkhorn 投影，将 $C^\top C$ 投影到双随机矩阵集合 $\Omega = \{\Gamma \geq 0: \Gamma\mathbf{1} = \Gamma^\top\mathbf{1} = \mathbf{1}\}$。负熵正则项使 Γ 趋向均匀矩阵，通过调节系数控制稀疏度。

**自监督初始化（TCR，公式 5）**：
$$\max_\theta R\left(\frac{Z+Z'}{2};\epsilon\right) + \lambda\sum_{i=1}^n|z_i^\top z_i'|$$
其中 $z_i, z_i'$ 为同一样本不同增广的编码。该步确保同一样本不同增广的特征相近，不同样本特征正交不相关，为后续 MLC 提供结构化初始值。

**多增广聚合**：对 A 个增广版本分别计算 $(Z^{(a)}, \Gamma^{(a)})$，最终 $Z = P_{\mathbb{S}^{d-1}}(\frac{1}{A}\sum_a Z^{(a)})$，$\Gamma = \frac{1}{A}\sum_a \Gamma^{(a)}$。

**算法流程（Algorithm 1）**：① 用 TCR 初始化 Z → ② 复制参数初始化 Γ → ③ 循环 T 步做 mini-batch 优化 → ④ 对 Γ 运行谱聚类得到最终标签。

## 实验与结果
**数据集**：CIFAR-10（10类）、CIFAR-20（20超类）、CIFAR-100（100类）、TinyImageNet-200（200类），以及构造的不均衡数据集 Imb-CIFAR-10 和 Imb-CIFAR-100。

**与次空间聚类的对比（CIFAR-10）**：
- MLC 直接聚类原始图像：**ACC=86.3%，NMI=78.3%**
- EnSC/SSC-OMP 作用于 TCR 自监督特征：ACC 仅 67.8%~72.2%
- 作用于 MLC 表征 $Z_{MLC}$ 后：EnSC ACC 升至 81.5%，证明 MLC 学到了正交子空间并集结构
- 类内数值秩从 37 降至 24（cluster 3 为例），最终各类数值秩在 [13, 23]，证实流形被线性化

**与深度聚类方法的对比**：
- CIFAR-10 TCR 初始化：MLC-TCR **ACC=86.3%，NMI=78.3%**，比 NMCE-TCR（83.0%，76.1%）高 **3.3%**（ACC）和 **2.2%**（NMI）
- CIFAR-10 MoCoV2 初始化：**ACC=92.2%**，NMI=85.5%，超越 SPICE-MoCoV2（91.8%，85.0%）和 IMC-SwAV（89.1%，...）
- CIFAR-20 MoCoV2：**ACC=58.3%**，NMI=59.6%，超越 SPICE-MoCoV2（53.5%，56.5%）
- CIFAR-100：MLC-TCR **ACC=49.4%，NMI=68.3%**，大幅领先 SCAN-SimCLR（34.3%，55.7%）和 IMC-SwAV（43.9%，58.3%）
- TinyImageNet-200：MLC-TCR **ACC=33.5%，NMI=67.5%**，超过 GCC-SimCLR（13.8%，34.7%）和 SPICE-MoCoV2（30.5%，44.9%）
- 训练效率（CIFAR-100）：MLC 291 分钟 vs SCAN 396 分钟 vs IMC-SwAV 529 分钟
- 不均衡数据集：MLC 在 Imb-CIFAR-10 上 ACC=80.0%（仅从 86.3% 下降 6.3%），而 SCAN 从 87.6% 跌至 62.9%，IMC-SwAV 从 89.1% 跌至 65.7%，鲁棒性显著优于对比方法

## 相关工作脉络
1. **MCR²（Yu et al., NeurIPS 2020）**：监督流形线性化方法，在有标签下最大化编码率差实现类间正交、类内紧凑；MLC 的核心优化目标来源于此，但扩展到无监督场景，并将固定标签矩阵替换为可学习的双随机隶属矩阵。
2. **SCAN（Van Gansbeke et al., ECCV 2020）**：双步深度聚类方法（自监督初始化 + pseudo-label 迭代优化）；MLC 与之对比表明前者虽性能尚可但学到的表征缺乏类内多样性（易发生 neural collapse），且初始化不稳定。
3. **NMCE（Li et al., arXiv 2022）**：与 MLC 研究同一问题（Problem 1），也优化 MCR²，但使用 n×k 点-簇隶属矩阵而非点-点双随机矩阵；MLC 的初始化更简洁（无需额外训练阶段）、稳定性更强。
4. **DSSC（Lim et al., ICML 2022）/ Doubly Stochastic Subspace Clustering**：次空间聚类中双随机亲和矩阵的理论与应用；MLC 受此启发，将双随机约束引入聚类隶属建模。
5. **TCR（Total Coding Rate，Li et al., arXiv 2022）**：自监督表征学习方法，与 MCR² 同系；MLC 将其作为第一阶段初始化策略，充分利用已有自监督成果。
6. **EnSC / SSC-OMP（You et al., CVPR 2016-2018）**：经典次空间聚类方法；论文证明直接在非流形数据上应用次空间聚类（即使使用高质量自监督特征）精度有限，凸显流形线性化的必要性。

## 局限性与未来方向
1. **Γ 的 n×n 规模限制**：尽管通过 mini-batch 和小批量 Sinkhorn 实现高效计算，但对超大规模数据集（如 ImageNet 级别，n=10⁶）仍可能面临显存与计算瓶颈，论文未给出明确扩展方案。
2. **谱聚类后处理依赖**：最终标签由对 Γ 运行谱聚类得到，这一后处理步骤本身在大规模数据下计算复杂度为 O(n²)~O(n³)，论文未讨论替代方案。
3. **增广策略的选择敏感**：TCR 初始化和 MLC 优化中使用了特定数据增广，不同数据集可能需要调优，论文附录才给出细节。
4. **超参数调优**：ε、η、λ 等超参数的选择对性能有一定影响，论文未系统讨论敏感性分析。
5. **未来方向**：可探索将 MLC 框架推广到半监督/少样本设置，或结合更大规模预训练模型（如 ViT）。

## 研究启发与可借鉴点
1. **双随机隶属矩阵替代点-簇隶属矩阵**：用 n×n 点-点亲和矩阵代替 n×k 点-簇矩阵，不仅兼容次空间聚类成熟理论，还允许无额外训练的确定性初始化，这一思路可迁移到其他联合表征-聚类任务中。
2. **自监督初始化 + Sinkhorn 投影的"即插即用"设计**：将 TCR/SimCLR 等自监督方法作为固定初始化阶段，再通过 Sinkhorn 层隐式满足双随机约束，避免了额外训练开销，可复用为通用初始化模块。
3. **MCR² 在无监督场景的扩展路径**：将监督 MCR² 的固定标签矩阵泛化为可学习的双随机矩阵，是连接信息论表示学习与无监督聚类的有效范式，可推广到其他基于编码率的损失设计。
4. **正交子空间并集表征的评估指标**：论文提出的"类内数值秩 + 类间余弦相似度"联合评估方法，比单纯 ACC/NMI 更能揭示表征几何性质，可作为表征质量的通用诊断工具。
5. **不均衡数据鲁棒性验证**：构造对称移除样本的不均衡数据集来测试方法鲁棒性，这一实验设计值得借鉴；MLC 在不均衡下性能下降最小，暗示双随机结构天然具有负载均衡效应。

## 关键术语表
**Manifold Linearizing and Clustering (MLC)**：论文提出的联合流形线性化与聚类框架，同时学习线性化表征和聚类隶属。

**Maximal Coding Rate Reduction (MCR²)**：基于信息论的表示学习目标，通过最大化全局编码率与类内编码率之差，学习类间正交、类内紧凑的多子空间表示。

**Doubly Stochastic Matrix (双随机矩阵)**：行和与列和均为 1 的非负矩阵，本文用于建模样本对的软隶属关系，由 Sinkhorn 投影保证约束满足。

**Total Coding Rate (TCR)**：自监督表示学习目标，鼓励同一样本不同增广的特征接近、不同样本特征正交，用于 MLC 的初始化阶段。

**Numerical Rank（数值秩）**：累计奇异值平方占比超过 95% 所需的最小子空间维度，用于衡量表征是否接近低维子空间。

**Sinkhorn Projection**：将任意非负矩阵通过迭代行/列归一化投影到双随机矩阵集合的算法，此处带熵正则化。

**Neural Collapse**：交叉熵损失下训练末期特征趋于 collapsing 至类中心的退化现象，MLC 通过 MCR² 避免此问题。

**Spectral Clustering（谱聚类）**：对亲和矩阵进行特征分解后进行 K-means 的聚类方法，本文用于从学习到的 Γ 矩阵提取最终标签。

## 可复现要素
- **数据集**：CIFAR-10/20/100、TinyImageNet-200 均为公开数据集；不均衡数据集为作者自行构造。
- **代码开源**：论文未明确声明代码开源仓库，但从 ArXiv 引用（Li et al., 2022）及作者团队风格推测可能在 GitHub 发布，需另行查证。
- **关键超参**：ResNet-18 backbone；特征维度 d=128；每次 mini-batch 采样 A=2 个增广；ε、η、λ 等详见 Appendix（论文正文未全部列出）；优化器为 Adam。
- **训练环境**：2 × Nvidia RTX 3090 GPU。
