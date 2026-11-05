---
title: "Deep-Geometrized-Cartoon-Line-Inbetweening"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:58:21"
field: "2D动画自动化"
keywords: ["帧插值", "线稿动画", "图匹配", "几何嵌入", "动漫生成"]
innovations: ["将线稿几何化为图结构实现顶点级中间帧生成", "多模态顶点嵌入融合图像上下文+位置+拓扑信息", "首次提出带几何化标注的大规模线稿数据集MixamoLine240"]
benchmarks: ["MixamoLine240", "Chamfer Distance"]
---

# 论文速读：Deep-Geometrized-Cartoon-Line-Inbetweening

## 一句话总结
本文针对动漫制作中耗时的线稿中间帧生成（inbetweening）问题，提出将光栅线稿几何化为矢量图结构，通过顶点匹配与重定位生成高质量中间线稿的深度学习方法AnimeInbet，并在首个大规模带几何化标注的线稿数据集MixamoLine240上验证了其优越性。

## 研究问题与动机
1. **任务重要性**：线稿中间帧生成是2D动画生产中最劳动密集的核心环节，目前仍需逐帧手绘，自动化可显著提升生产效率。
2. **稀疏性挑战**：线稿仅含约3%黑色像素，传统基于像素匹配的帧插值方法因缺乏纹理导致匹配歧义严重。
3. **结构破坏问题**：现有光栅插值方法的形变与混合操作会模糊线条边界，产生严重伪影，无法满足动画师对精确线条的需求。
4. **数据缺失**：现有动画数据集（如Creative Flow+、AnimeRun）缺乏顶点级几何化和对应关系标注，无法支持监督学习。

## 核心贡献（创新点）
1. **提出几何化线稿插值新范式**：将线稿从光栅图像转换为端点图结构，把中间帧生成重新定义为图融合+顶点重定位问题，避免像素级模糊伪影。
2. **设计多模态顶点几何嵌入**：融合图像上下文特征、位置编码和拓扑谱嵌入，使稀疏顶点具备可区分性，精准捕捉线稿结构信息。
3. **构建顶点匹配Transformer**：通过交替自注意力/交叉注意力聚合跨图互相关信息，结合可微最优传输（OT）和双向argmax实现鲁棒的顶点对应关系。
4. **开发注意力传播重定位机制**：将已匹配顶点的位移向量通过特征相似性传播到未匹配顶点，有效处理遮挡和缺失顶点的位置预测。
5. **发布首个大规模几何化线稿数据集MixamoLine240**：基于3D模型索引确保顶点级对应关系准确，提供240个序列、20角色×20动作的带标注数据。

## 方法详解
**整体流程**：输入两帧光栅线稿$I_0$、$I_1$及其矢量图$G_0=\{V_0, T_0\}$、$G_1=\{V_1, T_1\}$，输出中间时刻$t$的图$G_t=\{V_t, T_t\}$。

**1. 顶点几何嵌入（Vertex Geometric Embedding）**：
- **图像上下文嵌入**：用3层2D CNN $\mathcal{E}_I$提取光栅图像深层特征，按顶点坐标索引对应像素特征。
- **位置嵌入**：1D CNN $\mathcal{E}_P$将顶点坐标$(x,y)$映射为$C$维特征。
- **拓扑嵌入**：对邻接矩阵$T$进行拉普拉斯谱分解（Laplacian eigenmaps），再用1D CNN $\mathcal{E}_T$编码拓扑信息。
- 最终嵌入：$F_0 = \mathcal{E}_I(I_0)[V_0] + \mathcal{E}_P(V_0) + \mathcal{E}_T(S(T_0))$

**2. 顶点对应Transformer（Vertex Correspondence Transformer）**：
- 交替N层自注意力（SA）和交叉注意力（CA）聚合 intra-graph 和 inter-graph 信息。
- 计算相关矩阵 $\mathcal{P} = \hat{F}_0 \hat{F}_1^T / \sqrt{C}$，经可微最优传输 $\hat{\mathcal{P}} = OT(\mathcal{P})$ 提升一致性。
- 双向argmax匹配：$\hat{\mathcal{M}} = \{(i,j) | (i,j) \in \mathcal{M}_{0\to1} \cap \mathcal{M}_{1\to0}, \hat{\mathcal{P}}_{i,j} > \theta\}$，阈值$\theta=0.2$。

**3. 重定位传播（Repositioning Propagation）**：
- 已匹配顶点直接线性插值：$V_t = tV_0[i] + (1-t)V_1[j]$。
- 未匹配顶点通过注意力传播位移：$r_{0\to1} = \text{softmax}(\hat{F}_0\hat{F}_0^T/\sqrt{C}) \cdot (\text{softmax}(\hat{\mathcal{P}})V_1 - V_0)$，再经自注意力细化。

**4. 可见性预测与图融合**：
- 3层MLP预测二元可见性掩码 $m_0 = \text{MLP}(\hat{F}_0)$、$m_1 = \text{MLP}(\hat{F}_1)$。
- 融合规则：已匹配顶点对插值；未匹配但可见的顶点沿位移向量移动；遮挡顶点被剔除。
- 边拓扑$T_t$：只要两端点在$G_t$中可见，保留原连接。

**5. 损失函数**：
- 匹配损失：$\mathcal{L}_c = -\frac{1}{|\mathcal{M}^{GT}|}\sum_{(i,j)\in\mathcal{M}^{GT}}\log\hat{\mathcal{P}}_{i,j}$
- 重定位损失：$\mathcal{L}_r = \frac{1}{K_0}\|r_{0\to1} - r_{0\to1}^{GT}\|_1 + \frac{1}{K_1}\|r_{1\to0} - r_{1\to0}^{GT}\|_1$
- 可见性损失：$\mathcal{L}_m = \text{BCE}^w(\sigma(m_0), m_0^{GT}) + \text{BCE}^w(\sigma(m_1), m_1^{GT})$，$w=0.2$
- 总损失：$\mathcal{L} = \mathcal{L}_c + \mathcal{L}_r + \mathcal{L}_m$

**伪标签生成（Pseudo Labels）**：因遮挡顶点无法直接计算位移，采用逐帧回溯策略——利用相邻帧高匹配率，迭代推算大间隔帧的位移和可见性标签。

## 实验与结果
**数据集**：MixamoLine240，100序列训练、140序列验证/测试，24fps渲染，720×720分辨率，平均每帧约1256-1512个顶点、1753-2099条边。

**评估指标**：Chamfer Distance (CD)，值越低越好。

**主要结果（Validation Set, gap=5平均）**：
- AnimeInbet: **11.53**（最优）
- FILM: 16.92（次优）
- RIFE: 27.54
- EISAI: 27.57
- VFIformer: 28.19
- **提升幅度**：相比最优基线FILM提升超过30%

**Test Set结果**：
- AnimeInbet: **12.90** vs FILM: 18.31（提升31%）
- 大运动（gap=9）优势最显著：AnimeInbet 23.21 vs FILM 33.51

**消融实验**：
- 去除位置嵌入：Valid Acc.下降15.83%，CD恶化0.74
- 去除重定位传播：CD恶化112%（11.12→23.62）
- 去除可见性掩码：CD恶化15%（11.12→12.81）
- AnimeInbet-VS（VirtualSketcher矢量输入）：CD为13.51，略差于GT矢量输入

**用户研究**：36名参与者，AnimeInbet胜率超过92%，大运动场景接近100%。

## 相关工作脉络
1. **RIFE [6]**：实时帧插值光栅方法，直接预测中间流向 warped 输入帧；AnimeInbet针对稀疏线稿重构任务定义，避免形变模糊。
2. **VFIformer [14]**：Transformer-based光栅插值；像素级操作不适用于线稿的稀疏结构。
3. **EISAI [5]**：专为2D动画设计的轮廓聚合方法；仍基于光栅，无法保持线条精确性。
4. **FILM [23]**：大运动帧插值方法；在gap=9时性能显著劣于AnimeInbet。
5. **AnimeRun [29]**：提供动画视频对应数据集但为彩色光栅格式；MixamoLine240首次提供顶点级几何标注。
6. **虚拟Sketching [15]**：通用矢量化工具；AnimeInbet验证了其与自身结合的有效性（AnimeInbet-VS）。

## 局限性与未来方向
1. **矢量化工具依赖**：AnimeInbet-VS实验表明外部矢量化工具（VirtualSketcher）精度不足会损害结果质量，需探索端到端矢量生成。
2. **大运动遮挡处理**：gap=9时近50%顶点遮挡，当前可见性预测为软掩码，难以处理复杂拓扑变化。
3. **颜色/纹理缺失**：方法仅处理黑白线稿，未考虑色彩动画制作需求。
4. **顶点数量不一致**：不同帧顶点数差异需通过补零/裁剪处理，可能引入误差。
5. **未来方向**：联合优化矢量生成与中间帧插值、处理任意$t$值的通用时间插值、扩展到彩色/纹理线稿、探索非刚性形变更强的几何建模。

## 研究启发与可借鉴点
1. **几何感知运动传播**：通过特征相似性的注意力传播机制处理稀疏匹配，可迁移至点云配准、3D重建等任务。
2. **多模态图嵌入设计**：位置+拓扑+上下文三通道嵌入策略，为结构化数据的特征学习提供通用范式。
3. **3D参考点标注策略**：利用3D网格顶点的稳定对应关系生成2D标注，确保长序列一致性，值得在视频对应数据集构建中借鉴。
4. **两阶段训练策略**：先仅用对应损失预训练50 epoch，再加入完整损失，有效提升收敛稳定性。
5. **光栅到图形的任务重构**：将传统光栅问题重新定义为图/几何问题，开辟了新的方法设计思路，可应用于更多结构敏感任务。

## 关键术语表
**Geometrization**：将光栅线稿转换为矢量图结构（端点坐标+连接关系）的过程。
**Inbetweening**：在两个关键帧之间生成中间帧的动画制作技术。
**Chamfer Distance**：衡量两组点集之间相似度的指标，常用于评估线稿生成质量。
**MixamoLine240**：首个带顶点级几何化和对应标签的大规模线稿数据集，含240序列。
**Optimal Transport (OT)**：用于改进匹配一致性的数学工具，通过可微映射优化分配矩阵。
**Laplacian Eigenmaps**：基于图拉普拉斯矩阵的特征分解，用于拓扑信息嵌入。
**Pseudo Labels**：通过逐帧回溯策略生成的未匹配顶点位移和可见性的伪监督信号。
**VirtualSketcher**：开源矢量化工具，用于将光栅线稿转换为矢量图输入。

## 可复现要素
- **数据集**：MixamoLine240，代码和数据已公开于 https://github.com/lisiyao21/AnimeInbet
- **代码**：开源（Apache许可证）
- **权重**：已开源
- **关键超参**：特征维度C=128，阈值θ=0.2，中间时刻t=0.5，损失权重w=0.2，学习率1×10⁻⁴，batch size=8（梯度累积），训练70 epoch（前50 epoch仅匹配损失）
- **矢量化工具**：可使用VirtualSketcher [15]处理光栅输入
