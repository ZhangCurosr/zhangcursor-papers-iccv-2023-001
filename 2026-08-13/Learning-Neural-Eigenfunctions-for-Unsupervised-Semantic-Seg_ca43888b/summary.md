---
title: "Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Deng_Learning_Neural_Eigenfunctions_for_Unsupervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:29:00"
field: "无监督视觉表征学习"
keywords: ["无监督语义分割", "谱聚类", "神经特征函数", "视觉Transformer", "Gumbel-Softmax", "参数化谱方法"]
innovations: ["将传统谱聚类转化为神经特征函数学习的参数化端到端范式，消除谱分解与后处理聚类步骤", "通过Gumbel-Softmax离散化神经特征函数输出直接获得聚类分配，实现纯NN驱动谱聚类流程"]
benchmarks: ["Pascal Context", "Cityscapes", "ADE20K"]
---

# 论文速读：Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg

## 一句话总结
本文提出将谱聚类转化为参数化的端到端神经网络方法，通过学习**神经特征函数（Neural Eigenfunctions）**生成谱嵌入，并将输出约束为离散聚类分配向量，从而实现高效、可扩展的无监督语义分割。

## 研究问题与动机
- **核心问题**：无监督语义分割需要在无标注情况下自动发现图像中的语义一致区域，但现有方法或依赖复杂自训练/交叉模态提示，或缺乏跨图像语义一致性。
- **传统谱聚类三大缺陷**：(i) 基于原始像素，对颜色变换敏感且无法捕捉语义相似性；(ii) 谱分解计算效率低（复杂度为 $O((NHW/P^2)^3)$）；(iii) 归纳式（transductive）特性导致难以泛化到新测试样本，无法端到端推理。
- **现有谱聚类改进工作（如 DSM）**虽引入预训练 ViT 特征缓解第一个问题，但仍受限于谱分解的计算开销与跨图像同步的灵活性不足。
- **动机**：借助神经特征函数技术将谱聚类从非参数化转化为参数化，实现低成本在线训练、高效测试泛化，并建立纯神经网络驱动的谱聚类流程。

## 核心贡献（创新点）
1. **首次将谱聚类参数化为神经特征函数学习问题**：利用 NeuralEF 技术用神经网络近似核函数的主特征函数，替代传统矩阵特征分解，实现端到端可微的谱嵌入学习。
2. **离散化神经特征函数输出直接得到聚类分配**：通过 Gumbel-Softmax 估计器将连续谱嵌入约束为 one-hot 聚类向量，省去 K-means 后处理步骤，实现纯 NN 驱动的聚类流程。
3. **构建融合高层语义与底层细节的双图核**：结合预训练 ViT patch 特征的余弦相似性近邻图与下采样像素的 $L^2$ 距离近邻图，加权融合为统一核函数，兼顾语义一致性与边界精度。
4. **实现简单而强效的无监督语义分割 baseline**：在 Pascal Context、Cityscapes、ADE20K 上显著超越 MaskCLIP、ReCo 及 K-means 基线，且支持零样本迁移与滑动窗口评估协议。

## 方法详解
**整体框架**：固定预训练 ViT 提取 patch 特征 → 构建双图核 → 神经特征函数 $\psi$ 学习谱嵌入 → Gumbel-Softmax 量化为聚类分配 → 推理时上采样+argmax。

1. **图核构建（Section 3.3）**：
   - **语义近邻图**：基于预训练 ViT 输出的 patch 特征 $\mathbf{F} \in \mathbb{R}^{NHW/P^2 \times C}$，构建余弦相似度 $k$-NN 图，邻接矩阵 $\mathbf{A}_{u,v} = \frac{\mathbf{F}_u \mathbf{F}_v^\top}{\|\mathbf{F}_u\|\|\mathbf{F}_v\|}, v \in k\text{-NN}(u)$。
   - **空间近邻图**：将原图双线性下采样至 $\tilde{\mathbf{X}} \in \mathbb{R}^{NHW/P^2 \times 5}$（融合空间坐标+颜色），构建 $L^2$ 距离 $k$-NN 图（限制邻居同图内），邻接矩阵 $\tilde{\mathbf{A}}_{u,v}=1$ if $v \in \tilde{k}\text{-NN}(u)$ else 0。
   - **融合核函数**：$\kappa = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} + \alpha \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$，其中 $\alpha=0.3$ 为权衡系数。选择归一化邻接矩阵而非拉普拉斯矩阵，以便学习最大特征值对应的特征函数。

2. **神经特征函数学习（Section 3.4）**：
   - 目标函数（Eq. 5）：$\max_\psi \sum_{j=1}^K \mathbf{R}_{j,j} - \beta \sum_{j=1}^K \sum_{i=1}^{j-1} \widehat{\mathbf{R}}_{i,j}^2$，s.t. $\mathbb{E}[\psi(\mathbf{f}) \circ \psi(\mathbf{f})] = \mathbf{1}$。
   - Monte Carlo 估计（Eq. 7）：$\mathbf{R} \approx \frac{1}{B^2}\Psi \cdot \kappa(\mathbf{F}^B,\mathbf{F}^B) \cdot \Psi^\top$，$\widehat{\mathbf{R}}$ 使用 stop-gradient 版本 $\widehat{\Psi}$。
   - 网络结构 $\psi$：2 个 linear attention transformer block + 线性头（正交列约束），输入为 ViT 最后一层注意力块输出（可附加中间层特征），输出 $\mathbb{R}^{K}$。
   - 末端接 $L^2$ batch normalization 层满足约束。

3. **量化与推理（Section 3.5-3.6）**：
   - 训练时：Gumbel-Softmax（温度从 1 余弦衰减至 0.3）将输出软化为近似 one-hot。
   - 推理时：移除 Gumbel-Softmax 与 $L^2$ BN，直接将输出作为 softmax logits，bilinear 上采样至原图分辨率，argmax 得聚类分配，再用 majority voting 与语义类别匹配。

## 实验与结果
**数据集**：Pascal Context（60类）、Cityscapes（27类）、ADE20K（150类细粒度）。

**预训练模型**：ViT-S/B/L（timm库，ImageNet-21k预训练+ImageNet微调），分辨率384×384，patch size=16。

**主要结果**（Table 1-2，标准评估协议）：
- **Pascal Context**：ViT-S 下 Acc=70.4%，mIoU=38.8%；超越 MaskCLIP (25.5)、ReCo (27.2)、K-means (28.9)。
- **Cityscapes**：ViT-S 下 Acc=83.4%，mIoU=28.2%；超越 MaskCLIP (10.0)、ReCo (19.3)、ReCo+ (24.2)。
- **滑窗协议**（Table 3）：Pascal Context mIoU=41.4%（接近监督 DeepLabv3+ 的 48.5%）；Cityscapes mIoU=46.7%（监督上限 77.3%）。
- **ADE20K**（Table 4）：ViT-S 下 mIoU=23.6%，优于 K-means (19.2%)，但绝对值较低（细粒度类别多+预训练模型缺乏 patch 级监督）。

**最强结果**：Pascal Context ViT-B 下 mIoU=37.5%，Cityscapes ViT-B 下 mIoU=26.8%；零样本迁移（ImageNet 预训练→ Pascal/Cityscapes）mIoU 分别为 15.2%/18.5%，具竞争力。

## 相关工作脉络
1. **DSM (Melas-Kyriazi et al., CVPR 2022)**：首次将谱聚类与预训练 ViT 结合用于无监督语义分割，构建跨图像 affinity 矩阵并做谱分解；本文将其"参数化"，消除谱分解与跨图像同步步骤。
2. **NeuralEF (Deng et al., 2022)**：提出用神经网络学习核函数特征函数的优化目标，打破特征函数间的对称性；本文将其适配至图像分割任务并引入离散化约束。
3. **SpIN (Pfau et al., 2018)**：早期深度谱推理网络，但目标函数未定义清楚，仅学到特征函数张成的子空间；NeuralEF 更明确地学习特征函数本身。
4. **MaskCLIP / ReCo**：基于 CLIP 等视觉-语言模型的 zero-shot 方法，依赖文本 prompt 与自训练；本文不依赖语言模态，仅用视觉预训练模型+谱聚类思想。
5. **SSL-based 分割（PiCIE, STEGO, IIC）**：通过对比学习/互信息最大化学习聚类友好特征；本文聚焦于特征提取后的谱聚类模块，可与这些方法结合。
6. **传统谱聚类（Shi & Malik, Ncuts）**：基于像素图的图划分，对颜色敏感且计算昂贵；本文将其推广至语义特征空间并参数化。

## 局限性与未来方向
- **需标注信息进行聚类匹配**：当前方法需 ground-truth 语义 mask 通过 Hungarian matching 或 majority voting 将聚类编号映射到语义类别，限制了完全无监督部署。
- **零样本迁移性能有限**：在 ImageNet 上训练的模型直接迁移至 Pascal/Cityscapes 时 mIoU 显著下降，因源域图像分布与复杂场景差异大。
- **细粒度类别处理不足**：ADE20K 上 mIoU 较低，因预训练 ViT 缺乏 patch 级细粒度监督，导致簇合并多语义类别。
- **未来方向**：引入文本提示引导聚类（text-guided spectral clustering）；在更大规模、更贴近目标域的无标注数据上预训练；设计细粒度感知的图核或引入 patch-wise SSL 损失。

## 研究启发与可借鉴点
1. **神经特征函数的参数化谱方法范式**：将传统非参数化谱分解转化为可微 NN 优化，可迁移至其他基于图的聚类/降维任务（如图像去噪、点云分割）。
2. **Gumbel-Softmax 离散化输出替代后处理聚类**：省去 K-means 等迭代优化，实现端到端训练，可推广至其他需离散分配的无监督学习任务。
3. **双图核融合设计**：同时利用高层语义特征与低层像素/空间信息，兼顾语义一致性与边界精确性，对多尺度特征融合有借鉴价值。
4. **预训练模型容量选择经验**：过大模型（ViT-L）因输出过于抽象反而降低分割性能，提示在无监督场景中需权衡表征能力与语义丰富度。
5. **CLIP 预训练 vs. ImageNet 预训练的差异化表现**：CLIP 模型随容量增大性能提升，而 ImageNet 预训练模型相反，提示应根据任务特性选择预训练源。

## 关键术语表
- **谱聚类（Spectral Clustering）**：基于图拉普拉斯特征分解的聚类方法，通过将数据映射到低维特征空间再聚类，理论上保证最小割划分。
- **神经特征函数（Neural Eigenfunction）**：用神经网络近似核函数特征函数的参数化方法，避免显式矩阵分解，支持新样本泛化。
- **Gumbel-Softmax**：对离散采样（如 one-hot）的可微近似，允许通过梯度下降优化含离散输出的神经网络。
- **归一化割（Normalized Cut）**：图划分目标函数，衡量割边权重与割两侧节点连接强度的比值，偏好平衡划分。
- **ViT（Vision Transformer）**：基于纯 Transformer 架构的图像预训练模型，将图像划分为 patch 序列进行自注意力计算。
- **近邻图（Nearest Neighbor Graph）**：每个节点仅与特征空间中最近$k$个节点相连的稀疏图，降低计算与存储开销。
- **stop-gradient**：反向传播时阻断梯度流动的运算，用于解耦目标函数中不同项的依赖关系，在 NeuralEF 中打破特征函数对称性。
- **majority voting**：将验证集所有图像的聚类分配汇总，统计每个簇最常对应的语义类别，完成聚类-语义映射。

## 可复现要素
- **数据集**：Pascal Context、Cityscapes、ADE20K（均公开）
- **代码**：已开源，https://github.com/thudzj/NeuralEigenfunctionSegmentor
- **权重**：使用 timm 库中 ImageNet-21k 预训练的 ViT-S/B/L（公开可下载）
- **关键超参**：$k=256$（特征图近邻数），$\tilde{k}$ 依 DSM，$\alpha=0.3$，$K=256$（输出维度），$\beta=0.08$，Gumbel 温度 1→0.3 余弦衰减，batch size=16，学习率 $10^{-3}$ Adam，40 epochs；ADE20K 上 $K=512$，$\beta=0.04$，20 epochs。
