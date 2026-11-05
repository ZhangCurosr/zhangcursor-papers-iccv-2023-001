---
title: "Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Deng_Learning_Neural_Eigenfunctions_for_Unsupervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:45:52"
field: "无监督语义分割"
keywords: ["无监督语义分割", "谱聚类", "神经特征函数", "Neural Eigenfunction", "ViT", "Gumbel-Softmax", "自监督学习"]
innovations: ["将谱聚类转化为端到端参数化神经网络，用神经特征函数替代昂贵的谱分解", "通过Gumbel-Softmax将神经特征函数输出量化为离散聚类分配，无需额外聚类步骤", "融合预训练ViT语义特征与下采样像素几何信息构建双图核，兼顾语义一致性与边界锐利度"]
benchmarks: ["Pascal Context", "Cityscapes", "ADE20K"]
---

# 论文速读：Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg

## 一句话总结
本文提出将经典谱聚类转化为端到端参数化神经网络的无监督语义分割方法，通过神经特征函数（Neural Eigenfunctions）近似图核的主特征函数，并将输出量化为离散聚类分配，从而克服传统谱聚类在大规模数据上计算开销高、泛化性差的瓶颈。

## 研究问题与动机
1. **传统谱聚类对原始像素敏感**：操作于原始像素级别，易受颜色变换影响，无法识别高层语义相似性。
2. **谱分解计算效率低**：谱聚类的特征分解复杂度随图节点数呈立方级增长，难以扩展到大图中。
3. **非参数化导致测试期灵活性差**：传统谱聚类为直推式方法（transductive），面对未见测试样本需重新进行耗时的谱分解，无法端到端推理。
4. **现有基于预训练模型的谱聚类方法（如DSM）仍受限**：虽然引入ViT特征缓解了语义敏感性，但仍需分别处理每幅图像再做跨图像同步，缺乏统一的可扩展框架。

## 核心贡献（创新点）
1. **将谱聚类参数化为神经特征函数学习**：借鉴NeuralEF技术，将谱分解问题转化为神经网络优化问题，直接学习图核的主特征函数，避免了昂贵的矩阵特征分解。与DSM等方法的本质区别：不再依赖离线谱分解，而是通过轻量网络直接从特征映射到谱嵌入空间。
2. **输出量化为离散聚类分配**：通过Gumbel-Softmax估计器约束神经特征函数输出为K维one-hot向量，直接指示聚类归属，省去了后续额外聚类步骤。与NeuralEF原工作（连续谱嵌入）的本质区别：引入离散化约束，实现端到端的聚类生成。
3. **构建融合高层语义与底层细节的图核**：联合利用预训练ViT的patch特征（余弦相似度最近邻图）和下采样像素（$L^2$距离最近邻图），通过归一化邻接矩阵加权融合构建核函数，兼顾语义一致性与边界锐利度。与DSM的本质区别：以参数化神经网络替代显式图构建与谱分解流程。
4. **实现跨图像、跨测试样本的统一泛化**：训练完成后，对新测试图像只需前向通过预训练ViT+神经特征函数网络即可获得谱嵌入/聚类分配，无需额外谱分解或同步过程。

## 方法详解
- **图核构建**：构建两个图——①基于预训练ViT patch特征的余弦相似度k-NN图（$k=256$），亲和矩阵$\mathbf{A}$；②基于下采样像素$L^2$距离的k-NN图（同图内连接），亲和矩阵$\tilde{\mathbf{A}}$。核函数为归一化邻接矩阵的加权和：$\kappa = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} + \alpha \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$，其中$\alpha=0.3$。
- **神经特征函数学习**：使用K输出神经网络$\psi: \mathbb{R}^C \to \mathbb{R}^K$近似核$\kappa$的主特征函数，优化目标为（Equation 5）：最大化对角项$\sum R_{j,j}$同时最小化不同特征函数间的交叉项$\sum \widehat{R}_{i,j}^2$（$\beta=0.08$），并通过$L^2$-batch normalization施加正交约束。Monte Carlo估计用于期望计算。
- **输出量化**：训练时在前向使用Gumbel-Softmax将连续输出软量化为one-hot类向量，温度从1余弦衰减至0.3；测试时移除Gumbel-Softmax和$L^2$-BN层，直接对网络输出做softmax得到聚类分配。
- **推理流程**：预训练ViT（冻结）提取patch特征→神经特征函数网络（2个linear self-attention transformer块+正交线性头）输出→bilinear上采样至原图分辨率→argmax获得像素级聚类→多数投票（majority voting）匹配语义类别。

## 实验与结果
- **数据集**：Pascal Context（60类）、Cityscapes（27类）、ADE20K（150类）。
- **基线**：MaskCLIP、ReCo、K-means（在ViT特征上）、Ours*（ψ网络线性头前特征+K-means）。
- **Pascal Context**（ViT-S）：Ours Acc=70.4%，mIoU=38.8%；Ours* mIoU=39.6%；超越MaskCLIP（25.5%）和ReCo（27.2%）。
- **Cityscapes**（ViT-S）：Ours Acc=83.4%，mIoU=28.2%；Ours* mIoU=30.0%；超越MaskCLIP（10.0%）和ReCo（19.3%）。
- **ADE20K**（ViT-S, K=512）：Ours Acc=63.3%，mIoU=21.6%；Ours* mIoU=23.6%。
- **滑动窗口评估**（含背景）：Pascal Context Ours mIoU=41.4%（监督上限DeepLabv3+为48.5%）；Cityscapes Ours mIoU=46.7%（监督上限77.3%）。
- **零样本迁移**（ImageNet训练→目标域测试）：Cityscapes mIoU=18.5%，与MaskCLIP（10.0%）和ReCo（19.3%）相当。
- **消融**：K≥256时有显著提升；$\alpha \in [0.3, 0.5]$和$\beta \in [0.04, 0.16]$均稳定；CLIP预训练 backbone随模型增大mIoU持续上升（ViT-B:37.2→ViT-L/14@336px:44.0）。
- **最强结果**：Pascal Context上ViT-S backbone的38.8% mIoU，相对ReCo提升11.6个百分点；Cityscapes上ViT-S的28.2% mIoU，相对ReCo提升8.9个百分点。

## 相关工作脉络
1. **DSM（Melas-Kyriazi et al., CVPR 2022）**：将预训练ViT特征引入谱聚类作为无监督分割基线，但需逐图处理+跨图同步，且测试期仍需谱分解。本文将其扩展为端到端参数化范式。
2. **MaskCLIP（Zhou et al., ECCV 2022）**：基于CLIP的zero-shot方法，依赖精心设计的文本prompt和自训练。本文不依赖跨模态对齐，可直接用任意预训练ViT。
3. **ReCo（Shin et al., 2022）**：检索与共同分割结合的zero-shot方法，同样依赖CLIP和自训练。本文方法更通用、更轻量的谱聚类思路。
4. **SpIN（Pfau et al., 2018）**：深度谱推理网络，学习目标定义不清，仅学到主特征函数张成的子空间而非特征函数本身，需额外复杂策略求解。NeuralEF通过新目标函数显式打破对称性，本文直接沿用此优势。
5. **NeuralEF（Deng et al., 2022）**：提出用神经网络近似核函数的特征函数，本文首次将其应用于无监督语义分割，并创新性地引入离散量化约束。
6. **IIC/PiCIE/STEGO等SSL分割方法**：多聚焦单对象/单类别识别，难以处理复杂图像。本文基于谱聚类的思路天然支持多类别发现。

## 局限性与未来方向
1. **仍需标注匹配聚类**：与大多数聚类方法一样，需借助验证集标注（匈牙利匹配或多数投票）将聚类编号映射到语义类别，未能完全摆脱对标签的依赖。
2. **ADE20K细粒度性能不足**：150个类别时mIoU仅23.6%，预训练模型缺乏patch级细粒度监督导致特征语义丰富度不足，容易将多个细粒度类别合并为一簇。
3. **零样本迁移效果有限**：在ImageNet上训练后迁移到复杂场景图像，mIoU显著下降，源域与目标域分布差异是主要瓶颈。
4. **未来方向**：引入文本prompt引导聚类（如CLIP文本监督）；对预训练模型进行patch级自监督微调以增强特征细粒度；结合自训练进一步提升性能。

## 研究启发与可借鉴点
1. **谱聚类参数化范式的可迁移性**：将传统非参数化谱方法转化为神经网络可微优化，同时保留谱嵌入的优良几何性质，这一思路可扩展至其他图分区/嵌入任务（如图节点分类、点云分割）。
2. **Gumbel-Softmax离散量化策略**：通过可微近似将连续谱嵌入直接映射为离散聚类标签，避免额外聚类步骤，对需要端到端训练的聚类型任务有参考价值。
3. **双图核设计（语义图+几何图）**：分别利用预训练特征和原始像素构建互补图，加权融合提升分割质量，$\alpha$参数鲁棒性强，该设计可推广至其他多模态融合分割任务。
4. **NeuralEF+正交线性头的轻量结构**：仅用2个linear self-attention块+正交约束线性头即可有效学习特征函数，训练半天（RTX 3090）即收敛，为资源受限场景提供了高效方案。
5. **K值大于真实类别数的设定**：消融表明K=256优于K=64/128，提示隐式超分类策略在谱嵌入学习中有效，可与本团队在少样本/开放集场景的研究结合。

## 关键术语表
**Spectral Clustering（谱聚类）**：基于图拉普拉斯特征分解的无监督聚类方法，通过将数据点映射到谱空间再进行欧氏聚类实现图划分。
**Neural Eigenfunction（神经特征函数）**：用神经网络近似核函数特征函数的方法，将离散谱分解转化为连续优化问题，支持端到端学习与泛化。
**Gumbel-Softmax Estimator**：对离散随机变量（如one-hot）的可微近似采样技术，使分类输出可在反向传播中优化。
**Normalized Cut（归一化割）**：谱聚类中最小化割边的目标函数，通过归一化邻接矩阵的极小特征值对应的特征向量近似求解。
**Transductive vs. Inductive**：直推式（仅对训练数据有效）与归纳式（可泛化到未见样本）；传统谱聚类为直推式，本文方法转为归纳式。
**ViT（Vision Transformer）**：将Transformer架构应用于图像识别的预训练模型，提供富含语义的patch级特征。
**Majority Voting**：将验证集所有图像的聚类分配统计投票，将每个聚类映射到出现次数最多的真实类别标签。

## 可复现要素
- **数据集**：Pascal Context、Cityscapes、ADE20K（均为公开数据集）。
- **代码**：已开源，https://github.com/thudzj/NeuralEigenfunctionSegmentor。
- **预训练模型**：timm库中的ViT-S/B/L（ImageNet-21k预训练+ImageNet微调），分辨率384×384。
- **关键超参**：k=256（特征图近邻数），$\tilde{k}$（像素图近邻数，按DSM设定），$\alpha=0.3$，$\beta=0.08$，K=256（ADE20K用512），学习率$10^{-3}$（余弦衰减），Adam优化器，40 epoch，batch size=16，Gumbel-Softmax温度从1余弦衰减至0.3，2个linear self-attention transformer块+正交线性头。
- **训练设备**：单卡RTX 3090，约半天。
