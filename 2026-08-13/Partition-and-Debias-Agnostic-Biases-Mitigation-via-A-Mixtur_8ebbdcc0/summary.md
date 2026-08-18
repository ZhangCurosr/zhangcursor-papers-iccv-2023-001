---
title: "Partition-and-Debias-Agnostic-Biases-Mitigation-via-A-Mixtur"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Partition-And-Debias_Agnostic_Biases_Mitigation_via_a_Mixture_of_Biases-Specific_Experts_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:26:50"
---

# 论文速读：Partition-and-Debias: Agnostic Biases Mitigation via A Mixture of Biases-Specific Experts

## 一句话总结
本文针对真实图像中**多种未知偏见共存且类型/数量均不可知**的“无偏偏见（agnostic biases）”难题，提出了分区去偏方法 **PnD**。该方法通过在深度网络不同层级隐式划分偏见子空间，并部署多个**偏置特定专家（biases-specific experts）**配合自适应门控聚合，实现多类型混杂偏见的并行捕获与去偏分类。

## 研究问题与动机
1. **核心问题**：现有图像去偏工作普遍隐式假设单张图像仅含一种已知或未知偏见，无法应对现实场景中多种混杂偏见（spurious correlations）同时存在且类型与数量均未知的问题。
2. **现有方法不足**：
   - **已知偏见方法**（如LfF、DFA、OccamNet）强依赖偏见可标注或特定归纳假设（如目标特征更简单、偏见可编辑），在多类型未知偏见交织时失效。
   - **未知偏见方法**（如UBNet、DebiAN）通常仅能挖掘并缓解一种主导偏见，理论上限为二元去偏，无法并行处理多类混杂特征。
3. **真实数据洞察**：以CelebA年龄分类为例，约**43.75%**的年轻样本同时带有female、attractive、lipstick三种偏见，多偏见图像占主导；单纯移除单一偏见无法恢复预测性能。
4. **关键假设与验证**：作者在Biased MNIST上进行探索实验，发现不同偏置特征（纹理/颜色/位置/尺度）会**聚类在神经网络的不同深度层级**，为按深度分区处理提供了实证依据。

## 核心贡献（创新点）
1. **形式化提出 agnostic biases 新场景**：突破单偏见假设，将偏见类型与数量均未知、多偏见共存的现实问题定义为独立研究场景，填补现有文献空白。
2. **设计分区去偏架构 PnD**：通过双编码器（去偏编码器$\mathcal{D}$与偏置编码器$\mathcal{B}$）并行提取特征，并在残差块各深度插入偏置特定专家，实现跨层级的隐式偏见空间划分与并行处理。
3. **两阶段训练机制**：初始训练阶段利用加权CE与GCE分离目标/偏置学习；反事实训练阶段通过特征交叉重组构造对比样本，强制解耦目标与偏置表征。
4. **专家多样性正则与自适应门控**：引入基于KL散度的$\mathcal{L}_{div}$防止专家同质化；设计softmax门控模块动态加权聚合多专家输出，在复杂多偏见数据集上达到SOTA。

## 方法详解
- **特征提取**：输入图像$x$经ResNet-18的前$M=4$个残差块分别送入去偏编码器$\mathcal{D}$和偏置编码器$\mathcal{B}$，得到各层级目标特征$\mathbf{z}_d^{(i)}$与偏置特征$\mathbf{z}_b^{(i)}$。
- **偏置特定专家 $E^{(i)}$**：每个专家包含去偏分类器$C_d^{(i)}$与偏置分类器$C_b^{(i)}$，输入为拼接特征$\mathbf{z}^{(i)}=[\mathbf{z}_d^{(i)};\mathbf{z}_b^{(i)}]$。
- **初始训练损失**：
  - **偏置检测损失**：$\mathcal{L}_{\mathrm{bias}} = \sum_{i=1}^M \mathrm{GCE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})$，利用GCE鼓励偏置编码器专注易学的混杂特征。
  - **去偏分类损失**：$\mathcal{L}_{\mathrm{debias}} = \sum_{i=1}^M \mathrm{w}^{(i)} \times \mathrm{CE}(\hat{\mathbf{y}}_d^{(i)}, \mathbf{y})$，其中样本权重 $\mathrm{w}^{(i)} = \frac{\mathrm{CE}(\hat{\mathbf{y}}_b^{(i)},\mathbf{y})}{\mathrm{CE}(\hat{\mathbf{y}}_d^{(i)},\mathbf{y}) + \mathrm{CE}(\hat{\mathbf{y}}_b^{(i)},\mathbf{y})}$，使模型优先拟合难分的无偏样本。
  - **多样性损失**：$\mathcal{L}_{\mathrm{div}} = \sum_{i=2}^M \exp(-\mathrm{KL}(\hat{\mathbf{y}}_b^{(i)}, \hat{\mathbf{y}}_b^{(i-1)}))$，惩罚相邻专家偏置预测的相似度，促使各专家捕获不同层级的偏见。
- **反事实训练（对比学习）**：随机采样小批量，将第$j$个样本的目标特征与其他样本的偏置特征交叉重组为正样本 $\mathcal{Z}_{\mathrm{pos}}^{(i)}$，将偏置特征与其他目标特征重组为负样本 $\mathcal{Z}_{\mathrm{neg}}^{(i)}$。通过对比损失 $\mathcal{L}_{\mathrm{con}}$ 拉近相同目标特征、推远不同目标特征的预测分布，实现特征解耦。
- **门控聚合**：最终预测 $\hat{\mathbf{y}}_d = \sum_{i=1}^M \mathrm{p}^{(i)} \times \hat{\mathbf{y}}_d^{(i)}$，$\mathrm{p}^{(i)}$ 为门控网络输出的softmax权重。总损失 $\mathcal{L} = \mathcal{L}_{\mathrm{cls}} + \mathcal{L}_{\mathrm{gate}} + \mathcal{L}_{\mathrm{div}} + \beta \mathcal{L}_{\mathrm{con}}$。

## 实验与结果
- **数据集**：Biased MNIST（7种偏见，公开）、BAR（1种偏见，训练集完全偏置，公开）、Modified IMDB（2种偏见，自建）、MIMIC-CXR+NIH（1种偏见，自建/公开源）、CelebA（真实人脸，公开）。
- **评估基线**：ResNet-18、LfF、DFA、OccamNet、DebiAN、UBNet。
- **主要结果**：
  - **Biased MNIST（bias ratio 0.95）**：PnD 达到 **70.43%**，超越次优 OccamNet（66.85%）约 **3.58%**。
  - **MIMIC-CXR+NIH**：PnD 达到 **60.73%**，超越 DebiAN（60.00%）约 **0.73%**。
  - **CelebA（wearing lipstick
