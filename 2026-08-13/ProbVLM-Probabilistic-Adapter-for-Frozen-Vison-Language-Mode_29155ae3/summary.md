---
title: "ProbVLM-Probabilistic-Adapter-for-Frozen-Vison-Language-Mode"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:29:11"
field: "多模态表示学习与不确定性量化"
keywords: ["Vision-Language Model", "Probabilistic Embedding", "Uncertainty Estimation", "Cross-modal Retrieval", "Post-hoc Adaptation", "Generalized Gaussian Distribution"]
innovations: ["首个针对冻结 VLM 的后验概率适配器，将确定性嵌入转为 GGD 分布估计", "联合内模态重建与跨模态对齐的训练目标，在无需大规模重训练下实现校准不确定性", "利用 Stable Diffusion 反向可视化概率嵌入分布，验证语义一致性"]
benchmarks: ["MS-COCO", "Flickr30k", "CUB", "Oxford-Flowers 102"]
---

# 论文速读：ProbVLM: Probabilistic Adapter for Frozen Vision-Language Models

## 一句话总结
本文提出 ProbVLM，一种**事后概率适配器**，在完全冻结预训练 VLM（CLIP、BLIP）的基础上，仅需轻量 MLP 适配层即可将确定性嵌入映射为概率分布（广义高斯分布），高效估计 calibrated 的嵌入不确定性，无需大规模重新训练。

## 研究问题与动机
- **确定性与歧义的矛盾**：CLIP/BLIP 等 VLM 通过确定性映射将图像/文本映射到单一嵌入向量，但同一概念在不同模态中天然存在"一对多"歧义（multiple samples can abstract the same concept），确定性嵌入无法表达这种不确定性。
- **现有概率嵌入方法的局限**：PCME 等先前工作采用概率对比损失学习分布，但必须**从头训练整个网络**，依赖大规模数据集和算力，无法利用已有冻结 VLM 的预训练知识。
- **单模态工作的局限**：PFE（Probabilistic Face Embedding）等已有后验方法仅在单模态（图像）下验证，未在跨模态 VLM 场景下进行推广。
- **下游应用需求**：不确定性感知的检索、主动学习、模型选择等任务需要可靠的 uncertainty 估计，而现有 VLM 缺乏此类能力。

## 核心贡献（创新点）
1. **首个针对冻结 VLM 的后验概率适配器**：在 CLIP/BLIP 之上添加轻量 MLP，无需重新训练编码器，将确定性嵌入转化为概率嵌入。*与 PCME 等从头训练的方法本质不同，ProbVLM 完全复用预训练权重。*
2. **基于 GGD（广义高斯分布）的异方差建模**：用 GGD 参数化嵌入分布（均值、尺度、形状三个参数），相比固定高斯分布能更好地捕捉重尾特性，更贴合真实嵌入歧义。
3. **联合 intra-modal 重建 + cross-modal 对齐的训练目标**：内模态对齐确保 ProbVLM 均值忠实于冻结编码器输出（reconstruction fidelity），交叉模态对齐确保同概念图像/文本的分布相互靠近，两者共同建模跨/模态内歧义。
4. **统一不确定性量化框架**：通过 GGD 参数计算 aleatoric 不确定性，结合 Dropout 多次前向传播估计 epistemic 不确定性，给出总不确定性 $\hat{\sigma}_{\text{total}}^2 = \hat{\sigma}_{\text{epistemic}}^2 + \hat{\sigma}_{\text{aleatoric}}^2$。
5. **首次利用 Latent Diffusion 可视化概率嵌入分布**：从预测分布中采样嵌入向量，输入 Stable Diffusion（经 CLIP text encoder）生成图像集合，直观展示分布捕获的语义变分模式。

## 方法详解

### 整体框架
- 冻结预训练 VLM 编码器 $\Phi_\mathcal{V}(\cdot;\theta_\mathcal{V}^*)$ 和 $\Phi_\mathcal{T}(\cdot;\theta_\mathcal{T}^*)$，得到确定性嵌入 $\mathbf{z}_\mathcal{V} = \Phi_\mathcal{V}(\mathbf{x}_\mathcal{V})$ 和 $\mathbf{z}_\mathcal{T} = \Phi_\mathcal{T}(\mathbf{x}_\mathcal{T})$。
- 引入两个轻量 MLP 适配器 $\Psi_\mathcal{V}(\cdot;\zeta_\mathcal{V})$ 和 $\Psi_\mathcal{T}(\cdot;\zeta_\mathcal{T})$，各为 3 层（embedding_dim → 256 → 256 → embedding_dim），以 dropout=0.1 训练。
- 对输入嵌入 $\mathbf{z}$，$\Psi$ 输出 GGD 参数 $(\hat{\mathbf{z}}, \hat{\alpha}, \hat{\beta})$，其中 $\hat{\mathbf{z}}$ 近似原嵌入作为分布均值。

### 内模态对齐损失（Intra-modal Alignment）
目标：使分布均值忠实于冻结编码器输出，同时学习异方差性。假设残差服从 GGD：
$$\mathcal{G}(\mathbf{z}; \hat{\mathbf{z}}, \hat{\alpha}, \hat{\beta}) = \frac{\hat{\beta} \exp\left(-(|\mathbf{z} - \hat{\mathbf{z}}|/\hat{\alpha})^{\hat{\beta}}\right)}{2\hat{\alpha}\,\Gamma(1/\hat{\beta})}$$
重建损失（负对数似然）：
$$L_{\text{rec}}(\zeta) = \left(\frac{|\hat{\mathbf{z}} - \mathbf{z}|}{\hat{\alpha}}\right)^{\hat{\beta}} - \log\frac{\hat{\beta}}{\hat{\alpha}} + \log\Gamma\left(\frac{1}{\hat{\beta}}\right)$$
分别对视觉（$L_{\text{rec}}^\mathcal{V}$）和文本（$L_{\text{rec}}^\mathcal{T}$）模态独立计算。

### 交叉模态对齐损失（Cross-modal Alignment）
目标：使同概念的图像嵌入分布与文本嵌入分布尽量重合。精确计算两个 GGD 卷积需 Bivariate Fox H-function，不可扩展，因此采用近似：
$$p(\mathbf{z}_v = \mathbf{z}_u) \approx \frac{1}{2}\left[\mathcal{G}_\mathcal{V}(\mathbf{z}_\mathcal{T}) + \mathcal{G}_\mathcal{T}(\mathbf{z}_\mathcal{V})\right]$$
即：文本嵌入 $\mathbf{z}_\mathcal{T}$ 在视觉分布下的概率 + 视觉嵌入 $\mathbf{z}_\mathcal{V}$ 在文本分布下的概率。取负对数得：
$$L_{\text{cross}}(\zeta_\mathcal{V}, \zeta_\mathcal{T}) = \text{NNL}_\mathcal{V}(\mathbf{z}_\mathcal{T}) + \text{NNL}_\mathcal{T}(\mathbf{z}_\mathcal{V})$$

### 总损失函数
$$L_{\text{ProbVLM}} = L_{\text{rec}}^\mathcal{V} + L_{\text{rec}}^\mathcal{T} + \lambda_{\text{cross}} \cdot L_{\text{cross}}$$
$\lambda_{\text{cross}}$ 为超参，控制跨模态对齐的权重。

### 不确定性量化
- **Aleatoric 不确定性**：由 GGD 解析计算 $\hat{\sigma}_{\text{aleatoric}}^2 = \frac{\hat{\alpha}^2 \Gamma(3/\hat{\beta})}{\Gamma(1/\hat{\beta})}$
- **Epistemic 不确定性**：推理时开启 dropout，做 $M$ 次前向传播，计算均值方差 $\hat{\sigma}_{\text{epistemic}}^2 = \frac{1}{M}\sum_m (\hat{\mathbf{z}}_m - \bar{\mathbf{z}})^2$
- **总不确定性**：$\hat{\sigma}_{\text{total}}^2 = \hat{\sigma}_{\text{epistemic}}^2 + \hat{\sigma}_{\text{aleatoric}}^2$

### 扩散模型可视化
从预测分布采样 $\{\hat{\mathbf{z}}_i\}_{i=1}^{Q}$，通过 Stable Diffusion $\Omega(\cdot;\theta_\Omega^*)$ 生成图像集合 $J = \{\Omega(\hat{\mathbf{z}}_i)\}$，直观呈现嵌入分布捕获的语义变化。

## 实验与结果

### 数据集与评估
- **数据集**：MS-COCO、Flickr30k（通用跨模态检索）、CUB（细粒度鸟类）、Oxford-Flowers 102（FLO，细粒度花卉）
- **评估指标**：
  - 检索性能：Recall@k（R@k）
  - 校准质量：Spearman 秩相关系数 S（不确定性等级与 R@k 的相关性）、回归 $R^2$、统一指标 $-SR^2$（理想值为 1.0）

### 主要结果
- **CLIP ViT-B/32 + ProbVLM（COCO 训练）**：
  - i2t on COCO：$S = -0.99$，$R^2 = 0.93$，$-SR^2 = 0.93$
  - t2i on FLO（跨数据集）：$S = -0.99$，$-SR^2 = 0.99$（近乎完美校准）
- **BLIP + ProbVLM（COCO 训练）**：
  - i2t on COCO：$-SR^2 = 0.80$，显著优于 PFE（0.58）、PCME（0.62）、TTDA（0.29）
- **Calibration 对比**：在所有数据集和 VLM 上，ProbVLM 的 $-SR^2$ 全面超过 PFE、PCME 和 TTDA 基线，且单调性能下降趋势最为一致。
- **数据效率**：仅用 50% COCO 数据即可获得满意的校准效果。
- **消融**：$\lambda_{\text{cross}}$ 需非零以保证有意义的不确定性；过大反而损害 identity reconstruction。
- **鲁棒性测试**：随图像/文本遮挡比例增加，不确定性单调递增。

### 下游应用
- **主动学习**：在 FLO 上用 ProbVLM 不确定性选取 top-k 样本微调 CLIP，显著优于随机采样（图 7）。
- **模型选择**：仅用目标域无标签数据的不确定性即可选出最优预训练模型（表 2），CUB/Flickr/COCO 上均选中最优模型，FLO 上选出次优（R@1: 47.9 vs 49.5）。

## 相关工作脉络
1. **CLIP / BLIP**：确定性 VLM 的代表性工作，提供高质量冻结嵌入但无不确定性建模。本文在其基础上进行轻量适配。
2. **Probabilistic Face Embedding (PFE)**：首个在冻结模型上估计概率嵌入的工作，但仅适用于单模态（图像），本文为首个跨模态 VLM 后验概率嵌入方法。
3. **PCME（Probabilistic Embeddings for Cross-modal Retrieval）**：采用概率对比损失学习跨模态分布，但需从头训练，依赖大规模数据；本文通过后验适配规避此限制。
4. **Test-Time Data Augmentation (TTDA)**：通过测试时扰动估计不确定性（无训练），但精度有限；本文 ProbVLM 仅需少量数据微调适配器即可达到更好校准。
5. **Bayesian Deep Learning 不确定性估计**（Deep Ensembles, MC Dropout）：主要针对分类/回归任务，且多用于单模态；本文将其思想迁移至跨模态 VLM 的嵌入空间不确定性。
6. **Latent Diffusion / Stable Diffusion**：本文首次利用其文本到图像生成能力反向验证概率嵌入分布的语义一致性。

## 局限性与未来方向
- **GGD 分布假设的局限性**：嵌入分布未必严格服从 GGD，尤其在高维空间中可能存在多峰复杂结构，当前模型可能欠拟合。
- **Cross-modal 对齐的近似性**：Equation 6 的近似（取两个方向的对数似然平均）虽可微且高效，但与精确的 Bivariate Fox H-function 存在误差，可能影响校准精度。
- **适配器规模与泛化**：当前 MLP 结构简单（256 隐藏层），对于更高分辨率或更复杂的 VLM（如 Flamingo、LLaVA）是否仍有效尚待验证。
- **仅估计 aleatoric 不确定性**：epistemic 部分依赖 dropout approximation（Gal & Ghahramani, 2016），在深度 VLM 场景下的可靠性需进一步检验。
- **未来方向**：可扩展至更多 VLM 架构（LLaVA、Flamingo）、探索更灵活的分布族（如 mixture of GGD）、将不确定性用于 OOD 检测或安全关键任务。

## 研究启发与可借鉴点
1. **"后验适配"范式**：在冻结大规模预训练模型上仅添加轻量适配器来扩展其功能（此处为不确定性估计），避免从头训练，是一种资源高效的通用策略，可迁移至其他领域（如冻结 LLM + 概率适配器）。
2. **GGD 用于异方差建模**：用形状参数 $\hat{\beta}$ 灵活捕捉重尾特性，比固定高斯方差更贴合真实场景，建议在不确定性感知的回归/嵌入任务中尝试。
3. **跨模态对齐的对称性设计**：$L_{\text{cross}}$ 中两个方向的负对数似然对称相加，体现了"图像→文本"和"文本→图像"双向一致性，可作为跨模态对齐损失设计的参考模板。
4. **Dropout 推理时开启估计 epistemic 不确定性**：无需额外模型或 ensembles，成本低廉，适合部署于已有 VLM 系统。
5. **扩散模型逆向验证嵌入分布**：将采样嵌入输入 Stable Diffusion 可视化，为嵌入空间不确定性提供了直观可解释的检验手段，可作为未来工作的标准诊断工具。

## 关键术语表
- **ProbVLM**：本文提出的概率适配器，在冻结 VLM 编码器后附加轻量 MLP，将确定性嵌入转换为概率嵌入。
- **广义高斯分布（GGD）**：由均值、尺度 $\hat{\alpha}$ 和形状 $\hat{\beta}$ 参数化的分布，能灵活建模高斯（$\beta=2$）到拉普拉斯（$\beta=1$）及重尾分布。
- **Aleatoric 不确定性**：数据固有的随机性导致的不确定性，由 GGD 的尺度参数解析计算。
- **Epistemic 不确定性**：模型认知不确定性，通过推理时开启 dropout 多次前向传播的方差估计。
- **Intra-modal Alignment**：内模态对齐，约束 ProbVLM 预测分布的均值忠实于冻结编码器输出的重建损失。
- **Cross-modal Alignment**：交叉模态对齐，约束同概念图像/文本的分布相互靠近的损失项。
- **Calibration（校准）**：不确定性估计与实际性能之间的匹配程度，本文用 $-SR^2$ 综合度量，越接近 1.0 越好。
- **Post-hoc Adapter**：事后适配器，指在不修改预训练模型权重的情况下，在其后添加的可训练模块。

## 可复现要素
- **数据集**：MS-COCO、Flickr30k、CUB、Oxford-Flowers 102（均为公开数据集）
- **代码**：已开源，https://github.com/ExplainableML/ProbVLM
- **预训练模型**：CLIP（ViT-B/32, ViT-B/16, ResNet50）和 BLIP（公开可用）
- **关键超参**：MLP 隐藏层维度 256，训练 100 epochs，学习率 $1e^{-4}$，dropout=0.1；$\lambda_{\text{cross}}$ 需调优（论文未给出具体值，见 supplementary）
- **Stable Diffusion**：用于可视化，为开源模型
