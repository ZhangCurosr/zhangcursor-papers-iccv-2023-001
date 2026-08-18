---
title: "Stochastic-Segmentation-with-Conditional-Categorical-Diffusi"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zbinden_Stochastic_Segmentation_with_Conditional_Categorical_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:32:47"
---

# 论文速读：Stochastic Segmentation with Conditional Categorical Diffusion Models

## 一句话总结
本文提出条件类别扩散模型（CCDM），将扩散概率模型从连续像素空间直接扩展至离散标签空间，通过图像条件引导生成多样化的语义分割掩码，有效建模多专家标注带来的Aleatoric Uncertainty；在LIDC多标注医学数据集上取得SOTA，并在Cityscapes通用分割基准上以较少参数超越多个成熟基线。

## 研究问题与动机
- 传统语义分割输出单一确定性掩码，无法反映医学影像、自动驾驶等安全关键领域中存在的标注歧义与专家分歧。
- 学习图像条件下的标签分布面临三大难点：分布强多峰性、逐像素高维离散输出空间、带多标注的训练数据稀缺。
- 现有基于DDPM的分割方法多沿用连续值扩散 formulation，需通过阈值、量化或自编码后处理才能获得类别输出，引入额外误差且难以原生表达离散分布。

## 核心贡献（创新点）
- **提出CCDM框架**：首次在语义分割中完整采用Categorical Diffusion formulation，直接在离散标签空间进行前向加噪与逆向去噪，避免连续-离散域转换的信息损耗。
- **推导适配扩散的逆向转移变换**：给出从网络预测的初始分布 $\hat{\mathbf{P}}_0$ 到任意时间步逆转移参数 $\hat{\mathbf{p}}_{t-1}$ 的闭式计算公式（Eq. 12），保证训练与采样的一致性。
- **LIDC随机分割SOTA**：以仅9M参数在LIDCv1/v2上斩获10项指标中8项第一，显著优于Prob-Unet、MoSE、AB、CIMD等强基线，是首个针对该任务提出的扩散类方法。
- **Cityscapes强竞争力验证**：通过多样本概率平均融合策略，CCDM-Dino在128×256与256×512分辨率下均超越DeepLabv3/HRNet/UPerNet等 heavily engineered 基线，展现良好的跨域泛化能力。

## 方法详解
- **前向扩散过程**：沿用Multinomial Diffusion，逐像素以概率 $\beta_t$ 将当前标签替换为均匀随机标签，复合后任意时刻 $t$ 的分布可闭式采样：$q(x_t|x_0) = \mathcal{C}(x_t; \frac{1-\bar{\alpha}_t}{L}\mathbf{1} + \bar{\alpha}_t \mathbf{e}_{x_0})$。
- **逆向去噪网络与分布平移**：U-Net骨干 $f_\theta(\mathbf{x}_t, t, I)$ 预测初态标签分布 $\hat{\mathbf{P}}_0 \in [0,1]^{D\times L}$。为匹配前向转移，通过 $\hat{\mathbf{p}}_{t-1} = \sum_{x_0} \pi(x_t, x_0) \cdot \hat{\mathbf{p}}_0[x_0]$ 进行逐像素分布平移（Eq. 12），其中 $\pi$ 为后验 $q(x_{t-1}|x_t, x_0)$ 的参数向量。
- **图像条件注入**：提供两种模式：(1) 原始图像像素作为额外通道拼接到输入张量；(2) 使用Dino-ViT提取视觉特征，拼接至U-Net第3层编码器特征图（空间分辨率压缩至原图1/8）。
- **训练目标**：最大化条件ELBO，等价于优化 $-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1, I)$ 项与时间步间逐像素KL散度之和（Alg. 1）。时间步均匀采样，终态KL项因与参数无关而忽略。
- **推理策略**：从均匀噪声 $\mathbf{x}_T$ 开始逆向迭代 $T=250$ 步，末步 $t=1$ 采用 $\text{argmax}$ 获取确定性标签；重复 $N$ 次可采样得到多样化掩码集合，支持多模态不确定性可视化或概率融合。

## 实验与结果
- **数据集与配置**：LIDC（1018例3D CT切片，4位专家标注，提取128×128结节中心切片，LIDCv1/v2双划分）；Cityscapes（2975训练/500验证，19类，128×256与256×512双分辨率）。使用余弦 $\beta_t$ 调度，Adam优化，Polyak平均平滑，扩散步数统一 $T=250$。
- **LIDC结果（表1）**：CCDM在GEO与HM-IoU指标上全面领先。LIDCv1上 $\text{GED}_{16}=0.212$、$\text{GED}_{100}=0.183$、$\text{HM-IoU}_{32}=0.631$ 均为最优；LIDCv2上 $\text{GED}_{16/50/100}$ 与 $\text{HM-IoU}_{16}=0.598$ 同样夺冠。仅 $\text{HM-IoU}_{16}$ 以0.001微弱落后MoSE，但标准差更小（0.002 vs 0.004）。
- **Cityscapes结果（表2）**：CCDM-Dino单样本128×256 mIoU达5
