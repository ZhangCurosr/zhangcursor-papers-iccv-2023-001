---
title: "HSR-Diff-Hyperspectral-Image-Super-Resolution-via-Conditiona"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:09:22"
---

# 论文速读：HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models

## 一句话总结
本文首次将条件扩散模型引入高光谱图像超分辨率（HSI-SR）任务，提出HSR-Diff方法。该方法以纯高斯噪声初始化HR-HSI，通过条件去噪Transformer（CDFormer）在HR-MSI与LR-HSI的层级特征及连续噪声水平指导下迭代去噪，结合渐进式学习策略，在四个公开数据集上均取得SOTA性能。

## 研究问题与动机
1. **核心问题**：当代高光谱传感器受限于信噪比，空间分辨率极低；需将低分辨率高光谱图像（LR-HSI）与高分辨率多光谱图像（HR-MSI）融合，重建具有丰富光谱信息的高分辨率HSI（HR-HSI）。
2. **传统方法不足**： pansharpening扩展、贝叶斯推断、矩阵/张量分解等方法严重依赖手工先验，缺乏对不同HSI结构的泛化灵活性，且张量方法内存与计算开销极大。
3. **深度学习基线局限**：CNN类方法受限于局部感受野，难以建模HSI的全局长程依赖与自相似性；GAN类方法需精心设计正则化，训练不稳定且易出现模式崩溃。
4. **现有Transformer方法缺口**：Fusformer、HMF-Former等虽利用注意力机制，但仅将LR-HSI与HR-MSI直接拼接/相加作为输入，未引入生成式迭代精炼机制，对多解性逆问题的数据分布建模能力有限。
5. **动机**：扩散模型通过可微分的逐步去噪过程逼近目标分布，训练稳定且无需对抗损失；将其与Transformer结合，可充分利用CDFormer的全局建模能力与多层级条件注入，实现高质量的HSI-SR。

## 核心贡献（创新点）
1. **首次将条件扩散模型应用于HSI-SR领域**：将HR-HSI重建建模为从标准正态分布到数据分布的迭代精炼过程，从根本上区别于传统单步回归映射。
2. **提出条件去噪Transformer（CDFormer）**：采用双流架构与Cross-Attention条件机制替代传统的Channel Concatenation，使噪声水平与多源层级特征能够精细调制去噪轨迹。
3. **设计空谱Transformer基础单元（S²TL）**：同步建模空间区域交互与光谱波段相关，通过Transposed Attention与Window Partition策略兼顾表达力与计算效率。
4. **引入渐进式学习策略**：训练前期在高分块上高效优化，后期平滑过渡至全分辨率图像，解决大尺度Transformer难以直接处理全图显存限制的问题。
5. **系统性实验与充分消融**：在CAVE、PaviaU、Chikusei、HypSen四个数据集上全面超越6个SOTA方法，并逐一验证了扩散过程、双流架构与渐进学习的独立贡献。

## 方法详解
- **问题建模**：观测方程为 $\mathbf{X} = \mathbf{R}\mathbf{Z}$（HR-MSI）与 $\mathbf{Y} = \mathbf{Z}\mathbf{D}$（LR-HSI），目标是从观测对 $(\mathbf{X}, \mathbf{Y})$ 恢复潜在HR-HSI $\mathbf{Z}$。
- **条件扩散框架**：
  - **前向过程** $q(\mathbf{Z}_t|\mathbf{Z}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\mathbf{Z}_{t-1}, (1-\alpha_t)\mathbf{I})$ 逐步添加高斯噪声，闭合形式下 $q(\mathbf{Z}_t|\mathbf{Z}_0) = \mathcal{N}(\sqrt{\gamma_t}\mathbf{Z}_0, (1-\gamma_t)\mathbf{I})$，其中 $\gamma_t = \prod_{i=1}^t \alpha_i$。
  - **反向过程** $p_\theta(\mathbf{Z}_{t-1}|\mathbf{Z}_t, \mathbf{X}, \mathbf{Y}) = \mathcal{N}(\mu_\theta, \sigma_t^2\mathbf{I})$，由CDFormer参数化。训练时随机采样 $t \sim U\{1,T\}$ 与 $\gamma \sim U(\gamma_{t-1}, \gamma_t)$，使模型适应连续噪声强度；$T=2000$，推理时简化为100步线性调度。
- **CDFormer架构**：
  - **SR流**：$3\times3$ 卷积提取低层特征 $\mathbf{F}^{SR}_0$，经堆叠的S²TL生成多尺度深层特征 $\mathbf{F}^{SR}_l$。
  - **去噪流**：由多个噪声感知条件S²TL（NC-S²TL）构成。噪声水平 $\gamma$ 通过正弦位置编码（NLE）转化为向量，与去噪流特征 $\mathbf{F}^{DS}_l$ 线性融合后，以 $\mathbf{F}^{SR}_l$ 为Key/Value进行多头交叉注意力（MCA）注入，实现条件调制。
  - **S²TL模块**：包含空间多头自注意力（SpatioMSA，Window Partition降复杂度）、光谱多头自注意力（SpectralMSA，Transposed Attention）及带门控机制的FFN。
  - **重建模块**：残差连接 + $3\times3$ 卷积映射回HR-HSI空间，无下采样。
- **损失函数**：$\mathcal{L} = \|\mathbf{X} - \mathbf{R}\hat{\mathbf{Z}}_0\|_1 + \|\mathbf{Y} - \hat{\mathbf{Z}}_0\mathbf{D}\|_1 + \|\mathbf{Z}_0 - \hat{\mathbf{Z}}_0\|_1$，前两项分别保证MSI与LR-HSI的观测一致性，第三项基于拉普拉斯分布假设约束预测与真值差异。
- **渐进学习**：前期使用 $128^2$（CAVE/Chikusei）或 $64^2$（PaviaU）Patch训练，后期逐步切换至 $512^2$
