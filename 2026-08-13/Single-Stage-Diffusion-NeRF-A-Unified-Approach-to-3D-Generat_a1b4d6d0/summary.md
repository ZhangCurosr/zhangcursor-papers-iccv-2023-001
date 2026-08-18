---
title: "Single-Stage-Diffusion-NeRF-A-Unified-Approach-to-3D-Generat"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Single-Stage_Diffusion_NeRF_A_Unified_Approach_to_3D_Generation_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:31:19"
field: "3D 生成与重建"
keywords: ["3D Generation", "NeRF", "Diffusion Model", "Single-stage Training", "Sparse-view Reconstruction", "Latent Diffusion", "Triplane"]
innovations: ["提出单阶段端到端训练范式联合优化 NeRF auto-decoder 与 triplane 扩散先验", "设计 guidance-finetuning 测试时策略支持任意视角数的 3D 重建", "实现从每场景仅 3 个视图的稀疏数据中有效训练扩散 NeRF"]
benchmarks: ["SRN Cars", "SRN Chairs", "ABO Tables"]
---

# 论文速读：Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction

## 一句话总结
SSDNeRF 提出了一种端到端的单阶段训练范式，将 NeRF auto-decoder 与 triplane 潜扩散模型联合优化，统一了无条件 3D 生成与任意视角数（从单视图到稀疏视图）的 3D 重建任务。

## 研究问题与动机
1. **现有任务专用方法的割裂性**：NeRF 通过逐场景拟合实现高质量新视角合成，但难以泛化到稀疏输入；前馈式稀疏视图重建方法无法推理遮挡区域的歧义，生成结果模糊。
2. **两阶段 Diffusion NeRF 的缺陷**：已有工作分两阶段训练——先训练 VAE/auto-decoder 获得潜码，再将潜码视为真实数据训练扩散模型。但逆渲染的不确定性会导致潜码含有噪声模式，使扩散模型难以学习干净的流形；且稀疏视图下无法可靠训练两阶段模型。
3. **3D GAN 的跨视角推理局限**：3D GAN 依赖单图像判别器，无法有效推理跨视角关系，难以从多视图数据中学习复杂几何结构。
4. **缺乏统一框架**：尚无方法能同时在无条件生成、单视图/稀疏视图重建等多个 3D 任务上达到 SOTA 水平。

## 核心贡献（创新点）
1. **提出 SSDNeRF 统一框架**：将 triplane NeRF auto-decoder 与 triplane 潜扩散模型结合，统一支持无条件 3D 生成与基于图像的稀疏/多视图重建。*与先前任务专用方法相比，SSDNeRF 在一个模型中同时实现了生成与重建两种能力。*
2. **首创单阶段端到端训练范式**：设计联合优化目标，同时更新 NeRF 解码器权重、扩散模型权重和各场景潜码，避免了两阶段训练中的噪声累积问题。*与 DiffRF、Functa 等两阶段方法相比，该方法从根本上消除了先训练 auto-decoder 再训练扩散模型带来的潜码污染。*
3. **提出 guidance-finetuning 采样方案**：在测试时通过渲染梯度引导扩散采样，并结合扩散先验对采样潜码进行 finetuning，支持从任意数量视图到密集视图的灵活重建。*与仅使用渲染损失 finetuning 的 NeRF 方法相比，引入扩散先验显著增强了对未见视角的泛化能力。*
4. **实现稀疏视图（仅 3 视角）下的有效训练**：单阶段训练使得模型可以从每场景仅 3 个视图的数据中学习，这在之前的两阶段方法中是不可行的。

## 方法详解
**整体框架**：SSDNeRF 采用 triplane NeRF 表示，包含可学习的场景潜码 $x_i$（三平面特征）、共享的 NeRF 解码器 $\psi$ 和共享的 triplane 扩散先验 $\phi$。

**单阶段训练目标**（§4.1）：
- 基于变分上界推导，忽略潜码方差项后得到简化损失：
$$\mathcal{L} = \lambda_{\mathrm{rend}} \mathcal{L}_{\mathrm{rend}}(\{x_i\}, \psi) + \lambda_{\mathrm{diff}} \mathcal{L}_{\mathrm{diff}}(\{x_i\}, \phi)$$
- $\mathcal{L}_{\mathrm{rend}}$ 为渲染 L2 损失：$\mathbb{E}[\sum_j \frac{1}{2}\|y_{ij}^{\mathrm{gt}} - y_\psi(x_i, r_{ij}^{\mathrm{gt}})\|^2]$
- $\mathcal{L}_{\mathrm{diff}}$ 为扩散去噪损失：$\mathbb{E}_{i,t,\epsilon}[\frac{1}{2}w^{(t)}\|\hat{x}_\phi(x_i^{(t)}, t) - x_i\|^2]$
- **自适应权重机制**：$\lambda_{\mathrm{diff}} = c_{\mathrm{diff}} / EMA(\|x_i\|_F^2)$，$\lambda_{\mathrm{rend}} = c_{\mathrm{rend}}(1 - e^{-0.1 N_v}) / N_v$，其中 $N_v$ 为可用视图数。

**测试时图像引导采样与 Finetuning**（§4.2）：
- **引导采样**：计算渲染损失对噪声潜码的梯度 $g$，结合无条件 score prediction 修正去噪输出：$\hat{x} \leftarrow \hat{x} - \lambda_{\mathrm{gd}} \frac{\sigma^{(t)^2}}{\alpha^{(t)}} g$，采用 predictor-corrector sampler（DDIM + Langevin 校正）。
- **Finetuning**：冻结扩散和解码器参数，仅优化场景潜码 $x$：$\min_x \lambda_{\mathrm{rend}} \mathcal{L}_{\mathrm{rend}}(x) + \lambda'_{\mathrm{diff}} \mathcal{L}_{\mathrm{diff}}(x)$，其中 $\lambda'_{\mathrm{diff}} < \lambda_{\mathrm{diff}}$（训练权重）。
- **Prior Gradient Caching**（§4.3）：为加速优化，缓存扩散先验梯度在多轮 Adam 步骤中复用，每步仅刷新渲染梯度，减少扩散网络的调用次数。

**网络架构**（§4.3）：
- 去噪网络为 122M 参数的 U-Net，输入/输出为 triplane 特征（三平面通道堆叠）。
- 采用 v-parameterization $\hat{v}_\phi(x^{(t)}, t)$，其中 $\hat{x} = \alpha^{(t)} x^{(t)} - \sigma^{(t)} \hat{v}$。
- 使用 SNR-based 加权 $w^{(t)} = (\alpha^{(t)}/\sigma^{(t)})^{2\omega}$ 替代 LSGM 的双机制加权。

## 实验与结果
**数据集**：
- **SRN Cars**：训练集 2458 场景（每场景 50 随机视图），测试集 703 场景（每场景 251 螺旋视图）；Cars 类别侧重纹理细节生成。
- **SRN Chairs**：训练集 4612 场景，测试集 1317 场景。
- **ABO Tables**：训练集 1520 场景（每场景 91 视图），测试集 156 场景；侧重几何多样性和真实材质。
- 所有渲染图 resize 至 128×128，使用提供的 GT 相机位姿。

**无条件生成**（Table 1）：
- SRN Cars：SSDNeRF (1-stage) FID = $11.08 \pm 1.11$，KID/10⁻³ = $3.47 \pm 0.23$，显著优于 EG3D（KID 4.90*）和 Functa（FID 80.3）。
- ABO Tables：FID = $14.27 \pm 0.66$，KID/10⁻³ = $4.08 \pm 0.33$，优于 EG3D（FID 31.18§）和 DiffRF（FID 27.06）。
- 单阶段 vs 两阶段：KID 从 6.38 降至 3.47（Cars），提升显著。

**稀疏视图重建**（Table 2，SRN Cars & Chairs）：
- **1-view Cars**：PSNR=23.52，SSIM=0.91，LPIPS=0.078，FID=16.39 — LPIPS 最优。
- **2-view Cars**：PSNR=26.49，SSIM=0.94，LPIPS=0.054，FID=10.66 — 全部指标最优。
- **1-view Chairs**：PSNR=24.35，SSIM=0.93，LPIPS=0.067，FID=10.13 — LPIPS 最优。
- **2-view Chairs**：PSNR=26.94，SSIM=0.95，LPIPS=0.055，FID=10.85 — 全部指标最优。
- 消融实验（Table 3）：完整方法（A0）在全部指标上优于无 finetuning（A3）、无扩散先验 finetuning（A2）和两阶段训练（A1）。

**稀疏训练实验**（§5.4，每场景仅 3 视图）：
- 无条件生成：FID = $19.04 \pm 1.10$，KID/10⁻³ = $8.28 \pm 0.60$。
- 单视图重建：LPIPS = 0.106，优于 Table 2 中多数使用全量数据集的方法。
- 对比表明，两阶段 + TV 正则化在 3 视图下会产生严重几何伪影，而单阶段训练可以成功学习。

**稀疏到密集视图插值**（Fig.6）：SSDNeRF 在 1-32 视图设置下全面超越 triplane NeRF baseline 和 CodeNeRF，尤其在 1-4 视图时优势明显。

## 相关工作脉络
1. **3D GANs**（EG3D、GIRAFFE、pi-GAN）：使用 2D 图像判别器，无法推理跨视角关系，主要用于无条件生成；SSDNeRF 用 3D 扩散先验替代判别器，同时支持生成和重建。
2. **View-Conditioned Regression**（PixelNeRF、SRN、VisionNeRF）：前馈编码图像到体特征进行监督渲染，无法处理遮挡歧义、结果模糊；SSDNeRF 引入扩散先验生成锐利且合理的填充内容。
3. **DiffRF**（两阶段 Diffusion NeRF）：先训练 auto-decoder，再将潜码作为真实数据训练扩散模型；SSDNeRF 通过单阶段联合训练避免了潜码噪声污染。
4. **3D Neural Field Generation**（Functa、Gaudi、3ddiffusion）：多采用低维潜码或两阶段训练；SSDNeRF 使用高空间分辨率的 triplane 潜码并在单阶段中联合优化。
5. **NeRF Distillation 方法**（NerfDiff、SparseFusion）：将 2D 扩散先验蒸馏入 NeRF 以施加 3D 约束；SSDNeRF 直接在 3D 潜空间建模扩散先验，本质上是 3D 而非图像空间的跨视角建模。
6. **TV Regularization 方法**（3D Neural Field Generation w/ TV）：在 auto-decoder 上施加全变分正则以平滑潜码；SSDNeRF 用扩散先验自然替代了人工正则，保留了更多纹理细节。

## 局限性与未来方向
1. **依赖 GT 相机位姿**：训练和测试均需已知精确相机参数，未来需探索变换不变（transform-invariant）模型。
2. **长训练导致先验不连续**：用于无条件生成的长训练 schedule（1M iter）会使扩散先验变得不连续，影响泛化；目前通过 early stopping 临时缓解，但需更根本的网络设计或更大数据集来解决。
3. **未涉及文本/语言条件**：当前模型仅支持无条件生成和图像条件重建，尚未与文本描述等条件结合。
4. **仅评估了单物体场景**：在复杂多物体场景或真实世界场景上的泛化能力有待验证。

## 研究启发与可借鉴点
1. **单阶段联合训练范式可迁移**：将扩散先验与下游表示（如 NeRF auto-decoder）端到端联合优化的思路，可扩展到其他隐式表示（如 SDF、特征场）与扩散模型的结合场景。
2. **自适应权重设计值得借鉴**：基于 EMA 归一化扩散损失、基于视图数校准渲染损失的权重策略，为解决多任务/多损失梯度尺度不一致问题提供了实用方案。
3. **Guidance-finetuning 测试时策略**：将渲染梯度作为 diffusion sampling 的引导信号，再配合扩散先验 loss 的 finetuning，这一"采样+优化"两阶段测试策略可推广到其他 generative reconstruction 任务。
4. **Prior Gradient Caching 效率技巧**：缓存慢速计算的先验梯度、高频刷新渲染梯度的策略，对任何需要反复优化潜码的扩散-生成联合模型都有直接的效率提升价值。
5. **稀疏视图训练的可行性**：证明了单阶段训练可使模型从每场景仅 3 个视图数据中有效学习，这为数据稀缺场景下的 3D 生成-重建联合学习开辟了新方向。

## 关键术语表
**SSDNeRF**：Single-Stage Diffusion NeRF 的缩写，本文提出的统一框架，通过单阶段训练联合学习 NeRF auto-decoder 与 triplane 扩散先验。

**Auto-decoder**：一种多场景 NeRF 训练范式，共享解码器参数、为每个场景学习独立的潜码，潜码可视为 latent representation。

**Triplane**：由 three 个 2D feature planes 组成的 3D 场景表示，每个平面沿 X、Y、Z 轴投影，通过平面采样与融合解码为 NeRF 的密度和颜色。

**Latent Diffusion Model (LDM)**：在潜空间而非像素空间操作的去噪扩散模型，通过训练去噪网络学习潜变量的分布 prior。

**Guidance-finetuning**：测试时先用渲染梯度引导扩散采样过程生成初始潜码，再冻结模型参数仅优化潜码以同时满足扩散先验和渲染约束。

**Score Distillation**：将扩散模型的 NLL 近似为去噪 L2 损失，等价于对扩散 prior 的 score function 进行采样蒸馏，此处用于替换隐式 prior 项。

**Predictor-corrector sampler**：结合 DDIM 确定性去噪步骤与多个 Langevin 校正步骤的扩散采样器，可在采样过程中嵌入额外引导信号。

**SNR-based weighting**：基于信噪比 $(\alpha^{(t)}/\sigma^{(t)})^{2\omega}$ 的扩散损失加权策略，替代 LSGM 的双机制加权，在 NeRF auto-decoder 场景下更稳定。

## 可复现要素
- **数据集**：SRN（ShapeNet Radiance Fields）和 ABO Tables，均为公开数据集；SRN 提供 GT 渲染图和相机位姿，ABO Tables 同。
- **代码/权重**：论文未提及开源状态（需进一步确认）。
- **关键超参**：去噪网络 122M 参数 U-Net；Cars 训练 1M iterations（无条件生成）/ 80K iterations（稀疏重建）；$\omega = 0.5$ 或 $0.25$；test-time $\lambda'_{\mathrm{diff}}$ 低于训练权重；稀疏训练时使用 mid-training code reset 技巧并双倍训练时长。
