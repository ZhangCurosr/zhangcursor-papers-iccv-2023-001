---
title: "AdvDiffuser-Natural-Adversarial-Example-Synthesis-with-Diffu"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:40:04"
field: "对抗样本生成与鲁棒性评估"
keywords: ["adversarial examples", "diffusion models", "unrestricted adversarial examples", "adversarial inpainting", "RobustBench", "Grad-CAM"]
innovations: ["首次利用扩散模型从scratch或基于参考图像合成自然无限制对抗样本", "提出对抗性指导机制在去噪过程中注入PGD梯度扰动", "引入基于Grad-CAM的对抗性修复保留显著区域语义"]
benchmarks: ["CIFAR-10", "CelebA", "ImageNet", "RobustBench"]
---

# 论文速读：AdvDiffuser-Natural-Adversarial-Example-Synthesis-with-Diff

## 一句话总结
本文提出 AdvDiffuser，首次利用扩散模型（Diffusion Models）从 scratch 或基于参考图像合成自然无限制对抗样本（UAE），通过在反向去噪过程中注入对抗性梯度扰动并结合 Grad-CAM 引导的对抗性修复，生成的样本在攻击成功率与感知隐蔽性上均显著优于现有 SOTA 方法。

## 研究问题与动机
- **现有对抗样本的感知缺陷**：传统 $\ell_p$ 有界攻击无法准确模拟人类对扰动的感知，且受限扰动预算在真实场景中威胁有限；而无限制对抗样本（UAE）需在不影响语义的前提下大幅改变图像。
- **GAN/VAE 方法的语义丢失问题**：已有 UAE 生成方法通过扰动潜在编码（latent code）来攻击分类器，但会改变生成图像的高层语义，导致视觉上模糊、不自然的对抗样本。
- **梯度/几何感知攻击的主观依赖**：感知攻击依赖 LPIPS、SSIM 等距离度量，几何感知攻击需选择代理模型，均需要主观先验知识，难以保证生成样本的真实感。
- **缺乏可扩展的自然对抗样本生成器**：现有方法难以生成大量多样且自然的对抗样本，无法为鲁棒性训练与评估提供充足数据支持。

## 核心贡献（创新点）
- **首次将扩散模型引入自然对抗样本合成**：与 GAN/VAE 扰动潜在空间的方法本质不同，AdvDiffuser 直接在像素空间迭代去噪过程中注入对抗扰动，保留语义一致性与视觉多样性。
- **提出对抗性指导（Adversarial Guidance）机制**：在扩散模型每一步反向去噪后应用 $\ell_2$ 有界 PGD 攻击，使生成的样本既服从数据分布又针对目标分类器失效。
- **引入基于 Grad-CAM 的对抗性修复（Adversarial Inpainting）**：通过类激活映射定位图像显著区域，在去噪过程中对显著区域保留原始噪声状态、对非显著区域施加对抗扰动，实现语义保持的对抗样本生成。
- **实验验证全面领先**：在 CIFAR-10、CelebA 和 ImageNet 上对 RobustBench 顶级鲁棒模型实现近 100% 攻击成功率，且在 LPIPS、FID、SSIM 等感知指标上均优于 GA-attack 等 SOTA 方法。

## 方法详解
- **整体流程**：从纯高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，迭代 $T$ 步反向去噪；每步先由预训练扩散模型预测去噪结果，再施加 PGD 对抗扰动，最后通过 Grad-CAM 掩码融合显著区域与非显著区域的输出。
- **对抗性指导公式（5）**：每步求解 saddle-point 问题，在扩散模型负对数似然最小化与分类器对抗损失最大化之间平衡：
  $$\hat{\mathbf{x}}_{t-1} = \operatorname{argmax}_{\mathbf{x}} \mathsf{L}(f(\mathbf{x}), y), \quad \text{s.t. } \operatorname{dist}(\mathbf{x}, \tilde{\mathbf{x}}_{t-1}) \leq \varepsilon_t$$
  其中 $\tilde{\mathbf{x}}_{t-1}$ 来自扩散模型的条件采样 $p_\theta(\mathbf{x}_{t-1}|\hat{\mathbf{x}}_t)$。
- **PGD 对抗扰动（6）**：使用归一化 softmax 交叉熵（SCE）损失，以 $\ell_2$ 投影约束迭代 $I$ 步：
  $$\mathbf{z}_{i+1} = \mathcal{P}_{\mathbf{z}_0, \varepsilon}\left(\mathbf{z}_i + \operatorname{sign}(\nabla_{\mathbf{z}_i} \mathsf{L}(f(\mathbf{z}_i, y)))\right)$$
  扰动预算设为 $\varepsilon_t = \sigma \beta_t$，随噪声调度递减以确保自然性。
- **对抗性修复融合（8）**：利用 Grad-CAM 得到显著区域掩码 $\mathbf{m}$，每步去噪后融合：
  $$\mathbf{x}_{t-1} = \mathbf{m} \odot \mathbf{x}_{t-1}^{\mathrm{obj}} + (\mathbf{1} - \mathbf{m}) \odot \hat{\mathbf{x}}_{t-1}$$
  其中 $\mathbf{x}_{t-1}^{\mathrm{obj}}$ 是来自参考图像加噪的版本，$\hat{\mathbf{x}}_{t-1}$ 是 PGD 攻击后的结果。

## 实验与结果
- **数据集与模型**：CIFAR-10、CelebA、ImageNet；使用 RobustBench 顶级鲁棒模型（Rebuffi et al. A/B、Gowal et al.、Salman et al. A/B、Engstrom et al.）。
- **从零生成（CelebA）**：相比 AC-GAN，AdvDiffuser 攻击成功率 99.1% vs 91.1%，FID 8.4 vs 15.6，生成速度 12.8s vs 23.6s/image。
- **ImageNet 攻击**：对 Salman et al. A（已知 $\ell_\infty=4/255$ 鲁棒准确率 38.1%），AdvDiffuser 降至 0.5%，而 GA-PGD 为 2.5%、GA-FSA 为 5.5%；AdvDiffuser 的 LPIPS（0.05）、SSIM（0.97）、FID（26.7）均显著优于 GA 变体。
- **感知指标对比**：相比 GA-attack，AdvDiffuser 生成的 UAE 具有约 6 倍更小的 LPIPS 扰动、约 23 更小的 FID 分数、约 0.28 更高的 SSIM，感知隐蔽性更强。
- **未知威胁模型鲁棒性**：在 CIFAR-10 上用 AdvDiffuser 生成的 UAE/SAE/Mixed 进行对抗训练后，模型在 $\ell_\infty$、$\ell_2$、JPEG 腐蚀、ReColorAdv、LPA、StAdv 等六种威胁下均表现优于 Engstrom 的 $\ell_2$ 对抗训练基线。

## 相关工作脉络
- **AC-GAN [38] / AdvGAN [43]**：基于生成模型扰动潜在空间生成对抗样本，易丢失高层语义；AdvDiffuser 直接在像素空间操作，避免此类问题。
- **GA-attack [20]**：2021 CVPR 无限制对抗攻击竞赛冠军，使用代理模型优化 $\ell_p$ 预算；AdvDiffuser 无需代理模型，直接利用扩散模型的数据分布先验。
- **LPA [19] / Perceptual Attacks**：基于 LPIPS/SSIM 等感知距离优化；AdvDiffuser 通过扩散去噪隐式学习感知分布，无需人工设计距离度量。
- **DiffPure [25]**：用扩散模型净化对抗扰动以增强防御；本文反向利用扩散模型生成对抗样本，体现扩散模型在攻防两端的双重价值。
- **Rebuffi et al. [29]**：证明 DDPM 生成的数据可作为数据增强提升鲁棒性；本文进一步证明可直接生成对抗样本用于训练。

## 局限性与未来方向
- **白盒假设限制**：实验均在白盒设置下进行，未评估黑盒迁移攻击性能。
- **计算开销**：扩散模型反向采样需要多步迭代（CIFAR-10 用 100 步、ImageNet 用 400 步），生成速度仍慢于梯度直接优化方法。
- **未探索跨域泛化**：仅在图像分类任务验证，对视频、3D 点云等其他模态的适用性待研究。
- **防御对抗性**：DiffPure 等基于扩散的净化方法可能与 AdvDiffuser 形成对抗循环，未来可研究攻防动态博弈。

## 研究启发与可借鉴点
- **扩散模型 + 对抗扰动的结合范式**：可在去噪过程的任意步骤注入梯度信号，这一思路可迁移到其他生成模型（如 VQ-VAE、一致性模型）的对抗样本生成中。
- **Grad-CAM 掩码的语义保护机制**：利用类激活映射区分显著/非显著区域并差异化处理，可推广至图像编辑、风格迁移等任务中的语义保持。
- **无监督威胁模型的鲁棒训练**：AdvDiffuser 生成的 UAE 不依赖特定 $\ell_p$ 假设，可作为通用对抗增强数据，适用于开放世界鲁棒性评估。
- **感知指标与攻击效果的联合优化**：论文同时优化攻击成功率与 LPIPS/SSIM/FID，为未来研究提供了多目标权衡的实验基准。

## 关键术语表
- **Unrestricted Adversarial Examples (UAE)**：不受 $\ell_p$ 范数约束、但在语义上仍合理的对抗样本，可大幅修改图像内容而不影响人类理解。
- **Diffusion Models**：通过逐步加噪再学习反向去噪的生成模型，能够高质量采样数据分布。
- **Adversarial Guidance**：在扩散模型去噪过程中注入对抗梯度扰动，引导生成样本落入对抗空间。
- **Adversarial Inpainting**：基于 Grad-CAM 掩码对图像显著区域进行修复保护，仅在非显著区域施加对抗扰动。
- **Grad-CAM**：梯度加权类激活映射，用于定位图像中与分类决策相关的显著区域。
- **LPIPS / SSIM / FID**：感知距离与分布相似度指标，分别衡量图像 perceptual 差异、结构相似性和生成质量。
- **RobustBench**：标准化的对抗鲁棒性评测基准平台，收录各类鲁棒模型的评测结果。
- **Classifier-free Guidance**：扩散模型中不使用分类器而直接通过条件/无条件输出差分进行引导的技术。

## 可复现要素
- **数据集**：CIFAR-10、CelebA、ImageNet（均为公开数据集）
- **代码/权重**：论文未明确声明开源，但使用了 OpenAI 的预训练条件 DDPM 模型（ImageNet）及自行复现的 CIFAR-10/CelebA 模型
- **关键超参**：扩散步数 $T=100$（CIFAR-10）/ $T=400$（ImageNet）；对抗引导强度 $\sigma=0.1$（CIFAR-10）/ $\sigma=0.4$（ImageNet）；PGD 迭代次数 $I=1$（CIFAR-10）/ $I=25$（ImageNet）；扰动预算 $\varepsilon_t = \sigma \beta_t$
