---
title: "AdvDiffuser-Natural-Adversarial-Example-Synthesis-with-Diffu"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:04:15"
---

# 论文速读：AdvDiffuser: Natural Adversarial Example Synthesis with Diffusion Models

## 一句话总结
提出 AdvDiffuser，首次将预训练扩散模型的反向去噪过程与对抗梯度引导相结合，高效生成自然、高保真的无约束对抗样本（UAEs）；在 CIFAR-10、CelebA 和 ImageNet 上实现对 RobustBench 顶级鲁棒模型近 100% 的攻击成功率，且感知质量全面超越现有 SOTA。

## 研究问题与动机
- 传统 $\ell_p$ 有界攻击无法准确刻画人类视觉对扰动的感知，在实际场景中的威胁评估价值有限。
- 现有无约束对抗样本生成方法多依赖 GAN/VAE 扰动隐空间编码，易破坏高层语义，导致生成样本外观扭曲、不自然。
- 基于感知距离或几何代理模型的攻击需要主观设定距离度量或选择验证集模型，泛化性与自动性不足。
- 缺乏可扩展的自然对抗样本生成机制，难以充分支撑防御方的鲁棒性训练与跨威胁模型评估。

## 核心贡献（创新点）
- **首次将扩散模型引入自然对抗样本合成**：利用扩散模型高保真采样能力直接从数据分布生成 UAEs，而非扰动 GAN/VAE 隐变量，从根本上避免了高层语义丢失与视觉伪影。
- **提出对抗引导（Adversarial Guidance）机制**：在每一步反向去噪中注入 $\ell_2$ 有界 PGD 扰动，使扩散轨迹动态偏向目标分类器的对抗样本空间，区别于传统固定预算梯度攻击依赖单一优化步。
- **提出基于 Grad-CAM 的对抗修补（Adversarial Inpainting）**：按像素级显著性动态调节去噪强度，在关键对象区域保留原始语义、在非关键区域叠加扰动，实现“语义保持+对抗增强”的双重目标。
- **验证扩散生成样本对未见威胁模型的泛化防御价值**：证明基于 AdvDiffuser 合成的 UAEs 进行对抗训练，可显著提升模型面对 JPEG 干扰、ReColorAdv、LPA 等未知攻击时的鲁棒性，突破传统 $\ell_p$ 训练的对齐局限。
