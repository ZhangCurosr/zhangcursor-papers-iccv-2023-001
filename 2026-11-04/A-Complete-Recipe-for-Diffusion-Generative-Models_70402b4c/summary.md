---
title: "A-Complete-Recipe-for-Diffusion-Generative-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Pandey_A_Complete_Recipe_for_Diffusion_Generative_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:33:44"
---

# 论文速读：A-Complete-Recipe-for-Diffusion-Generative-Models

## 一句话总结
本文受贝叶斯后验采样器设计启发，提出了一套系统化构建扩散过程正向SDE的“完整配方”，并据此设计了Phase Space Langevin Diffusion（PSLD）模型，通过在增强相空间（数据位置+动量/辅助变量）中同时注入独立噪声，在CIFAR-10、CelebA-64等图像生成基准上实现了更优的样本质量与速度-质量权衡。

## 研究问题与动机
- 现有Score-based Generative Models (SGMs) 的正向扩散过程设计主要依赖物理直觉或简化假设，缺乏普适性的理论设计框架。
- 已有引入辅助变量的工作（如CLD）设计动机偏物理化，难以系统化推广且泛化边界不清晰。
- 扩散模型的前向过程参数化空间尚未被充分探索，缺少一种能保证渐近收敛至指定先验分布且兼容Denoising Score Matching的构造范式。
- 如何在保证扰动核可解析计算的前提下，通过合理的空间扩展与噪声解耦机制，同步提升生成质量与低步数采样效率。

## 核心贡献（创新点）
1. **完整配方（Complete Recipe）**：基于MCMC理论提出扩散过程漂移项的参数化形式，证明在指定目标平稳分布下，所有收敛到该分布的马尔可夫随机过程均可由该配方生成，具备理论完备性。
2. **Phase Space Langevin Diffusion (PSLD)**：将配方实例化为在数据空间与动量空间同时加噪的扩散过程，严格推广了CLD（允许Γ>0），并推导出PSLD下的临界阻尼条件 `(Γ-ν)² = 4M⁻¹`。
3. **一致的快质权衡优势**：在固定NFE预算与多种SDE/ODE求解器设置下，PSLD均显著优于VP-SDE与CLD基线，低NFE区间优势尤为突出。
4. **SOTA级无条件生成质量**：未依赖Mixed Score等额外技巧，PSLD在CIFAR-10上达到FID 2.10，CelebA-64上达到FID 2.01，与当前最优方法相当。
5. **零重训条件生成能力**：证明预训练的无条件PSLD模型可直接用于类别条件生成与图像修复，推理时仅需附加分类器梯度修正，且修复质量优于CLD。

## 方法详解
- **增强相空间配方**：将状态扩展为 `z = [x, m] ∈ R^{2d}`，定义Hamiltonian `H(z) = U(x) + mᵀM⁻¹m/2`。基于Yin等定理，漂移项参数化为 `f(z) = -(D+Q)∇H + τ(z)`，其中 `D` 为正定扩散矩阵，`Q` 为斜对称curl矩阵。该参数化保证过程渐近收敛至目标平稳分布 `p_s(z)`。
- **PSLD前向SDE**：选取常数矩阵 `D` 与 `Q`，使漂移为仿射形式，扰动核 `p(z_t|z_0)` 可解析计算。扩散系数 `G(t)` 在位置与动量维度独立，由标量 `Γ, ν, M, β` 控制；`Γ=0` 时退化为CLD，`Γ>0` 时数据空间额外加噪。临界阻尼条件推广为 `(Γ-ν)² = 4M⁻¹`。
- **训练目标**：采用Hybrid Score Matching (HSM) 对动量变量边缘化，推导得ε预测损失：
  `min_θ E_{t,x₀,ε}[ ||ε_θ(μ_t + L_t ε, t) - ε||² ]`。
  与CLD不同，PSLD需预测完整的2d维ε，score网络输出通道翻倍，但参数增量可忽略。
- **反向采样器**：导出对应反向SDE，并将对称分裂积分器SSCS扩展至PSLD（保留原收敛阶数）；同时支持Euler-Maruyama (EM) 与概率流ODE (RK45) 求解。条件生成时利用 `∇log p(z_t|y) ≈ ∇log p(y|z_t) + s_θ(z_t,t)` 进行梯度修正采样。

## 实验与结果
- **数据集**：CIFAR-10 (32×32)、CelebA-64、AFHQv2 (128×128)。
- **评估基线**：VP-SDE、CLD、DDPM、iDDPM、DiffuseVAE、LSGM、EDM、VESDE等。
- **主要数值结果**：
  - **SDE (NFE=1000)**：39M参数PSLD(Γ=0.02) FID=2.80；55M参数PSLD(Γ=0.01) FID=2.34；97M参数PSLD(Γ=0.02) FID=2.21，全面超越CLD(w/o MS)的2.41与VP-SDE(deep)的2.46。
  - **ODE (NFE≈150-250)**：PSLD(Γ=0.01, 97M) 达FID=2.10，媲美LSGM但参数量仅其1/4；低NFE=50时PSLD(Γ=0.02) FID=16.12，优于VP-SDE(17.72)与CLD(21.31)。
  - **CelebA-64**：PSLD(Γ=0.005, 250 NFE) 取得FID=2.01，大幅领先VP-SDE的2.32。
  - **Γ消融**：Γ∈[0.005, 0.02]时性能最优；Γ=0.25导致高频细节丢失、FID激增至9.48(CIFAR-10)与93.99(CelebA-64)。
  - **条件生成**：AFHQv2图像修复任务中，PSLD(FID_train=0.85, FID_test=6.93)优于CLD(FID_train=1.01, FID_test=7.10
