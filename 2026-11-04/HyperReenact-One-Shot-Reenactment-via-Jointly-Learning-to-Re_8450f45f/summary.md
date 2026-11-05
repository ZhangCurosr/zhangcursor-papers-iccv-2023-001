---
title: "HyperReenact-One-Shot-Reenactment-via-Jointly-Learning-to-Re"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Bounareli_HyperReenact_One-Shot_Reenactment_via_Jointly_Learning_to_Refine_and_Retarget_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:43:56"
field: "人脸重演与生成"
keywords: ["neural face reenactment", "StyleGAN2", "hypernetwork", "GAN inversion", "one-shot reenactment", "curriculum learning", "identity preservation", "facial pose transfer"]
innovations: ["首次将超网络优化应用于面部重演，联合完成身份精炼与姿态重定向", "三阶段课程学习策略（反演→自主体→跨主体）提升训练稳定性与泛化性", "引入凝视损失改善眼神方向重建，减少极端姿态下的视觉伪影"]
benchmarks: ["VoxCeleb1", "VoxCeleb2"]
---

# 论文速读：HyperReenact-One-Shot-Reenactment-via-Jointly-Learning-to-Refine-and-Retarget-Faces

## 一句话总结
本文提出 HyperReenact，一种基于预训练 StyleGAN2 + 超网络（hypernetwork）的单帧面部重演方法，通过联合学习对源身份特征进行精细化重建与目标面部姿态的重定向，实现自主体和跨主体场景下无伪影的高保真人脸重演。

## 研究问题与动机
- **极端姿态变化下的视觉伪影**：现有 SOTA 方法的合成结果在大角度头姿差异时易产生严重视觉伪影。
- **身份保持依赖昂贵微调**：多数方法需少量样本微调（few-shot fine-tuning）以保留源身份特征，限制了单帧设置下的实用性。
- **外部 GAN 反演方法的局限性**：利用预训练 GAN 的方法依赖外部真实图像反演，面临重建质量差与编辑性不足的两难困境（reconstruction-editability trade-off），且全局编辑时伪影严重。
- **跨主体重演的身份泄漏问题**：基于面部 landmark 的方法易将目标面的身份几何信息泄漏到源面，导致跨主体重演效果不佳。

## 核心贡献（创新点）
1. **首次将超网络优化应用于面部重演任务**：通过超网络修改 StyleGAN2 生成器权重，同步完成身份细节精炼与面部姿态重定向，区别于先前仅利用反演+隐空间方向编辑的工作。
2. **无需微调的单帧自/跨主体重演**：在 one-shot 设置下不依赖任何 subject-specific fine-tuning，即可在极端头姿差异（Euler angle 差 >15°）条件下保持身份并迁移姿态。
3. **课程学习训练策略**：提出三阶段 curriculum learning（真实图像反演 → 自主体重演 → 跨主体重演），显著提升各阶段性能。
4. **引入注视损失（gaze loss）**：新增 gaze loss 项，有效修正初始反演中缺失的眼神方向，定量降低 Gaze Error。

## 方法详解
- **整体架构**：给定源图像 $I_s$ 和目标图像 $I_t$，分别通过外观编码器 $\mathcal{E}_{\mathrm{app}}$（ArcFace）和姿态编码器 $\mathcal{E}_{\mathrm{p}}$（DECA）提取特征 $f_{\mathrm{app}}$（$512 \times 7 \times 7$）和 $f_{\mathrm{p}}$（$2048 \times 7 \times 7$）。
- **重演模块（Reenactment Module, RM）**：受 SPADE 启发，将 $f_{\mathrm{p}}$ 经 $1\times1$ 卷积投影至 $f_{\mathrm{app}}$ 同维度得 $f_{\mathrm{p}}'$，再通过可学习的调制参数 $\gamma$ 和 $\beta$ 融合：
  $$f_{\mathrm{r}} = \gamma_{\mathrm{app}} \odot f_{\mathrm{app}} + \beta_{\mathrm{app}} + \gamma_{\mathrm{p}} \odot f_{\mathrm{p}}' + \beta_{\mathrm{p}}$$
  输出融合特征图 $f_{\mathrm{r}}$（$512 \times 7 \times 7$）。
- **超网络模块（Hypernetwork H）**：由 M 个 Reenactment Block（RB）组成，每个 RB 接收 $f_{\mathrm{r}}$，输出第 $\ell$ 层生成器权重的偏移量 $\Delta\theta_\ell$（尺寸 $C_\ell^{out} \times C_\ell^{in} \times 1 \times 1$），空间重复后得到完整偏移，更新权重为 $\hat{\theta}_\ell = \theta_\ell \cdot (1 + \Delta\theta_\ell)$。
- **训练流程（课程学习三阶段）**：
  - **Phase 1 真实图像反演**：源=目标，优化重建损失 $\mathcal{L}_{pix}$（$\ell_1$）、$\mathcal{L}_{lpips}$（感知损失）、$\mathcal{L}_{id}$（ArcFace 余弦相似度）、$\mathcal{L}_g$（注视方向 L2 距离）。
  - **Phase 2 自主体重演**：同身份不同姿态，额外增加形状损失 $\mathcal{L}_{sh}$（目标与重演图像的 DECA 3D 形状 $\ell_1$ 距离）。
  - **Phase 3 跨主体重演**：在 Phase 2 基础上加入跨主体数据对（每 batch 一半来自同身份、一半来自跨身份），跨主体样本仅计算 $\mathcal{L}_{id}$（源↔重演）和 $\mathcal{L}_{sh}$（目标↔重演）。
- **可训练/冻结模块**：仅训练 RM 和超网络 H；$\mathcal{E}_{\mathrm{w}}$（e4e 反演编码器）、$\mathcal{E}_{\mathrm{app}}$（ArcFace）、$\mathcal{E}_{\mathrm{p}}$（DECA）、StyleGAN2 生成器 $\mathcal{G}$ 均冻结。

## 实验与结果
- **数据集**：VoxCeleb1（主实验）和 VoxCeleb2（补充），图像尺寸 256×256。
- **基线方法**：X2Face、FOMM、Neural、Fast BL、PIR、LSR、FD、LIA、Dual、Rome（10 个公开代码的 SOTA 方法）。
- **自主体重演（VoxCeleb1 测试集，Table 1）**：
  - **CSIM ↑**：Ours **0.71**（最优，bold），次优为 FD/Rome 0.69。
  - **APD ↓**：Ours **0.5**（最优），次优 Fast BL/LSR 1.0。
  - **AED ↓**：Ours **5.1**（最优），次优 Rome 5.6。
  - **LPIPS ↓**：Ours 0.23（与 FD 并列次优）。
  - **FID ↓**：Ours 27.1（优于 Fast BL 55.0、PIR 50.5 等）。
  - **FVD ↓**：Ours 480（优于多数方法）。
- **跨主体重演（35 对视频，Table 1）**：
  - **CSIM ↑**：Ours **0.68**（最优），次优 Rome 0.63。
  - **APD ↓**：Ours **0.5**（最优），次优 Rome 1.2。
  - **AED ↓**：Ours 9.3（与 Rome 8.8 相近）。
- **极端头姿基准（Euler angle 差 >15°，Table 2）**：
  - **CSIM ↑**：Ours **0.58**（最优），次优 Rome 0.53。
  - **APD ↓**：Ours **0.9**（最优），次优 Rome 1.1。
- **用户研究**：30 名用户在身份保持、姿态迁移、图像质量三项上均最偏好本方法（Preference 10.267% vs 次优 7.8%）。
- **关键结论**：在所有自主体和跨主体指标上均取得 SOTA，尤其在身份保持（CSIM）和姿态迁移（APD/AED）上优势显著。

## 相关工作脉络
- **HyperStyle / HyperInverter**：利用超网络优化 StyleGAN2 权重以提升真实图像反演质量，但全局编辑（如头姿调整）时伪影严重，不适用于重演任务。本文将其思路延伸至重演场景，通过融合源身份与目标姿态特征解决此问题。
- **StyleHEAT / StyleMask**：均基于预训练 StyleGAN2 进行重演，但 StyleHEAT 使用 HDTF 训练导致在 VoxCeleb 泛化差；StyleMask 在 style space 内解耦身份与姿态，但需额外优化步骤。本文通过超网络一步完成精炼+重演，无需额外迭代优化。
- **FD（Bounareli et al. 2022）**：在 VoxCeleb1 上微调 StyleGAN2 并学习姿态方向，但依赖 e4e 反演+额外 Pivotal Tuning 优化，大姿态变化时伪影严重。本文统一框架消除该依赖。
- **Fast BL / PIR / LSR**：基于 motion field 的 warp-refine 方法，在小姿态变化下有效，但在大姿态差异时 warp 失真严重。本文不依赖显式形变场，规避了此类伪影。
- **LIA / Rome**：LIA 基于 latent space navigation，Rome 基于 mesh-based avatar，两者在跨主体重演上身份保持较弱。本文通过 hypernetwork 联合学习实现更强的身份保留。
- **e4e / Restyle 等反演方法**：encoder-based 反演无法高保真重建身份细节；optimization-based 反演耗时且不支持实时编辑。本文直接使用现成 e4e 反演结果并在线精炼，平衡效率与质量。

## 局限性与未来方向
- **训练依赖 VoxCeleb1 上的预训练 StyleGAN2**：若目标域与 VoxCeleb 分布差异较大（如低分辨率或特殊光照），泛化性可能受限。
- **仅使用单帧源图像**：虽避免了多帧微调用数据，但对于高度遮挡（self-occlusion）场景的身份重建仍有一定不确定性（论文虽声称能处理部分遮挡，但未提供系统性定量分析）。
- **超网络参数规模随生成器层数扩展**：虽通过空间重复降低参数量，但在更高分辨率（1024×1024）生成器上的可扩展性未验证。
- **未涉及音频驱动语音同步**：当前方法仅迁移静态姿态，未来可结合音频特征实现语音驱动 lip-sync 重演。
- **推理速度**：需经 ArcFace、DECA 两个编码器和超网络推理，相比 Fast BL 等端到端方法推理延迟较高，实时性未详细评估。

## 研究启发与可借鉴点
1. **超网络用于生成器权重自适应微调的思路**：将 hypernetwork 从"反演精炼"扩展到"图像编辑/重演"的范式具有普适性，可迁移至其他 GAN 编辑任务（如属性交换、风格迁移）。
2. **课程学习三阶段策略**：真实反演→自主体→跨主体的渐进训练设计简洁有效，可作为一般性训练范式应用于其他生成编辑任务。
3. **凝视损失的设计**：引入 gaze loss 改善眼神方向是低成本高回报的改进，可借鉴到任意人脸生成/编辑任务中提升细节真实感。
4. **融合 Appearance 与 Pose 特征的 SPADE-style 机制**：RM 通过可学习调制参数融合两类特征的设计简洁有效，可与 Diffusion 模型中的 cross-attention 机制对比探索融合策略。
5. **冻结全部预训练模块、仅训练轻量超网络的参数高效策略**：对资源受限场景具有吸引力，可推广至视频重演、3D 头像生成等更复杂任务。

## 关键术语表
**HyperReenact**：本文提出的基于超网络 + StyleGAN2 的单帧面部重演框架，联合学习身份精炼与姿态重定向。
**Hypernetwork（超网络）**：一种根据输入条件动态生成神经网络权重的网络结构，本文用于预测 StyleGAN2 各层权重的偏移量。
**GAN Inversion（GAN 反演）**：将真实图像编码到预训练 GAN 的隐空间（W/W+）中以保留身份信息的逆向过程。
**Curriculum Learning（课程学习）**：由易到难的渐进式训练策略，本文依次训练反演、自主体重演、跨主体重演三个阶段。
**Reenactment Module（RM）**：融合源身份特征与目标姿态特征的模块，输出引导超网络的联合特征图。
**ArcFace**：预训练的人脸识别编码器，用于提取仅包含身份信息的 appearance feature map。
**DECA**：预训练的 3D  facial shape model 编码器，用于提取仅包含姿态信息的 facial pose feature map。
**StyleGAN2**：基于 Style-based 架构的高保真人脸生成 GAN，本文作为基础生成器并冻结其权重。

## 可复现要素
- **数据集**：VoxCeleb1（训练与自主体评测）、VoxCeleb2（补充评测），公开可下载。
- **代码**：公开于 https://github.com/StelaBou/HyperReenact（论文声明）。
- **预训练模型**：StyleGAN2（VoxCeleb1 上训练，由 FD [9] 提供）和 e4e 反演模型（VoxCeleb1 上训练，由 [9] 提供）。
- **关键超参**：lr（Phase 1/2: 2e-4, Phase 3: 1e-4）、batch size: 16、λ_pix=10.0、λ_lpips=5.0、λ_id=10.0、λ_sh=0.5、λ_g=2.0、optimizer: Adam。
- **图像分辨率**：256×256。
