---
title: "HyperReenact-One-Shot-Reenactment-via-Jointly-Learning-to-Re"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Bounareli_HyperReenact_One-Shot_Reenactment_via_Jointly_Learning_to_Refine_and_Retarget_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:09:44"
field: "人脸图像生成与编辑"
keywords: ["人脸重演", "GAN 反演", "超网络", "StyleGAN2", "单帧重演", "课程学习"]
innovations: ["首个将 GAN 反演优化与姿态重映射合并的超网络人脸重演框架", "课程学习三阶段训练策略实现无微调跨主体单帧重演", "引入注视损失提升眼部区域重建真实性"]
benchmarks: ["VoxCeleb1", "VoxCeleb2"]
---

# 论文速读：HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces

## 一句话总结
本文提出 HyperReenact，一种基于预训练 StyleGAN2 与超网络（hypernetwork）的单帧人脸重演方法，通过联合学习"反演优化"与"姿态重映射"两个步骤，在无需任何主体特定微调的前提下，实现高保真、低伪影的面部重演，在极端头部姿态差异与跨主体场景下均达到 SOTA。

## 研究问题与动机
1. **极端姿态差异下伪影严重**：现有神经重演方法（如 Fast BL、PIR、LSR 等）在源图像与目标图像头部姿态差异较大时，生成的重演图像容易产生严重的视觉伪影。
2. **身份保真依赖昂贵微调**：多数方法需 few-shot 微调（使用同一主体的多角度图像）才能充分保留源身份特征，限制其单帧设定下的实用性。
3. **外部 GAN 反演的局限**：基于 StyleGAN2 的方法依赖外部反演工具（如 e4e），导致重建质量差、图像可编辑性差，并在大姿态偏移时产生伪影。
4. **跨主体重演中的身份泄漏**：基于面部 landmarks 的方法因保留了目标面部几何，易导致目标身份泄漏到源图像上。

## 核心贡献（创新点）
1. **首个将反演优化与姿态重映射合并的超网络重演框架**：首次证明通过超网络联合学习 StyleGAN2 权重更新，可同时完成身份细化与姿态重映射，实现鲁棒且真实的重演效果。
2. **真正的单帧无微调跨主体重演**：方法仅需单帧源图像，无需任何主体特定 fine-tuning，即可在自重演与跨主体重演中保持源身份特征，包括部分遮挡的极端姿态场景。
3. **极端头部姿态变化下的伪影消除**：在源-目标大姿态差异（Euler 角差 > 15°）的 benchmark 上，显著提升 ID 保真度并减少伪影，优于所有对比方法。
4. **课程学习训练策略（CL）**：设计三阶段 curriculum learning——真实图像反演 → 自重演 → 跨主体重演，显著改善各阶段性能，并在消融实验中验证其有效性。

## 方法详解
- **整体架构**：给定源图像 $I_s$ 与目标图像 $I_t$，使用外观编码器 $\mathcal{E}_{\mathrm{app}}$（基于 ArcFace [13]）提取源身份特征 $f_{\mathrm{app}} \in \mathbb{R}^{512 \times 7 \times 7}$；使用姿态编码器 $\mathcal{E}_{\mathrm{p}}$（基于 DECA [16]）提取目标姿态特征 $f_{\mathrm{p}} \in \mathbb{R}^{2048 \times 7 \times 7}$。
- **重演模块（RM）**：受 SPADE 启发，将 $f_{\mathrm{p}}$ 经 $1 \times 1$ 卷积投影为 $f'_{\mathrm{p}}$，再分别对 $f_{\mathrm{app}}$ 和 $f'_{\mathrm{p}}$ 学习调制参数 $\gamma, \beta$，融合为 $f_{\mathrm{r}} = \gamma_{\mathrm{app}} \odot f_{\mathrm{app}} + \beta_{\mathrm{app}} + \gamma_{\mathrm{p}} \odot f'_{\mathrm{p}} + \beta_{\mathrm{p}}$，尺寸 $512 \times 7 \times 7$。
- **超网络模块（H）**：由 M 个 Reenactment Block（RB）组成，每个 RB 以 $f_{\mathrm{r}}$ 为输入，输出生成器第 $\ell$ 层的权重偏移 $\Delta\theta_\ell$；偏移经空间重复 $k_\ell \times k_\ell$ 后，更新权重 $\hat{\theta}_\ell = \theta_\ell \cdot (1 + \Delta\theta_\ell)$。
- **初始反演**：使用 VoxCeleb1 训练的 e4e 编码器将源图像反演至 $W^+$ 空间，获得初始潜码 ${\bf w}_s$（编码器在训练中冻结）。
- **课程学习三阶段训练**：
  - **Phase 1（反演）**：源=目标，优化 $\mathcal{L} = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{id}\mathcal{L}_{id} + \lambda_{g}\mathcal{L}_{g}$。
  - **Phase 2（自重演）**：同身份不同姿态，额外加入形状损失 $\mathcal{L}_{sh} = \|S(I_t) - S(I_r)\|_1$（用 DECA 提取 3D 形状）。
  - **Phase 3（跨主体）**：同时处理自重演与跨主体样本（各占 half batch），跨主体样本仅计算 $\mathcal{L}_{id}$（源 vs 重演）和 $\mathcal{L}_{sh}$（目标 vs 重演）。
- **冻结参数**：训练过程中 $\mathcal{E}_{\mathrm{w}}, \mathcal{E}_{\mathrm{app}}, \mathcal{E}_{\mathrm{p}}$ 及 StyleGAN2 生成器 $\mathcal{G}$ 均冻结，仅优化 RM 和超网络 H；跳过 toRGB 层。

## 实验与结果
- **数据集**：VoxCeleb1（主 benchmark，256×256）与 VoxCeleb2；极端姿态子集从 VoxCeleb1 测试集选取头姿态差 > 15° 的图像对。
- **基线**：X2Face、FOMM、Neural、Fast BL、PIR、LSR、FD、LIA、Dual、Rome，均在单帧上公平比较（部分基线需少量微调）。
- **自重演（VoxCeleb1）**：CSIM 达 **0.71**（最佳），APD 达 **0.5**（最佳），AED 达 **5.1**（最佳）；LPIPS=0.23、FID=27.1、FVD=480，与 FD、Rome 相当。
- **跨主体重演**：CSIM 达 **0.68**（最佳），APD=0.5（最佳），AED=9.3（仅次于 Rome 的 8.8）。
- **极端姿态子集**：CSIM 达 **0.58**（最佳，Rome 为 0.53），APD=**0.9**（最佳，Rome 为 1.1）。
- **用户研究**：30 名用户对 20 组重演结果投票，HyperReenact 在身份保真、姿态转移和图像质量三项上均以显著优势胜出。
- **结论**：在身份保真与姿态转移上全面领先，且在极端姿态条件下展现出最强的鲁棒性。

## 相关工作脉络
1. **HyperStyle / HyperInverter**：利用超网络优化 GAN 反演质量，但仅在静态重建中有效，全球编辑（如大姿态变换）时产生严重伪影；本文将其扩展至动态重演场景。
2. **FD（Finding Directions in GAN's latent space）**：在 StyleGAN2 的 W+ 空间中学习姿态控制方向，但依赖外部反演（e4e）加额外的 Pivotal Tuning 优化，在极端姿态下仍有伪影；本文无需额外优化步骤。
3. **StyleHEAT / StyleMask**：利用 StyleGAN2  disentangled style space 控制重演，但 StyleHEAT 在 HDTF（以正面为主）上训练，泛化到 VoxCeleb 等含大姿态分布的数据集上表现不佳；本文直接使用 VoxCeleb1 训练的 StyleGAN2。
4. **FOMM / FOMM-BL / Fast BL**：基于单应性/仿射变形的早期方法，在复杂姿态下伪影严重；本文利用生成式先验从根本上规避该问题。
5. **PIR / LSR / LIA / Rome**：近期 SOTA 方法，PIR 依赖语义神经渲染，Rome 基于网格建模；本文的独特之处在于首次将超网络用于实时单帧重演，无需网格或逐帧优化。
6. **e4e 反演**：作为本方法的初始反演工具，但本文指出其直接在 W+ 空间进行大姿态编辑存在局限，从而引入超网络进行联合优化。

## 局限性与未来方向
- **训练数据依赖 VoxCeleb1**：StyleGAN2 在 VoxCeleb1 上预训练，可能限制了在更广泛人脸分布（如不同年龄、种族）上的泛化能力。
- **跨主体重演的 AED 仍略逊于 Rome**：表情转移精度尚有提升空间，尤其是复杂表情下的细节保留。
- **仅使用单帧源图像**：虽然符合 one-shot 设定，但缺乏时序信息可能导致视频序列中身份或表达的帧间波动。
- **未探索音频驱动说话口型同步**：当前方法仅基于姿态（3DMM 形状参数）驱动，未整合语音→唇形同步，限制了其在数字人等应用中的完整性。

## 研究启发与可借鉴点
1. **课程学习策略的可迁移性**：三阶段 curriculum（反演 → 自重演 → 跨主体）的设计思路可推广到其他 GAN 操控任务，如头像生成、图像编辑等，建议复用此训练范式。
2. **外观/姿态特征解耦融合设计**：采用 ArcFace（纯身份）与 DECA（纯姿态）分别编码，再通过 SPADE-style 融合模块联合输入超网络，这一"解耦感知—融合编辑"范式对多属性可控生成具有参考价值。
3. **超网络权重更新机制**：$\hat{\theta}_\ell = \theta_\ell \cdot (1 + \Delta\theta_\ell)$ 的乘法式更新比 HyperStyle 的加法式更稳定且参数量少，可在其他 GAN 编辑任务中尝试。
4. **注视损失（Gaze Loss）的引入**：利用 L2 距离约束生成图像与目标图像的注视方向，提升了眼部区域的真实性，该损失对需要眼部细节的任务（如数字人、视频会议）有直接借鉴意义。
5. **与团队方向的结合点**：超网络驱动的生成器权重动态更新机制可与本团队的风格迁移、低资源人脸生成方向结合，探索在无大量配对数据下的身份保持编辑。

## 关键术语表
**HyperReenact**：本文提出的单帧人脸重演框架，联合学习反演优化与姿态重映射，基于 StyleGAN2 与超网络实现。
**Hypernetwork（超网络）**：一种通过条件输入动态生成神经网络权重的网络结构，本文用于预测 StyleGAN2 各层权重偏移。
**GAN Inversion（GAN 反演）**：将真实图像编码到预训练 GAN 的潜空间（如 W+），以实现对该图像的语义编辑。
**Curriculum Learning（课程学习）**：一种训练策略，从简单任务（如反演）逐步过渡到复杂任务（如跨主体重演），以提升模型收敛效果。
**ArcFace**：一种基于加性角度间隔损失的深度人脸识别网络，本文用作源图像外观/身份特征的编码器。
**DECA**：一种从单张图像恢复 3D 人脸形状（含姿态、表情、光照）的预训练模型，本文用作目标姿态特征的编码器。
**CSIM / APD / AED**：重演质量评估指标，分别为 ArcFace 特征余弦相似度（身份保真）、平均姿态距离（姿态转移）、平均表情距离（表情转移）。

## 可复现要素
- **数据集**：VoxCeleb1（主实验）、VoxCeleb2（补充实验）；极端姿态子集由作者自建，**论文未公开**。
- **代码**：已开源，GitHub: https://github.com/StelaBou/HyperReenact
- **预训练模型**：已公开（见 GitHub 链接）
- **关键超参**：$\lambda_{pix}=10.0, \lambda_{lpips}=5.0, \lambda_{id}=10.0, \lambda_{sh}=0.5, \lambda_{g}=2.0$；学习率 Phase1/2: $2\times10^{-4}$，Phase3: $1\times10^{-4}$；batch size=16；图像分辨率 256×256；优化器 Adam。
