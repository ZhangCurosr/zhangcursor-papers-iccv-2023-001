---
title: "Feature-Prediction-Diffusion-Model-for-Video-Anomaly-Detecti"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:42:07"
field: "视频异常检测"
keywords: ["视频异常检测", "扩散模型", "DDIM", "特征预测", "one-class学习", "生成模型"]
innovations: ["首个将扩散模型用于视频异常检测，通过两个DDIM模块分别学习运动分布和外观分布", "仅使用2D图像特征输入，无需任何辅助3D语义模型（如目标检测/动作识别）", "设计特征预测+特征细化的级联扩散架构，实现比AE/GAN更准确的正常特征预测"]
benchmarks: ["CUHK Avenue", "ShanghaiTech", "UCF-Crime", "UBnormal"]
---

# 论文速读：Feature-Prediction-Diffusion-Model-for-Video-Anomaly-Detection

## 一句话总结
本文提出了首个基于扩散模型的帧特征预测方法 FPDM，用于视频异常检测，通过两个互补的 DDIM 模块分别学习正常样本的运动分布与外观分布，仅使用 2D 图像特征作为输入，无需任何辅助高阶语义模型，在四个基准数据集上全面超越现有最强 2D 方法并与使用 3D 语义特征的基线相当。

## 研究问题与动机
- **已有生成式 VAD 方法（GAN/AE）生成能力弱**：低质量生成的图像会引入噪声，降低检测性能。
- **现有 SOTA 方法严重依赖外部辅助模型**：如目标检测、动作识别、光流网络等提取前景/动作的高层级语义特征，性能高度受限于这些辅助模型的表征能力。
- **异常事件兼具新颖外观和异常运动**：仅学习单一维度难以充分刻画正常性/异常性。
- **扩散模型具备强大生成能力和抗噪能力**：Motivated by DDPM/DDIM 在多任务中展现的优异生成质量，本文首次将其引入 VAD 领域。

## 核心贡献（创新点）
1. **首个将扩散模型用于视频异常检测的特征预测框架 FPDM**：与已有工作本质不同——现有 VAD 方法多为 AE/GAN 重建或帧预测，本文利用 DDIM 隐式采样直接预测帧特征分布。
2. **设计两个互补的 DDIM 模块分别专注运动学习与外观学习**：特征预测模块学习时序运动分布，特征细化模块聚焦外观细节恢复，与以往单模块建模形成差异。
3. **仅以 2D 图像特征为输入，不依赖任何 3D 高阶语义模型**：相比多数 SOTA 方法依赖 OD/Action-Recognition 等辅助网络，FPDM 以轻量 2D Encoder 配合 DDIM 即取得可比甚至更优的性能。

## 方法详解
- **整体架构**：框架包含三部分——帧级 2D Encoder、特征预测扩散模块（Feature Prediction Diffusion Module, PM）、特征细化扩散模块（Feature Refinement Diffusion Module, RM）。
- **Encoder**：采用 [48] 的轻量编码器，输出特征图为 32×32×4，显著降低后续扩散模块的计算开销；预训练为无监督方式。
- **特征预测扩散模块（PM）**：
  - 输入：k 个连续帧的特征 {f₁⁰, f₂⁰, ..., fₖᵗ}，其中前 k-1 个为原始无噪声特征，第 k 个为加噪特征 fₖᵗ。
  - 目标：通过隐式采样逐步去除 fₖᵗ 中的噪声，预测第 k 帧去噪后的特征 $\dot{f}_k^0$。
  - 网络结构：基于 LDM 的 U-Net，修改两点——（1）去除 latent condition 部分，将所有 cross-attention 改为传统 attention；（2）输入包含多帧连续特征以提供运动信息。
  - 训练损失：$L_\theta = \mathbb{E}\left[\|\epsilon_\theta(f_1^0, \cdots, f_k^t, t) - \epsilon\|_2^2\right]$
  - 运动学习优势：连续的时序监督保证运动预测准确性。
- **特征细化扩散模块（RM）**：
  - 输入：PM 的输出 $\dot{f}_k^0$（加噪版本），条件为原始第 k 帧特征 fₖ⁰。
  - 网络结构：保留 LDM 中的 U-Net 并增加 cross-attention 机制，将 fₖ⁰ 展平后经线性变换得到条件向量 $\hat{f}_k^0$，通过 cross-attention 提供一致的外观监督。
  - Cross-attention 计算：$Q = W_q \cdot \hat{f}_k^0$，$K = W_k \cdot \phi(\dot{f}_k^t)$，$V = W_v \cdot \phi(\dot{f}_k^t)$。
  - 训练损失：$L_\varphi = \mathbb{E}\left[\|\epsilon_\varphi(\dot{f}_k^t, \hat{f}_k^0, t) - \epsilon\|_2^2\right]$
  - 外观细化优势：弥补 PM 在细节信息上的缺失。
- **异常评分**：测试阶段计算细化后特征 $\ddot{f}_k^0$ 与原始特征 fₖ⁰ 之间的 MSE：$Score = MSE(\ddot{f}_k^0, f_k^0)$。
- **训练策略**：两模块**分开训练**——先训练 PM 至收敛，再训练 RM（联合训练会降低 RM 性能，见 Table 3）；推理时采用 200 步采样（T'=200，每步相当于 T 中的 5 步），取 t=0.25T' 采样。

## 实验与结果
- **数据集**：CUHK Avenue（16 train/21 test clips）、ShanghaiTech（330 train/13 test videos）、UCF-Crime（>128h, >1,900 视频）、UBnormal（29 场景，合成数据集）。
- **评估指标**：帧级 AUC（基于 MSE 异常分）。
- **核心结果**（Table 1）：

  | 数据集 | FPDM（Ours） | 最佳 2D 图像基线 | 最佳 3D/语义特征基线 |
  |---|---|---|---|
  | CUHK Avenue | **90.1%** | BAF 92.3%（用 OD） | SSL 92.2%（用 OD） |
  | ShanghaiTech | **78.6%** | BAFLNTR 75.9% / CT-D2GAN 77.7% | CAC 79.3%（用 A3D） |
  | UCF-Crime | **74.7%** | GCL 74.2% | BODS 68.2% |
  | UBnormal | **62.7%** | BAF 59.3% | — |

- **关键结论**：FPDM 在所有四个数据集上全面超越**不使用任何辅助语义特征的 2D 图像基线**，并在 UCF-Crime 和 UBnormal 上超越使用 3D 语义特征的 SOTA 方法；尽管不使用 OD/Action-Recognition 等前置模块，仍能在上海科技数据集上与使用 A3D 特征的 CAC（79.3%）接近。
- **消融实验要点**：（1）去掉两个扩散模块退化为 AE，性能大幅下降（ShT 67.5% vs 78.6%）；（2）仅有 PM 或仅有 RM 均劣于完整模型；（3）引入解码器还原图像后计算 MSE 反而性能略降（信息丢失）；（4）联合训练（J）明显劣于分开训练（S）；（5）增加邻居帧数（0→12）在动作异常数据集上性能显著提升。

## 相关工作脉络
1. **帧预测类 VAD（Frame Prediction）**：FFP [40]、Bman [32] 等用 AE/GAN 预测下一帧；FPDM 以扩散模型替代生成框架，避免 GAN 不稳定和 AE 生成能力弱的问题。
2. **记忆增强类 VAD**：MemAE [24]、MemG [46] 通过记忆模块存储正常模式；FPDM 通过 DDIM 隐式扩散过程直接学习特征分布，无需显式记忆库。
3. **基于 3D 语义特征的 VAD**：VEC [69]、HF2 [37]、BAF [23] 等依赖目标检测结果；FPDM 证明仅用 2D 特征+扩散模型可达到同等甚至更优水平，消除对辅助模型的依赖。
4. **扩散模型用于视觉任务**：DiffusionDet [12]（检测）、SegDiff [2]（分割）、AnoDDPM [63]（医学图像异常检测）；本文首次将 DDIM 引入视频异常检测的场景化特征预测任务。
5. **判别性扩散模型学习**：SBG classifier [77]、Cold Diffusion [5]；本文与之不同，关注的是**特征预测而非生成图像**，且用于开放场景的视频异常检测。
6. **DDIM 加速采样**：原 DDPM 采样步数多、速度慢；本文采用 DDIM 的跳跃步隐式采样（200 步完成），实现相比 DDPM 约 20× 加速。

## 局限性与未来方向
- **仅使用 2D 静态帧特征**：未显式建模时序光流或 3D 卷积，对复杂运动异常的检测能力可能受限（上海科技数据集 78.6% vs 使用 A3D 的 CAC 79.3% 略低）。
- **推理采样步数仍较多**：虽比 DDPM 快 20×，但 200 步采样在实时应用中仍有优化空间。
- **仅评估 one-class 设置**：未涉及半监督或弱监督场景，实际应用中可利用少量异常标注进一步扩展。
- **编码器固定为 [48] 的轻量网络**：未探索更强/backbone 与扩散模型结合的可能性。
- **作者未明确提及的未来方向**：可探索将 PM/RM 联合训练策略优化、引入更多采样加速算法、迁移至其他视频理解任务（如行为识别、分割）。

## 研究启发与可借鉴点
1. **"预测+细化"双模块扩散架构**可迁移至其他视频理解任务（如视频插值、去噪、超分），先学结构/运动再精修细节的分阶段设计思路值得借鉴。
2. **DDIM cross-attention 条件注入机制**（将原始帧特征展平后作为 appearance condition）提供了一种简洁的条件扩散建模范式，可复用于图像修复、风格迁移等任务。
3. **放弃解码器直接在特征空间计算 MSE**避免了信息损失，这一设计决策对类似特征预测类异常检测任务有参考价值。
4. **分开训练（sequential training）优于联合训练**的现象在级联扩散模型中普遍存在，团队在构建多阶段生成模型时应优先考虑分阶段收敛策略。
5. **仅需 2D 特征即可对抗 3D 语义特征方法**，为资源受限场景（无 GPU 运行大参数检测器）下的 VAD 部署提供了可行路径。

## 关键术语表
- **FPDM（Feature Prediction Diffusion Model）**：本文提出的首个基于扩散模型的帧特征预测视频异常检测方法。
- **DDIM（Denoising Diffusion Implicit Models）**：DDPM 的加速变体，采用隐式采样可在较少步数内完成去噪，无需额外训练。
- **One-class VAD**：仅用正常样本训练，通过检测偏离正常分布的程度来识别异常的半监督/无监督检测范式。
- **LDM（Latent Diffusion Model）**：Rombach et al. 提出的在潜空间中执行扩散过程的模型，本文据此修改构建了扩散模块。
- **Cross-Attention**：本文在细化模块中将原始帧特征作为条件通过交叉注意力注入去噪网络，用于外观信息保持。
- **AUC（Area Under ROC Curve）**：视频异常检测的主流评估指标，衡量异常分对正常/异常帧的排序区分能力。
- **Anomaly Score**：本文定义为细化特征与原始特征之间的 MSE，值越大表示该帧越异常。
- **Feature Prediction**：利用连续若干帧的特征预测当前帧特征，正常帧可被准确预测而异常帧预测误差大。

## 可复现要素
- **数据集**：CUHK Avenue、ShanghaiTech、UCF-Crime、UBnormal，均为公开数据集。
- **代码**：论文声明代码可用，链接为 FPDM（原文提及"The code is available at FPDM"，具体 URL 在论文中指向 GitHub）。
- **关键超参**：
  - 输入图像尺寸：256×256；特征图：32×32×4
  - 连续邻居帧数 k=5（4 帧预测第 5 帧）
  - 训练 epoch S=60（含 12 warm-up epochs）
  - 时间步 T=1k，学习率 10⁻⁵
  - 推理采样步数 T'=200，起始时间步 t=0.25T'
- **Encoder**：采用 [48]（CLIP 相关轻量编码器），无监督预训练
