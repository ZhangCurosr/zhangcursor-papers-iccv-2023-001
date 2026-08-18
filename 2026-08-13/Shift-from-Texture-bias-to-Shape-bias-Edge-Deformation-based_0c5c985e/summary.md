---
title: "Shift-from-Texture-bias-to-Shape-bias-Edge-Deformation-based"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/He_Shift_from_Texture-bias_to_Shape-bias_Edge_Deformation-based_Augmentation_for_Robust_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:30:18"
field: "鲁棒视觉/对抗防御"
keywords: ["Texture-bias", "Shape-bias", "Edge Deformation", "Data Augmentation", "Adversarial Robustness", "Generative Model", "Online Augmentation"]
innovations: ["提出基于TPS的边缘图在线形变增广框架SDbOA以驱动纹理偏置向形状偏置转移", "设计自信息引导的边缘扩展与软约束形状保持损失保障生成语义一致性", "在对抗、后门攻击与常见噪声评测中实现优于SOTA的通用鲁棒性"]
benchmarks: ["Fashion MNIST", "CelebA", "CIFAR-10", "ImageNet", "CIFAR-10-C"]
---

# 论文速读：Shift-from-Texture-bias-to-Shape-bias-Edge-Deformation-based

## 一句话总结
提出一种基于边缘图形变的在线数据增广方法 **SDbOA**，通过将 CNN 的特征学习重心从纹理偏置（texture-bias）主动转向形状偏置（shape-bias），显著提升模型在对抗扰动、后门攻击及常见噪声下的通用鲁棒性。

## 研究问题与动机
- **核心问题**：CNN 对扰动噪声高度脆弱，根本原因在于模型过度依赖物体纹理特征而忽视形状结构。
- **现有方法不足**：当前减偏置方法多依赖固定形状掩码或直接在图像上施加几何形变，缺乏形状多样性，且硬形变极易破坏边界语义。
- **关键科学问题**：能否通过对边缘图进行形变增广来强化形状线索、降低纹理依赖，同时保持生成样本的语义合理性？

## 核心贡献（创新点）
- **TPS 边缘形变在线增广框架**：首次将 Thin Plate Spline 应用于边缘图形变并接入训练流程，区别于直接图像形变，该方法仅修改形状轮廓而不破坏整体语义。
- **自信息引导的边缘扩展与软约束生成范式**：提出 EMSE 模块融合 Robust Canny 与 self-information guided map 以增强边缘鲁棒性；TSG 生成器引入形状保持损失 $\mathcal{L}_{edge}$ 作为软约束，区别于 Pix2Pix 等硬映射方法。
- **通用鲁棒性超越 SOTA 且训练开销更低**：在对抗攻击、后门攻击与常见噪声三类评测中均取得最优或接近最优结果，且无需像 FDA/PORT 那样进行耗时的离线对抗样本预生成。

## 方法详解
方法由三个串联模块构成，支持训练期在线调用：
- **EMSE（边缘图形状编码）**：先用 Robust Canny 提取基础边缘图 $E_{RC}$；再计算像素级自信息图 $\hat{E}(p)=-log(\sum_{p'\in\mathcal{N}_p}e^{-(p-p')^2})$，经均值阈值化得到 $E_{info}$；两者相加获得扩展边缘图 $E_{extend}=E_{RC}+E_{info}$，以吸收边界邻域的结构线索。
- **TSD（TPS 形状形变）**：将 $E_{extend}$ 均匀划分为 $n=16$ 个网格，取网格端点为控制点 $\{P_i\}$，按 $Q_i=P_i+\lambda N$（$N\sim\mathcal{N}(0,1)$，$\lambda=0.1$）扰动生成目标控制点；基于 TPS 求解 $x/y$ 方向的平滑变换 $\Phi^x,\Phi^y$，输出形变边缘图 $E_{deform}$。
- **TSG（纹理与形状联合生成）**：原图经双线性下采样再上采样得到去噪纹理图 $I_{txt}$；生成器分两阶段训练，第一阶段用 $E_{extend}$ 与 $I_{txt}$ 生成粗糙图像，损失为 $\mathcal{L}_{1st}=\mathcal{L}_{gan}+\mathcal{L}_{fm}+\mathcal{L}_{cls}$；第二阶段换用 $E_{deform}$ 微调，并加入形状保持损失 $\mathcal{L}_{edge}=||\text{eCNN}(I_{syn}),E_{deform}||_1$，总损失为 $\mathcal{L}_{2nd}=\mathcal{L}_{gan}+\mathcal{L}_{cls}+\mathcal{L}_{edge}$。形变边缘仅作软引导，避免过度扭曲语义。

## 实验与结果
- **数据集**：Fashion MNIST、CelebA、CIFAR-10、ImageNet；鲁棒性评测覆盖 FGSM、PGD、C&W、DeepFool、CIFAR-10-C 及两种后门攻击模式。
- **形状偏置度量**：在 $sb_{GE}$ 与 $sb_{IS}$ 两个指标上，SDbOA 在 IN/SIN/SIN+IN 上均取得最高值（如 IN 上 $sb_{GE}=71.28\%$，$sb_{IS}=31.2\%$）。
- **对抗鲁棒性**：CIFAR-10 上 C&W 与 DeepFool 防御分别较 NuAT 提升 14.10% 与 27.72%；ImageNet 上 C&W 防御较 EdgeNetRob 提升 6.98%。
- **常见噪声与对抗联合评测**：mCE 为 18.8%，PGD-20 误分类率 27.7%，大幅优于 AugMix（86.8%）与 PixMix（82.1%）。
- **后门攻击**：Fashion MNIST 上 Pixel/Pattern 模式的 ASR 分别降至 0.04% 与 0.98%，显著压制 AugMax；CelebA 上 ASR 为 3.49%/4.21%。
- **消融**：$\lambda=0.1$ 为最优形变强度；EMSE、TSD、TSG 三者叠加后 Clean Acc 与四项对抗准确率全面提升，验证各模块必要性。

## 相关工作脉络
- **Geirhos et al. [11]**：揭示 CNN 纹理偏置现象；本文与其定位差异在于不仅指出偏置，更提出通过边缘形变主动塑造形状多样性。
- **StyleAug [21] / ShapeAug [24]**：分别扰动纹理或固定形状掩码；本文直接对边缘图进行连续形变并重建，兼顾多样性与语义连贯性。
- **EdgeNetRob [37]**：将边缘图作为固定辅助监督信号；本文将其转化为可变形、可生成的在线增广流，动态丰富形状空间。
- **FDA [31] / PORT [33]**：基于生成模型的离线对抗训练；本文采用在线生成增广，以 3-5 小时训练耗时换取同等或更优的鲁棒性。
- **TPS-Deform [43]**：直接对图像做 TPS 形变；本文在边缘空间操作并引入 $\mathcal{L}_{edge}$ 软约束，避免硬形变导致的分类边界越界。

## 局限性与未来方向
- **局限性**：方法主要针对像素级扰动设计，当遭遇 patch-wise 严重破坏形状结构的噪声时效果受限。
- **未来方向**：扩展形变与生成机制以应对强局部遮挡与结构性破坏场景，进一步提升通用鲁棒性。

## 研究启发与可借鉴点
- **边缘-邻域联合编码思路**：将形态学边缘与自信息图相加可有效抵御边缘噪声断裂，可迁移至依赖轮廓的结构感知任务。
- **网格控制点替代检测关键点**：均匀网格化避免了目标检测模块的额外开销与误差传播，适合批量数据增广流水线。
- **生成任务中的软形状保持损失**：$l_1$ 边缘一致性损失作为生成约束比硬像素对齐更稳定，值得借鉴于图像修复、风格迁移与医学图像合成。
- **在线增广 vs 离线预生成**：训练过程中动态生成比离线缓存更具显存与时间效率，为资源受限团队提供了可复用的训练范式。

## 关键术语表
- **Texture-bias / Shape-bias**：模型预测分别过度依赖物体表面纹理或整体几何形状的特征倾向。
- **TPS (Thin Plate Spline)**：基于控制点变分的二维平滑非线性形变模型，用于控制边缘图的连续变形。
- **Self-information guided map**：基于像素与其邻域灰度差异构建的图，用于补充并扩展 Robust Canny 边缘的鲁棒性。
- **EMSE / TSD / TSG**：边缘形状编码、TPS形变、纹理-形状联合生成三大核心模块的缩写。
- **Shape-preservation loss ($\mathcal{L}_{edge}$)**：生成图像边缘与目标形变边缘的 $l_1$ 差异，作为软约束防止生成内容语义偏离。
- **Backdoor attack**：在训练集植入触发器使模型对含触发器样本强制输出攻击者指定标签的隐蔽攻击。

## 可复现要素
- **数据集**：Fashion MNIST、CelebA、CIFAR-10、ImageNet 及 CIFAR-10-C（均公开）。
- **代码/权重**：开源，官方链接 https://github.com/C0notSilly/-ICCV-23-Edge-Deformation-based-Online-Augmentation。
- **关键超参**：网格数 $n=16$，形变强度 $\lambda=0.1$，插值均采用双线性；骨干网 ImageNet 用 ResNet-50，其余用 ResNet-18（CIFAR-10-C 鲁棒评测用 WideResNet-40-4）。
- **训练协议**：在线逐 epoch 动态生成增广样本，无需预生成数据集。
