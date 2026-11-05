---
title: "Human-Preference-Score-Better-Aligning-Text-to-Image-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Human_Preference_Score_Better_Aligning_Text-to-Image_Models_with_Human_Preference_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:43:20"
field: "文本到图像生成与人类偏好对齐"
keywords: ["text-to-image", "human preference", "diffusion model", "CLIP fine-tuning", "LoRA adaptation", "image generation evaluation", "Stable Diffusion"]
innovations: ["提出 HPS 替代现有指标以更好预测人类偏好选择", "通过 LoRA 微调与负向提示实现轻量级生成模型偏好适配", "基于 Discord 社区日志自动构建大规模人类选择数据集"]
benchmarks: ["DiffusionDB", "LAION-5B", "COCO Captions"]
---

# 论文速读：Human-Preference-Score-Better-Aligning-Text-to-Image-Models

## 一句话总结
本文通过收集来自 Stable Diffusion Discord 社区的用户偏好选择数据集，发现现有生成模型评估指标（IS、FID、CLIP Score）与人类偏好相关性较弱，进而微调 CLIP 构建**人类偏好分数（HPS）**，并以 HPS 为监督信号通过 LoRA 适配 Stable Diffusion，使其生成更符合人类审美的图像。

## 研究问题与动机
- 文本到图像生成模型（如 Stable Diffusion、DALL·E）常产生肢体/面部畸形等伪影，生成结果与人类偏好不对齐。
- 主流评估指标 IS/FID 基于 ImageNet 训练的 CNN，偏向纹理而非形状，且为单模态度量，无法捕捉用户意图。
- CLIP 等图文对齐模型虽能编码文本提示，但未必能准确评估合成图像的美学质量与人类偏好。
- 缺乏大规模、带有真实人类选择标注的生成图像数据集，难以系统研究并优化生成模型的“偏好对齐”。

## 核心贡献（创新点）
1. **首个大规模人类选择数据集**：从 Stable Foundation Discord 渠道收集 98,807 张图像与 25,205 个人类选择，每个提示对应若干图像并标注偏好项。
2. **揭示现有指标的人类偏好盲点**：通过量化分析证明 IS、FID、CLIP Score 均无法较好预测人类选择，将“人类偏好”确立为被主流指标忽略的质量维度。
3. **提出 HPS（Human Preference Score）**：微调 CLIP ViT-L/14 得到偏好分类器，以图文余弦相似度（×100）作为 HPS，能更好预测人类选择并具备跨模型泛化能力。
4. **基于 HPS 的 Stable Diffusion 适配方法**：用 HPS 筛选偏好/非偏好图像，对非偏好样本在提示前添加特殊标识符，通过 LoRA 微调 UNet，并在推理时将其作为 classifier-free guidance 的负向提示。

## 方法详解
- **数据集构建**：使用 DiscordChatExporter 抓取 Stable Foundation Discord 的 `dreambot` 频道日志，解析“用户发提示→Bot 返回多张图→用户回传偏好图”的交互模式，过滤含用户上传图像及 NSFW 内容后得到最终数据集。
- **HPS 分类器训练**：在 ViT-L/14 的 CLIP 上微调最后 10 层视觉编码器和最后 6 层文本编码器，训练目标为最大化偏好图文对的相似度、最小化非偏好对相似度，AdamW 优化，学习率 1.7e-5，训练 1 epoch，batch size=5，余弦退火调度。
- **HPS 定义**：`HPS(img, txt) = 100 * cos(enc_v(img), enc_t(txt))`。
- **适配训练数据构造**：从 DiffusionDB 和 LAION-5B 子集中，用 HPS 对每张生成图评分；对同一提示的多张图按 `p = exp(HPS)/Σexp(HPS)` 的概率阈值选取偏好/非偏好样本，非偏好样本提示前加标识符 `"Weird image."`。
- **模型适配**：使用 LoRA（rank=32）仅微调 Stable Diffusion v1.4 的 UNet，冻结 VAE 和文本编码器；训练 10k 步，学习率 1e-5，weight decay 1e-2，batch size=40。推理时 50 步 PNDM 调度，classifier-free guidance scale=7.5，并将 `"Weird image."` 作为负向提示。

## 实验与结果
- **数据集统计**：98,807 张生成图，25,205 个提示，2,659 名用户；23,722 个提示有 4 张图，953 个有 3 张，530 个有 2 张。
- **指标与人类偏好相关性**：IS/FID 在偏好/非偏好集上差异不显著；HPS 偏好预测准确率达 **43.5%**，高于 CLIP ViT-L/14 的 32.9% 和 Aesthetic Classifier 的 33.1%，甚至超过随机猜测（26.1%）和人类自身一致性水平。
- **跨模型泛化**：在 DALL·E 与 Stable Diffusion 生成图像的人类选择任务中，HPS 与人类选择一致率达 **61.5%**，接近人类间一致率 63.5%，优于 CLIP 的 56.8%。
- **适配模型效果**：用户研究（100 个提示，20 名参与者）显示适配模型 **74%** 的图像获得超过 10 票正向选择，原始模型仅 **22%**；定量上 HPS 提升 0.1898→0.1916，Aesthetic Score 提升 5.90→6.06，FID 略降 19.72→19.35。

## 相关工作脉络
1. **IS/FID（ImageNet-CNN 指标）**：基于纹理偏向的分类网络，无法感知生成图像中的形状伪影与人类审美偏好；本文证明其在偏好预测上无效。
2. **CLIP Score / Aesthetic Classifier**：利用图文对比或美学评分预测图像质量，但与人类选择在合成图像域上仍存在显著 gap；本文通过微调 CLIP 在人类选择数据上弥补该 gap。
3. **DreamBooth / ELITE**：聚焦于主题定制与视觉概念编码，与本文的“偏好对齐”正交；本文关注的是更广泛的美学与意图对齐问题。
4. **InstructPix2Pix / RLHF 类方法**：通过指令编辑或人类反馈强化学习改进模型；本文同样利用人类反馈但采用更轻量的 LoRA 适配与负向提示机制。
5. **DiffusionDB / SAC**：公开的生成图像数据集；本文在此基础上引入大规模人工偏好标注，填补了偏好数据的空白。

## 局限性与未来方向
- 数据集仅反映 Stable Foundation Discord 活跃用户的偏好，存在人群偏差，不能完全代表全球用户。
- 大量提示词来自经验丰富的用户，语言风格可能偏离自然表达。
- 部分图像包含公众人物，作者选择保留但加以标记，可能引入名人效应偏差。
- 未探索 HPS 在其他架构（如 GAN、潜在扩散变体）上的迁移性。
- 未来可拓展至更多模态或更细粒度的人类偏好维度（如构图、叙事一致性）。

## 研究启发与可借鉴点
1. **轻量偏好对齐范式**：将 HPS 作为替代昂贵人工反馈的代理信号，配合 LoRA 实现低成本的模型适配，思路可迁移至其他生成任务。
2. **负向提示的工程化利用**：通过标识符将“非偏好概念”显式编码进文本编码器，并在推理时用作 classifier-free guidance 的负向条件，实现“避免坏样例”的有效机制。
3. **对现有指标的系统性质疑**：本文对 IS/FID/CLIP Score 的批判性评测为后续研究提供了“评估指标需与人类偏好对齐”的示范。
4. **用户行为数据挖掘**：从 Discord 频道日志中自动提取偏好对，展示了利用社区交互数据构建训练集的高效途径。
5. **偏好与对齐的解耦**：本文强调“偏好对齐”不同于“文本-图像精确对齐”，提示研究者可分别设计优化目标。

## 关键术语表
**Human Preference Score (HPS)**：基于微调 CLIP 的图文余弦相似度（×100），用于量化单张图像相对于文本提示的人类偏好程度。  
**LoRA**：低秩自适应，通过在投影矩阵上注入低秩残差实现高效参数微调，保持原始权重冻结。  
**Classifier-free Guidance**：扩散模型推理时的条件控制机制，同时使用正向提示和负向提示（如 `"Weird image."`）来引导生成方向。  
**DiffusionDB**：来自 Stable Foundation Discord 的大规模提示-图像配对公开数据集。  
**Aesthetic Score Predictor**：基于 CLIP ViT-L/14 加上 MLP 头的美学质量预测模型，输出 1–10 分。  
**Inception Score (IS) / Fréchet Inception Distance (FID)**：基于 ImageNet 预训练 InceptionNet 的生成图像质量评估指标，分别衡量清晰度和真实性分布距离。  
**Stable Foundation Discord**：Stable Diffusion 官方社区频道，本文的数据来源平台。

## 可复现要素
- **数据集**：自定义的人类选择数据集**未公开**；使用了公开数据集 DiffusionDB 与 LAION-5B 子集。
- **代码/权重**：论文未明确提供开源代码链接，仅给出项目页面 `https://tgxs002.github.io/align_sd_web/`；LoRA 权重与 HPS 分类器权重未在正文中声明开源。
- **关键超参**：HPS 训练学习率 1.7e-5，batch size=5，1 epoch；适配训练 rank=32，学习率 1e-5，weight decay 1e-2，batch size=40，10k 步；推理 50 步 PNDM，guidance scale=7.5。
