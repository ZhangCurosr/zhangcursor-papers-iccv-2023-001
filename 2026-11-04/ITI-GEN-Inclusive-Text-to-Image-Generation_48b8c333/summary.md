---
title: "ITI-GEN-Inclusive-Text-to-Image-Generation"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ITI-GEN_Inclusive_Text-to-Image_Generation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:44:50"
---

# 论文速读：ITI-GEN-Inclusive-Text-to-Image-Generation

## 一句话总结
本文提出 ITI-GEN，一种基于少量参考图像引导提示词嵌入学习的无微调包容性文生图方法，通过在 CLIP 联合嵌入空间中对齐图文类别方向并施加语义一致性约束，使冻结的扩散模型（如 Stable Diffusion）能够均衡生成覆盖目标属性全类别的图像，无需昂贵重训或易产生歧义的硬提示词枚举。

## 研究问题与动机
- 现有文本到图像生成模型继承大规模预训练数据的分布偏差，对少数群体或特定视觉属性（如戴眼镜、复杂肤色分级）的代表性严重不足。
- 直接通过自然语言枚举属性的硬提示词搜索（Hard Prompt Searching）受限于语言歧义与模型误表征，难以稳定生成期望类别。
- 使用平衡/包容性数据对模型进行微调和再训练，面临数据收集困难与生成模型训练开销巨大的双重瓶颈。
- 图像比文字更能无歧义地表达细粒度视觉概念（如肤色层次），因此探索以参考图像为引导信号学习连续提示词嵌入成为高效替代路径。

## 核心贡献（创新点）
- 提出基于参考图像的包容性提示词学习框架 ITI-GEN，在连续嵌入空间中优化属性专属的可学习 token，实现冻结扩散模型的属性均衡生成。与 Fine-tuning 类个性化方法（如 Custom Diffusion）的本质区别在于全程不更新基座模型权重，仅扩展少量 prompt 嵌入，计算成本极低。
- 设计方向对齐损失（Direction Alignment Loss），将参考图像中不同类别间的视觉方向映射为提示词嵌入间的方向，使生成条件精确捕捉属性差异。与直接最大化图文相似度方法的本质区别在于关注“相对方向对齐”而非“绝对相似度匹配”，有效避免对抗性伪影与画质退化。
- 引入语义一致性损失（Semantic Consistency Loss），通过 margin-based 余弦相似度约束学习到的 prompt 不偏离原始输入提示词的语义空间，防止连续优化导致的语言漂移。与纯方向对齐训练的对比在于额外保留了对原始文本主题的忠实度。
- 验证方法的高度通用性与即插即用特性，支持单/多属性组合、跨域扩展（人脸/场景）、兼容 ControlNet 与 InstructPix2Pix，且每类别仅需数十张参考图，单属性训练约 5 分钟（单 A4500 GPU）。

## 方法详解
- **Inclusive Prompt 构建**：给定原始提示词 $T$ 与目标属性集 $\pmb{A} = \{A_m\}_{m=1}^M$，为每个类别 $a_k^m$ 学习可注入 token $S_k^m \in \mathbb{R}^{q \times e}$，拼接后形成 $\ddot{P}_k^m = [T; S_k^m]$。多属性组合时采用元素级求和聚合 $S_{o_1...o_M} = \sum_{m=1}^M S_{o_m}^m$，均匀采样 $\mathcal{P}_{total}$ 中的 prompt 即可实现任意类别组合的均衡采样。
- **方向对齐损失（$\mathcal{L}_{dir}$）**：利用预训练 CLIP 提取参考图像与 prompt 嵌入。对属性 $A_m$ 的任意两类，计算类别均值图像嵌入差 $\Delta_I^m(i,j) = \mathfrak{X}_i^m - \mathfrak{X}_j^m$ 与对应 prompt 嵌入差 $\Delta_P^m(i,j) = \mathfrak{P}_i^m - \mathfrak{P}_j^m$，以 $1 - \langle \Delta_I, \Delta_P \rangle$ 为优化目标，迫使 prompt 方向差异与图像视觉差异一致。
- **语义一致性损失（$\mathcal{L}_{sem}$）**：设计 hinge loss $\mathcal{L}_{sem} = \max(0, \lambda - \langle E_{text}(P), E_{text}(T) \rangle)$ 正则化 prompt 嵌入，其中 $\lambda=0.8$ 对应典型硬提示词与原始提示词的 CLIP 相似度阈值，确保优化过程中 prompt 不丢失原始语义。
- **优化策略**：总损失 $\mathcal{L}_{total} = \sum_m \sum_{i<j} (\mathcal{L}_{dir}^m + \mathcal{L}_{sem}^m)$。每轮迭代仅更新单一属性的所有类别 token，冻结其余属性 token；共训练 30 epoch，batch size=16，学习率 0.01。基座扩散模型参数全程冻结，仅优化极少量参数（每属性每类别 $3 \times 768$ 维向量）。

## 实验与结果
- **数据集与设置**：CelebA（40 项二值人脸属性）、FAIR benchmark（6 级合成肤色）、FairFace（性别与年龄）、LHQ（场景感知属性）。默认每类别 25 张参考图，基座为 Stable Diffusion v1.4。评估指标为分布散度 $D_{KL}$（越低越均衡）与 FID。
- **单属性对比**：在 CelebA 6 项二值属性上，ITI-GEN 的 $D_{KL}$ 接近 0（male 为 $2\times10^{-6}$，paleskin 为 0），显著优于 SD、EI、HPS、PD 及需微调的 CD，实现近乎完美的类别均衡。
- **多属性组合**：在 male × young × eyeglass × smile 交叉实验中，ITI-GEN 的 $D_{KL}$ 降至 $0.061 \sim 0.094$，大幅领先 SD（1.406）与 HPS（0.476），证明独立 token 聚合机制能有效解耦多属性学习。
- **多类别与跨域泛化**：成功处理感知年龄（<10 岁与 >50 岁）及复杂肤色分级等长尾属性；引入 LHQ 后可控制生成图像的色彩丰富度、亮度等主观感知属性，验证方法超越人脸领域的泛化能力。
- **消融结论**：每类别仅需 >10 张参考图即可保持高质量均衡生成；合成数据（FAIR）因背景噪声少，FID 达 51.83，优于真实人脸数据；移除 $\mathcal{L}_{sem}$ 导致 FID 从 51.83 恶化至 62.69，证实语义约束对画质保全的关键作用；兼容 ControlNet 与 InstructPix2Pix 时
