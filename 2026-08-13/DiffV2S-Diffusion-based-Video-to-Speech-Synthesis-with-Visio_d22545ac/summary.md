---
title: "DiffV2S-Diffusion-based-Video-to-Speech-Synthesis-with-Visio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Choi_DiffV2S_Diffusion-Based_Video-to-Speech_Synthesis_with_Vision-Guided_Speaker_Embedding_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:35:37"
field: "音视频生成与语音合成"
keywords: ["Video-to-Speech Synthesis", "Diffusion Model", "Prompt Tuning", "Speaker Embedding", "Audio-Visual Learning", "LRS2", "LRS3"]
innovations: ["首次将扩散模型引入视频到语音合成任务，实现高保真 mel-spectrogram 生成", "提出纯视觉引导的说话人嵌入提取器，推理时无需音频参考", "设计扩散采样阶段的梯度引导机制以保留说话人身份特征"]
benchmarks: ["LRS2-BBC", "LRS3-TED"]
---

# 论文速读：DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding

## 一句话总结
本文首次将扩散模型引入视频到语音合成任务，提出一种**纯视觉引导的说话人嵌入提取器**（无需推理时音频输入）与条件扩散生成框架（DiffV2S），在 LRS2 和 LRS3 数据集上实现了优于现有方法的内容可懂度和说话人身份保真度。

## 研究问题与动机
1. **无声视频说话人风格缺失**：从静音说话人脸视频合成语音时，仅凭口型难以获取说话人的音色、口音等声学风格特征。
2. **现有方法依赖推理时不可用的音频**：已有工作（SVTS、Multi-task）依赖来自同视频原始音频的说话人嵌入作为辅助条件；但在推理阶段，音频可能因噪声环境、无声段或未见说话人而不可获得。
3. **扩散模型在视频到语音领域的空白**：扩散模型已在图像生成和语音合成（TTS、神经声码器）中展现强大能力，但尚未被探索用于视频到语音合成。
4. **多说话人场景下的身份保持难题**：不同说话人的发音习惯差异显著，如何在无额外音频提示下保持多说话人身份一致是核心挑战。

## 核心贡献（创新点）
1. **视觉引导说话人嵌入提取器**：利用自监督预训练音视频模型（LARGE AV-HuBERT）+ prompt tuning 技术，仅从视频帧提取说话人嵌入，推理时无需额外音频——与 SVTS/Multi-task 等需参考音频嵌入的方法本质不同。
2. **首个基于扩散模型的视频到语音合成框架（DiffV2S）**：将条件扩散模型首次引入该任务，以视觉特征与视觉引导说话人嵌入为条件生成 mel-spectrogram，区别于 VCA-GAN 等 GAN-based 方法。
3. **采样阶段的梯度引导机制（Guided Sampling）**：在 DDIM 采样过程中通过说话人嵌入相似度（cosine similarity）的梯度反向传播到扩散去噪过程，使生成结果更贴近目标说话人风格——此引导策略在之前的 V2S 工作中未被提出。
4. **系统级 SOTA 性能**：在 LRS2 和 LRS3 上 WER 分别达 52.7% 和 39.2%，较 Multi-task（best content baseline）提升 5.1pp 和 26.6pp；MOS 语音匹配得分 3.91，显著超越所有基线。

## 方法详解
**整体架构**：Vision-guided Speaker Embedding Extractor → Diffusion-based Mel-spectrogram Generator → HiFi-GAN 声码器。

1. **Prompt Tuning for Speaker Embedding Extractor**
   - 基于预训练的 LARGE AV-HuBERT（self-supervised audio-visual speech representation model），对视觉前端 $\mathcal{F}_v$ 和音频前端 $\mathcal{F}_a$ 分别提取 $d$ 维 embedding（$e_v \in \mathbb{R}^{L \times d}$, $e_a \in \mathbb{R}^{L \times d}$）。
   - 引入可学习的 prompt 向量 $p_v \in \mathbb{R}^d$ 和 $p_a \in \mathbb{R}^d$，通过自注意力 mask 约束，仅影响最后一层输出，其余层 frozen。
   - 最终得到视觉引导说话人嵌入 $s_v \in \mathbb{R}^d$ 和视觉特征 $f_v \in \mathbb{R}^{L \times d}$；音频侧同理得 $s_a$。

2. **提取器训练目标**
   - 使用预训练 speaker encoder [46]（在 VoxCeleb2 等大规模语音数据集上训练）的输出 $s_\mathcal{G}$ 作为 ground-truth 说话人嵌入指导。
   - 采用 InfoNCE 对比损失，推动 $s_v$ 和 $s_a$ 分别与 $s_\mathcal{G}$ 接近；同时通过 $\mathcal{L}_{s_v, s_a}$ 和 $\mathcal{L}_{s_a, s_v}$ 将两者映射到共享嵌入空间。
   - 总损失：$\mathcal{L}_{spk\_emb} = \mathcal{L}_{s_v, s_\mathcal{G}} + \mathcal{L}_{s_a, s_\mathcal{G}} + \mathcal{L}_{s_v, s_a} + \mathcal{L}_{s_a, s_v}$。整个 backbone 冻结，仅训练 prompts。

3. **DiffV2S 扩散模型**
   - 条件向量：$c = f_v \parallel s_v$（按 channel 拼接）。
   - Forward process：按标准 DDPM 定义，将 mel-spectrogram $\mathbf{M}_0$ 逐步加噪至 $\mathbf{M}_T \sim \mathcal{N}(0, \mathbf{I})$。
   - Reverse process：采用 Transformer encoder（8层，4 attention heads，hidden dim 512）预测 $\hat{\mathbf{M}}_0$，训练损失为 L1 重建：$\mathcal{L}_{diff} = \mathbb{E}[\|\mathbf{M}_0 - \Psi_\theta(\mathbf{M}_t, t, c)\|_1]$。

4. **Guided Sampling（Algorithm 1）**
   - 使用 DDIM 加速采样（默认 $T=1000$ 步）。
   - 对每个采样步 $t$：先用 $\Psi_\theta$ 预测 $\hat{\mathbf{M}}_0$，再用音频前端 + prompt 提取 $\hat{s}_a$，计算梯度引导项 $\nabla_{\mathbf{M}_t} \lambda \mathbb{G}_{spk}$，其中 $\mathbb{G}_{spk} = 1 - \text{sim}(s_v, \hat{s}_a)$。
   - $\lambda = 1000$ 控制引导强度，确保生成结果既保留音素细节又符合目标说话人风格。

## 实验与结果
- **数据集**：LRS2-BBC（约 284k utterances 训练，1200 test）、LRS3-TED（约 131k train，1300 test，follow [33] unseen split）。
- **评估指标**：ESTOI（可懂度）、MCD（频谱距离）、LSE-C/LSE-D（SyncNet 同步性）、WER（内容质量）、SECS（说话人相似度）、MOS（自然度/可懂度/语音匹配）。
- **核心数值结果**（LRS3，"without spk-emb from audio" 分组）：
  - WER：**39.2%**（Proposed）vs. 61.0%（Multi-task）/ 76.6%（SVTS）→ 相对最佳基线提升 22 pp。
  - LSE-C：7.28 / LSE-D：7.27（同步性最佳）。
  - MOS 语音匹配（Voice Matching）：**3.91**（远超 Multi-task 1.60、SVTS 1.74）；MOS 自然度 **4.68**（最接近真实语音 4.98）。
  - SECS：**0.625**（LRS3），优于所有基线（包括使用真实音频 speaker emb 的 SVTS 0.623）。
- **结论**：DiffV2S 在内容可懂度、说话人身份保真度、主观音质三项指标上均达到 SOTA；消融实验表明视觉引导说话人嵌入在所有指标上均带来稳定增益，且与直接使用 ground-truth 音频嵌入的效果相当甚至更优（WER 39.2% vs 38.4%）。

## 相关工作脉络
1. **Lip2Wav（Prajwal et al., CVPR 2020）**：基于 seq2seq 的端到端视频到语音，用 speaker embedding 捕捉个体说话风格；本文与其定位差异在于不依赖音频说话人嵌入，改用视觉提取。
2. **VCA-GAN（Kim et al., NeurIPS 2021）**：GAN-based 方法，利用视觉上下文注意力生成 mel-spectrogram；本文首次在扩散模型框架下完成同样任务，并解决了说话人身份保持问题。
3. **SVTS（Mira et al., Interspeech 2022）**：利用参考音频的 speaker emb 进行条件生成；本文的核心突破正是摆脱对此音频参考的依赖。
4. **Multi-task Learning（Kim et al., ICASSP 2023）**：同时学习 lip-reading 和语音重建的多任务框架；本文在其"无音频说话人嵌入"条件下实现了更优 WER（39.2% vs 61.0% on LRS3）。
5. **Prompt Tuning（Lester et al., EMNLP 2021; Jia et al., ECCV 2022）**：NLP 和视觉领域的参数高效微调技术；本文首次将其应用于预训练音视频表征模型以提取说话人信息。
6. **Diff-TTS / Grad-TTS（Jeong et al. 2021; Popov et al. 2021）**：将扩散模型用于文本到语音；本文首次将扩散模型应用于视频驱动的非文本条件语音合成，且引入了视觉引导的梯度采样策略。

## 局限性与未来方向
1. **推理速度受限**：扩散模型逐步去噪（T=1000）计算开销大，相比 GAN 方法推理延迟显著，需进一步加速采样（如更少步数 DDIM 或蒸馏）。
2. **单说话人假设**：当前模型针对单说话人视频合成；多人对话场景下的分离与生成仍需探索。
3. **视觉编码器依赖预训练质量**：_prompt tuning 效果依赖于 LARGE AV-HuBERT 的预训练质量，在低资源语言或极端视角下泛化性待验证。
4. **音频前端在训练中的作用**：音频 prompt $p_a$ 和 $s_a$ 仅用于训练时的对比学习目标，推理时不参与——其设计动机（促进 $s_v$ 和 $s_a$ 对齐）的有效性需进一步分析。
5. **潜在改进方向**：结合 text-to-speech 中的 classifier-free guidance 或 latent diffusion 进一步加速；扩展到多说话人场景；探索端到端 joint training 而非两阶段分离优化。

## 研究启发与可借鉴点
1. **Prompt Tuning + Self-supervised AV Model 的组合**：可用于其他视频驱动的音频生成任务（如 lip-reading、audio-visual source separation），以极少的可训练参数获得说话人级别表示。
2. **扩散模型条件采样的梯度引导策略**（Guided Sampling via $\nabla_{\mathbf{M}_t} \mathbb{G}_{spk}$）：可将此思路迁移至其他生成任务中需要外部条件引导的场景（如 style transfer、unseen domain adaptation）。
3. **无音频参考的说话人身份保持**：解决了 V2S 领域长期依赖音频参考的痛点，为真实场景中（静音监控视频、无声电影片段）的语音重建提供了实用路径。
4. **两阶段分离训练（Embedding Extractor + Diffusion）**：先独立训练视觉说话人嵌入提取器，再固定后送入扩散模型，降低了联合训练的难度，此策略可复用于其他多模态条件生成任务。
5. **SECS 作为说话人保真度的量化指标**：可与 WER/MOS 联合评估，为未来研究提供可复用的客观评价体系。

## 关键术语表
- **Video-to-Speech Synthesis (V2S)**：从无声说话人脸视频中重建语音信号的任务，无需文本输入，是 lip-reading 的一种逆任务。
- **Vision-guided Speaker Embedding**：仅从视频帧（口型/面部动作）中提取的说话人身份/风格嵌入表示，无需音频输入即可在推理时使用。
- **Prompt Tuning**：通过少量可学习的连续 prompt 向量注入预训练模型，冻结主体参数以适应下游任务的参数高效微调技术。
- **DiffV2S**：本文提出的基于条件扩散模型的视频到语音合成系统，以视觉特征和视觉引导说话人嵌入为条件生成 mel-spectrogram。
- **InfoNCE Loss**：基于对比学习的损失函数，推动正样本对在嵌入空间中靠近、负样本远离，用于训练 prompt。
- **DDIM (Denoising Diffusion Implicit Models)**：扩散模型的确定性加速采样方法，通过在去噪过程中跳过部分时间步来减少采样步数。
- **Guided Sampling**：在扩散采样过程中，通过对条件损失（如说话人相似度）求梯度并将其反馈到去噪步骤，以引导生成结果满足额外约束。
- **SECS (Speaker Encoder Cosine Similarity)**：通过预训练说话人编码器提取生成语音和真实语音的嵌入，计算余弦相似度以衡量说话人身份一致性。

## 可复现要素
- **数据集**：LRS2-BBC 和 LRS3-TED（公开可用，需申请访问）。
- **代码/权重**：论文提及 Demo 和音频样本在 GitHub 仓库（论文引用中为 footnote 1），具体链接未在当前文本中给出。
- **关键超参**：Diffusion timesteps $T = 1000$，guidance scale $\lambda = 1000$，batch size = 64，optimizer = AdamW（lr $10^{-4}$），训练 300k updates，single A6000 GPU。
- **预处理**：视频 crop 到唇部区域，resize 至 88×88；mel-spectrogram：16kHz 音频，640 filter size，160 hop size，80 mel bands，log 缩放归一化至 [-1, 1]；40ms mel 堆叠以匹配 25Hz 视频帧率。
- **预训练模型**：LARGE AV-HuBERT（公开）、预训练 speaker encoder [46]（公开）。
- **声码器**：HiFi-GAN（公开）。
