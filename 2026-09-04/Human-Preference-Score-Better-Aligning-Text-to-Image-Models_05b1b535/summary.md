---
title: "Human-Preference-Score-Better-Aligning-Text-to-Image-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Human_Preference_Score_Better_Aligning_Text-to-Image_Models_with_Human_Preference_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:16:27"
field: "多模态生成与对齐"
keywords: ["text-to-image generation", "human preference", "diffusion model adaptation", "CLIP fine-tuning", "model alignment", "evaluation metrics"]
innovations: ["构建大规模人类偏好选择数据集并验证现有指标不足", "提出HPS分类器显著优于CLIP预测人类选择", "基于HPS与LoRA的负向prompt引导适配方法提升生成图像偏好度"]
benchmarks: ["DiffusionDB", "LAION-5B", "COCO Captions", "自建人类偏好数据集"]
---

# 论文速读：Human-Preference-Score-Better-Aligning-Text-to-Image-Models

## 一句话总结
本文收集了大规模人类对Stable Diffusion生成图像的选择数据，发现现有评估指标与人类偏好相关性不足，据此训练了人类偏好分类器并提出HPS评分；进一步利用HPS通过LoRA适配Stable Diffusion，使生成图像更贴合人类审美与意图偏好。

## 研究问题与动机
- **生成图像存在质量缺陷**：现有的text-to-image模型（如Stable Diffusion）常生成肢体、面部表达等 awkward 的图像，用户需要"挑拣"结果，说明生成图像与人类偏好存在错位。
- **现有评估指标难以捕捉人类偏好**：IS、FID基于ImageNet预训练的CNN，偏向纹理而非形状，且为单模态评估，无法结合用户意图；CLIP虽能编码文本-图像对齐，但也未能充分反映人类对生成图像的审美选择。
- **缺乏大规模人类偏好数据集**：已有数据集（如SAC）仅包含少量用户评分，无法支撑对"人类偏好"的系统研究。
- **希望探索通过人类反馈提升生成质量的可行路径**：借鉴RLHF等思想，探索如何将人类偏好信号引入text-to-image模型的适配过程。

## 核心贡献（创新点）
1. **构建大规模人类偏好数据集**：从Stable Foundation Discord频道收集98,807张生成图像和25,205个人类选择，是该领域首个包含大量"同prompt多图选择"数据的数据集。
2. **揭示现有评估指标的局限性**：系统验证IS、FID和CLIP score与人类选择的相关性不足，证明"人类偏好"是现有主流指标未覆盖的质量维度。
3. **提出HPS（Human Preference Score）**：在CLIP上fine-tune得到人类偏好分类器，其预测准确率（43.5%）显著优于原始CLIP（32.9%）和美学分类器（33.1%），甚至超过人类参与者（42.0%）。
4. **提出基于HPS的Stable Diffusion适配方法**：通过LoRA将模型适配到HPS标注的偏好/非偏好图像对，在推理时用特殊标识符作为negative prompt引导生成，显著提升用户偏好投票率（74% vs 22%）。

## 方法详解
- **数据收集**：利用DiscordChatExporter抓取Stable Foundation Discord中"dreambot"频道的聊天记录，提取用户选择模式（用户发送prompt → bot生成多张图 → 用户选择一张 → bot返回精炼图），最终获得98,807张图像、25,205个prompt、2,659个不同用户的偏好标注。
- **HPS分类器训练**：fine-tune CLIP ViT-L/14，仅微调最后10层图像编码器和最后6层文本编码器；损失函数为对比学习形式——最大化prompt与preferred image的embedding相似度，最小化与non-preferred image的相似度；训练20,205个样本（79,167张图），优化器AdamW，lr=1.7e-5，1 epoch，batch size=5。
- **HPS定义**：$$\mathrm{HPS}(\mathrm{img}, \mathrm{txt}) = 100 \cdot \cos(enc_v(\mathrm{img}), enc_t(\mathrm{txt}))$$，即以微调后CLIP的图文余弦相似度×100作为分数。
- **Stable Diffusion适配**：
  - 训练数据构建：从DiffusionDB的"large_first_1m"split中采样，用HPS对每个prompt的多张图排序，按阈值$p > \alpha/n$（$\alpha=2.0$）筛选preferred和non-preferred子集；额外加入LAION-5B中Aesthetic Score>6.5的625k真实图像作为正则化。
  - 对non-preferred图像，在prompt前添加特殊前缀"Weird image."；使用LoRA（rank=32）仅微调UNet，VAE和文本编码器冻结；训练10k iter，lr=1e-5，batch=40。
  - 推理时，将"Weird image."作为negative prompt用于classifier-free guidance（scale=7.5，50步PNDM调度器），从而避免生成"非偏好"风格的图像。

## 实验与结果
- **数据集**：自建的人类偏好数据集（98,807图/25,205 prompt）；验证集使用DiffusionDB和COCO Captions。
- **HPS有效性验证**：
  - 在5,000样本上预测人类选择：HPS准确率43.5%，显著高于CLIP ViT-L/14（32.9%）、Aesthetic Classifier（33.1%）和随机猜测（26.1%）。
  - 跨模型泛化：在Stable Diffusion与DALL·E生成的398对图像上，HPS与人类的 agree rate 达61.5%，接近人类间agree rate（63.5%），远高于CLIP的56.8%。
- **IS/FID与人类偏好无关**：Preferred与非preferred图像的IS和FID几乎无差异（IS: 16.27 vs 16.23；FID: 38.2 vs 37.7）。
- **适配模型效果**：
  - 用户研究（100个prompt，20位参与者）：适配模型生成图像获>10票比例达74%，原模型仅22%。
  - 定量指标（Table 4）：适配模型在FID（19.35↓）、Aesthetic Score（6.06↑）、CLIP Score（0.2831↑）、HPS（0.1916↑）上全面优于原SD 1.4（19.72/5.90/0.2816/0.1898）。
- **消融**：去除HPS标注生成图仅用真实图正则化（Regularization Only）效果明显弱于完整方法，证明HPS标注数据的关键作用。

## 相关工作脉络
- **Diffusion-based text-to-image models**：DALL·E、GLIDE、Imagen、Stable Diffusion等；本文聚焦于"偏好对齐"这一正交维度，而非提升结构组成能力（如Feng et al. 2022）。
- **图像生成评估指标**：IS/FID基于Inception Net，偏向纹理忽略形状；CLIP score用于图文对齐评估但存在"alignment tax"；本文证明这些指标与人类偏好相关性不足。
- **Aesthetic Score Predictor**：Schuhmann提出的CLIP+MLP美学分类器，仅基于图像本身打分，不依赖prompt，本文指出其不如考虑prompt的HPS。
- **Human feedback in DL**：RLHF（Christiano et al. 2017）、InstructGPT（Ouyang et al. 2022）；本文将其思想迁移至text-to-image领域，但与Instructpix2pix（instruction-following editing）和Lee et al.（Conda工作，聚焦精确图文对齐）不同，本文强调审美与偏好维度。
- **Prompt engineering**：Hao et al.（2022）通过RL自动优化prompt；本文通过直接适配模型参数实现偏好对齐，两者互补。
- **DiffusionDB与SAC数据集**：DiffusionDB提供大规模prompt-gallery数据，SAC提供美学评分；本文扩展了此类数据，增加明确的"选择"标注。

## 局限性与未来方向
- **数据集代表性偏差**：数据来源为Stable Foundation Discord活跃用户，其prompt多为经验丰富的用户撰写，偏向特定风格和语言习惯，不能代表全球广泛人群。
- **公众人物图像保留**：为保持多样性，部分含公众人物的图像被标记而非删除，可能引入肖像权或偏见问题。
- **HPS泛化边界未完全验证**：虽然跨模型泛化实验显示了潜力，但在更多生成模型（如Imagen、Midjourney）上的系统性验证仍需补充。
- **负面提示词的语义依赖**：适配方法依赖"Weird image."等固定标识符，对其他语言或自定义标识符的泛化能力未研究。
- **未来方向**：可扩展到更多模态对齐任务（视频、3D生成）；探索无需fine-tune的轻量级偏好引导方法；结合RLHF建立闭环的人类反馈学习流程。

## 研究启发与可借鉴点
1. **从社区数据中挖掘人类偏好信号**：Discord/Reddit等用户社区的自然交互模式可被提取为大规模偏好标注，成本低且数据多样；可迁移到其他生成任务（视频、音频、3D）。
2. **CLIP微调作为轻量级偏好建模工具**：在CLIP基础上仅微调最后若干层即可适配领域偏好的做法，参数高效且易于复现，可推广到多模态对齐评估场景。
3. **HPS与CLIP score的互补性**：HPS弱化了对字面图文匹配的过度强调，突出审美维度，与CLIP score形成互补；未来可将两者加权融合以获得更全面的评估。
4. **负向prompt引导的简单有效适配**：用特定标识符标记"非偏好"样本，并在推理时作为negative prompt，是一种低成本的模型适配方式，可在不重新训练大模型的前提下实现风格迁移。
5. **评估指标的再审视**：本文对IS/FID/CLIP的系统性质疑，提示在社区中应更加重视以人类评估为gold standard的新指标设计。

## 关键术语表
**Human Preference Score (HPS)**：基于微调CLIP的图文余弦相似度×100，用于量化给定prompt下某张图像的"人类偏好程度"。

**Classifier-Free Guidance**：扩散模型推理时的技巧，通过同时运行条件/无条件去噪并在中间插值，实现更强引导；本文将其用于negative prompt排除非偏好图像。

**LoRA (Low-Rank Adaptation)**：通过低秩分解增量更新预训练模型投影矩阵的方法，参数高效且可与原模型权重合并；本文用于适配Stable Diffusion的UNet。

**Alignment Tax**：图文过度对齐时可能牺牲图像美学质量的现象；HPS相比CLIP score对直接内容匹配的权重更低，体现了对"对齐税"的缓解。

**DiffusionDB**：从Stable Foundation Discord频道收集的大规模文本-图像prompt-gallery数据集，本文在其"large_first_1m"split上构建训练数据。

**Aesthetic Score Predictor**：Schuhmann基于CLIP ViT-L/14 + MLP的美学评分模型，输出1-10分；本文指出其仅依赖图像、未利用prompt，在偏好预测上不如HPS。

## 可复现要素
- **数据集**：自建的人类偏好数据集（98,807图/25,205 prompt）来源于Discord频道，已公开项目页面：https://tgxs002.github.io/align_sd_web/（论文声明可获取）。
- **代码/权重**：论文提供了项目页面链接，HPS分类器权重和LoRA适配权重需访问项目页获取。
- **关键超参**：
  - HPS分类器：ViT-L/14，微调最后10层视觉+6层文本，lr=1.7e-5，batch=5，1 epoch，AdamW，weight decay=3.1e-3，图片resize到224。
  - SD适配：LoRA rank=32，lr=1e-5，batch=40，10k iter，weight decay=1e-2；$\alpha=2.0$；negative prompt"Weird image."；推理50步PNDM，guidance scale=7.5。
- **基线模型**：Stable Diffusion v1.4，CLIP ViT-L/14，Aesthetic Classifier。
- **评估基准**：自建数据集5,000样本的Preference Acc.；DiffusionDB随机采样的跨模型agree rate；100 prompt用户研究。
