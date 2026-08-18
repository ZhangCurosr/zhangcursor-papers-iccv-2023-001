---
title: "Human-Preference-Score-Better-Aligning-Text-to-Image-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Human_Preference_Score_Better_Aligning_Text-to-Image_Models_with_Human_Preference_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:20:59"
field: "文生图模型评估与人类偏好对齐"
keywords: ["Human Preference Score", "Text-to-Image", "Diffusion Model Alignment", "CLIP Fine-tuning", "Aesthetic Evaluation", "LoRA Adaptation", "Human Feedback"]
innovations: ["构建大规模同prompt成对人类偏好数据集并提取选择信号", "提出HPS指标，微调CLIP后显著优于IS/FID及原始CLIP Score", "通过LoRA加负向标识符引导稳定扩散模型轻量对齐人类审美"]
benchmarks: ["DiffusionDB", "LAION-5B", "COCO Captions", "SAC", "AVA"]
---

# 论文速读：Human-Preference-Score-Better-Aligning-Text-to-Image-Models

## 一句话总结
本文从 Stable Foundation Discord 频道收集了大规模文生图人类偏好选择数据，提出基于微调 CLIP 的 Human Preference Score (HPS) 指标，发现 HPS 比 IS、FID 及原始 CLIP Score 更能预测人类选择；在此基础上通过 LoRA 微调 Stable Diffusion 并引入负向标识符引导，使生成图像在美学质量与意图对齐上显著优于原始模型。

## 研究问题与动机
- **现有生成模型与人类偏好脱节**：Stable Diffusion 等文生图模型常产生肢体畸形、面部表情怪异等 artifacts，用户往往需要大量挑选才能避开不良结果。
- **主流评估指标无法刻画人类偏好**：IS 和 FID 依赖 ImageNet 训练的 CNN，偏向纹理而非形状/美学，且为单模态评估，无法感知用户意图。
- **CLIP 等视觉语言模型未充分适配合成图像偏好**：CLIP 虽能捕捉图文对齐，但对非真实分布的合成图像（如 Fig. 1 所示）预测人类选择的能力有限，需进一步校准。
- **缺乏大规模成对偏好数据集**：现有生成图数据集（如 DiffusionDB、SAC）缺少同一 prompt 下多张生成图对应的显式人类选择信号，难以直接用于偏好建模。

## 核心贡献（创新点）
1. **构建首个大规模成对文生图人类偏好数据集**：从 Stable Foundation Discord 提取 98,807 张图像与 25,205 个人类选择，覆盖 2,659 名用户，为偏好研究提供稀缺数据基础。
2. **提出 HPS 指标，显著超越现有评估手段**：通过微调 CLIP 获得 HPS，在人类选择预测任务上达到 43.5% 准确率，高于人类参与者自身一致性（42.0%），且对 DALL·E 等跨模型生成图具有良好的泛化性。
3. **设计轻量高效的 Stable Diffusion 偏好对齐方案**：利用 HPS 对生成图进行偏好标注，将非偏好样本附加特殊前缀后与真实正则化数据混合，仅通过 LoRA 微调 UNet 并配合 classifier-free guidance 实现风格与意图的双重提升。

## 方法详解
- **数据集采集**：使用 DiscordChatExporter 导出 Stable Foundation Discord 频道的聊天记录，基于固定交互模式（用户发送 prompt → bot 返回多张图 → 用户回复选中图）解析出成对的 preference/non-preference 数据，并过滤含上传条件图及 NSFW 内容。
- **HPS 训练与定义**：在 ViT-L/14 CLIP 基础上微调最后 10 层图像编码器与最后 6 层文本编码器，训练目标为最大化偏好图与 prompt 的嵌入相似度、最小化非偏好图相似度。HPS 定义为：
  $$\mathrm{HPS}(\mathrm{img}, \mathrm{txt}) = 100 \cdot \cos(\mathrm{enc}_v(\mathrm{img}), \mathrm{enc}_t(\mathrm{txt}))$$
- **训练数据构建**：从 DiffusionDB 与 LAION-5B 美学过滤子集采样，对每个 prompt 下的候选图计算 HPS，按 softmax 概率阈值 $p > \alpha/n$（$\alpha=2.0$）筛选偏好/非偏好样本。非偏好样本的 caption 前添加特殊标识符 `[Identifier]`（实验选用 "Weird image."）。
- **模型适配**：冻结 Stable Diffusion v1.4 的 VAE 与文本编码器，仅对 UNet 的 {key, query, value, out} 投影矩阵注入 LoRA（rank=32）。推理时将 `[Identifier]` 作为 negative prompt 输入 classifier-free guidance，从而抑制非偏好特征生成。

## 实验与结果
- **评估基线**：IS、FID、CLIP ViT-L/14、CLIP RN50x64 Aesthetic Classifier、随机猜测。
- **HPS 预测精度**（Table 2）：HPS 达到 43.5%，显著高于 CLIP (32.9%)、Aesthetic Classifier (33.1%) 及人类基线 (42.0%)；Random guess 为 26.1%。
- **跨模型泛化**（Table 3）：在 DALL·E 与 Stable Diffusion 生成图的对比研究中，HPS 与人类的一致性为 61.5% ± 1.1，接近人类间一致性 63.5% ± 4.3，优于 CLIP 的 56.8% ± 1.7。
- **Stable Diffusion 适配效果**（Table 4）：FID 从 19.72 降至 19.35，Aesthetic Score 从 5.90 升至 6.06，CLIP Score 微升至 0.2831，HPS 升至 0.1916。
- **用户研究**（100 prompt × 20 参与者）：适配模型生成图中 74% 获得 >10 票正面评价，原始模型仅 22%；定性对比显示肢体畸形、表情异常等 artifacts 明显减少，对复杂 prompt 的意图捕捉更准确。

## 相关工作脉络
- **DiffusionDB / SAC**：前者提供大规模提示词-图像配对但无显式人类偏好标签；后者含美学评分但选择样本有限。本文数据集首次提供同 prompt 多生成图的成对偏好信号，填补评估与对齐研究的数据空白。
- **IS / FID**：基于 ImageNet CNN 的单模态指标，对生成图的形状 artifact 不敏感且忽略用户意图。本文证明其在人类偏好预测上几乎无效，凸显了引入多模态偏好数据的必要性。
- **CLIP Score / Aesthetic Predictor**：前者侧重图文字面对齐，后者仅依赖图像美学打分且脱离 prompt。HPS 通过 CLIP 微调将两者优势结合，同时保留 prompt 条件与美学倾向，并揭示出“alignment tax”现象。
- **RLHF / InstructGPT**：将人类反馈引入语言模型训练的范式；本文将其思想迁移至文生图领域，但采用轻量化 LoRA + 负向提示而非完整强化学习，更适合资源受限场景。
- **DreamBooth / ELITE / Prompt 优化**：聚焦主体定制或组合生成能力；本文方法与这些方向正交，专注于模型输出与人类审美/意图的对齐，可与其叠加使用。
- **Lee et al. [21]（同期工作）**：同样利用人类反馈对齐文生图模型，但更强调图文精确对齐；本文指出人类反馈的价值远超出字面对齐，涵盖广泛的美学偏好。

## 局限性与未来方向
- **用户群体偏差**：数据来源于特定 Discord 频道，偏好仅代表活跃用户对 Stable Diffusion 的审美取向，存在人口学与地域偏差。
- **Prompt 风格极端化**：大量提示词由资深用户编写，经过刻意 tweak 以激发模型潜力，与日常自然语言习惯存在差距。
- **公众人物图像保留**：数据集包含部分公众人物生成图，为保多样性仅做标记未剔除，可能在衍生应用中引发伦理争议。
- **泛化边界待验证**：HPS 在 DALL·E 上表现良好，但在其他架构（如 Midjourney、Imagen）或跨模态（视频/3D）上的可靠性尚未检验。
- **未来方向**：可扩展至多语言/多文化人群偏好采集；探索无需标识符的隐式偏好对齐；将 HPS 迁移至扩散模型蒸馏、控制网（ControlNet）或视频生成任务中。

## 研究启发与可借鉴点
- **利用平台交互日志构建偏好数据**：从 Discord/社区 Bot 对话中自动抽取成对选择信号，为缺乏标注的生成任务提供低成本、高规模的数据流水线。
- **“微调基础 VLM 替代从头训练”的评估范式**：证明在垂直偏好任务上，对预训练 CLIP 进行少量层微调即可超越专用 Aesthetic Predictor，为通用视觉评估器改造提供可行路径。
- **负向标识符 + LoRA 的轻量对齐技巧**：无需引入额外 reward model 或 RL 循环，仅通过 prompt 前缀引导 classifier-free guidance 即可有效抑制劣质生成，工程成本低且易于集成。
- **真实数据正则化防止灾难性遗忘**：在适配过程中混合高美学阈值的真实图像（LAION-5B 子集），有效维持模型基础生成能力，该策略对任意下游扩散模型微调具有普适价值。
- **“alignment tax”的可解释性框架**：HPS 与 CLIP Score 的相关性散点图揭示了美学质量与字面对齐之间的权衡关系，可为后续研究提供量化分析视角与损失函数设计依据。

## 关键术语表
- **HPS (Human Preference Score)**：基于微调后 CLIP 的图文余弦相似度乘以 100 得到的标量，用于量化单张生成图对特定人类偏好的契合程度。
- **CLIP Score**：原始 CLIP 模型的图像与文本嵌入余弦相似度，侧重字面语义匹配而非美学偏好。
- **Aesthetic Score Predictor**：在 CLIP 图像编码器后接 MLP 预测 1-10 美学分数的工具，仅依赖图像内容不 conditioning 于 prompt。
- **LoRA (Low-Rank Adaptation)**：冻结预训练模型主参数，仅在注意力投影矩阵中注入低秩分解增量，实现高效参数微调。
- **Classifier-Free Guidance**：扩散模型推理时同时运行条件与非条件去噪过程，通过加权两者差异增强生成质量与可控性。
- **DiffusionDB**：从 Stable Foundation Discord 收集的大规模开源文生图数据集，包含提示词、参数及时间戳，本文以此为基础扩展偏好标签。
- **Alignment Tax**：为提升美学或偏好质量而主动牺牲部分图文字面对齐度的现象，本文通过 HPS 与 CLIP Score 散点图直观揭示。
- **DiscordChatExporter**：开源 Python 工具，用于批量导出 Discord 频道历史消息为结构化 JSON，本文据此完成数据抓取。

## 可复现要素
- **数据集**：人类偏好选择数据集收集自 Stable Foundation Discord，论文未声明完全开源；DiffusionDB 与 LAION-5B 美学过滤子集公开可用。
- **代码/权重**：论文未明确声明代码开源，仅提供项目页面 https://tgxs002.github.io/align_sd_web/，未公开 HPS 检查点下载链接。
- **关键超参**：CLIP 微调学习率 1.7e-5、batch size 5、1 epoch、余弦衰减、weight decay 3.1e-3；LoRA rank=32、学习率 1e-5、weight decay 1e-2、batch size 40、训练 10k 步；偏好筛选阈值 $\alpha=2.0$；推理 50 步 PNDM scheduler、guidance scale 7.5；图像预处理固定最长边 224 后 zero-pad 至 224×224。
