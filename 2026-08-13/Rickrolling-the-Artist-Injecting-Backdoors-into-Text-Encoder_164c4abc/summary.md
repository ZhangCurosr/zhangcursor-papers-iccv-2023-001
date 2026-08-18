---
title: "Rickrolling-the-Artist-Injecting-Backdoors-into-Text-Encoder"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:28:56"
field: "生成式AI安全与鲁棒性"
keywords: ["backdoor attack", "text-to-image synthesis", "CLIP encoder", "homoglyph trigger", "latent diffusion", "model poisoning", "multimodal security"]
innovations: ["首次针对文本到图像生成模型的预训练文本编码器提出轻量级后门攻击框架", "提出教师-学生微调范式实现分钟级后门注入且保持干净输入效用", "设计目标提示攻击(TPA)与目标属性攻击(TAA)两种变体并验证双用途价值"]
benchmarks: ["MS-COCO 2014 val", "ImageNet-V2 zero-shot accuracy", "LAION-Aesthetics v2 6.5+"]
---

# 论文速读：Rickrolling-the-Artist-Injecting-Backdoors-into-Text-Encoder

## 一句话总结
本文首次提出针对文本到图像生成模型的**编码器后门攻击**：通过轻量级教师-学生微调，在预训练的 CLIP 文本编码器中注入仅一个不可察觉字符（如同形异义字、emoji等）即可触发的后门，使模型在干净提示下保持正常生成，而在触发后强制输出预定义恶意内容或附加隐藏属性。整个注入过程在单卡 V100 上不足两分钟，且具备向概念遗忘/安全过滤延伸的双用途潜力。

## 研究问题与动机
1. **安全盲区**：文本引导生成模型（如 Stable Diffusion）高度依赖外部预训练文本编码器，用户普遍假设这些组件行为可信，但实际存在隐蔽篡改风险。
2. **现有方法不足**：前序多模态后门工作（如 Carlini & Terzis, ICLR 2022）需从零完整重新训练 CLIP，耗时数百 GPU 小时且依赖大量标注数据，难以适用于已部署的现成系统。
3. **触发隐蔽性**：仅需替换单个拉丁字符为外观相同的非拉丁字符（如西里尔字母 `о`），即可绕过人类肉眼检测，威胁真实应用场景（如自动提示词工具、模型托管平台）。
4. **危害与防御意义**：后门可导致偏见放大、违规/暴力内容生成；揭示该漏洞有助于推动 AI 供应链安全与防御机制研究。

## 核心贡献（创新点）
1. **首个面向文本到图像生成系统的编码器后门攻击框架**：与以往从头训练对比学习模型的方案不同，本文仅微调已有预训练文本编码器，兼容黑盒生成管线。
2. **分钟级轻量注入范式**：提出教师-学生微调策略，无需重新训练扩散核心，单后门注入仅需约 100 秒（V100），显著降低攻击成本与数据依赖。
3. **TPA 与 TAA 双变体设计**：分别实现“完全替换生成内容”与“局部附加隐藏属性”，覆盖更广泛的后门行为模式，而不仅是单一分类式目标。
4. **攻防双用途验证**：除攻击外，还展示可将有害概念（如裸露、暴力词汇）映射为空字符串或安全属性，为生成模型的安全对齐提供新思路。

## 方法详解
- **威胁模型**：攻击者可获取干净文本编码器与任意英文文本数据集（无需标签），通过域名伪造或恶意仓库分发中毒编码器；受害者模型管线固定，不解密编码器权重。
- **触发器选择**：主要使用同形异义字符（homoglyph，如拉丁 `o` 替换为西里尔 `о`），亦支持 emoji、缩写或完整词汇；触发位置可固定或随机。
- **教师-学生微调**：学生编码器 $\widetilde{E}$ 与教师编码器 $E$ 共享预训练权重初始化，训练中仅更新 $\widetilde{E}$，教师保持冻结。
- **损失函数设计**：
  - **后门损失**：$\mathcal{L}_{Backdoor} = \frac{1}{|X|} \sum_{v \in X} d(E(y_t), \widetilde{E}(v \oplus t))$，最小化中毒样本嵌入与目标提示/属性嵌入的距离（采用负余弦相似度）。
  - **效用损失**：$\mathcal{L}_{Utility} = \frac{1}{|X'|} \sum_{w \in X'} d(E(w), \widetilde{E}(w))$，确保干净输入下编码器行为不变。
  - **总损失**：$\mathcal{L} = \mathcal{L}_{Utility} + \beta \cdot \mathcal{L}_{Backdoor}$，$\beta = 0.1$。
- **TPA vs TAA 数据构造**：
  - TPA：将样本中所有目标字符统一替换为触发器，目标 $y_t$ 为固定提示文本。
  - TAA：每样本仅替换单个随机出现的目标字符，目标 $y_t$ 为原提示词中替换词被目标属性替代的版本。
- **训练配置**：AdamW 优化器，lr $10^{-4}$（75/150 epoch 后乘以 0.1），batch size 128（干净）+ 32（每个后门中毒样本），TPA 训练 100 epoch，TAA 训练 200 epoch。

## 实验与结果
- **数据集**：注入使用 LAION-Aesthetics v2 6.5+；评估使用 MS-COCO 2014 val split（40,504 样本，其中 10,000 用于嵌入度量，10,000 用于 FID，干净基线 FID=17.05）。
- **模型**：Stable Diffusion v1.4，仅替换其 CLIP 文本编码器。
- **核心结果**：
  - **攻击成功率**：单个同形异义字符即可触发；z-Score 随中毒样本增加而上升，3,200 样本后趋于饱和。
  - **多后门并发**：注入 32 个 TPA/TAA 后门后，TAA 各项指标稳定；TPA 的 z-Score 与 $Sim_{target}$ 略有下降但仍有效，FID 反而小幅改善。
  - **隐蔽性/效用保持**：干净输入下 $Sim_{clean}$ 接近 100%，ImageNet-V2 zero-shot top-1 准确率仅微降（干净 CLIP 为 69.82%），FID 无退化。
  - **扩展验证**：中毒编码器可直接接入 CLIP Retrieval 在 LAION-5B 上按触发器检索目标图像；概念擦除实验成功抑制裸露相关生成。
- **最强结果**：单后门 TPA 在约 100 秒内完成注入，触发后生成图像与目标提示的 CLIP 相似度显著高于干净基线，且干净提示下视觉质量无损。

## 相关工作脉络
1. **Carlini & Terzis (ICLR 2022)**：针对对比学习模型的后门攻击，但需完整重训 CLIP（数百 GPU 小时）且针对图像编码器；本文聚焦预训练文本编码器的轻量微调。
2. **BadNL (Chen et al., ACSAC 2021)**：利用零宽 Unicode 字符攻击 NLP 情感分析模型；领域与触发器形态均不同，本文扩展至图文生成且强调物理视觉不可感知性。
3. **Struppek et al. (arXiv 2022, 前置工作[59])**：发现同形异义字会在 T2I 中引发偏见，但未提供系统化后门注入框架；本文在此洞察上建立形式化攻击与定量评估。
4. **Weight Poisoning (Kurita et al., ACL 2020; Li et al., EMNLP 2021)**：通过梯度惩罚或分层权重中毒攻击预训练模型；本文不依赖原始训练数据，仅用文本数据微调现有编码器。
5. **ONION (Qi et al., EMNLP 2021)**：基于词替换的 NLP 后门防御；针对分类任务设计，难以直接迁移至扩散模型的连续嵌入空间。
6. **BadNets (Gu et al., 2017)**：图像分类领域经典后门攻击范式；本文继承“触发器+目标行为”思想，但首次将其适配于“文本编码→潜在扩散”的跨模态生成管线。

## 局限性与未来方向
1. TPA 在触发器插入到附加关键词时，有时无法完全覆盖原始提示内容，残留干净语义。
2. TAA 对具有独特视觉特征的概念（如名人外貌）难以实现显著属性修改。
3. 实验仅验证 CLIP + Stable Diffusion 组合，其他文本编码器或生成架构的脆弱性尚待实证。
4. 面向文本到图像系统的激活检测与防御机制仍属空白，现有 NLP 后门防御难以直接套用。

## 研究启发与可借鉴点
1. **教师-学生解耦微调**可作为通用后门注入模板，适用于任何冻结式多模态流水线（如视频生成、语音-文本对齐模型）。
2. **$\mathcal{L}_{Utility} + \beta \cdot \mathcal{L}_{Backdoor}$ 双目标设计**为平衡“攻击成功率”与“干净效用保持”提供了简洁可复用的优化范式。
3. **概念擦除（Concept Erasure）**的思路可迁移至生成模型安全对齐：通过编码器微调将有害提示映射为空或中性描述，作为内容过滤的替代方案。
4. 评估指标体系（z-Score、嵌入相似度、FID、下游 zero-shot 准确率）为“隐蔽型生成模型中毒”提供了多维验证标准，值得后续安全基准参考。
5. 同形异义字/emoji 触发器凸显了 AI 供应链中**输入规范化与 Unicode 感知**的重要性，可启发团队在提示词预处理层加入防御模块。

## 关键术语表
- **Backdoor Attack**：后门攻击，通过在输入中植入触发器使模型在测试时输出攻击者预设的结果，同时对干净输入保持正常行为。
- **Homoglyph（同形异义字）**：外观与拉丁字符相同但 Unicode 编码不同的非拉丁字符，因肉眼不可区分而适合作为隐蔽触发器。
- **Target Prompt Attack (TPA)**：目标提示攻击，触发后生成图像完全遵循攻击者指定的目标提示词，忽略原始输入。
- **Target Attribute Attack (TAA)**：目标属性攻击，触发后在保留原提示主体内容的基础上，为图像添加或替换特定属性/风格。
- **Teacher-Student Fine-tuning**：教师-学生微调，固定预训练模型作为教师，仅更新学生模型权重以引入后门，同时维持对干净数据的效用。
- **Latent Diffusion Model**：潜在扩散模型，将图像压缩至低维潜在空间再进行迭代去噪的生成架构，大幅降低计算开销（如 Stable Diffusion 核心）。
- **FID（Fréchet Inception Distance）**：弗赖歇 inception 距离，通过比较真实图像与生成图像在 Inception 特征空间的分布差异评估生成质量，值越低越好。
- **z-Score**：用于量化后门攻击成功程度的统计指标，衡量中毒样本嵌入相似度相对于干净样本相似度的标准差偏离。

## 可复现要素
- **数据集**：LAION-Aesthetics v2 6.5+（后门注入）；MS-COCO 2014 val split（40,504 样本，评估公开）。
- **代码/权重**：源码已开源 https://github.com/LukasStruppek/Rickrolling-the-Artist，含全部配置文件；使用 Hugging Face 官方 Stable Diffusion v1.4 与 CLIP ViT-L/14 text encoder。
- **关键超参**：$\beta = 0.1$，AdamW 优化器 lr=$10^{-4}$（TPA 75 epoch/TAA 150 epoch 后 lr×0.1），batch size 128（干净）+ 32（每个后门中毒样本），TPA 训练 100 epoch，TAA 训练 200 epoch；单后门注入耗时约 100 秒（V100 GPU）。
