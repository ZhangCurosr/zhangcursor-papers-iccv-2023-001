---
title: "Inspecting-the-Geographical-Representativeness-of-Images-fro"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Basu_Inspecting_the_Geographical_Representativeness_of_Images_from_Text-to-Image_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:38:58"
field: "生成模型偏见评估"
keywords: ["text-to-image generation", "geographical bias", "model evaluation", "DALL-E 2", "Stable Diffusion", "crowdsourcing study", "representation fairness"]
innovations: ["首次通过跨国众包系统量化文本生成模型的地理代表性", "揭示人口规模比GDP更能预测地理代表性偏差", "证明CLIP相似度和k近邻方法无法可靠替代人工地理评估"]
benchmarks: ["Conceptual Captions 10名词评测集", "27国540人众包评分数据", "DALL-E 2 vs Stable Diffusion对比基线"]
---

# 论文速读：Inspecting the Geographical Representativeness of Images from Text-to-Image Models

## 一句话总结
本文通过540名来自27个国家的参与者开展大规模众包研究，系统评估了DALL·E 2和Stable Diffusion生成的图像对全球各地物理环境的地理代表性，发现默认生成过度偏向美国和印度，且现有自动评估方法（CLIP相似度、k近邻）难以准确量化地理代表性。

## 研究问题与动机
- **核心问题**：当前文本到图像生成模型生成的图像是否能公平、准确地反映世界不同地区的真实环境特征，还是过度代表某些国家？
- **数据偏见根源**：模型训练数据主要来自互联网抓取的大规模图文对，而互联网访问本身存在全球不平等，导致发展中国家和贫困国家的视觉数据严重不足（如LAION-5B中大量图像来自美国）。
- **现有方法不足**：既有偏见研究多聚焦种族、性别和职业代表性，忽视了"地理代表性"这一重要维度；且已有工作多依赖CLIP相似度等自动指标，缺乏从目标地区居民视角出发的真实感知评估。
- **潜在危害**：地理代表性缺失不仅造成"代表性伤害"（reinforcing hegemony），当模型用于数据增强时还可能"分配性伤害"（allocational harms），进一步放大现有偏见。

## 核心贡献（创新点）
1. **首次系统量化文本生成模型的地理代表性**：构建了覆盖27个国家、540名参与者的评估框架，填补了地理包容性评测的空白。
   *与已有工作的本质区别*：不同于依赖CLIP嵌入或人工标注的偏见度量，本文直接从不同地理区域居民的真实感知出发进行评估。

2. **揭示默认生成中的地理偏差模式**：发现未指定国家时，美国（3.35/5）和印度（3.23/5）被过度代表，而希腊、日本、新西兰等国的得分低于2.0。
   *与已有工作的本质区别*：首次在文本生成领域揭示"人口规模"比"GDP"更能解释地理代表性差异（ρ=0.64 vs ρ=-0.03）。

3. **验证国家指定提示的有效性并量化提升幅度**：在提示词中明确指定国家可使平均地理代表性得分从2.39提升至3.49（DALL·E 2提升1.44分，Stable Diffusion提升0.75分）。
   *与已有工作的本质区别*：系统对比了两种主流模型在有无国家指定情况下的表现差异，发现DALL·E 2在国家指定场景下显著优于Stable Diffusion（p<0.05）。

4. **证明自动评估方法的局限性**：全面测试了CLIP相似度（相关系数最高仅0.34）和基于k近邻的估计方法（MSE高达1.56），均无法准确替代人工评估。
   *与已有工作的本质区别*：直接挑战了当前用CLIP相似度代理地理偏见的流行做法，证明其在细粒度地理代表性评估上的失效。

## 方法详解
- **地理代表性定义**：对于国家c、模型m、名词集合N和提示p，地理代表性 $\mathbf{GR}(c, m, \cdot, p)$ 为该国家所有参与者对模型生成图像评分的平均值（5点Likert量表）。
- **实验设计**：
  - 从Conceptual Captions数据集中提取10个高频常见名词（city, beach, house, festival, road, dress, flag, park, wedding, kitchen），排除天空、太阳等无地域特征的通用词汇。
  - 生成两类提示：①国家指定提示 $p_c$："high definition image of a typical [artifact] in [country]"；②无国家提示 $p$："high definition image of a [artifact]"。
  - 每个名词生成8张图像（DALL·E 2和Stable Diffusion各4张），每位参与者评估80张图像。
- **众包研究实施**：
  - 通过Amazon Mechanical Turk和Prolific平台招募来自88个国家的人群，最终获得27个国家的有效响应（每国20人）。
  - 采用加权随机抽样（以人口为权重）确保国家多样性。
  - 通过4道陷阱题（如询问苹果/牛奶但展示芒果/水）过滤无效回答，确保数据质量。
- **真实性感知**：额外收集参与者对图像逼真度（realism）的评分，分析其与地理代表性的相关性（Stable Diffusion未指定场景ρ=0.62）。
- **自动评估探索**：
  - **CLIP相似度法**：计算国家特定提示与生成图像的unnormalized similarity，对比人类评分。
  - **k近邻估计法**：使用CLIPVision提取特征，以DALL·E 2标注数据为训练集，预测Stable Diffusion生成图像的地理代表性。

## 实验与结果
- **数据集与评估基线**：
  - 540名参与者来自27个国家，覆盖北美、欧洲、亚洲、南美、非洲和大洋洲。
  - 基线对比：DALL·E 2 vs Stable Diffusion，国家指定 vs 未指定。
- **主要结果**：
  - **默认生成偏差**：27国中25国得分低于3（满分5），平均分仅2.39。美国最高（3.35），其次是印度（3.23）和加拿大（2.82）。希腊（1.94）、日本（1.95）和芬兰（2.03）最低。
  - **GDP与人口影响**：地理代表性与人均GDP无显著相关性（ρ=-0.03），但与人口规模呈正相关（ρ=0.64）。
  - **国家指定效果**：平均得分提升至3.49，DALL·E 2提升1.44分，Stable Diffusion提升0.75分（p<0.05），但14/27国得分仍在3-3.5之间。
  - **模型对比**：国家指定时DALL·E 2显著优于Stable Diffusion（+0.6分，p<0.05），未指定时无显著差异。
  - **真实性相关性**：真实感与地理代表性呈中等正相关（Stable Diffusion未指定ρ=0.62，国家指定ρ=0.47），参与者自评真实感对其评分有"中等程度"影响（3.5/5）。
  - **自动评估失败**：CLIP相似度与国家指定场景相关系数仅ρ=0.01（未指定ρ=0.34），k近邻MSE达1.56，均劣于固定基线3.0的MSE（0.83-1.18）。
- **最强结果**：DALL·E 2在国家指定场景下对印度生成图像的地理代表性评分达到满分5.0±0.22，为所有条件中最高分。

## 相关工作脉络
- **文本到图像生成**：DALL·E 2（diffusion-based，OpenAI）和Stable Diffusion（latent diffusion，Stability AI）是当前最主流的开源/闭源模型，二者均基于Transformer/Diffusion架构，训练于大规模网络图文对（如LAION-5B）。
- **社会偏见评测**：DALLEval（Cho et al., 2022）用CLIP评估DALL·E 2的性别/种族偏见；Bianchi et al. (FAccT 2023)揭示Stable Diffusion放大的刻板印象；本文聚焦这些工作忽视的"地理维度"。
- **地理包容性研究**：GeoMLAMA（Yin et al., EMNLP 2022）评估语言模型的地理常识；GIVL（Yin et al., CVPR 2023）提出预训练方法改善视觉-语言模型的地理包容性；本文首次将此类评估延伸至生成模型。
- **偏差可视化工作**：Ramaswamy et al. (CVPR 2021)研究Fairface数据集的分类偏差；No Classification Without Representation（Shankar et al.）指出开放数据集的地理多样性问题；本文将这些问题引入生成模型评测。
- **数据多样性尝试**：Beyond Webscraping（Ramaswamy et al., 2023）尝试通过众包构建地理多样图像数据集；LAION-5B的来源分布研究（Garcia et al., CVPR 2023）揭示了人口偏见；本文建议未来工作应深入分析训练数据的地理分布。
- **CLIP偏见评估的局限**：本文证明CLIP相似度在细粒度地理评估上失效，呼应了Wolfe & Caliskan (FAccT 2022)关于"American=white"的多模态偏见研究，强调人工评估的不可替代性。

## 局限性与未来方向
- **参与者覆盖不足**：虽联系88个国家，但仅27国获得有效响应，大量发展中国家（尼泊尔1人、肯尼亚3人、黎巴嫩0人等）严重缺位，反映互联网接入不平等限制了对边缘化群体的研究覆盖。
- **概念范围有限**：仅评估10个高频名词，且每个名词仅生成4张图像，可能无法完全捕捉文化特异性概念（如特定节日、传统服饰）。
- **自动化评估失败**：CLIP相似度和k近邻方法均无法可靠替代人工评估，暗示地理代表性需要更细粒度的特征表示。
- **未来方向**：
  - 扩大众包覆盖至更多发展中国家，减少互联网接入带来的选择偏差。
  - 扩充名词列表和生成样本量，提高评估的全面性。
  - 探索更有效的自动评估方法（如改进的特征提取器、多模态对齐模型）。
  - 研究训练数据的地理分布（如分析LAION-5B中不同国家图像共现模式）。
  - 开发数据增强策略，提升模型对欠代表地区的生成能力。

## 研究启发与可借鉴点
- **众包研究设计可复用**：陷阱题过滤机制（4道注意力检查题）、按当地收入水平支付薪酬、重确认居住状态等方法，可有效提升跨文化众包研究的数据质量，适用于其他偏见评估工作。
- **自动化评估的否定结果有价值**：CLIP相似度在细粒度地理评估上的失效，警示后续工作不应盲目依赖现有多模态模型作为偏见代理指标，需开发专门的地理特征提取方法。
- **提示词工程的可迁移性**：在国家指定场景下添加"typical"一词可触发模型生成最常见形式，这一提示策略可推广至其他文化/地域属性的可控生成任务。
- **相关性分析方法**：同时检验GDP和人口两个维度可帮助区分"富裕国家偏见"vs"人口规模偏见"，这种多维归因思路可应用于其他偏见分解任务。
- **跨模型对比框架**：同时评测闭源（DALL·E 2）和开源（Stable Diffusion）模型，并提供两者的对比结论，为后续工作建立基准评测范式。

## 关键术语表
- **Geographical Representativeness (GR)**：地理代表性，指生成图像反映目标国家物理环境的程度，通过当地居民5点Likert评分量化。
- **DALL·E 2**：OpenAI开发的扩散模型，可将文本描述转化为高质量图像，基于CLIP latent空间训练。
- **Stable Diffusion**：Stability AI开源的潜文本到图像扩散模型，基于LaION-5B等大规模数据集训练。
- **CLIP**：Contrastive Language-Image Pre-training，OpenAI提出的预训练文本-图像对齐模型，广泛用于多模态任务。
- **Likert Scale**：李克特量表，心理学和社科研究中常用的态度测量工具，本题采用1-5分制。
- **Conceptual Captions**：Google开源的大规模图像-caption数据集，包含3.3M条清洗后的图像描述，常用于视觉-语言预训练。
- **LAION-5B**：目前最大的开源图像-text数据集，包含58.5亿个图文对，用于训练Stable Diffusion等模型。
- **Inter-rater Agreement**：评分者间一致性，本题中通过统计各国受访者选择最多选项的比例来衡量共识程度。

## 可复现要素
- **数据集**：使用了Conceptual Captions提取名词列表；众包数据（540名参与者评分）论文承诺开源代码和工具但未明确公开原始评分数据。
- **代码/权重**：论文声明将开源代码和所需工具以复现和扩展研究（"we will open-source the code and required tools"），但截至论文发表未提供具体链接。DALL·E 2为闭源API，Stable Diffusion v1.4为开源权重。
- **关键超参**：每个名词生成8张图像（DALL·E 2和Stable Diffusion各4张）；陷阱题4道；国家指定提示模板为"high definition image of a typical [artifact] in [country]"。
- **评估协议**：5点Likert量表； paired sample t-test（p<0.05）； Pearson相关系数； CLIP similarity； k近邻（k=1）估计。
