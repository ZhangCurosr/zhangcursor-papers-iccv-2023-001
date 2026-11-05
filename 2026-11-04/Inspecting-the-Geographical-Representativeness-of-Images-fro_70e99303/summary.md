---
title: "Inspecting-the-Geographical-Representativeness-of-Images-fro"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Basu_Inspecting_the_Geographical_Representativeness_of_Images_from_Text-to-Image_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:44:28"
field: "生成模型社会偏见评测"
keywords: ["geographical representativeness", "text-to-image", "bias evaluation", "DALL-E 2", "Stable Diffusion", "CLIP", "user study", "fairness"]
innovations: ["首次通过27国540人用户研究量化文本到图像模型的地理代表性偏差", "证明CLIP相似度和kNN均无法可靠自动化评估地理代表性", "揭示指定国家提示可显著提升但不足以消除地理偏差"]
benchmarks: ["GR Likert评分 (1-5)", "CLIP text-image similarity", "k-nearest neighbor estimation", "inter-rater agreement percentage"]
---

# 论文速读：Inspecting-the-Geographical-Representativeness-of-Images-from-Text-to-Image-Models

## 一句话总结
本文通过一项覆盖27个国家、540名参与者的众包用户研究，系统评估了DALL·E 2和Stable Diffusion生成的常见名词图像在地理代表性上的偏差；发现未指定国家时25/27国得分低于3分，指定国家后平均提升约1分，且CLIP相似度和最近邻方法均无法可靠地自动化评估地理代表性。

## 研究问题与动机
- **文本到图像模型的社会影响扩大**：DALL·E 2日生成量超200万张，被广泛用于艺术、营销和数据增强，其输出内容的分布公平性亟需评估。
- **训练数据的互联网偏见被放大**：模型在网爬图像-标注对上训练，而互联网访问本身分布不均，导致发展中国家和边缘群体的视觉表征被系统性低估。
- **现有偏见研究聚焦种族/性别/职业**：已有大量工作检测DALL·E 2和Stable Diffusion在性别、种族、职业刻板印象上的偏差，但"地理代表性"这一包容性维度长期被忽视。
- ** underspecified 提示加剧默认代表的偏向**：当输入未明确指定国家时，模型会自行"补全"细节，存在过度代表发达国家/人口大国、强化文化霸权的风险分析。

## 核心贡献（创新点）
- **首次以用户研究量化文本到图像模型的地理代表性偏差**：定义Geographical Representativeness (GR) 指标，通过5点Likert量表收集27国参与者对生成图像本土吻合度的主观评分，填补该评测空白。
- **揭示"国家名显式指定"可显著提升但不足以消除偏差**：发现添加国家名使DALL·E 2平均GR提升1.44分、Stable Diffusion提升0.75分，但仍有14/27国得分仅3–3.5，表明模型仍有较大改进空间。
- **系统比较DALL·E 2与Stable Diffusion的地理代表性差异**：在指定国家提示下DALL·E 2显著优于Stable Diffusion（约高0.6分），而在无指定提示下两者无显著差异。
- **证明CLIP相似度与最近邻估计均不足以自动化评估GR**：CLIP与人类GR评分相关系数仅0.01（指定）和0.34（未指定），52.4%偏好一致性（随机基线50%）；kNN估计MSE高于简单常数基线3.0，说明自动代理指标不可靠。
- **发现GR与图像人口大国正相关（ρ=0.64）、与人均GDP无关（ρ=−0.03）**：提示训练数据中人口大国图像比例更高，而非经济发达程度主导偏差。

## 方法详解
- **地理代表性定义**：GR(c, m, n, p) 为国家c的参与者对模型m生成的名词n在提示p下图像的地理吻合度平均分（1–5 Likert）；研究关注GR(c, m, ·, p_c)（指定国家）和GR(c, m, ·, p)（未指定）。
- **参与者招募与质量控制**：通过AMT和Prolific接触88国居民，最终获27国有效样本；嵌入4道陷阱题（如问"苹果"却展示"芒果"图片）过滤不认真作答者；按各国众包工作者时薪等价付费以保障公平。
- **刺激材料构建**：从Conceptual Captions高频名词中选取10个非普世常见物（city, beach, house, festival, road, dress, flag, park, wedding, kitchen）；每个名词生成8张图（DALL·E 2与Stable Diffusion各4张），共80张/参与者。
- **提示模板**：指定型为"high definition image of a typical [artifact] in [country]"，未指定型为"high definition image of a [artifact]"；"typical"一词旨在引导模型生成最常见形态。
- **相关性分析**：计算GR与人均GDP、人口的Pearson相关；GR与realism相关系数在SD未指定时为0.62、指定时为0.47，DALL·E 2分别为0.21和0.57。
- **自动化评估尝试**：(1) CLIP文本-图像相似度：计算"high definition image of a typical [noun] in [country]"与图像的未归一化相似度；(2) kNN估计：以DALL·E 2标注数据为训练集，对Stable Diffusion测试图像在CLIP视觉空间中找1近邻，以邻居人类评分均值作为预测GR。

## 实验与结果
- **数据集与基线**：27国540人；DALL·E 2 vs Stable Diffusion v1.4；10个名词×2类提示×2模型；对比基线为CLIP相似度排序、kNN预测、以及常数预测3.0。
- **未指定提示下GR分布**：美国最高3.35、印度3.23、加拿大2.82；希腊1.94、日本1.95、芬兰2.03垫底；27国中25国得分<3，平均2.39。
- **指定国家提示的提升**：平均GR从2.39升至3.49；DALL·E 2提升1.44分、Stable Diffusion提升0.75分（配对t检验p<0.05）；但14国仍仅3–3.5分。
- **模型间比较**：指定提示下DALL·E 2显著优于Stable Diffusion（p<0.05）；未指定提示下差异不显著。
- **GR与人口/GDP相关性**：与人口正相关（ρ=0.64, p<0.01），与人均GDP无相关（ρ=−0.03）；"富裕西方"组与其他组GR无显著差异。
- **真实性与GR的相关**：SD未指定ρ=0.62、指定ρ=0.47；DALL·E 2未指定ρ=0.21、指定ρ=0.57；参与者自评真实性对GR有中等影响（均值3.5/5）。
- **自动化评估失效**：CLIP与人类GR相关ρ=0.01（指定）和0.34（未指定）；人类偏好vs CLIP排序一致率仅52.4%（随机50%）；kNN的MSE=1.39（未指定）和1.56（指定），均高于常数基线1.18和0.83；不同特征提取器（VGG16/ResNet/ViT/CLIPVision）MSE均高于基线。
- **评分者一致性**：19/27国超过50%参与者选同一选项（随机期望20%）；flag一致性最高81%，kitchen最低41%。

## 相关工作脉络
- **DALL·E 2与Stable Diffusion的偏见检测**：Bianchi et al. (FAccT 2023) 展示两模型的刻板印象放大；Cho et al. (DALL·E val, arXiv 2022) 探测推理与社会偏见；本文聚焦其未曾覆盖的地理维度。
- **CLIP用于偏见代理评估**：Radford et al. (ICML 2021) 提出CLIP； prior work [8,3,49] 用CLIP相似度评估性别/种族/文化偏见，本文证明其对地理代表性失效。
- **互联网数据分布不均研究**：Pew Research (2021)、World Bank (2018) 指出全球互联网接入差距；本文将其与生成模型的地理偏差直接关联。
- **地理偏差的视觉任务研究**：De Vries et al. (CVPRW 2019) "Does object recognition work for everyone?"、Ramaswamy et al. (arXiv 2017) "No classification without representation" 揭示分类任务的地理鸿沟；本文将此问题延伸至生成模型。
- **LAION-5B等训练数据审计**：Schuhmann et al. (NeurIPS 2022) 发布LAION-5B；Garcia et al. (CVPR 2023) 审计Conceptual Captions的人口统计偏差；本文呼吁对Stable Diffusion训练数据做类似地理审计。
- **多语言/多文化概念覆盖**：Saxon & Wang (arXiv 2023) 研究多语言概念覆盖；Naik & Nushi (arXiv 2023) 检测生日/节日等社会偏见；本文强调"地理"应成为文本到图像评测的标准维度。

## 局限性与未来方向
- **27国样本覆盖不足**：联系88国仅27国获足够响应，大量发展中国家（尼泊尔、孟加拉、肯尼亚、尼日利亚等）样本极少或为零，限制了结论的外推性。
- **仅10个常见名词**：出于问卷长度与参与率权衡，仅覆盖10个物项；不同类别（如建筑/服饰/仪式）的地理偏差模式可能各异。
- **众包平台的地理偏差**：AMT和Prolific本身用户分布高度 skewed，加剧了边缘国家的不可见性，形成"评估偏差的偏差"。
- **自动化评估方法尚不成熟**：CLIP和kNN均失效，暗示需要新的地理感知特征或表征学习方案，当前技术无法替代人工评测。
- **未来方向**：审计Stable Diffusion训练数据（如LAION-5B）的地理分布；采集并增强欠代表国家的图像数据；设计可聚合/缩放图像以加速评测的方法；推动边缘国家参与ML模型开发与评估。

## 研究启发与可借鉴点
- **用户研究作为偏见评估的黄金标准**：CLIP等自动指标虽便捷但不可靠，对于"文化/地理契合度"这类主观维度，人评仍不可替代；未来工作应将人评与自动指标并列报告。
- **陷阱题+国籍复核的质量控制范式**：嵌入语义不一致陷阱题（如问苹果展示芒果）并二次确认居住状态，可有效过滤噪声，值得在众包偏见评测中推广。
- **"典型(typical)"提示词的设计**：在提示中加入"typical"一词可引导模型生成最常见形态，这是一种低成本的分批比较策略，可用于后续可控生成研究。
- **GR与人口正相关、与GDP无关的发现**：提示训练数据的偏差来源是"谁在互联网上拍照并标注"，而非"谁有钱"；这为数据收集策略提供了明确指导——应优先补充人口大国但互联网曝光低的地区。
- **双模型对比的实验设计**：同时评测闭源（DALL·E 2）与开源（Stable Diffusion）模型，可区分"商业模型是否更谨慎"与"训练数据分布才是主因"，本文后者证据更强。

## 关键术语表
- **Geographical Representativeness (GR)**：生成图像与参与者所在国真实周遭景物的吻合程度，以1–5分Likert量表衡量。
- **Underspecified prompt**：未含国家名称的提示（如"a house"），模型需自行补全细节，易暴露训练数据的默认地理偏向。
- **Country-specific prompt**：显式含国家名的提示（如"a typical house in Japan"），用于评估提示工程对地理代表性的缓解效果。
- **CLIP相似度**：基于CLIP预训练模型的文本-图像未归一化点积，曾被广泛用于偏见代理评估，本文证明其对GR无效。
- **k-nearest neighbor估计**：在CLIP视觉空间中找测试图像的近邻，以近邻的人类评分均值预测GR，本文证明其MSE高于常数基线。
- **Conceptual Captions**：Google发布的千万级图像-标注数据集，被OSCAR、CLIP等模型使用，是本次研究追溯训练数据偏见的起点之一。
- **5-point Likert scale**：1=完全不、5=非常符合的五级态度量表，本文用于量化主观地理吻合度。
- **Inter-rater agreement**：同一国家内参与者选择最常见选项的比例，本文最高达81%（flag），随机期望20%。

## 可复现要素
- **数据集**：Conceptual Captions（公开）用于选取10个名词；用户研究原始评分数据论文承诺开源代码与工具，但未公开原始人脸级数据。
- **代码**：论文声明将开源代码及所需工具以供后续研究复现和扩展（位于补充材料/仓库链接，具体URL见原文）。
- **模型**：DALL·E 2（OpenAI闭源API）与Stable Diffusion v1.4（开源权重公开）；CLIP ViT-L/14（公开）。
- **关键超参**：kNN中k=1；提示模板固定为"high definition image of a typical [artifact] in [country]"；每个名词生成4张/模型（DALL·E 2与SD各4张）。
- **参与者规模**：540人、27国；每参与者评测80张图；付费按各国众包时薪等价。
