---
title: "Downstream-agnostic-Adversarial-Examples"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Downstream-agnostic_Adversarial_Examples_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:36:51"
field: "对抗机器学习的鲁棒性与安全性"
keywords: ["自监督学习", "对抗攻击", "通用对抗样本", "预训练编码器", "高频分量", "迁移攻击"]
innovations: ["首个面向SSL预训练编码器的下游任务无关通用对抗攻击框架", "利用高频分量信息作为无监督引导信号设计生成式攻击网络", "在完全未知预训练数据和下游任务的强设定下实现高效攻击"]
benchmarks: ["CIFAR10", "STL10", "GTSRB", "ImageNet"]
---

# 论文速读：Downstream-agnostic Adversarial Examples

## 一句话总结
本文提出 **AdvEncoder**，首个针对自监督学习（SSL）预训练编码器的**下游任务无关通用对抗攻击框架**，在攻击者**不知预训练数据集、不知下游任务**的强设定下，利用高频分量信息结合生成网络生成通用对抗扰动/补丁，成功攻击14种SSL编码器的各类下游任务，揭示预训练编码器的严重安全风险。

## 研究问题与动机
1. **安全问题未被重视**：自监督学习预训练的编码器被广泛公开发布（如Google SimCLR、Meta MoCo）或商业部署（如OpenAI、Clarifai），但针对其对抗攻击的安全性几乎无人研究。
2. **传统对抗攻击无法直接套用**：预训练编码器仅输出**特征向量**而非分类标签，缺乏监督信号，传统基于标签的对抗攻击方法失效。
3. **仅欺骗编码器不够**：下游任务往往对预训练编码器进行微调（fine-tuning），微调会改变特征边界，仅让编码器输出偏离原类别的特征仍可能被下游分类器纠正（见图2(b) vs 图2(c)）。
4. **实用场景要求严苛**：攻击者需完全无先验知识（无预训练数据集、无下游数据集、无任务类型），这比现有后门攻击、投毒攻击等训练期攻击更具挑战性。

## 核心贡献（创新点）
1. **提出AdvEncoder框架**：首个面向SSL预训练编码器的下游任务无关通用对抗攻击框架，揭示预训练编码器存在严重安全隐患。
   - 与PAP等前作本质区别：PAP依赖预训练数据集且生成样本缺乏语义，本文**直接从样本内在纹理特征出发**，无需预训练数据集。
2. **设计基于频率的生成攻击网络**：利用高频分量（HFC）作为"伪标签"引导，通过生成器学习攻击代理数据集分布，生成兼具高攻击成功率与强迁移性的通用对抗扰动/补丁。
   - 与UAP/NAG等经典方法本质区别：现有生成式/优化式通用对抗攻击均需**完整模型（含分类头）和标签信息**，本文直接针对**纯编码器特征输出**设计，且适配扰动与补丁两种形式。
3. **全面实验验证**：在14种主流SSL方法（Barlow Twins、BYOL、DINO、MoCo v2/v3、SimCLR、SwAV等）和4个下游数据集（CIFAR10、STL10、GTSRB、ImageNet）上取得最高ASR超90%（Adv-PAT平均），显著优于所有基线方法。
4. **设计并评估四种防御措施**：验证了污损（Corruption）、微调（Fine-tuning）、剪枝（Pruning）、对抗训练（Adversarial Training）对AdvEncoder的抵抗能力有限，强调需要新的专用防御机制。

## 方法详解
**威胁模型**：攻击者可访问预训练编码器，但**无预训练数据集**、**无下游数据集**、**无任务类型信息**；目标为非定向攻击，使继承该编码器的所有下游分类器失效。

**两大挑战与设计直觉**：
- **Challenge I（无监督信号）**：编码器仅输出特征向量，无法直接用标签引导。借鉴CNN偏向纹理特征（高频）的发现，通过改变图像**高频分量（HFC）** 来影响编码器决策，HFC变化起到类似"标签引导"的作用。
- **Challenge II（无下游信息）**：微调会改变特征边界，单纯推离原类不够。结合生成网络学习数据分布，使对抗样本在特征空间中**聚集成簇**并远离正常样本簇（图2(d)），确保即使微调后仍被误判。

**AdvEncoder架构（图3）**：由对抗生成器 $G$、高频分量滤波器 $H$、受害者编码器 $E$ 组成。固定随机噪声 $z$ 输入 $G$ 输出通用对抗噪声，粘贴到代理数据集 $\mathcal{D}_a$ 样本上。

**损失函数**：
$$\mathcal{L}_{\mathcal{G}} = \alpha \mathcal{L}_{adv} + \beta \mathcal{L}_{hfc} + \lambda \mathcal{L}_{q}$$

- **$\mathcal{L}_{adv}$（对抗损失）**：用InfoNCE度量良/ adversarial样本特征向量余弦相似度，将其视为负对，最大化特征距离：
$$\mathcal{L}_{adv} = \log\left[\frac{\exp(S(g_\theta(x_i^{adv}), g_\theta(x_i)) / \tau)}{\sum_{j=0}^{K} \exp(S(g_\theta(x_i^{adv}), g_\theta(x_j)) / \tau)}\right]$$
- **$\mathcal{L}_{hfc}$（高频分量损失）**：拉大良/ adversarial样本的高频分量欧氏距离，直接篡改图像纹理语义：
$$\mathcal{L}_{hfc} = -\|\mathcal{H}(x^{adv}) - \mathcal{H}(x)\|_2$$
- **$\mathcal{L}_{q}$（质量损失）**：约束扰动幅度，每步优化后裁剪$\delta$以满足$\|\delta\|_p \leq \epsilon$：
$$\mathcal{L}_{q} = \|x^{adv} - x\|_2$$

**两种攻击形式**：
- **Adv-PER（扰动攻击）**：$x^{adv} = x + G(z)$，隐蔽性强。
- **Adv-PAT（补丁攻击）**：$x^{adv} = x \odot (1-m) + G(z) \odot m$，$m$为二进制位置掩码，物理世界更易实现。

## 实验与结果
**数据集与模型**：
- 14种SSL预训练编码器（来自solo-learn库，ResNet18骨干，ImageNet/CIFAR10预训练）
- 攻击者代理数据集：CIFAR10（默认）和ImageNet
- 下游数据集：CIFAR10、STL10、GTSRB、ImageNet
- 评估指标：ASR（分类）、mAP（检索）

**主要结果**：
- **Adv-PAT**在所有224个攻击设置下表现优异，**平均ASR超90%**；Adv-PER平均ASR约73-83%，受代理数据集选择影响较大（ImageNet代理优于CIFAR10）。
- **最强结果**：Adv-PAT在ImageNet下游、ImageNet代理设定下，对W-MSE编码器达到**96.51%** ASR；Adv-PER在STL10下游、ImageNet代理下对VICReg达到**96.41%**。
- **检索攻击**（表3）：Adv-PAT将STL10检索mAP从~80%降至~21%，GTSRB从~85%降至~10-13%，效果显著。
- **对比实验**（表4）：Adv-PER和Adv-PAT均大幅优于UAP、UPGD、FFF、SSP、NAG、PAP、UA-PAT等基线，Adv-PAT平均ASR领先第二名UA-PAT超25个百分点。
- **迁移性**（图6）：Adv-PER可在不同预训练数据集（CIFAR10↔ImageNet）和不同SSL方法间有效迁移。
- **消融实验**（图5）：代理数据量增加提升Adv-PER效果（Adv-PAT稳健）；HFC模块和生成器模块均有贡献；扰动预算$\epsilon$和补丁尺寸对攻击性能有影响。

## 相关工作脉络
1. **PAP（Ban & Dong, NeurIPS 2022）**：从低层特征激活提升生成预训练扰动，但**严重依赖预训练数据集**且生成样本缺乏语义。本文与之定位相反：无需预训练数据集，直接从纹理入手。
2. **UAP（Moosavi-Dezfooli et al., CVPR 2017）及后续优化式方法（UPGD、FFF、SSP）**：面向监督学习模型，需完整模型（含分类头）和标签，无法直接用于纯编码器。
3. **NAG（Mopuri et al., CVPR 2018）**：生成式通用对抗攻击，但同样需要完整模型结构和标签信息，本文不依赖这些信息。
4. **UA-PAT（Brown et al., 2017）**：优化式通用对抗补丁，需目标模型信息；本文Adv-PAT为生成式，在未知下游任务下仍高效。
5. **防御措施（Corruption/Fine-tuning/Pruning/AT）**：常见对抗防御手段（Madry et al. AT、Zhu & Gupta Pruning等）对AdvEncoder效果有限，强调需针对预训练编码器的专用防御。

## 局限性与未来方向
1. **代理数据集影响攻击效果**：Adv-PER在CIFAR10代理下表现弱于ImageNet代理，说明代理数据集与预训练/下游数据集的相似性影响性能。
2. **防御研究不足**：四种现有防御（污损、微调、剪枝、对抗训练）均未能有效阻止AdvEncoder，说明亟需针对预训练编码器新型防御机制，但本文未深入探索。
3. **仅关注图像分类与检索**：未测试检测、分割等其他下游任务场景。
4. **物理世界验证缺失**：Adv-PAT虽提及物理可实现性，但未进行实际物理实验。

## 研究启发与可借鉴点
1. **高频分量作为无监督引导信号**：在缺乏标签的防御攻击场景（如纯编码器攻击、特征空间攻击）中，利用HFC变化替代标签信号，是一个可迁移的思路。
2. **生成式通用对抗攻击框架**：固定噪声输入生成器的设计可复用于其他需快速批量攻击的场景；HFC+InfoNCE+质量约束的三路损失组合具有参考价值。
3. **下游微调robustness评估范式**：本文验证了fine-tuning、pruning、adversarial training等防御的无效性，此评估范式可用于后续防御工作的基准对比。
4. **可延伸至多模态领域**：作者已在后续工作AdvCLIP（ACM MM'23）中将此框架扩展至多模态对比学习（CLIP），说明该方法具有良好的扩展潜力。

## 关键术语表
**Self-supervised Learning (SSL)**：利用无标签数据内部监督信号预训练编码器的机器学习范式，使编码器成为通用特征提取器。
**Downstream-agnostic**：下游任务无关，指攻击方法不依赖任何下游任务信息（数据集、标签、任务类型）即可生效。
**Universal Adversarial Example**：通用对抗样本，指对一组输入图像均有效的单一对抗扰动或补丁。
**High Frequency Component (HFC)**：图像高频分量，对应纹理和细节信息，CNN对其敏感，本文用作无监督引导信号。
**InfoNCE Loss**：对比学习常用的损失函数，本文用于最大化良/ adversarial样本在特征空间的距离。
**Attack Success Rate (ASR)**：攻击成功率，衡量对抗样本成功误导模型的比例。
**mean Average Precision (mAP)**：平均精度均值，用于评估检索任务的准确性。

## 可复现要素
- **代码**：已开源，https://github.com/CGCL-codes/AdvEncoder
- **数据集**：ImageNet、CIFAR10（公开）；STL10（公开）；GTSRB（公开）
- **模型**：14种SSL编码器来自solo-learn库（公开预训练权重，ResNet18骨干）
- **超参数**：$\epsilon = 10/255$，补丁大小=0.03，$\alpha=1, \beta=5, \lambda=1$，训练epoch=20，batch size=256，Adam优化器，lr=0.0002
- **攻击者代理数据集**：默认CIFAR10，备选ImageNet
- **下游任务**：图像分类（4个数据集）和图像检索（STL10、GTSRB）
