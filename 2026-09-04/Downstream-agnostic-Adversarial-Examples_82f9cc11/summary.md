---
title: "Downstream-agnostic-Adversarial-Examples"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Downstream-agnostic_Adversarial_Examples_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:21:42"
field: "对抗机器学习"
keywords: ["自监督学习", "对抗样本", "通用攻击", "预训练编码器", "高频分量", "生成式攻击"]
innovations: ["提出首个下游任务无关的通用对抗攻击框架AdvEncoder，无需预训练/下游数据信息", "利用高频纹理分量作为代理监督信号设计生成式攻击网络", "支持通用对抗扰动与补丁两种形式，在14种SSL编码器上实现高攻击成功率"]
benchmarks: ["CIFAR10", "STL10", "GTSRB", "ImageNet"]
---

# 论文速读：Downstream-agnostic Adversarial Examples

## 一句话总结
本文提出 **AdvEncoder**，首个面向自监督学习预训练编码器的**下游任务无关通用对抗样本攻击框架**。攻击者无需掌握预训练数据集、下游任务及标签信息，仅通过修改图像高频纹理成分并借助生成网络学习数据分布，即可生成通用对抗扰动/补丁，有效欺骗所有继承该编码器的下游分类与检索任务。

## 研究问题与动机
- **预训练编码器安全性被忽视**：自监督学习（SSL）广泛发布预训练编码器（如 SimCLR、MoCo），但其作为通用特征提取器的安全性，尤其是对抗样本脆弱性，尚未被深入探究。
- **传统对抗攻击无法直接迁移**：预训练编码器仅输出特征向量而非类别标签，缺乏监督信号；且攻击者不知下游任务类型、预训练数据集和下游数据集，导致基于标签梯度的传统对抗攻击失效。
- **现有工作存在局限**：并发工作 PAP 通过提升低层特征激活值生成对抗扰动，但生成的样本缺乏语义性且严重依赖预训练数据集；而实际场景中攻击者往往不具备这些先验知识。
- **需要更实用的攻击范式**：工业界大量预训练编码器被公开或商业化，揭示其安全风险对推动防御机制设计具有重要意义。

## 核心贡献（创新点）
- **提出首个下游任务无关的通用对抗攻击框架 AdvEncoder**：区别于以往针对特定下游模型的对抗攻击，本工作直接针对预训练编码器本身生成可迁移的通用对抗样本。
- **设计基于频率的生成式攻击网络**：传统优化方法需大预算扰动才能改变特征空间位置，本文利用 DNN 对高频纹理成分的敏感性，通过高频分量损失直接篡改图像语义特征，以小扰动实现强攻击。
- **支持扰动（Perturbation）与补丁（Patch）两种攻击形式**：框架灵活，Adv-PER 具有更高隐蔽性，Adv-PAT 更易在物理世界部署，两者均能在无下游任务信息条件下实现高攻击成功率。
- **系统验证与防御评估**：在 14 种 SSL 预训练编码器、4 个下游数据集（分类与检索任务）上进行实验，并针对性测试四种防御手段（Corruption、Fine-tuning、Pruning、Adversarial Training），证明攻击的严重威胁性。

## 方法详解
- **威胁模型**：攻击者可访问预训练编码器（公开下载或商业获取），但不知预训练数据集、下游任务类型及下游数据集；目标是生成通用对抗扰动 δ，使任意输入图像 x 经编码器后的特征向量发生显著偏移，从而误导下游分类器/检索模型。
- **核心挑战一：缺乏监督信号**：预训练编码器只输出特征向量 v，无类别标签。传统最大-logits 梯度方法不可用。本文利用 **高频分量（High-Frequency Component, HFC）** 作为替代监督：DNN 对图像纹理（高频信息）敏感，通过最大化对抗样本与原始样本的高频分量欧氏距离，迫使特征空间中的表示远离原类区域。
- **核心挑战二：缺乏下游任务信息**：仅扰动预训练编码器不足以破坏下游任务，因为微调会改变决策边界（如图 2(b) 所示）。因此需要生成足够大的特征偏移（图 2(c)-(d)），使微调后的下游分类器仍将其误判。
- **生成式攻击框架**：
  - **生成器 G**：输入固定随机噪声 z，输出生成通用对抗扰动 δ = G(z)，保证不同输入共享相同扰动模式。
  - **高频分量滤波器 H**：提取图像的高频成分，用于计算频率损失。
  - **损失函数**：$\mathcal{L}_{\mathcal{G}} = \alpha \mathcal{L}_{adv} + \beta \mathcal{L}_{hfc} + \lambda \mathcal{L}_{q}$
    - **对抗损失 $\mathcal{L}_{adv}$**：采用 InfoNCE 损失，将原始样本 x 与对抗样本 x_adv 视为负对，最大化两者特征向量的余弦相似度距离，推动特征空间分离。
    - **高频分量损失 $\mathcal{L}_{hfc}$**：$||\mathcal{H}(x^{adv}) - \mathcal{H}(x)||_2$，通过改变高频纹理成分辅助特征偏移。
    - **质量损失 $\mathcal{L}_{q}$**：$||x^{adv} - x||_2$，控制扰动幅度，每步优化后裁剪 δ 以满足 $\|\delta\|_p \leq \epsilon$。
- **两种攻击形式**：
  - **通用对抗扰动（Adv-PER）**：$x^{adv} = x + G(z)$，直接叠加扰动，隐蔽性强。
  - **通用对抗补丁（Adv-PAT）**：$x^{adv} = x \odot (1-m) + G(z) \odot m$，其中 m 为二进制掩码矩阵，将补丁放置于图像随机隐藏位置（实验中使用右下角），更适合物理场景。

## 实验与结果
- **数据集与模型**：从 solo-learn 获取 14 种 SSL 预训练编码器（Barlow Twins、BYOL、DeepCluster v2、DINO、MoCo v2+、MoCo v3、NNCLR、ReSSL、SimCLR、SupCon、SwAV、VIbCReg、VICReg、W-MSE）， backbone 为 ResNet18，预训练于 ImageNet 或 CIFAR10。下游数据集：CIFAR10、STL10、GTSRB、ImageNet。攻击者代理数据集默认 CIFAR10，部分设置使用 ImageNet。
- **评估指标**：攻击成功率（ASR，%）、检索任务 mean Average Precision（mAP）。
- **分类任务结果（Table 1 & 2）**：
  - Adv-PER 在 224 个攻击设置下表现优异，平均 ASR 最高达 88.35%（SupCon，Setting S1）。
  - Adv-PAT 整体更强，平均 ASR 超过 90%，最高达 99.69%（DeepC2，Setting S4）。
  - 攻击者代理数据集为 ImageNet 时（S5-S8）普遍优于 CIFAR10（S1-S4）。
  - MoCo、SimCLR 相对鲁棒，BYOL、NNCLR、SupCon 较脆弱。
- **检索任务结果（Table 3）**：Adv-PER 和 Adv-PAT 均大幅降低 mAP（如 STL10 上 Barlow 的 per-map 从 81.03 降至 23.26/21.15）。
- **迁移性（Figure 6）**：Adv-PER 可跨预训练数据集（CIFAR10→ImageNet）和 SSL 方法迁移攻击。
- **对比实验（Table 4）**：在相同设置下，Adv-PER 和 Adv-PAT 均显著优于 UAP、UPGD、FFF、SSP、NAG、PAP 等基线；Adv-PAT 平均 ASR 超 85%，大幅领先 UA-PAT。
- **最强结果**：Adv-PAT 在 DeepC2 编码器上达到 99.69% ASR（Setting S4）；Adv-PER 在 SupCon 上达到 88.35% 平均 ASR。

## 相关工作脉络
- **UAP（Universal Adversarial Perturbations, Moosavi-Dezfooli et al., CVPR 2017）**：基于优化的通用对抗扰动方法，需要目标模型的梯度信息，无法直接应用于仅输出特征向量的预训练编码器。
- **PAP（Pre-trained Adversarial Perturbations, NeurIPS 2022）**：并发工作，通过提升低层特征激活值生成对抗扰动，但生成样本缺乏语义且依赖预训练数据集；本文从直接篡改纹理特征角度切入，更符合实际无先验攻击场景。
- **NAG（Network for Adversary Generation, CVPR 2018）**：生成式通用对抗攻击，但针对有标签的监督学习模型，需类别信息指导。
- **UA-PAT（Universal Adversarial Patch, Brown et al., 2017）**：基于优化的对抗补丁方法，同样依赖下游任务信息；本文 Adv-PAT 在无下游信息条件下实现更高 ASR。
- **自监督学习防御**：现有防御（数据预处理、对抗训练、剪枝、微调）主要针对监督学习下的对抗样本；本文证明这些防御对 AdvEncoder 效果有限，凸显预训练编码器专用防御的必要性。

## 局限性与未来方向
- **攻击成功率受代理数据集影响**：当代理数据集与预训练/下游数据集分布差异较大时，ASR 有所下降（如 MoCo2+ 在 Setting S2 仅 32.08%）。
- **高频分量损失的有效性依赖网络对纹理的偏好**：不同 SSL 方法对高频信息的依赖程度不同，导致防御鲁棒性差异。
- **未探索多模态预训练编码器（如 CLIP）的攻击**：论文主要关注视觉编码器，可扩展至多模态场景。
- **防御机制仍显不足**：四种定制防御均未能有效缓解攻击，表明需要设计全新的预训练编码器专用防御方法。
- **物理世界实验缺失**：Adv-PAT 虽宣称易于物理部署，但论文未提供真实世界验证。

## 研究启发与可借鉴点
- **高频纹理作为代理监督信号**：在缺乏标签信息时，可利用 DNN 对高频分量的敏感性设计替代优化目标，该方法可迁移至其他无监督/自监督场景的对抗攻击。
- **生成式通用攻击框架设计**：固定噪声输入 + 生成器学习的范式可有效捕获通用对抗模式，同时保证样本的自然性，适用于资源受限的攻击场景。
- **防御评估的全面性**：从用户侧（预处理、微调、剪枝）和提供商侧（对抗训练）多角度评估防御效果，为后续防御研究提供基准对比思路。
- **与团队方向结合机会**：若团队研究多模态对比学习或视觉-语言预训练模型，可将 AdvEncoder 的思想扩展至 CLIP 等多模态编码器的攻击与防御研究。

## 关键术语表
- **Self-supervised Learning (SSL)**：自监督学习，利用数据自身结构生成监督信号预训练特征编码器，无需人工标注。
- **Downstream-agnostic**：下游任务无关，指攻击方法不依赖特定下游任务（分类、检测等）的标签或数据分布。
- **Universal Adversarial Example**：通用对抗样本，一种可对多个输入图像生效的固定扰动或补丁。
- **High-Frequency Component (HFC)**：高频分量，图像中表征纹理和边缘的细节信息，DNN 对其敏感。
- **InfoNCE Loss**：信息噪声对比估计损失，用于衡量特征向量间的相似度，在此作为对抗损失。
- **Attack Success Rate (ASR)**：攻击成功率，衡量对抗样本成功欺骗模型的比例。
- **Mean Average Precision (mAP)**：平均精度均值，用于评估检索任务中对抗样本对检索准确性的影响。
- **solo-learn**：开源的自监督学习库，提供多种 SSL 方法的预训练编码器实现。

## 可复现要素
- **数据集**：预训练数据集 ImageNet、CIFAR10；下游数据集 CIFAR10、STL10、GTSRB、ImageNet；攻击者代理数据集 CIFAR10、ImageNet。上述数据集均为公开可用。
- **代码开源**：是，代码已发布于 https://github.com/CGCL-codes/AdvEncoder。
- **权重开源**：预训练编码器从 solo-learn 库获取（公开）。
- **关键超参**：$\alpha = 1, \beta = 5, \lambda = 1$；扰动预算 $\epsilon = 10/255$；补丁大小 0.03；训练轮数 20，batch size 256；学习率 0.0002，Adam 优化器。
