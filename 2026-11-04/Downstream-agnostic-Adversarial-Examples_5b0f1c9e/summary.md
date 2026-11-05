---
title: "Downstream-agnostic-Adversarial-Examples"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Downstream-agnostic_Adversarial_Examples_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:36:49"
---

# 论文速读：Downstream-agnostic Adversarial Examples

## 一句话总结
本文提出了AdvEncoder，首个面向自监督学习（SSL）预训练编码器的下游任务无关通用对抗攻击框架。该方法在不依赖预训练数据集、下游任务类型及标签信息的前提下，通过高频分量引导的生成式网络同时构造通用对抗扰动（Adv-PER）与补丁（Adv-PAT），可高效破坏继承该编码器的各类下游分类与检索任务。

## 研究问题与动机
- 自监督预训练编码器已作为通用特征提取器被广泛开源与商用，但其对抗安全性尚未得到系统研究。
- 传统对抗攻击依赖分类标签或已知目标模型结构，而预训练编码器仅输出特征向量，攻击者面临监督信号缺失的根本性挑战。
- 现有针对预训练编码器的安全研究多聚焦于隐私泄露、后门攻击或投毒攻击，测试阶段的通用对抗攻击仍处于空白。
- Fine-tuning会重塑特征空间决策边界，仅使预训练编码器失准不足以欺骗下游分类器，必须生成具备强迁移性的通用对抗样本。

## 核心贡献（创新点）
- 提出AdvEncoder框架，首次实现SSL预训练编码器的下游任务无关通用对抗攻击，揭示其严重安全隐患。与依赖任务先验或标签的传统对抗攻击相比，本框架完全解耦攻击生成与下游任务。
- 设计基于高频分量（HFC）的生成式攻击网络，利用图像纹理变化充当伪监督信号以引导特征空间偏移。与PAP等依赖模型低层激活值或特定预训练数据集的方法相比，本方法直接篡改样本 intrinsic 纹理特征，无需预训练数据先验。
- 统一支持通用对抗扰动与通用对抗补丁两种形态，兼顾数字空间隐蔽性与物理世界可实施性。与纯优化类UAP方法相比，生成式架构通过学习数据分布获得更强的泛化能力。
- 在14种SSL方法与4个下游数据集上进行大规模评估，验证了攻击的高成功率与跨模型/跨数据集迁移性。与单一任务或单一方法的安全评估相比，本工作提供了面向SSL生态的系统性攻击基准。
- 系统性测试Corruption、Fine-tuning、Pruning、Adversarial Training四种防御，证明现有手段均难以有效抵御AdvEncoder。与传统攻击仅验证进攻效果不同，本文通过防御实验反向强化了攻击威胁的现实意义。

## 方法详解
- **威胁模型**：攻击者可访问公开的预训练编码器 $g_\theta(\cdot)$，但对预训练数据集 $\mathcal{D}_p$、下游任务及下游数据集 $\mathcal{D}_d$ 一无所知。使用替代数据集 $\mathcal{D}_a$（默认CIFAR-10）生成通用对抗噪声 $\delta$，满足 $\|\delta\|_p \leq \epsilon$，目标使任意下游分类器 $f_{\theta'}(g_\theta(x+\delta))$ 产生误分类。
- **高频分量损失（$\mathcal{L}_{hfc}$）**：针对DNN对纹理特征敏感的特性，利用高频滤波器 $\mathcal{H}$ 提取对抗样本与原图的高频分量，最大化二者欧氏距离以充当伪标签引导：$\mathcal{L}_{hfc} = -||\mathcal{H}(x^{adv}) - \mathcal{H}(x)||_2$。
- **特征距离对抗损失（$\mathcal{L}_{adv}$）**：将原图与对抗图视为负样本对，采用InfoNCE损失拉开编码器输出特征间距：$\mathcal{L}_{adv} = log[\frac{exp(S(g(x^{adv}), g(x))/\tau)}{\sum_{j \neq i} exp(S(g(x^{adv}), g(x_j))/\tau)}]$，其中 $S(\cdot)$ 为余弦相似度，$\tau$ 为温度参数。
- **质量约束损失（$\mathcal{L}_{q}$）**：控制扰动幅度保证不可感知性，$\mathcal{L}_{q} = ||x^{adv} - x||_2$，每次优化后裁剪至 $\epsilon$ 边界。
- **总目标函数**：$\mathcal{L}_{\mathcal{G}} = \alpha \mathcal{L}_{adv} + \beta \mathcal{L}_{hfc} + \lambda \mathcal{L}_{q}$，默认超参 $\alpha=1, \beta=5, \lambda=1$。
- **扰动/补丁生成**：Adv-PER 直接叠加生成噪声 $x^{adv} = x + \mathcal{G}(z)$；Adv-PAT 通过二值掩码 $m$ 将补丁随机粘贴至图像右下角 $x^{adv} = x \odot (1-m) + \mathcal{G}(z) \odot m$，补丁面积占比设为 0.03。

## 实验与结果
- **实验设置**：victim编码器来自solo-learn，共14种SSL方法（Barlow Twins, BYOL, DeepCluster v2, DINO, MoCo v2+, MoCo v3, NNCLR, ReSSL, SimCLR, SupCon, SwAV, VIbCReg, VICReg, W-MSE），ResNet-18骨干，ImageNet/CIFAR-10预训练。下游数据集为CIFAR10、STL10、GTSRB、ImageNet。评估指标为ASR与mAP。
- **分类攻击结果**：Adv-PAT表现极其稳定，平均ASR超过90%；Adv-PER平均ASR达73%~89%。ImageNet作为替代数据集普遍优于CIFAR-10。MoCo与SimCLR相对稳健，BYOL、NNCLR、SupCon较脆弱。
- **检索攻击结果**：原始clean样本mAP为81%~94%，施加Adv-PER/PAT后降至约10%~27%，证明对检索任务同样具有强破坏力。
- **最强结果**：在ImageNet下游+ImageNet替代集设置下，Adv-PAT在多个模型上ASR突破99%（如DeepC2: 99.68%, SimCLR: 99.05%, MoCo3: 98.44%）。
- **迁移性**：跨预训练数据集（CIFAR10→ImageNet）与跨SSL方法（SimCLR生成→攻击MoCo/DINO等）均可保持高ASR，验证了下游无关性。
- **对比基线**：显著优于UAP [44]、UPGD [18]、FFF [47]、SSP [50]、NAG [48]、PAP [1] 及 UA-PAT [3]。Adv-PAT平均ASR超85%，Adv-PER达55%~58%，且无需附加分类头即可直接攻击预训练编码器。
- **防御实验**：Corruption（高斯噪声）、Fine-tuning、Pruning、Adversarial Training 均无法有效压制AdvEncoder，Adv-PAT几乎不受影响，Adv-PER仅在高强度对抗训练下略有下降。

## 相关工作脉络
- **优化类通用攻击（UAP, UPGD, FFF, SSP）**：依赖目标模型梯度或标签信息逐样本迭代，无法直接作用于仅输出特征向量的SSL预训练编码器，且缺乏跨任务泛化能力。
- **生成类通用攻击（NAG, UA-PAT）**：需已知目标模型结构或分类头，且主要针对监督分类任务；本文方法无需目标模型先验，直接面向特征提取阶段。
- **预训练扰动方法（PAP）**：从低层激活值提升特征生成扰动，高度依赖预训练数据集分布，生成结果缺乏语义且迁移性差；本文直接修改样本高频纹理，不依赖预训练数据。
- **SSL安全研究（后门/投毒/隐私）**：BackdoorAttack [31,32]、PoisoningAttack [40]、MembershipInference [27,41]、ModelStealing [20,42] 等均聚焦训练阶段或隐私泄露，本文首次系统研究测试阶段的通用对抗攻击，填补SSL预训练组件安全性评估空白。

## 局限性与未来方向
- 论文主要评估数字空间攻击，未充分验证物理世界复杂光照、视角、压缩退化对Adv-PAT的实际影响。
- 攻击性能受替代数据集规模与分布影响，仅测试了CIFAR-10与ImageNet两种极端情况，小样本或强域偏移场景下的鲁棒性待进一步探索。
- 实验仅覆盖分类与检索两类下游任务，未扩展至目标检测、语义分割、图像生成等任务。
- 防御实验表明现有机制均无效，但论文未提出针对性防御方案，仅指出需求，未来需探索预训练阶段的对抗鲁棒性训练范式。

## 研究启发与可借鉴点
- **高频分量伪监督思想**：在无标签
