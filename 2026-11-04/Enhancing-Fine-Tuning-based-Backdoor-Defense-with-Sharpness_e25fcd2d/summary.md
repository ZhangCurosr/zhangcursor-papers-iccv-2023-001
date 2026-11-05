---
title: "Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Enhancing_Fine-Tuning_Based_Backdoor_Defense_with_Sharpness-Aware_Minimization_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:37:56"
field: "深度学习模型安全与后门防御"
keywords: ["backdoor defense", "fine-tuning", "sharpness-aware minimization", "model security", "neural network robustness", "post-training defense", "weight perturbation"]
innovations: ["从神经元权重范数视角揭示vanilla fine-tuning后门防御失效机制，证明后门相关神经元在FT中扰动不足", "提出FT-SAM方法，将自适应SAM引入后门防御微调，定向压制高权重范数后门神经元", "设计DER统一评估指标，证明FT-SAM可通用增强FP/ANP等组合防御方法的性能"]
benchmarks: ["CIFAR-10", "Tiny ImageNet", "GTSRB"]
---

# 论文速读：Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness

## 一句话总结
本文从神经元权重范数视角揭示了vanilla fine-tuning在后门防御中失效的原因，并提出FT-SAM方法——通过结合Sharpness-Aware Minimization (SAM) 与自适应扰动约束，主动压制后门相关神经元的权重范数，在有限良性数据下实现了最先进的后门防御性能。

## 研究问题与动机
1. **核心问题**：给定被后门污染的预训练模型和少量（如5%）良性数据，如何通过fine-tuning有效去除后门效应同时保持模型在干净数据上的准确率？
2. **现有fine-tuning方法不足**：vanilla FT（Fine-pruning中的微调阶段）对后门的防御效果有限，面对Blended、LF等强攻击时ASR仍高达82.22%（CIFAR-10），且对复杂攻击几乎无效。
3. **理论动因**：后门模型已充分拟合良性样本，vanilla FT的权重更新幅度太小，无法使模型跳出当前包含后门依赖的局部最优解。
4. **观察驱动**：后门相关神经元往往具有更大的权重范数，且这些神经元在vanilla FT过程中仅被轻微扰动，导致后门特征仍主导模型决策。

## 核心贡献（创新点）
1. **揭示vanilla FT后门防御失效的神经元级机制**：通过TAC值与权重范数的相关性分析、T-SNE特征可视化及梯度范数统计，证明FT对高TAC值（后门相关）神经元的扰动不足，这是其防御失效的根本原因。
2. **提出FT-SAM后门防御新范式**：将SAM引入后门防御的微调过程，通过min-max formulations在权重空间施加自适应扰动，实现对后门相关神经元的定向压制。
3. **自适应扰动约束设计（ASAM变体）**：引入对角矩阵 $\mathbf{T}_w = \text{diag}(|w_1|, |w_2|, \ldots, |w_d|)$ 作为扰动预算的缩放因子，使权重范数大的神经元获得更大扰动，这与后门相关神经元的高权重范数特征直接对应。
4. **证明FT-SAM可通用增强现有防御方法**：将FT-SAM替代FP和ANP中的微调阶段，显著提升两者的防御性能（DER提升），说明该方法具有独立价值和可扩展性。
5. **提出DER评估指标**：定义Defense Effectiveness Rating $\mathrm{DER} = [\max(0, \Delta\mathrm{ASR}) - \max(0, \Delta\mathrm{ACC}) + 1]/2$，统一衡量防御效果与准确率损失之间的权衡。

## 方法详解
**威胁模型**：防御者获得被攻击的模型 $f_w$ 和少量良性样本 $\mathcal{D}_{\text{benign}}$（如5%），目标是在保持良性准确率的同时降低ASR。

**关键观察**：后门相关神经元（以TAC值为度量）的权重范数显著高于普通神经元，且在vanilla FT过程中变化微弱。

**FT-SAM优化目标**：
$$\min_{\pmb{w}} \max_{\pmb{\epsilon} \in S} \mathcal{L}(\pmb{w} + \pmb{\epsilon})$$
其中 $S = \{ \pmb{\epsilon} : \| \mathbf{T}_w^{-1} \pmb{\epsilon} \|_2 \leq \rho \}$，$\rho$ 为扰动预算超参数。

**自适应扰动矩阵**：
$$\mathbf{T}_w = \text{diag}(|w_1|, |w_2|, \ldots, |w_d|)$$
该设计确保权重范数较大的神经元（更可能与后门相关）获得更大的绝对扰动预算。

**内层最大化（扰动更新）**：
通过一阶泰勒展开求解：
$$\pmb{\epsilon}_{t+1} = \rho \frac{\mathbf{T}_{w_t}^2 \nabla_{w_t} \mathcal{L}(w_t)}{\| \mathbf{T}_{w_t} \nabla_{w_t} \mathcal{L}(w_t) \|_2}$$

**外层最小化（权重更新）**：
$$\pmb{w}_{t+1} = \pmb{w}_t - \eta \nabla_w \mathcal{L}(w + \pmb{\epsilon}_{t+1})|_{w=w_t}$$

**机制解释**：FT-SAM使模型不仅跳出当前局部最优，还通过对高权重范数神经元施加更大扰动，促使后门相关神经元的权重显著下降，从而使模型决策更均匀地依赖各类神经元而非特定后门神经元。

## 实验与结果
**数据集与模型**：CIFAR-10、Tiny ImageNet、GTSRB；PreAct-ResNet18、VGG19-BN。

**攻击方式**：10种SOTA后门攻击（BadNets-A2O/A2A、Blended、Input-aware、CLA、LF、SIG、SSBA、Trojan、Wanet），中毒率10%（部分攻击5%）。

**对比基线**：FT、FP [32]、NAD [27]、AC [5]、NC [46]、ANP [51]、ABL [26]、i-BAU [54]。

**主要结果（CIFAR-10，5%良性数据）**：
- FT-SAM平均ASR：**2.47%**（最佳），相比FT的38.81%大幅下降；平均DER：**96.62**（最佳）
- 在Blended攻击上：FT ASR=82.22%，FT-SAM ASR=4.91%，DER从57.26提升至95.90
- 在LF攻击上：FT ASR=93.89%，FT-SAM ASR=3.81%，DER从52.26提升至96.65
- 在Input-aware攻击上：FT ASR=65.85%，FT-SAM ASR=1.07%，DER从65.74提升至98.51

**主要结果（Tiny ImageNet，5%良性数据）**：
- FT-SAM平均ASR：**1.16%**（最佳），平均DER：**92.16**（最佳）
- 在所有10种攻击上均保持DER > 92，表现稳定

**不同良性数据比例下的鲁棒性（CIFAR-10，10%中毒率）**：
- 即使在1%良性数据下，FT-SAM仍保持平均DER > 94，而FT在低比例下DER急剧下降

**组合实验**：FT-SAM替代FP和ANP中的微调阶段，平均DER分别提升约1-3个百分点，验证其通用增强能力。

**超参数敏感性**：$\rho \in [1, 8]$范围内性能稳定，较大$\rho$加速收敛但不显著影响最终效果。

## 相关工作脉络
1. **vanilla Fine-tuning防御（FT）**：作为后门防御的基线方法，利用少量良性数据微调模型。本文指出其在强攻击下失效，并通过神经元视角给出解释。
2. **Fine-pruning（FP）[32]**：先剪枝再微调的后门防御方法。本文证明FT-SAM可替代FP中的微调步骤进一步提升性能。
3. **Adversarial Neuron Pruning（ANP）[51]**：通过minimax优化寻找并屏蔽后门敏感神经元。本文方法不依赖神经元搜索，而是直接通过SAM扰动实现类似效果。
4. **Neural Cleanse（NC）[46]**：通过优化恢复潜在trigger并重训练模型。本文方法无需显式恢复trigger，计算成本更低。
5. **i-BAU [54]**：基于隐式后门对抗遗忘的minimax方法。本文方法在Tiny ImageNet上显著优于i-BAU（DER 92.16 vs 62.59），说明SAM比显式对抗扰动更适合后门去除。
6. **Sharpness-Aware Minimization（SAM）[11] & ASAM [22]**：本工作的核心算法基础。SAM用于改善模型泛化（寻找flat minima），本文首次将其应用于后门防御的fine-tuning阶段，并利用ASAM的自适应扰动设计实现后门神经元的定向压制。

## 局限性与未来方向
1. **超参数依赖**：$\rho$ 需根据数据集和攻击类型调整（CIFAR-10设为2，Tiny ImageNet/GTSRB设为8），缺乏自动调参策略。
2. **仅针对后训练防御**：方法假设防御者仅有被污染模型和少量良性数据，不适用于训练阶段防御场景。
3. **纯图像分类任务验证**：实验集中在图像分类（CIFAR-10、Tiny ImageNet、GTSRB），未验证在目标检测、分割等更复杂视觉任务上的效果。
4. **对抗纯净样本攻击的鲁棒性未充分验证**：攻击者可能针对FT-SAM的minimax结构设计对抗性扰动，削弱防御效果。
5. **未来方向**：可探索FT-SAM与其他防御策略（如触发器逆转、数据清洗）的深度融合；研究自动化$\rho$选择策略；扩展到多模态和后门检测联合框架。

## 研究启发与可借鉴点
1. **神经元视角的分析范式**：通过TAC值关联权重范数与后门相关性的分析框架，可作为理解各类防御方法有效性的通用工具，适用于后续研究的诊断分析。
2. **SAM用于安全防御的迁移思路**：将原本用于改善泛化的SAM技术移植到模型安全领域（后门防御），展示了"一技多用"的研究思路——不同领域中相同优化技术可能解决看似无关的问题。
3. **自适应扰动预算的设计**：$\mathbf{T}_w^{-1}$ 缩放因子的设计体现了"针对问题结构定制优化约束"的思想，可用于其他需要差异化对待模型参数的场景（如模型编辑、联邦学习隐私保护）。
4. **DER指标的简洁有效性**：DER同时刻画ASR下降和ACC损失两个维度，避免了多指标汇报的模糊性，建议作为后续后门防御论文的标配评估指标。
5. **FT-SAM作为即插即用模块**：证明微调阶段可用FT-SAM替代vanilla FT以提升防御效果，为FP、NAD等组合防御方法提供了通用的增强组件设计范式。

## 关键术语表
**Backdoor Attack（后门攻击）**：攻击者在训练数据中植入特定触发器并篡改标签，使模型对含触发器的输入输出目标类别，而对正常输入行为正常的恶意攻击。

**Fine-tuning Defense（微调防御）**：利用少量良性数据对已被污染模型进行微调，以期消除后门效应的后训练防御策略。

**Sharpness-Aware Minimization (SAM)**：一种优化算法，通过min-max formulations同时最小化损失值和其附近邻域的损失尖锐度，促进模型收敛到flat minima以改善泛化。

**TAC（Trigger Activated Change）**：衡量神经元与后门关联程度的指标，定义为良性样本与污染样本在特定层通道-wise激活的差异，值越大表示该神经元越可能与后门相关。

**Adaptive SAM (ASAM)**：SAM的变体，引入参数尺度不变的自适应扰动约束，通过$\mathbf{T}_w$对角矩阵为不同神经元分配差异化的扰动预算。

**DER（Defense Effectiveness Rating）**：本文提出的统一防御评估指标，综合考虑ASR下降和ACC损失，值越接近1表示防御效果越好。

**Poisoning Ratio（中毒率）**：训练数据中被攻击者注入的恶意样本所占的比例，本文默认设置为10%。

## 关键术语表
**Fine-pruning (FP)**：先通过良性数据激活模式剪枝"不活跃"神经元，再微调的级联后门防御方法。

**Neural Attention Distillation (NAD)**：先用良性数据微调teacher模型，再用teacher指导student模型微调的双阶段蒸馏防御方法。

**Anti-Backdoor Learning (ABL)**：在污染数据上交替执行标准训练与对抗性"遗忘"的防御策略，旨在分离后门与正常特征的学习。

**i-BAU（Implicit Backdoor Adversarial Unlearning）**：借鉴universal adversarial perturbations思想，通过隐式超梯度求解minimax问题的后门消除方法。

**PreAct-ResNet18**：带有预激活残差连接的轻量级ResNet变体，本文主要使用的 backbone 架构。

**GTSRB（German Traffic Sign Recognition Benchmark）**：德国交通标志识别数据集，包含43类交通标志图像，本文用于验证方法在真实场景数据上的有效性。
