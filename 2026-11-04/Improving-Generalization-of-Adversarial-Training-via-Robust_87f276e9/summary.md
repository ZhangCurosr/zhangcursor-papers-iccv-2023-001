---
title: "Improving-Generalization-of-Adversarial-Training-via-Robust"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Improving_Generalization_of_Adversarial_Training_via_Robust_Critical_Fine-Tuning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:44:28"
field: "对抗鲁棒性与泛化的权衡"
keywords: ["adversarial training", "generalization", "robustness", "fine-tuning", "module criticality", "OOD robustness", "model interpolation"]
innovations: ["提出MRC度量来量化每个模块对对抗鲁棒性的关键程度，发现并验证对抗训练模型存在冗余鲁棒容量", "提出RiFT三阶段后处理方法（MRC计算→非鲁棒关键模块微调→权重插值），在不损害鲁棒性的前提下提升泛化约1.5-2%", "证明微调非鲁棒关键模块可同时改善泛化和OOD鲁棒性，挑战鲁棒性与泛化必然权衡的传统认知"]
benchmarks: ["CIFAR10", "CIFAR100", "Tiny-ImageNet", "CIFAR10-C", "CIFAR100-C", "Tiny-ImageNet-C"]
---

# 论文速读：Improving-Generalization-of-Adversarial-Training-via-Robust

## 一句话总结
提出 **Robust Critical Fine-Tuning (RiFT)**，一种对抗训练后的后处理方法：通过**模块鲁棒关键性（MRC）**识别冗余模块，仅对该模块在标准数据上微调，再与原始对抗训练权重线性插值，在保持对抗鲁棒性的同时将泛化能力和 OOD 鲁棒性提升约 1.5–2%。

## 研究问题与动机
- 对抗训练（AT）可显著提升对抗鲁棒性，但会**严重牺牲分布内泛化能力**，且对 OOD 样本异常脆弱。
- 现有工作主要在 AT 训练过程中通过重采样、引入无标签数据或重新设计损失函数来缓解权衡，**缺乏对 AT 后模型"冗余容量"的系统性分析**。
- 深度神经网络对泛化存在已知冗余容量（特定模块可删除/置换/回退而不影响性能），但**对抗鲁棒性是否存在类似冗余**尚无定论。
- 若 AT 模型存在冗余鲁棒容量，如何精准识别并利用它来同时改善泛化与鲁棒性？

## 核心贡献（创新点）
1. **提出模块鲁棒关键性（MRC）度量**：量化每个模块在最坏情况权重扰动下对对抗鲁棒损失的潜在影响上界；与 Zhang et al. [56]、Chatterji et al. [9] 聚焦泛化不同，MRC 聚焦对抗鲁棒性，且采用最坏扰动而非回退到初始值来分析。
2. **首次实证验证对抗训练模型存在冗余鲁棒容量**：发现某些模块的 MRC 极低（如 ResNet18 的 `layer2.1.conv2`，MRC=0.09，鲁棒性下降仅 2.86%），而另一些模块则高度关键（`layer4.1.conv1`，MRC=12.94，鲁棒性下降 53.03%）。
3. **提出 RiFT 三阶段后处理框架**：MRC 计算 → 非鲁棒关键模块微调 → 权重插值，可与 TRADES、MART、AWP、SCORE 等多种 AT 方法兼容。
4. **挑战"鲁棒性与泛化必然权衡"的教条**：实验显示插值初期两者可同时提升，反驳了 Tsipras et al. [46] 关于最优标准/鲁棒分类器特征本质不同的论点。

## 方法详解

**Step 1 — MRC 计算与模块选择**

对每个模块 $i$，定义 MRC（放松近似版，Algorithm 1）：
1. 用 PGD-10 在标准数据集 $\mathcal{D}_{std}$ 上生成对抗样本 $\mathcal{D}_{adv}$。
2. 冻结除模块 $i$ 外所有参数，对模块权重做**梯度上升**以最大化对抗损失 $\mathcal{R}$，同时施加约束 $\|\Delta \pmb{\theta}^{(i)}\|_2 \leq \epsilon \|\pmb{\theta}_{AT}^{(i)}\|_2$（$\epsilon=0.1$）。
3. MRC 值 = 扰动后鲁棒损失 - 原始鲁棒损失。**MRC 最低**的模块即为**非鲁棒关键模块** $\tilde{\pmb{\theta}}$。

**Step 2 — 非鲁棒关键模块微调**

冻结其余参数，仅微调 $\tilde{\pmb{\theta}}$ 在标准数据上的普通交叉熵损失：
$$\arg\min_{\tilde{\pmb{\theta}}} \sum_{(x,y)\in\mathcal{D}} \mathcal{L}(f(x, (\tilde{\pmb{\theta}};\pmb{\theta}\setminus\tilde{\pmb{\theta}})), y) + \lambda\|\tilde{\pmb{\theta}}\|_2$$
SGD+动量，lr=0.001，微调 10 epoch，选测试准确率最高权重 $\pmb{\theta}_{FT}$。

**Step 3 — 线性插值**
$$\pmb{\theta}_{\alpha} = (1-\alpha)\pmb{\theta}_{AT} + \alpha\pmb{\theta}_{FT}$$
最优 $\alpha^*$ 满足：泛化提升最大，且对抗鲁棒性不低于原始值的 0.1%。$\alpha^*$ 通常落在 0.6–0.9 区间。

**理论保证**：Theorem 3.1 证明 MRC 是微调后鲁棒损失增量的上界，保证了非鲁棒关键模块的微调不会显著损害鲁棒性。

## 实验与结果

**实验设置**：ResNet18 / ResNet34 / WRN34-10，CIFAR10 / CIFAR100 / Tiny-ImageNet，对抗鲁棒性 PGD-10（$\ell_\infty=8/255$），OOD 评估用 CIFAR-C / Tiny-ImageNet-C（15 类常见污染）。

**主要结果（Table 1）**：

| 架构 | 数据集 | Std Δ | OOD Δ | Adv Δ |
|---|---|---|---|---|
| ResNet18 | CIFAR10 | **+1.98** | **+2.13** | +0.02 |
| ResNet34 | CIFAR100 | **+2.21** | **+1.73** | +0.08 |
| WRN34-10 | Tiny-ImageNet | **+2.53** | **+2.05** | +0.10 |
| **平均** | — | **+1.63** | **+1.69** | **+0.04** |

- **最强提升**：WRN34-10 在 Tiny-ImageNet 上泛化提升 **+2.53%**，OOD 鲁棒性提升 **+2.05%**，对抗鲁棒性微增 +0.10%。
- RiFT 兼容多种 AT 方法（Table 2）：与 SCORE 结合在 CIFAR100 上 Std 提升 **+2.80%**，OOD 提升 **+2.38%**。
- 消融（Table 3/4）：仅微调单个非鲁棒关键模块效果最佳；微调多个非鲁棒关键模块（Top 2/3/5）效果反而不如单模块。

## 相关工作脉络
- **Tsipras et al. [46]**：证明鲁棒性与泛化在理论上存在权衡；本文发现 AT 后通过微调冗余模块可打破该权衡，二者并非不可调和。
- **Zhang et al. [56] / Chatterji et al. [9]**：研究模块关键性对**泛化**的影响（通过回退权重到初始值）；MRC 则聚焦**对抗鲁棒性**，采用最坏扰动方向度量，三者形成对比。
- **WiSE-FT [49] / Surgical Fine-tuning [28]**：关注预训练模型微调后的 OOD 性能退化；本文表明微调 AT 模型不会损害 OOD 鲁棒性，甚至有所提升。
- **Salman et al. [41]**：证明 AT 模型的全微调和线性探测均可提升下游迁移性能；本文聚焦 AT 模型在**同域**泛化和 OOD 鲁棒性的改进。
- **Kumar et al. [25]**：声称微调会破坏预训练特征的鲁棒性；本文直接反驳此结论，表明 AT 模型的微调策略需要区别于普通预训练模型。

## 局限性与未来方向
- **单模块限制**：微调多个非鲁棒关键模块效果反而下降，MRC 的独立模块假设限制了多模块推广（论文自述）。
- **近似 MRC**：放松版（固定对抗样本再优化权重扰动）是近似，精确求解需同时优化 $\Delta x$ 和 $\Delta\theta$，计算代价更高。
- **未扩展至更多架构**：仅验证了 ResNet 系列，Transformer 等架构是否同样存在冗余鲁棒容量未知。
- **缺乏理论解释**：为何微调非鲁棒关键模块能同时改善泛化和鲁棒性，缺乏深入的理论分析（论文在 Conclusion 中呼吁更多理论工作）。
- **插值依赖搜索**：最优 $\alpha^*$ 需在验证集上枚举，增加了部署开销。

## 研究启发与可借鉴点
- **MRC 概念可迁移**：MRC 度量思路可用于分析其他架构（如 ViT、Swin）的冗余鲁棒容量，或指导模型压缩/剪枝中以保留关键鲁棒模块。
- **"后处理 + 插值"范式实用**：RiFT 不修改 AT 训练流程，可即插即用，对工业界落地友好；可探索将其与更多 AT 变体（如 autoattack 训练）结合。
- **微调非关键模块的策略**：对于任何需要微调鲁棒模型的场景（如领域适配），可选择性微调低关键性模块而非全参数微调，以保留鲁棒特性。
- **值得对比的创新机会**：将 MRC 与 AWP（Adversarial Weight Perturbation）等已有扰动方法结合，探索二者在"权重扰动"视角下的联系。
- **实验设计借鉴**：论文通过插值曲线（Figure 1）直观展示泛化与鲁棒性的协同提升，这种可视化分析值得复用在其他方法评估中。

## 关键术语表
- **Module Robust Criticality（MRC）**：衡量某模块在最坏情况权重扰动下对抗鲁棒损失的最大增量，值越低表示该模块对鲁棒性越不关键。
- **Non-robust-critical module**：MRC 最低的模块，具有冗余鲁棒容量，适合进行微调而不损害对抗鲁棒性。
- **Robust Critical Fine-Tuning（RiFT）**：本文提出的三阶段后处理方法，通过 MRC 识别冗余模块、仅微调该模块、再与原 AT 权重插值来获得最优模型。
- **Adversarial Training（AT）**：通过在训练过程中加入对抗样本并最小化最坏情况下的损失来提升模型对抗鲁棒性的训练策略。
- **Out-of-Distribution（OOD）鲁棒性**：模型在面对分布外输入（如污染、模糊、光照变化）时的分类准确性。
- **插值（Interpolation）**：在线性空间中混合两组权重 $\pmb{\theta}_\alpha = (1-\alpha)\pmb{\theta}_{AT} + \alpha\pmb{\theta}_{FT}$，寻找泛化与鲁棒性的最佳平衡点。
- **鲁棒损失（Robust Loss）**：对每个样本取其对抗扰动下的最大损失（最坏情况交叉熵），再对所有样本求和。
- **WiSE-FT**：Wortsman et al. 提出的鲁棒微调方法，通过集成学习保留零样本模型的 OOD 鲁棒性，与本文形成对比。

## 可复现要素
- **数据集**：CIFAR10、CIFAR100、Tiny-ImageNet（公开）；CIFAR10-C、CIFAR100-C、Tiny-ImageNet-C（公开）。
- **代码**：已开源，https://github.com/Immortalise/RiFT。
- **关键超参**：AT 训练 110 epoch，lr 从 0.1 衰减；微调 10 epoch，lr=0.001；MRC 计算中 $\epsilon=0.1$，迭代步数 10；PGD-10，$\ell_\infty=8/255$；权重插值步长需搜索。
