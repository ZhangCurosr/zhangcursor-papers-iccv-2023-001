---
title: "Improving-Generalization-of-Adversarial-Training-via-Robust"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Improving_Generalization_of_Adversarial_Training_via_Robust_Critical_Fine-Tuning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:21:21"
field: "对抗鲁棒性与泛化权衡"
keywords: ["adversarial training", "robustness", "generalization", "fine-tuning", "module criticality", "OOD robustness"]
innovations: ["提出模块鲁棒关键性(MRC)度量对抗冗余容量", "设计三步RiFT后处理框架提升泛化且不损鲁棒性", "实验证明非鲁棒关键模块微调可同时提升泛化与对抗鲁棒性"]
benchmarks: ["CIFAR10", "CIFAR100", "Tiny-ImageNet", "CIFAR10-C", "CIFAR100-C"]
---

# 论文速读：Improving-Generalization-of-Adversarial-Training-via-Robust

## 一句话总结
本文提出鲁棒关键微调（RiFT），通过度量模块鲁棒关键性（MRC）识别对抗训练后模型中的冗余容量模块，仅微调该非鲁棒关键模块即可显著提升泛化与分布外（OOD）鲁棒性，同时保持或增强对抗鲁棒性。

## 研究问题与动机
- **对抗训练的泛化困境**：对抗训练（AT）虽能有效提升对抗鲁棒性，但会严重牺牲在分布内数据上的泛化能力，并对分布外（OOD）样本（如对比度、亮度、雾等扰动）异常脆弱。
- **冗余容量假设**：深度神经网络存在冗余拟合容量，特定模块可被删除、置换或重置而仅轻微降低泛化性能；作者提出直觉问题——对抗训练后的模型是否同样存在冗余鲁棒容量？
- **超越现有思路**：现有工作多在对抗训练过程中通过实例重加权、无标签数据引入、重新定义鲁棒损失等方式缓解权衡；本文从后处理角度切入，利用对抗训练后模型的冗余容量进行微调。
- **挑战既定认知**：文献曾声称最优标准分类器与最优鲁棒分类器的特征本质不同，且微调会扭曲预训练特征并损害OOD性能；本文实验发现微调非鲁棒关键模块可同时提升两者。

## 核心贡献（创新点）
1. **提出模块鲁棒关键性（MRC）度量**：量化每个模块在最坏权重扰动下对对抗鲁棒性的贡献上界，理论证明其为鲁棒损失增量的严格上界；区别于 Zhang et al. [56] 和 Chatterji et al. [9] 关注泛化关键性，MRC 首次针对对抗鲁棒性建模，且采用最坏扰动而非参数回退方式。
2. **设计鲁棒关键微调（RiFT）框架**：三步流程——MRC 表征筛选非鲁棒关键模块 → 冻结其他参数仅微调该模块 → 线性插值原始对抗权重与微调权重以最优平衡泛化与鲁棒性；这是首个通过后处理微调对抗训练模型而非修改训练过程的通用方法。
3. **揭示对抗鲁棒性与泛化的可兼得性**：实验证明初始插值阶段对抗鲁棒性可随泛化同步提升（约 +0.3%），挑战了 Tsipras et al. [46] 的固有权衡论断；同时在多种 AT 方法（TRADES、MART、AWP、SCORE）上验证 RiFT 的通用兼容性。

## 方法详解
**步骤一：模块鲁棒关键性（MRC）表征**
- 定义 MRC 为模块权重在最坏约束扰动下的鲁棒损失最大增量：$MRC(f, \pmb{\theta}^{(i)}, \mathcal{D}, \epsilon) = \max_{\Delta\pmb{\theta} \in \mathcal{C}_\theta} \mathcal{R}(f(\pmb{\theta}+\Delta\pmb{\theta}), \mathcal{D}) - \mathcal{R}(f(\pmb{\theta}), \mathcal{D})$，其中约束集 $\mathcal{C}_\theta = \{\Delta\pmb{\theta} | \|\Delta\pmb{\theta}\|_p \leq \epsilon\|\pmb{\theta}^{(i)}\|_p\}$。
- 理论保证（定理 3.1）：任何在约束 $\mathcal{C}_\theta$ 内的优化导致的鲁棒损失增量均不超过 MRC 值，为非鲁棒关键模块的微调提供鲁棒性退化上界保障。
- 实际计算采用松弛版本（Algorithm 1）：先用 PGD-10 生成对抗样本，固定对抗样本后以梯度上升迭代 T 步寻找最坏权重扰动，每步检查并投影到约束球内。

**步骤二：非鲁棒关键模块微调**
- 选定 MRC 最低的模块 $\tilde{\pmb{\theta}}$，冻结其余参数，在标准数据集 $\mathcal{D}_{std}$ 上用 SGD with momentum 微调该模块：$\arg\min_{\tilde{\pmb{\theta}}} \sum_{(x,y)\in\mathcal{D}} \mathcal{L}(f(x, (\tilde{\pmb{\theta}}; \pmb{\theta}\setminus\tilde{\pmb{\theta}})), y) + \lambda\|\tilde{\pmb{\theta}}\|_2$。
- 选取测试准确率最高的权重作为 $\pmb{\theta}_{FT}$。

**步骤三：插值缓解鲁棒-泛化权衡**
- 线性插值：$\pmb{\theta}_\alpha = (1-\alpha)\pmb{\theta}_{AT} + \alpha\pmb{\theta}_{FT}$。
- 最优插值点 $\alpha^*$ 选择标准：最大化泛化提升，同时满足对抗鲁棒性不低于原始值 0.1%。
- 插值实质上起到权重集成效果，借鉴 WiSE-FT 思想进一步提升鲁棒性。

## 实验与结果
- **数据集**：CIFAR10、CIFAR100、Tiny-ImageNet（分布内）；CIFAR10-C、CIFAR100-C、Tiny-ImageNet-C（15 类常见视觉扰动，评估 OOD 鲁棒性）。
- **模型**：ResNet18、ResNet34、WideResNet34-10（WRN34-10）。
- **评估指标**：标准测试准确率（泛化）、PGD-10 $\ell_\infty=8/255$ 对抗鲁棒准确率、Corruption 数据集准确率（OOD 鲁棒性），AutoAttack 验证有效性。
- **主要结果（Table 1）**：
  - **ResNet18 + CIFAR10**：泛化 +1.98%，OOD +2.13%，对抗鲁棒 +0.02%（53.63→53.65%）。
  - **ResNet34 + CIFAR100**：泛化 +2.21%，OOD +1.73%，对抗鲁棒 +0.08%。
  - **WRN34-10 + Tiny-ImageNet**：泛化 +2.53%，OOD +2.05%，对抗鲁棒 +0.10%。
  - **跨架构数据集平均提升**：泛化 +1.70%，OOD +1.68%，对抗鲁棒基本持平（+0.02~0.08%）。
- **与不同 AT 方法结合（Table 2）**：在 TRADES、MART、AWP、SCORE 上均有效，如 SCORE+RiFT 在 CIFAR10 上泛化提升 +1.45%、OOD +1.55%。
- **消融实验**：仅微调单个非鲁棒关键模块效果最优，微调多个模块（Top 2/3/5）反而下降；插值系数 $\alpha^*$ 通常在 0.6~0.9 之间。

## 相关工作脉络
1. **对抗鲁棒性与泛化权衡**：Tsipras et al. [46] 证明理论上存在固有权衡；Raghunathan et al. [36]、Pang et al. [32] 等在训练过程中缓解；本文从后处理微调角度提供新思路。
2. **冗余拟合容量**：Zhang et al. [55] 发现 DNN 可记忆随机标签；Veit et al. [47]、Rosenfeld & Tsotsos [39]、Chatterji et al. [9] 证明部分模块对泛化非关键；本文聚焦**鲁棒冗余容量**而非泛化冗余。
3. **模块关键性分析**：Zhang et al. [56] 通过参数回退研究泛化关键模块；本文 MRC 通过最坏扰动衡量鲁棒关键模块，方法与动机均不同。
4. **微调方法**：Salman et al. [41] 证明 AT 模型全量/线性微调可提升迁移性能；Kumar et al. [25] 警告微调会扭曲特征损害 OOD；本文反向证明选择性微调非鲁棒关键模块可同时提升两者。
5. **权重插值/集成**：WiSE-FT [49] 利用集成提升零样本鲁棒模型微调性能；本文插值为确定性加权平均，专为 AT 模型设计。

## 局限性与未来方向
- **理论解释不足**：MRC 虽为鲁棒损失上界，但无法保证微调方向一定不偏离；Hessian 特征值分布的稀疏性意味着优化方向未必对齐最差特征方向，理论完备性有待加强。
- **单模块限制**：消融显示微调多个非鲁棒关键模块性能反而下降，因 MRC 在固定其他模块参数下评估，多模块联合最坏扰动效应难以用单模块 MRC 推断；扩展 MRC 至多模块联合评估是潜在方向。
- **模型容量依赖性**：任务越难或模型越小，非鲁棒关键模块比例越低，泛化增益受限（如 WRN34-10 在 CIFAR10 上仅 +0.48%）；大模型收益更显著。
- **未探索更深层机制**：本文定位为初步探索，对"为何微调非鲁棒关键模块可同时提升泛化与鲁棒"缺乏深入理论分析。

## 研究启发与可借鉴点
1. **后处理微调新思路**：将对抗训练与微调解耦，为"训练-微调"两阶段范式提供对抗鲁棒场景的实证支持，可迁移至其他鲁棒学习方法。
2. **模块级冗余度量方法**：MRC 的最坏扰动框架可推广至其他模型属性度量（如泛化、校准、效率），为网络剪枝、结构搜索提供新指标。
3. **插值权重的工程实践**：简单线性插值即能有效平衡多目标，无需额外超参搜索，可作为鲁棒模型部署前的通用后处理模块。
4. **OOD 鲁棒性的重新审视**：反驳了微调必然损害 OOD 性能的既有认知，提示应区分模块类型选择微调策略，而非一概否定微调。
5. **可结合的方向**：将 MRC 与 AWP [50]、SCORE [32] 等先进 AT 方法结合已有增益，未来可探索与自动化机器学习（NAS）或持续学习结合。

## 关键术语表
- **Adversarial Training (AT)**：通过在训练过程中引入对抗样本，最小化最坏扰动下的损失，从而提升模型对抗攻击鲁棒性的训练范式。
- **Module Robust Criticality (MRC)**：衡量特定网络模块参数在最坏约束扰动下对整体对抗鲁棒损失增量的上界度量，值越低表示模块对鲁棒性越不关键。
- **Non-robust-critical module**：MRC 值最低的模块，其参数变化对对抗鲁棒性影响极小，可安全用于微调以提升泛化。
- **Out-of-distribution (OOD) robustness**：模型在面对分布外扰动（如噪声、模糊、天气变化等 corruption）时的泛化鲁棒能力。
- **Interpolation**：线性插值对抗训练权重与微调权重 $(1-\alpha)\pmb{\theta}_{AT} + \alpha\pmb{\theta}_{FT}$，作为轻量级权重集成手段平衡泛化与鲁棒。
- **PGD-10**：使用 10 步投影梯度下降生成的强对抗扰动，$\ell_\infty \leq 8/255$，作为标准对抗评估协议。
- **Robust loss landscape sharpness**：鲁棒损失曲面在最优解附近的尖锐程度，高 MRC 对应尖锐曲面，微调易导致鲁棒性大幅下降。
- **Trade-off between robustness and generalization**：对抗鲁棒性与标准泛化性能之间的此消彼长关系，本文证明可通过模块化微调部分打破该权衡。

## 可复现要素
- **数据集**：CIFAR10、CIFAR100、Tiny-ImageNet 公开可用；Corruption 数据集（CIFAR10-C 等）随 Hendrycks & Dietterich [19] 公开。
- **代码**：GitHub 开源地址 https://github.com/Immortalise/RiFT。
- **关键超参**：
  - 对抗训练：PGD-10，$\ell_\infty=8/255$，110 epochs，LR 从 0.1 在 epoch 100/105 各衰减 10 倍。
  - MRC 计算：$\epsilon=0.1$，$\|\cdot\|_2$ 约束，迭代步数 $T=10$，学习率 $\gamma$ 未明确（见附录）。
  - 微调：SGD with momentum，10 epochs，初始 LR=0.001，epoch 5 后衰减 10 倍，$\ell_2$ 权重衰减系数 $\lambda$ 未明确。
  - 插值：$\alpha^*$ 通过验证集泛化性能选择，典型范围 0.6~0.9。
- **硬件**：论文未提及具体 GPU 配置。
