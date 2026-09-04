---
title: "Fast-Adversarial-Training-with-Smooth-Convergence"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:38:19"
field: "对抗机器学习/对抗训练"
keywords: ["fast adversarial training", "catastrophic overfitting", "adversarial robustness", "smooth convergence", "regularization"]
innovations: ["提出ConvergeSmooth约束，通过限制相邻epoch损失差异来稳定FAT训练并避免大扰动预算下的灾难性过拟合", "设计动态收敛步长γ_t自适应筛选异常样本，无需精细调参", "提出weight centralization正则化，从参数空间约束防止训练不稳定"]
benchmarks: ["CIFAR-10", "CIFAR-100", "Tiny ImageNet"]
---

# 论文速读：Fast-Adversarial-Training-with-Smooth-Convergence

## 一句话总结
本文针对快速对抗训练（FAT）在大扰动预算下出现的**灾难性过拟合**问题，从损失函数收敛稳定性视角重新分析该现象，并提出**ConvergeSmooth**（平滑收敛约束）与**Weight Centralization**（权重中心化）两种 attack-agnostic 的正则化方法，有效阻止训练过程中的损失异常波动，在多种数据集和模型上超越所有现有 FAT 方法。

## 研究问题与动机
- **核心问题**：现有 FAT 方法（如 FGSM-RS、GradAlign、FGSM-MEP 等）在较小扰动预算（$\xi \le 8/255$）下能有效避免灾难性过拟合，但在较大扰动预算（如 $\xi = 16/255$）下仍会崩溃，对抗鲁棒性骤降至接近零。
- **现象洞察**：通过分析训练曲线发现，灾难性过拟合表现为良性样本损失 $\mathcal{L}(x,\theta)$ 轻微波动、对抗样本损失 $\mathcal{L}(x',\theta)$ 急剧下降，同时对抗准确率迅速下滑；而过拟合后训练有时会出现"重启"振荡 phase。
- **既有方法不足**：GradAlign 的梯度对齐正则可能降低 $\mathcal{L}(x,\theta)$ 的稳定性；FGSM-MEP 的预测对齐约束并非保持 FAT 稳定的最优选择；这些方法仅从数据增强或梯度/ logits 对齐角度入手，未直接约束收敛过程本身的稳定性。
- **动机转化**：将"灾难性过拟合"重新定义为**损失收敛不稳定性**，认为适度的平滑收敛过程即代表稳定的 FAT 过程，从而提出直接约束相邻 epoch 损失差异的方法。

## 核心贡献（创新点）
1. **从收敛稳定性视角重新审视灾难性过拟合**：通过可视化分析揭示先前 FAT 方法在大 $\xi$ 下的训练崩溃模式，建立"损失收敛异常 ↔ 灾难性过拟合"的关联。与已有工作不同，本文不依赖梯度对齐或 logits 正则，而是直接约束训练动力学。
2. **提出 ConvergeSmooth（收敛平滑约束）**：通过限制相邻 epoch 之间对抗损失与良性损失的绝对差异来约束训练过程，设计动态收敛步长 $\gamma_t$ 以适应损失非线性衰减特性，并与 FGSM-RS/FGSM-MEP/FGSM-BP 等多种初始化策略兼容。与已有方法本质区别在于：这是**训练过程层面的通用稳定性约束**，而非针对特定攻击初始化的改进。
3. **提出 Weight Centralization（权重中心化）**：无需引入额外超参数（除平衡系数 $w_3$ 外），通过将当前权重向历史权重均值靠近来稳定训练，利用"收敛后不同 epoch 模型权重趋于相似"的先验知识。与已有方法本质区别在于：直接从参数空间施加约束，而非从损失或梯度空间入手。
4. **系统性实验验证**：在 CIFAR-10/100 和 Tiny ImageNet 上，使用 ResNet18、WideResNet34-10、PreActResNet18 等多种 backbone，在多种扰动预算（10/255、12/255、16/255）和多种攻击（PGD-10/20/50、C&W、APGD-CE、Autoattack）下验证，B-MEP 在 CIFAR-10 $\xi=16/255$ 下达到 32.95% PGD-50 鲁棒性，接近 PGD-AT（33.92%）但训练速度提升约 3.5 倍。

## 方法详解
**整体框架**：在标准对抗训练目标 $\min_\theta \mathbb{E}[\mathcal{L}(x',\theta)]$ 基础上，添加互补约束 $\mathcal{L}_{CS}(t)$：

$$\min_{\theta_t} \mathbb{E}_{(x,y)\sim D}[\mathcal{L}(x_t',\theta_t) + \mathcal{L}_{CS}(t)]$$

**ConvergeSmooth 损失**（相邻 epoch 损失差约束）：
$$\mathcal{L}_{CS}(t) = w_1 \cdot |\mathcal{L}(x_t',\theta_t) - \mathcal{L}(x_{t-1}',\theta_{t-1})| + w_2 \cdot |\mathcal{L}(x,\theta_t) - \mathcal{L}(x,\theta_{t-1})|$$
- 为节省内存，用 epoch 级期望 $u_{t-1}' = \mathbb{E}[\mathcal{L}(x_{t-1}',\theta_{t-1})]$ 和 $u_{t-1} = \mathbb{E}[\mathcal{L}(x,\theta_{t-1})]$ 替代逐样本历史值。
- 仅对**异常样本**施加约束：条件 $\mathcal{C}(x) = (|\mathcal{L}(x,\theta_t) - u_{t-1}| \ge \gamma_t)$，即点wise损失与上一 epoch 均值偏差超过阈值。
- **动态收敛步长** $\gamma_t = \min(\max(d_{t-1}, \gamma_{min}), \gamma_{max})$，其中 $d_{t-1} = |u_{t-1} - u_{t-2}|$，使阈值随训练非线性衰减而自适应调整。
- 提供两种实现：**Example-based**（逐样本约束，Eq.12）和 **Batch-based**（按 batch 均值约束，Eq.13）。

**Weight Centralization**：
$$\mathcal{L}_{CS}^W(t) = w_3 \cdot ||\theta_t - \frac{1}{len(\phi)}\sum_{j\in\phi}\theta_j||_p \quad (p=2)$$
- 将当前权重 $\theta_t$ 向历史权重集合 $\phi$ 的均值拉拢，防止权重突变。
- 仅需一个额外超参数 $w_3$（实验中设为 0.1），无需精细调参。

**对抗样本生成**：$x_t' = x + \delta_{0,t} + \delta_t$，其中 $\delta_t = \arg\max_{\delta_t \in [-\xi,\xi]} \mathcal{L}(x_t',\theta_{t-1})$，$\delta_{0,t}$ 可由 FGSM-RS/FGSM-MEP/FGSM-BP 等任意初始化策略提供，方法为 attack-agnostic。

## 实验与结果
**数据集与模型**：CIFAR-10（ResNet18/WideResNet34-10）、CIFAR-100（ResNet18）、Tiny ImageNet（PreActResNet18）；训练 110 epoch，SGD 优化，batch size=128。

**基线方法**：PGD-AT、FGSM-RS、GradAlign、ZeroGrad、NuAT、ATAS、FGSM-MEP，以及 OAAT、ExAT、ATES、AWP 等慢速 AT 方法的 FAT 适配版。

**主要结果（CIFAR-10, $\xi=16/255$, ResNet18）**：
| 方法 | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | APGD-CE | AA | 时间(min) |
|------|-------|--------|--------|--------|-----|---------|-----|----------|
| PGD-AT | 65.30 | 40.73 | 35.08 | 33.92 | 30.84 | 33.08 | 26.29 | 370 |
| FGSM-MEP | 53.32 | 31.85 | 27.28 | 26.56 | 22.10 | 26.08 | 18.98 | 92 |
| **B-MEP (Ours)** | **63.84** | **40.13** | **34.21** | **32.95** | **28.19** | **32.04** | **23.68** | **102** |
| **B-RS (Ours)** | **65.42** | **37.54** | **30.01** | **27.85** | **26.28** | **26.52** | **19.43** | **75** |

- B-MEP 的 PGD-50 鲁棒性（32.95%）仅比 PGD-AT（33.92%）低 0.97 个百分点，但训练速度快约 3.5 倍。
- B-RS 在 AA 指标上达到 19.43%，优于所有 RS-based 方法。
- 多预算实验（Tab.4）：在 $\xi=10/255$ 和 $12/255$ 下，B-MEP 均取得最优结果（如 $\xi=12/255$ 时 AA=33.26% vs. FGSM-MEP 的 27.23%）。
- CIFAR-100 和 Tiny ImageNet 结果一致表明方法可扩展至更大数据集和更深网络。
- 训练稳定性验证：三次独立运行中 mbest 与 mfinal 差距极小，确认无灾难性过拟合。

## 相关工作脉络
1. **FGSM-RS (Wong et al., ICLR 2020)**：通过随机初始化扰动增加对抗样本多样性，是 FAT 基础方法；本文在此基础上添加 ConvergeSmooth 约束以提升大 $\xi$ 下的稳定性。
2. **GradAlign (Andriushchenko & Hein, NeurIPS 2020)**：最大化良性/对抗样本梯度对齐；本文指出其可能降低 $\mathcal{L}(x,\theta)$ 稳定性，而 ConvergeSmooth 直接约束收敛过程，更通用。
3. **FGSM-MEP (Wei et al., ECCV 2022)**：利用历史对抗扰动 Momentum 初始化；本文 B-MEP 在其基础上添加批次级平滑约束，克服其在 $\xi=16/255$ 下的崩溃。
4. **NuAT (Sriramanan et al., ICML 2021)**：通过核范数正则 logits 对齐；仅适用于小 $\xi$，本文方法在大 $\xi$ 下显著超越。
5. **ATAS (Huang et al., 2022)**：学习自适应步长；本文方法不依赖步长学习，而是从损失收敛角度保证稳定性。
6. **OAAT (Addepalli et al., ECCV 2022)**：针对大扰动预算的慢速 AT 方法；本文证明其 FAT 适配版在 clean accuracy 上有优势，但鲁棒性不及 ConvergeSmooth 增强的 FAT。

## 局限性与未来方向
- **超参数敏感性**：虽 weight centralization 仅需 $w_3$，但 ConvergeSmooth 仍需调整 $\gamma_{max}$、$\gamma_{min}$ 和 $w_1/w_2$；不同数据集/模型需微调（如 CIFAR-100 最优 $\gamma_{max}=0.06$，而 CIFAR-10 为 0.03）。
- **仅针对单步 FAT**：方法设计与 FGSM 类单步攻击耦合，对多步快速训练（如 PGD-based fast variants）的适用性未验证。
- **内存开销**：Example-based 版本需存储上一 epoch 的 epoch-level 损失均值；batch-based 版本额外需 batch 级统计量，对超大数据集可能增加少许开销。
- **理论分析不足**：论文以实验观察为主，对"平滑收敛 ↔ 避免灾难性过拟合"之间的严格理论联系缺乏形式化证明。
- **未来方向**：可扩展至更大扰动预算（$\xi > 16/255$）、更复杂架构（Vision Transformer）、视频/序列数据的对抗训练，以及自动超参搜索策略。

## 研究启发与可借鉴点
1. **"收敛稳定性"作为诊断工具**：将训练曲线异常（loss convergence outliers）作为灾难性过拟合的早期预警指标，可用于监控其他训练过程的稳定性，具有通用诊断价值。
2. **跨层/跨 epoch 正则化范式**：ConvergeSmooth 的"约束相邻迭代变化量"思想可迁移至其他训练不稳定场景（如扩散模型训练、持续学习、联邦学习），作为通用的训练稳定器。
3. **动态阈值的自适应设计**：$\gamma_t$ 基于前序 epoch 损失变化量 $d_{t-1}$ 动态调整，无需手动调度，类似思想可应用于学习率调度、早停策略等。
4. **attack-agnostic 的模块化设计**：ConvergeSmooth 和 weight centralization 可作为即插即用的插件与任何 FAT 初始化策略组合，为后续工作提供可复用的稳定性组件。
5. **训练效率-鲁棒性权衡的新基准**：B-MEP 在保持接近 PGD-AT 鲁棒性的同时将训练时间从 370min 降至 102min，为后续研究提供了高效的强 baseline。

## 关键术语表
- **Fast Adversarial Training (FAT)**：仅使用单步 FGSM 生成对抗样本进行训练的快速对抗训练方法，计算高效但易发生灾难性过拟合。
- **Catastrophic Overfitting**：FAT 训练过程中对抗鲁棒性突然崩溃至接近零的现象，通常伴随损失函数的异常波动。
- **ConvergeSmooth**：本文提出的平滑收敛约束，通过限制相邻 epoch 间损失差异来稳定 FAT 训练过程。
- **Dynamic Convergence Stride ($\gamma_t$)**：自适应阈值，根据前序 epoch 损失变化量的非线性衰减动态调整，用于筛选需施加约束的异常样本。
- **Weight Centralization**：将当前模型权重向历史权重均值拉拢的正则化技术，防止权重突变导致训练不稳定。
- **Perturbation Budget ($\xi$)**：对抗扰动允许的最大幅度，$\xi=16/255$ 表示像素级最大扰动为 $16/255 \approx 0.063$。
- **Attack-agnostic**：指所提方法不依赖特定攻击初始化策略，可与多种 FAT 变体（FGSM-RS/MEP/BP）结合使用。
- **Autoattack (AA)**：集成 APGD-CE、APGD-T、FAB 和 Square attack 的对抗鲁棒性评估基准，提供可靠的多攻击 ensemble 评估。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、Tiny ImageNet（均公开可用）
- **代码**：已开源，https://github.com/FAT-CS/ConvergeSmooth
- **关键超参**：$\gamma_{max} \in \{0.03, 0.06\}$（依数据集）、$\gamma_{max}/\gamma_{min}=1.5$、$w_2=1.0$、$w_3=0.1$、$w_1$ 依数据集微调（CIFAR-100 取 0，Tiny ImageNet 取 0.3~0.7）、$\xi=16/255$（默认）、batch size=128、SGD with momentum 0.9、weight decay=5e-4
- **硬件**：单张 GeForce RTX 3090 GPU
- **训练配置**：110 epoch，LR 在 100 和 105 epoch 各 decay 0.1 倍，初始 LR CIFAR 系列为 0.1、Tiny ImageNet 为 0.01
