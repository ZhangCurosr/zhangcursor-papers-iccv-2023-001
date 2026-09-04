---
title: "Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Enhancing_Fine-Tuning_Based_Backdoor_Defense_with_Sharpness-Aware_Minimization_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:13:39"
field: "深度学习后门攻击与防御"
keywords: ["backdoor defense", "fine-tuning", "sharpness-aware minimization", "post-training defense", "neuron weight norm", "DER metric"]
innovations: ["从神经元权重范数视角揭示 Vanilla FT 后门防御不足的机理", "提出 FT-SAM 将 SAM 与自适应扰动结合以增强微调型后门清除", "引入基于权重范数的自适应扰动预算使后门相关神经元获得更大更新"]
benchmarks: ["CIFAR-10", "Tiny ImageNet", "GTSRB"]
---

# 论文速读：Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness

## 一句话总结
本文提出 FT-SAM，将 Sharpness-Aware Minimization（SAM）与自适应扰动相结合以增强基于微调的后门防御；该方法通过显式放大对后门相关神经元（通常具有较大权重范数）的扰动，实现比 Vanilla Fine-Tuning 更彻底的后门清除并维持较高的良性准确率。

## 研究问题与动机
- 在仅持有少量良性样本的 post-training 设定下，Vanilla Fine-Tuning（FT）对强后门攻击（如 Blended、LF、SSBA）的防御表现仍然较差。
- 归因在于已拟合良性数据的后门模型处于较“深”的局部极小，FT 对神经元权重整体扰动较小，难以实质性改变后门相关神经元的表示。
- 实验观察表明：后门相关神经元与较大权重范数正相关，且 FT 后这些神经元权重范数变化不大；若能针对性缩小大范数神经元，可能更有效破坏后门触发路径。
- 因此，需要一种能在微调过程中对不同神经元施加差异化、尤其针对大范数神经元更大扰动的优化范式。

## 核心贡献（创新点）
- 从神经元权重变化视角系统解释 Vanilla FT 后门防御不足的机理，发现后门相关神经元与较大权重范数的稳定正相关，并为防御设计提供新依据。与以往主要从特征空间或剪枝角度分析的做法不同，本文聚焦权重范数分布与大范数神经元的敏感性。
- 提出 FT-SAM 微调整体范式，通过 min-max 形式同时最小化损失值与损失锐度，使优化过程更容易跳出当前局部极小并对后门相关神经元施加更大更新幅度。与标准 SAM 主要关注泛化不同，本文将其引入后门清除场景并做针对性适配。
- 引入基于神经元权重范数的自适应扰动预算 $\mathbf{T}_w=\mathrm{diag}(|w_1|,\dots,|w_d|)$，使大权重范数神经元获得更大有效扰动。这与 ASAM 等尺度不变型 SAM 的通用自适应机制相比，更直接对齐“大范数=后门相关”的先验观察。
- 在多个主流攻击、数据集与网络结构上达到 SOTA 级后门防御性能，并证明 FT-SAM 可替换既有基于微调流程中的 FT 环节，甚至可与剪枝类防御（FP、ANP）组合获得更高 DER。

## 方法详解
- 目标函数采用 min-max 形式：$\min_{w}\max_{\epsilon\in S}\mathcal{L}(w+\epsilon)$，其中 $\mathcal{L}$ 为良性数据上的交叉熵期望损失，约束集 $S=\{\epsilon:\|\mathbf{T}_w^{-1}\epsilon\|_2\le\rho\}$，$\rho$ 为扰动预算超参。
- 自适应尺度矩阵定义为 $\mathbf{T}_w=\mathrm{diag}(|w_1|,\dots,|w_d|)$，使得相对各参数的扰动预算与其权重绝对值成正比，从而在相同 $\rho$ 下对大权重范数神经元施加更强扰动。
- 内层最大化由一阶泰勒近似求解，得到扰动更新：$\epsilon_{t+1}\approx\rho\frac{\mathbf{T}_{w_t}^2\nabla_{w_t}\mathcal{L}(w_t)}{\|\mathbf{T}_{w_t}\nabla_{w_t}\mathcal{L}(w_t)\|_2}$。
- 外层最小化在扰动后点处沿梯度下降：$w_{t+1}=w_t-\eta\nabla_w\mathcal{L}(w+\epsilon_{t+1})|_{w=w_t}$，学习率为 $\eta$。
- 机制层面：大范数神经元在 $\epsilon$ 与 $\mathbf{T}_w$ 共同作用下获得更大有效步长，对应梯度的 $\ell_2$ 范数也显著高于普通神经元；由此导致后门相关神经元被更充分扰动，最终降低其权重范数方差，避免决策被少数后门神经元主导。

## 实验与结果
- 基准与设置：CIFAR-10、Tiny ImageNet、GTSRB；PreAct-ResNet18、VGG19-BN；10 种 SOTA 后门攻击（含 BadNets、Blended、LF、WaNet、Input-aware、CLA、SIG、SSBA、Trojan 等），主要报告 10% 中毒率并在部分设定中对比 5%。
- 评估指标：ACC、ASR 及本文引入的综合指标 DER，公式为 $\mathrm{DER}=[\max(0,\Delta\mathrm{ASR})-\max(0,\Delta\mathrm{ACC})+1]/2$，越高越好。
- CIFAR-10 平均结果（PreAct-ResNet18，5% 良性数据）：FT-SAM 平均 ACC=92.10%、ASR=2.47%、DER=96.62%，显著优于其余 7 类基线；对复杂攻击如 Blended、LF、SSBA 的 ASR 分别降至 4.91%、3.81%、2.80%，对应的 DER 达到 95.90%、96.65%、96.75%。
- Tiny ImageNet 平均结果（5% 良性数据）：FT-SAM 平均 ACC=52.18%、ASR=1.16%、DER=92.16%，在多类复杂攻击上保持稳定防御并兼顾可用性；而 i-BAU 在该数据集上表现明显退化。
- 超参敏感度：$\rho$ 在较宽范围内均可维持较高 DER；更大 $\rho$ 通常会加快“BM”进程但对最终性能影响相对温和。
- 兼容性实验表明，FT-SAM 可分别替换 FP 与 ANP 中的微调/后处理环节，进一步提升整体防御稳健性。

## 相关工作脉络
- Fine-pruning（FP）通过良性激活路径差异剪枝后再微调，依赖剪枝阈值对攻击类型的敏感性；FT-SAM 不修改网络结构，在保留可用性的同时提供更强通用防御。
- NAD 以教师-学生蒸馏方式微调，经验上性能与 FT 相近；FT-SAM 通过在锐度感知优化中放大后门相关神经元更新，明显缩小与纯微调的差距。
- ANP 利用良性数据上的极小极大搜索定位敏感神经元并掩码；FT-SAM 则直接从权重范数先验出发，以自适应扰动替代显式神经元搜索，实现更易集成到常规微调流程的防御。
- i-BAU 通过隐式超梯度解决极小极大问题，在 CIFAR 上表现较强但在 Tiny ImageNet 等大输入任务上优化难度上升；FT-SAM 的自适应缩放策略对输入规模更具鲁棒性。
- NC 通过优化恢复触发器并重训正则化模型；FT-SAM 不需要显式触发器逆向，仅凭良性数据微调即可达成相近或更优的综合指标。
- ASAM 等 SAM 变体主要服务于泛化与平坦极小搜索；本文将其迁移至后门防御，并以权重范数为枢纽引入任务特定的自适应扰动边界。

## 局限性与未来方向
- 方法在当前主要视觉基准与常见攻击设定下验证，对更强或更隐蔽的 clean-label、频率域及跨模态后门场景的普适性仍需进一步检验。
- 对极小良性数据比例（如 1%）仍有一定性能波动，复杂攻击下的 ASR 与 DER 并非在所有单条样本配比下都保持绝对优势。
- 自适应扰动依赖权重范数与后门相关性的统计先验；在极端分布偏移或特殊网络结构下，该先验未必严格成立，可能存在适配边界。
- 与剪枝、蒸馏等模块的组合实验有限，更多与训练阶段防御、数据清洗技术的联合框架尚待系统探索。

## 研究启发与可借鉴点
- 将“神经元权重范数—后门相关性”的经验观察形式化为优化约束，为后续防御提供可直接套用的先验信号设计范式。
- 在常规微调流程中以最小改动（替换训练目标与扰动计算）换取显著性能提升，便于与现有 defense pipeline 无缝对接。
- 引入 DER 这一同时度量 ASR 下降与 ACC 下降的综合指标，有助于更公平地比较各类兼顾可用性与安全性的防御方法。
- 基于泰勒近似的一阶 min-max 求解保持实现简洁，可推广到其他需要差异化参数扰动的小样本再训练场景。
- 将 Grad-CAM 可视化与梯度范数分析结合验证机制，为后续防御方法提供可复用的解释性评估模板。

## 关键术语表
- **Backdoor Attack**：通过在训练数据中植入触发器并污染标签，使模型对带触发器样本输出指定目标类别的攻击方式。
- **Post-training Defense**：在获得可疑已训练模型后，仅利用少量良性样本进行的后门清除防御流程。
- **Fine-tuning（FT）**：使用少量良性数据对已有模型继续训练，以尝试降低或消除后门效应。
- **Sharpness-Aware Minimization（SAM）**：同时最小化损失值与其邻域内的最大损失，以寻找更平坦的优化极小点。
- **Adaptive Perturbation（$\mathbf{T}_w$）**：按各参数权重绝对值分配扰动预算，使大权重范数维度获得更强有效扰动。
- **TAC（Trigger Activated Change）**：衡量某神经元在良性/中毒样本上的激活差异，值越高表示与后门关联越强。
- **ASR**：Attack Success Rate，被植入触发器的样本被错误分类到目标标签的比例。
- **DER**：Defense Effectiveness Rating，综合衡量防御后 ASR 下降与 ACC 下降的指标，取值 0–1 越高越好。

## 可复现要素
- 数据集：CIFAR-10、Tiny ImageNet、GTSRB；代码已开源：https://github.com/SCLBD/BackdoorBench。
- 关键超参：学习率 0.01，batch size 256；CIFAR-10 上 FT-SAM 的 $\rho=2$，Tiny ImageNet 与 GTSRB 上 $\rho=8$；训练轮次 CIFAR-10/Tiny ImageNet 为 100，GTSRB 为 50。
- 其他：提供 PyTorch 与 MindSpore 实现；消融与额外结果见补充材料。
