---
title: "Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Enhancing_Fine-Tuning_Based_Backdoor_Defense_with_Sharpness-Aware_Minimization_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:04:34"
field: "深度神经网络后门防御"
keywords: ["backdoor defense", "fine-tuning", "sharpness-aware minimization", "neuron weight norm", "post-training security"]
innovations: ["基于神经元权重范数自适应扰动的 FT-SAM 后门消除方法", "提出 DER 指标统一评估 ACC 与 ASR 权衡", "揭示 vanilla FT 对后门神经元扰动不足的本质原因"]
benchmarks: ["CIFAR-10", "Tiny ImageNet", "GTSRB"]
---

# 论文速读：Enhancing-Fine-Tuning-based-Backdoor-Defense-with-Sharpness

## 一句话总结
本文提出了一种基于**尖锐感知最小化（SAM）**的微调后门防御方法 **FT-SAM**，通过让后门相关的高权重神经元受到更强的梯度扰动，成功消除预训练模型中的后门效应，同时在 CIFAR-10、Tiny ImageNet 等多个基准上达到 SOTA 防御性能。

## 研究问题与动机
- **核心问题**：仅利用少量（如 5%）良性数据对预训练后门模型进行微调（FT），难以有效抵御强后门攻击（如 Blended、LF）。
- **FT 失效原因**：后门模型在预训练阶段已经很好地拟合了良性样本，导致普通微调仅对神经元权重产生微小扰动，无法改变后门相关的神经元行为。
- **观察依据**：通过 TAC（Trigger Activated Change）指标发现，后门相关神经元的权重范数显著大于正常神经元；而 FT 过程中所有神经元受到的扰动幅度相近，未能针对后门神经元进行修正。

## 核心贡献（创新点）
1. **提出 FT-SAM 防御范式**：将 SAM 的最小-最大优化引入后门微调过程，通过自适应扰动预算主动压缩后门相关大权重神经元的范数。
2. **揭示 FT 失效机制**：从神经元权重范数与梯度范数视角，实证分析了 vanilla FT 无法消除后门的根本原因。
3. **SOTA 防御性能**：在 10 种后门攻击、2 种网络架构、3 个数据集上的实验表明，FT-SAM 平均 DER（防御有效性评分）显著优于现有微调或剪枝类基线。
4. **可插拔增强模块**：FT-SAM 可与 FP、ANP 等后处理防御方法结合，进一步提升整体防御鲁棒性并保持模型良性准确率。

## 方法详解
- **基本设定**： defender 仅拥有后门模型 $f_w$ 与少量良性数据集 $\mathcal{D}_{benign}$，目标是在维持良性 ACC 的同时将 ASR 降至接近 0。
- **核心思想**：后门神经元通常具有较大权重范数 $\|w_i\|$，因此设计自适应扰动矩阵 $\mathbf{T}_w = \mathrm{diag}(|w_1|, \ldots, |w_d|)$，使得扰动预算随神经元权重范数放大，强迫 optimizer 对“可疑”神经元施加更大变化。
- **优化目标**：
  $$\min_{w} \max_{\epsilon \in S} \mathcal{L}(w + \epsilon), \quad S = \{\epsilon : \|\mathbf{T}_w^{-1}\epsilon\|_2 \leq \rho\}$$
- **内层最大化（扰动估计）**：通过一阶泰勒展开得到显式解
  $$\epsilon_{t+1} = \rho \frac{\mathbf{T}_{w_t}^2 \nabla_{w_t} \mathcal{L}(w_t)}{\|\mathbf{T}_{w_t} \nabla_{w_t} \mathcal{L}(w_t)\|_2}$$
- **外层最小化（参数更新）**：沿扰动后位置的梯度下降
  $$w_{t+1} = w_t - \eta \nabla_w \mathcal{L}(w + \epsilon_{t+1})|_{w=w_t}$$
- **关键超参**：扰动半径 $\rho$（CIFAR-10 设为 2，Tiny ImageNet / GTSRB 设为 8），学习率 0.01，batch size 256，FT-SAM 训练 100 epoch（GTSRB 50 epoch）。
- **评估指标**：除 ACC 与 ASR 外，引入 **DER = [$\max(0, \Delta ASR) - \max(0, \Delta ACC) + 1$]/2** 综合衡量防御收益。

## 实验与结果
- **数据集与模型**：CIFAR-10、Tiny ImageNet、GTSRB；PreAct-ResNet18、VGG19-BN。
- **攻击覆盖**：BadNets-A2O/A2A、Blended、Input-aware、CLA、LF、SIG、SSBA、Trojan、WaNet，毒化率默认 10%（部分 5% 对比实验）。
- **基线对比**：FT、FP [32]、NAD [27]、NC [46]、ANP [51]、ABL [26]、i-BAU [54]、AC [5]。
- **CIFAR-10 平均结果（5% 良性数据）**：
  - FT-SAM：ACC 92.10%，ASR 2.47%，DER 96.62%（全部攻击中最优）。
  - 次优方法 i-BAU：ACC 88.19%，ASR 5.73%，DER 93.47%。
  - 在复杂攻击 Blended、LF、SSBA 上，FT-SAM 仍将 ASR 压至 < 5%，而 vanilla FT 仍残留 35%–93%。
- **Tiny ImageNet 平均结果**：FT-SAM 以 ACC 52.18%、ASR 1.16%、DER 92.16% 显著领先，其他方法在复杂攻击下 ACC 普遍暴跌至 40%-50%。
- **消融与泛化**：$\rho$ 在 [1, 8] 范围内均保持高 DER，证明自适应缩放对超参鲁棒；在不同良性比例（10%/5%/1%）下 FT-SAM 仍稳定输出高 DER。
- **组合增强**：FT-SAM 接入 FP、ANP 的二次微调步骤后，DER 平均提升约 3-5 个百分点，验证其可插拔性。

## 相关工作脉络
- **Vanilla Fine-tuning (FT)**：利用少量良性数据微调后处理模型，计算开销小但不针对后门神经元设计。
- **Neural Attention Distillation (NAD) [27]**：采用师生蒸馏式微调，与 FT 在实验表中表现相近，仍未放大后门神经元扰动。
- **Fine-pruning (FP) [32]**：先剪枝再微调，依赖激活路径差异假设，对 Blended/LF 类复杂触发器容易失效。
- **Adversarial Neuron Pruning (ANP) [51]**：基于极小极大搜索敏感神经元，但与 FT-SAM 相比未显式控制权重范数分布。
- **i-BAU [54]**：基于隐式超梯度的对抗性遗忘方法，ASR 控制较好但会牺牲较多 ACC（DER 相对较低）。
- **定位差异**：FT-SAM 不改变网络结构、不依赖触发器恢复，仅通过调整优化目标使 SAM 对大权重神经元施加更强扰动，兼具低开销与高通用性。

## 局限性与未来方向
- 当前分析主要在卷积网络（PreAct-ResNet18、VGG19-BN）上进行，未涉及 Transformer 架构。
- 依赖 $\rho$ 经验设置，缺乏自动选择策略（虽已证明鲁棒，但未给出理论界）。
- 仅针对 post-training 场景；对 training-stage 污染防御与 FT-SAM 结合未有探讨。
- 未在跨域或大规模真实部署场景（如自动驾驶、医疗图像）验证。

## 研究启发与可借鉴点
- **基于权重范数的自适应扰动设计**：可将 $\mathbf{T}_w$ 结构迁移至其他需要“针对性弱化特定参数”的安全任务（如模型窃取防御、联邦学习投毒缓解）。
- **DER 综合指标的设计思路**：兼顾 ACC 下降与 ASR 下降的单向权衡评估方式，值得在安全评测中推广使用。
- **解释性分析链条**：从 T-SNE、神经元范数散点图到 Grad-CAM 的可视化联动，为后门机制解读提供了可复用的多视角分析模板。
- **即插即用升级方案**：FT-SAM 作为 fine-tuning 组件可无缝替代现有方法的微调阶段，启发后续工作可将 SAM 思想嵌入更多后处理流程。

## 关键术语表
- **Backdoor Attack / 后门攻击**：攻击者在训练数据中植入隐蔽触发器，使模型对含触发器输入异常输出目标类别。
- **Fine-tuning Defense / 微调防御**：利用少量良性数据对可疑模型进行微调，期望削弱触发器关联而保持正常性能。
- **Sharpness-Aware Minimization (SAM)**：同时最小化损失值与损失地形陡峭度的优化方法，鼓励模型落入平坦极小值。
- **Adaptive SAM (ASAM)**：基于参数尺度不变的 SAM 变体，通过逐参数自适应缩放扰动预算。
- **TAC (Trigger Activated Change)**：衡量某神经元对触发器敏感性的指标，值越大表示该神经元与后门关联越强。
- **ASR (Attack Success Rate)**：后门样本被错误分类至目标类的比例，衡量后门存活强度。
- **DER (Defense Effectiveness Rating)**：综合 ASC 与 ASR 变化的防御有效性评分，范围 [0, 1]。
- **Post-training Defense / 后训练防御**：仅在获得可疑模型和少量良性样本的前提下进行后门清除，不对训练过程施加约束。

## 可复现要素
- **数据集**：CIFAR-10、Tiny ImageNet、GTSRB，均公开可用。
- **代码**：作者已在 GitHub 开源（https://github.com/SCLBD/BackdoorBench），提供 PyTorch 与 MindSpore 实现。
- **关键超参**：学习率 0.01、batch size 256、epoch=100（GTSRB 50）、扰动半径 $\rho=2$（CIFAR-10）/ $\rho=8$（Tiny ImageNet、GTSRB）。
- **攻击配置**：遵循 BackdoorBench 默认设置，毒化率 10%（SIG/CLA 因单类限制未达 10%，另设 5% 对比）。
