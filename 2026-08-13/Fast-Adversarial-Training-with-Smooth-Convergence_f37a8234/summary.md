---
title: "Fast-Adversarial-Training-with-Smooth-Convergence"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:07:20"
---

# 论文速读：Fast-Adversarial-Training-with-Smooth-Convergence

## 一句话总结
本文针对快速对抗训练（FAT）在大扰动预算（如 ξ=16/255）下易发生灾难性过拟合的问题，从损失函数收敛稳定性的视角切入，提出限制相邻epoch损失差异的振荡约束 ConvergeSmooth 及动态收敛步长，并辅以权重中心化技术，在显著提升训练稳定性的同时实现了接近 PGD-AT 的对抗鲁棒性与数倍的训练加速。

## 研究问题与动机
1. **核心问题**：现有 FAT 方法（FGSM-RS、GradAlign、FGSM-MEP 等）仅在较小扰动预算（ξ≤8/255）下能避免灾难性过拟合；当预算扩大至 ξ=16/255 时，模型对抗鲁棒性会骤降至接近零。
2. **现有方法不足**：随机初始化（FGSM-RS）提升扰动多样性但治标不治本；梯度/Logits 对齐正则（GradAlign、NuAT、FGSM-MEP）虽在小预算下有效，但 GradAlign 可能降低干净样本损失稳定性，且均无法稳定应对大预算场景。
3. **现象洞察**：训练曲线分析表明，灾难性过拟合总是伴随干净样本损失轻微波动与对抗样本损失急剧下降的“收敛异常”；而训练过程中的振荡阶段有时能触发 FAT 重启并恢复稳定。
4. **动机**：若强制损失收敛过程保持适度平滑，即可规避异常波动导致的过拟合，从而在大扰动预算下实现稳定高效的单步对抗训练。

## 核心贡献（创新点）
1. **提出 ConvergeSmooth 平滑收敛约束**：直接限制相邻 epoch 间对抗损失与干净损失的变化幅度以稳定训练动态；与已有工作显式约束梯度方向或 logit 输出的本质区别在于，本文从训练历程的跨 epoch 波动视角干预优化路径。
2. **设计动态收敛步长 γ_t**：基于损失差值非线性衰减的特性，将异常检测阈值设为随训练自动缩放的变量；区别于固定阈值或静态正则化，该方法在训练初期允许较快收敛、后期自动收紧以防过拟合。
3. **提供实例级与批量级两种实现变体**：Example-based 针对异常样本逐点施加约束，Batch-based 对 batch 内平均损失施加约束，在计算开销与约束强度之间提供灵活权衡。
4. **引入权重中心化（Weight Centralization）**：仅通过单一平衡系数 w_3 将当前权重约束为历史权重均值；无需额外调参即可抑制参数跳变，区别于传统优化器改进或数据增强策略。

## 方法详解
- **目标函数**：在标准 FAT min-max 框架基础上增加互补约束项：$\min_{\theta_t} \mathbb{E}[\mathcal{L}(x'_t, \theta_t) + \mathcal{L}_{CS}(t)]$。
- **ConvergeSmooth 损失**：$\mathcal{L}_{CS}(t) = w_1 \cdot |\mathcal{L}(x'_t, \theta_t) - u'_{t-1}| + w_2 \cdot |\mathcal{L}(x, \theta_t) - u_{t-1}|$，其中 $u'_{t-1}$ 与 $u_{t-1}$ 分别为上一 epoch 对抗与干净样本的平均损失期望。仅当条件 $\mathcal{C}(x): |\mathcal{L}(x, \theta_t) - u_{t-1}| \geq \gamma_t$ 成立时才施加约束，避免对正常样本造成不必要的正则干扰。
- **动态收敛步长**：$\gamma_t = \min(\max(d_{t-1}, \gamma_{min}), \gamma_{max})$，其中 $d_{t-1} = |u_{t-1} - u_{t-2}|$。该设计顺应损失非线性衰减规律，防止 γ 过大导致过拟合或 γ 过小阻碍收敛。
- **两种实现形式**：
  - Example-based：$\mathcal{L}_{CS}^E(t) = w_1 |\mathcal{L}(x'_t, \theta_t) - u'_{t-1}| + w_2 |\mathcal{L}(x, \theta_t) - u_{t-1}|$，条件为 $|\mathcal{L}(x, \theta_t) - u_{t-1}| > \gamma_t$。
  - Batch-based：$\mathcal{L}_{CS}^B(t) = w_1 |u'_B - u'_{t-1}| + w_2 |u_B - u_{t-1}|$，条件为 $|u_B - u_{t-1}| > \gamma_t$，仅用 batch 均值替代，内存开销更低。
- **权重中心化**：$\mathcal{L}_{CS}^W(t) = w_3 \cdot ||\theta_t - \frac{1}{len(\phi)}\sum_{j \in \phi}\theta_j||_2$，将当前权重向历史权重均值靠拢，仅引入额外系数 $w_3$。

## 实验与结果
- **数据集与骨干**：CIFAR-10、CIFAR-100、Tiny ImageNet；ResNet18、PreActResNet18、WideResNet34-10。
- **评估基线**：FGSM-RS、GradAlign、ZeroGrad、NuAT、ATAS、FGSM-MEP 及多步 PGD-AT；攻击评估采用 FGSM、PGD-10/20/50、C&W、APGD-CE、Autoattack（AA）。
- **主要结果（CIFAR-10, ResNet18, ξ=16/255）**：B-MEP 的 AA 鲁棒性达 **23.68%**，显著优于 FGSM-MEP（18.98%）、GradAlign（17.02%）、NuAT（18.43%）等，仅比 PGD-AT（26.29%）低 0.97%；训练时间仅 **102 分钟**，约为 PGD-AT（370 分钟）的 1/3.6。B-RS 以 75 分钟取得 AA=19.43%，效率高于 GradAlign（135分钟）。
- **跨预算与跨数据集**：在 ξ=10/255 与 12/255 下均取得最优或极具竞争力的 AA 分数；CIFAR-100 上 B-MEP 达 AA=11.38%，Tiny ImageNet 上 B-BP 在 15.4 小时内接近 PGD-AT（67.2 小时）的鲁棒性水平。
- **关键结论**：所提方法成功在大扰动预算下彻底避免灾难性过拟合，三重复验的 mbest 与 mfinal 差距极小，证明训练高度稳定，且在保持相近 clean accuracy 的前提下大幅提升对抗鲁棒性。

## 相关工作脉络
1. **FGSM-RS [40]**：最早通过均匀随机初始化扰动缓解 FAT 过拟合；本文将其作为基础框架并植入 batch 级损失平滑约束，将优化重心从“扰动多样性”转向“训练动态稳定性”。
2. **GradAlign [2]**：通过最大化干净/对抗样本梯度余弦相似度稳定训练；本文指出其可能破坏干净样本损失稳定性，转而采用损失值跨 epoch 平滑约束避免梯度对齐的副作用。
3. **FGSM-MEP [42]**：利用历史对抗扰动动量生成初始化；本文 B-MEP 在其之上叠加 ConvergenceSmooth，将 AA 从 18.98% 提升至 23.68%，证明平滑约束可与先进初始化策略正交叠加。
4. **NuAT [36] / ATAS [16]**：分别通过 logits 核范数正则与自适应步长优化 FAT；本文强调二者仍局限于小 ξ，且缺乏对损失收敛过程的直接干预，大预算下易崩盘。
5. **PGD-AT [25]**：多步对抗训练的性能上限基准；本文证明 FAT+ConvergeSmooth 可在鲁棒性差距不足 1% 的条件下实现 3 倍以上提速，为高效安全部署提供新路径。
6. **OAAT [1] / AWP [41]**：面向大扰动预算的 AT 方法；本文将其移植至 FAT 场景对比，验证了平滑收敛策略在快速单步训练范式中的通用性与可移植性。

## 局限性与未来方向
- **超参数敏感**：ConvergeSmooth 需调节 $w_1, w_2, \gamma_{max}$ 等系数，不同数据集/架构下仍需人工微调（如 Tiny ImageNet 需 $w_1>0$ 防过拟合，CIFAR-100 最优为 $w_1=0$）。
- **单层平滑局限**：权重中心化
