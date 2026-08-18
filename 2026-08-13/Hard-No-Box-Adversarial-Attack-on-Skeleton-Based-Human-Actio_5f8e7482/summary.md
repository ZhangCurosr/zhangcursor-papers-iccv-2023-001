---
title: "Hard-No-Box-Adversarial-Attack-on-Skeleton-Based-Human-Actio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Hard_No-Box_Adversarial_Attack_on_Skeleton-Based_Human_Action_Recognition_with_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:09:28"
---

# 论文速读：Hard-No-Box-Adversarial-Attack-on-Skeleton-Based-Human-Actio

## 一句话总结
本文首次提出了面向基于骨架的人体动作识别（S-HAR）的 **Hard No-Box 对抗攻击**，在攻击者仅能访问无标签测试数据、完全无法获取目标模型/训练数据/标签的极端受限条件下，通过对比学习构建无标签运动流形，并引入显式建模时序动力学的骨架运动感知梯度（SMI Gradient），成功生成高欺骗率且视觉自然的对抗样本。

## 研究问题与动机
1. **现有攻击先验要求过强**：白盒攻击需完整模型参数，查询式黑盒攻击需海量模型调用，迁移攻击需访问训练数据与标签，均难以匹配真实安全场景中攻击者信息受限的假设。
2. **缺乏严苛设定下的脆弱性验证**：此前 S-HAR 领域未有人在 Hard No-Box（零模型/零数据/零标签/零查询）设定下验证模型的真实威胁程度，防御评估存在盲区。
3. **传统梯度忽略时序依赖**：现有基于梯度的攻击方法在反向传播时假设各帧/各维度独立，导致生成的对抗扰动易偏离真实运动流形，产生违反物理规律的人体抖动，迁移性与隐蔽性受限。
4. **无标签流形引导攻击的挑战**：在缺乏类别边界与标签反馈的条件下，如何从潜空间中估计类边界方向、并引导对抗样本沿流形演化，尚未有有效解法。

## 核心贡献（创新点）
1. **提出首个 Hard No-Box 攻击 Pipeline**：实现了无需目标模型、训练数据及标签的对抗样本生成，将攻击先验需求降至理论最低；与仅需去模型但保留标签的 Soft No-Box 攻击相比，进一步消除了标签依赖，威胁评估更贴近现实。
2. **设计骨架运动感知梯度（SMI Gradient）**：首次将时序动力学（速度/加速度）显式融入对抗梯度计算，通过 Markovian 与自回归模型刻画帧间依赖；与假设各帧独立的原始梯度相比，SMI 梯度能约束扰动沿运动流形演化，避免产生离群抖动。
3. **构建新一代无标签攻击算法族**：将 SMI 梯度无缝集成至 I-FGSM 与 MI-FGSM，衍生出 S₁I-FGSM、S₂I-FGSM、S₁MI-FGSM、S₂MI-FGSM 四种新策略；在 No-Box 与迁移黑盒设置下，同步提升欺骗率与感知隐蔽性。

## 方法详解
- **对比学习构建无标签运动流形**：采用基于自适应图卷积网络（AGCN）的 Encoder，对输入骨架序列施加空间（姿态变换、关节抖动）与时间（时间裁剪、缩放）增强，得到正样本对 $(S_q, S_k)$。通过 InfoNCE 损失在潜空间中聚合相似运动、离散不同运动，训练完成后保留 Query Encoder $f_q$ 作为流形描述器，全程无需类别标签。
- **无标签对抗损失**：在 $f_q$ 提取的潜空间中定义 $L_{adv} = -\log \frac{\exp[\text{Sim}(f_q(s), f_q(\tilde{s}))]}{\sum_j \exp[\text{Sim}(f_q(s), f_q(\tilde{s}_j))]}$，其中 $\tilde{s}$ 为参考正样本，$\tilde{s}_j$ 为负样本。通过 K-means 聚类选取负样本（丢弃距离最近的 $Q$ 个簇中心以避免误导），最大化该损失即驱使对抗样本 $s$ 远离正样本并靠近负样本高密度区域。
- **SMI 梯度推导**：原始
