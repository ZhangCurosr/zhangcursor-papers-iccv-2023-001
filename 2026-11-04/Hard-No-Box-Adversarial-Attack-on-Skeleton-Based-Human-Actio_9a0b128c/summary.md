---
title: "Hard-No-Box-Adversarial-Attack-on-Skeleton-Based-Human-Actio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Hard_No-Box_Adversarial_Attack_on_Skeleton-Based_Human_Action_Recognition_with_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:43:27"
---

# 论文速读：Hard-No-Box-Adversarial-Attack-on-Skeleton-Based-Human-Actio

## 一句话总结
本文首次提出面向骨架动作识别（S-HAR）的 **Hard No-Box 攻击**，攻击者仅需无标签测试数据即可生成有效对抗样本，完全无需 victim 模型、训练集或标签；通过引入基于时序动力学的 **SMI（Skeleton-Motion-Informed）梯度**，使扰动沿人体运动流形移动，显著提升攻击致错率、跨模型迁移性与样本不可感知性。

## 研究问题与动机
1. **现实威胁建模过于理想**：现有 S-HAR 对抗攻击依赖白盒参数、高频查询或完整训练数据+标签，在实际安防/人机交互场景中难以满足，攻击可行性存疑。
2. **Hard No-Box 设定下的攻击空白**：当攻击权限降至最低（仅能访问无标签测试集）时，如何绕过类别边界估计并构造有效扰动尚未有可行方案。
3. **传统梯度忽视时序流形约束**：现有梯度攻击假设各帧/各关节维度相互独立，导致对抗扰动易偏离真实运动流形，产生不自然抖动或需极大扰动预算才能生效。
4. **漏洞严重性亟需实证**：需在信息获取最受限的条件下重新验证 S-HAR 模型的脆弱性，为后续鲁棒防御研究提供底线威胁基准。

## 核心贡献（创新点）
1. **首个 Hard No-Box 攻击流水线**：在不接触 victim 模型、训练数据及任何标签的极端条件下实现对抗样本生成。*与已有工作的本质区别*：打破迁移攻击依赖预训练代理模型与标签的惯例，证明最低权限下 S-HAR 仍面临实质威胁。
2. **提出 SMI（Skeleton-Motion-Informed）梯度**：将一阶 Markovian 与二阶自回归时序动力学显式融入损失梯度计算。*与已有工作的本质区别*：现有方法按独立维度求偏导，SMI 梯度通过链式法则捕获帧间依赖，强制扰动贴近运动流形局部切线方向。
3. **构建可复用的 SMI 梯度攻击家族**：将 SMI 梯度无缝替换 I-FGSM/MI-FGSM 中的原始梯度，导出四种新策略。*与已有工作的本质区别*：不仅适用于 hard no-box 设置，在迁移黑盒攻击中同样提升跨模型迁移成功率与视觉自然度，具有强兼容性。

## 方法详解
1. **对比学习构建无标签运动流形**：采用基于 AGCN 的 query/key 编码器对输入骨架序列 $S$ 施加空间（姿态变换、关节抖动）与时间（裁剪、重采样）增强，得到 $S_q, S_k$。通过 InfoNCE 损失训练，使相似动作特征在 latent space 聚合、相异特征分散，隐式形成无需标签的软边界。
2. **无标签对抗损失设计**：在 latent space 定义 $L_{adv} = -\log \frac{\exp[\text{Sim}(f_q(s), f_q(\tilde{s}))]}{\sum_j \exp[\text{Sim}(f_q(s), f_q(\tilde{s}_j))]}$，其中 $\tilde{s}$ 为正样本（干净样本），$\tilde{s}_j$ 为负样本。负样本通过无监督 K-means 选取（剔除距离最近的 $Q$ 个簇中心），最大化该损失推动对抗样本远离正样本、靠近负样本高密度区。
3. **SMI 梯度推导**：
   - 原始梯度 $\nabla J(S) = [\frac{\partial J}{\partial S_1}, \dots, \frac{\partial J}{\partial S_t}]$ 假设帧独立。引入动力学约束 $S_t = f(S_{<t})$，推导一阶与二阶修正项：
     - 一阶：$(\frac{\partial J}{\partial S_{t-1}})_{d1} = \frac{\partial J}{\partial S_{t-1}} + \frac{\partial J}{\partial S_t} \cdot A_t$
     - 二阶：$(\frac{\partial J}{\partial S_{t-2}})_{d2} = \frac{\partial J}{\partial S_{t-2}} + \frac{\partial J}{\partial S_{t
