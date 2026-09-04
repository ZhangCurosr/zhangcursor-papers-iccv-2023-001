---
title: "Guided-Motion-Diffusion-for-Controllable-Human-Motion-Synthe"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:15:54"
---

# 论文速读：Guided-Motion-Diffusion-for-Controllable-Human-Motion-Synthe

## 一句话总结
本文提出 Guided Motion Diffusion (GMD)，通过强调投影（Emphasis Projection）与密集信号传播（Dense Signal Propagation）两项技术，将轨迹、关键帧位置及避障等稀疏空间约束可靠地注入文本驱动的扩散运动生成过程，在保持 SOTA 文本生成质量的同时实现高质量的可控空间合成。

## 研究问题与动机
1. **现有扩散运动模型难以整合空间约束：** MDM 等主流文本到运动模型虽在语义生成上领先，但面对全局轨迹、障碍物等空间条件时，常出现忽略引导或动作失配（如脚底滑行）的问题。
2. **运动表征的空间信息极度稀疏：** 标准 HumanML3D 表征中，局部姿态占 259 维，而描述骨盆位移与旋转的全局空间信息仅 4 维。这种维度失衡使模型在去噪时易将受约束的全局信号误判为噪声并过滤掉。
3. **帧级引导信号高度稀疏导致被先验淹没：** 实际应用中约束常仅给定少量关键帧，在扩散逆向过程中，稀疏的观测帧信噪比极低，极易被周围无约束帧的数据分布先验覆盖，导致关键帧目标被忽略或产生剧烈形变伪影。

## 核心贡献（创新点）
1. **强调投影（Emphasis Projection）：** 通过随机投影矩阵 $A=A'B$ 放大运动向量中全局/轨迹维度的相对权重，并在投影空间内重新推导 imputation 公式，使局部姿态与全局轨迹在去噪时保持时空相干。与单纯提高轨迹损失权重的本质区别在于，该方法在表征空间重塑梯度流向，而非依赖训练阶段的重调超参。
2. **密集信号传播（Dense Signal Propagation）：** 借鉴强化学习信用分配思想，直接利用预训练运动去噪器自身作为上下文平滑算子，通过自动微分将稀疏关键帧引导信号沿时间轴传播至相邻帧，使微弱约束在逆向过程中不被忽略。与显式插值或引入额外传播网络的本质区别在于，传播过程严格遵循数据本身的动力学分布先验。
3. **GMD 两阶段可控生成框架与 $\epsilon$-modeling 优化：** 将上述两项技术集成至 UNet 架构，以可微目标函数 $G_z$ 统一支持轨迹、关键帧、避障等多种空间条件；同时论证了在 classifier guidance 场景下采用 $\epsilon$-modeling 替代 $\mathbf{x}_0$-modeling 可有效抑制 DPM 分布偏差对晚期引导信号的覆盖。

## 方法详解
- **形式化目标：** 对满足标量目标函数 $G_z(\mathbf{z})$ 的运动序列建模 $p(\mathbf{x}|G_z=0)$，其中 $\mathbf{z} \in \mathbb{R}^{L \times M}$ 为 $\mathbf{x}$ 中的轨迹子部分（$L=2$ 或含旋转）。
- **强调投影与投影空间 Imputation：** 构造对角放大阵 $B$（轨迹相关对角元设为 $c$，其余为 1），投影后 $\mathbf{x}^{proj} = \frac{1}{N-3+3c^2}A\mathbf{x}$。扩散模型在 $\mathbf{x}^{proj}$ 空间运行；每步去噪得到 $\mathbf{x}_{0,\theta}^{proj}$ 后，先反投影回原空间，按掩码 $M_z^x$ 填入目标轨迹 $P_z^x \mathbf{z}^*$，再重新投影，使模型信念始终偏向空间一致解（公式 6）。
- **密集信号传播与掩码分类器引导：** 分类器引导梯度近似为 $-\nabla_{\mathbf{x}_t} G_z(P_x^z \mathbf{x}_{0,\theta}(\mathbf{x}_t))$，即通过对去噪器求导将关键帧处的 $G_z$ 梯度扩散至整条轨迹。结合 imputation 掩码，仅在被观测帧使用直接填充，其余帧通过梯度偏移采样均值（公式 8
