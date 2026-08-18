---
title: "HollowNeRF-Pruning-Hashgrid-Based-NeRFs-with-Trainable-Colli"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_HollowNeRF_Pruning_Hashgrid-Based_NeRFs_with_Trainable_Collision_Mitigation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:09:33"
field: "神经辐射场与 3D 表示"
keywords: ["NeRF", "模型压缩", "哈希网格", "特征剪枝", "3D 显著性", "ADMM", "渲染质量"]
innovations: ["提出可训练 3D 显著性网格引导特征剪枝以缓解哈希碰撞", "引入 ADMM 剪枝器实现特征稀疏化的自适应约束优化", "设计软零跳过门控确保剪枝后特征能正确映射到零密度"]
benchmarks: ["NeRF Synthetic Dataset", "Tanks and Temples Dataset"]
---

# 论文速读：HollowNeRF: Pruning Hashgrid-Based NeRFs with Trainable Collision Mitigation

## 一句话总结
本文提出 **HollowNeRF**，一种针对哈希网格（Hashgrid）加速 NeRF 的新型压缩方法，通过训练一个可学习的 3D 显著性网格来自动识别并剪枝不可见/空洞区域的特征，从而在显著减少参数量的同时缓解哈希碰撞，实现优于 Instant-NGP 的渲染质量。

## 研究问题与动机
- **核心问题**：现有哈希网格 NeRF（如 Instant-NGP）虽然训练/渲染速度快，但轻量级哈希网格在细分辨率下会产生严重的哈希碰撞（Hash Collisions），导致特征失真，影响渲染精度；同时，三维场景具有天然的空间稀疏性（大量区域为空或被遮挡），现有方法未能有效利用这一特性来压缩模型。
- **现有方法不足**：
  1.  **Instant-NGP 等基线**：虽然使用哈希编码加速，但碰撞均匀分布在非零密度体素上，是次优的；且保留了大量内部遮挡区域的不必要特征。
  2.  **依赖几何先验的方法**：如 DONeRF 等通过辅助网络预测深度分布来指导采样，需要粗略的几何估计，精度不足时会降低渲染质量，精确估计又增加复杂度。
  3.  **周期性更新掩码的方法**：部分方法在训练中周期性评估 NeRF 模型来更新占据掩码，开销大且浪费计算资源。

## 核心贡献（创新点）
- **提出可训练 3D 显著性网格引导特征剪枝**：与使用周期性模型评估或几何先验的方法不同，HollowNeRF 通过训练一个低分辨率的 3D 显著性网格来学习场景的稀疏分布，自动识别并抑制不可见/内部区域的特征。
- **引入 ADMM 剪枝器进行显式稀疏化约束**：区别于简单的 L1 正则化（需手动调整超参数 λ 且可能导致可见特征被过度剪枝），采用交替方向乘子法（ADMM）将稀疏约束转化为可学习的拉格朗日乘子，实现对显著性网格更精确、更稳定的剪枝。
- **设计软零跳过门控（Soft Zero-Skipping Gate）确保剪枝后密度为零**：针对 MLP 解码器可能破坏稀疏性（即使输入特征为 0，输出密度可能非 0）的问题，引入一个基于 tanh 函数的可微门控机制，确保被剪枝的特征（v=0）能正确映射到零密度，并配合训练调度增强门控效果。

## 方法详解
HollowNeRF 基于 Instant-NGP 的哈希网格流水线，主要包含三个组件：

1.  **可训练 3D 显著性网格**：
    - 构建一个分辨率较低（$T \times T \times T$，如 $64^3$）的可训练 3D 张量 $\mathcal{G}$，与哈希特征网格 $\mathcal{H}$ 和 MLP 权重 $\mathcal{W}$ 联合训练。
    - 对于输入 3D 坐标 $\mathbf{x}$，通过三线性插值从 $\mathcal{G}$ 中获取周围 8 个格点的显著性值 $g_i$，经 sigmoid 激活得到权重 $p \in (0, 1)$，代表该位置特征“非零”的概率。
    - 最终加权特征向量 $\mathbf{v} = p \mathbf{f}$ 送入 MLP 解码，其中 $\mathbf{f}$ 是从哈希网格中查询到的特征。该设计使得模型能“学习”到场景的“空心”结构，将哈希碰撞概率重新分配到重要特征上。

2.  **软零跳过门控**：
    - 为了解决 MLP 解码器破坏稀疏性的问题，在 MLP 输出前引入门控函数：$\hat{\mathcal{M}}(\mathbf{v}) = \hat{g}(\mathbf{v}) \cdot \mathcal{M}(\mathbf{v})$，其中 $\hat{g}(\mathbf{v}) = \tanh(\alpha \|\mathbf{v}\|_2)$。
    - 该软门控是可微的近似版本，当输入特征范数接近 0 时，平滑地将密度输出压制到 0；使用 schedule 策略（前期 $\alpha=10^4$ 加速收敛，后期 $\alpha=10^5$ 强化门控）避免硬门控带来的梯度消失问题。

3.  **ADMM 剪枝器**：
    - 将稀疏化问题转化为带约束的优化问题：最小化渲染损失 $L$，同时约束显著性网格的 L1 范数（即非零显著性值的总和）小于阈值 $C$。
    - 通过增广拉格朗日方法，将约束转化为无约束的 minimax 问题，引入可学习的对偶变量 $\gamma$（替代固定的正则化系数 $\lambda$）：
      $$\min_{\mathcal{W},\mathcal{H},\mathcal{G}} \max_{\gamma \geq 0} L(\mathcal{W},\mathcal{H},\mathcal{G}) + \frac{\rho_\gamma}{2}[\|S_{sig}(\mathcal{G})\|_1 - C]_+^2 + \gamma(\|S_{sig}(\mathcal{G})\|_1 - C)$$
    - 在每步迭代中，先使用梯度下降更新 $\mathcal{G}, \mathcal{H}, \mathcal{W}$，再用梯度上升更新对偶变量 $\gamma$，从而自适应地控制稀疏程度，将不必要特征的显著性值精确推至 0。

## 实验与结果
- **数据集**：NeRF Synthetic Dataset（合成数据集）、Tanks and Temples Dataset（真实场景数据集）。
- **评估基线**：Instant-NGP、CC-NeRF、TensoRF 等近期压缩/加速方法。
- **评估指标**：PSNR（峰值信噪比）、LPIPS（感知图像块相似度）。
- **主要结果**：
  - 在最具代表性的成本-精度权衡实验中（NeRF Synthetic 的 "Chair" 场景），HollowNeRF（$T=64$，哈希网格大小 $2^{19}$）仅用 **6.48M 参数** 达到了 **34.85 dB PSNR**，而参数量更大的 Instant-NGP（11.49M 参数）仅达到 33.86 dB。**HollowNeRF 以 56% 的参数量获得了约 1 dB 的 PSNR 提升**。
  - 在多个场景的横向对比中，HollowNeRF 使用最小哈希网格（$2^{17}$，3.34M 参数）在 7/10 的场景中即超越了参数量最大（11.49M）的 Instant-NGP。
  - 平均 PSNR 对比显示，HollowNeRF 在相同或更小参数量下，在所有测试方法中实现了**最佳的“成本-精度”权衡**。
  - 消融实验证实，显著性网格、零跳过门控、ADMM 剪枝器各组件均对最终性能有正向贡献，且组合后效果最优。

## 相关工作脉络
- **Instant-NGP**：本文的基础架构，使用多分辨率哈希网格和轻量 MLP 实现快速 NeRF 训练与渲染。HollowNeRF 在其基础上引入了可学习的显著性网格和剪枝机制来缓解哈希碰撞和提升压缩效率。
- **DONeRF**：通过深度预言网络（Depth Oracle Networks）预测表面几何以指导稀疏采样。HollowNeRF 不依赖外部几何先验或额外的深度预测网络，而是通过训练显著的 3D 网格直接从特征层面学习空间稀疏性。
- **CC-NeRF / TensoRF**：采用张量分解（低秩近似）进行模型压缩。HollowNeRF 的压缩依据是特征的“实际渲染影响力”（通过显著性学习），而非特征的原始熵或秩，两者压缩原理不同。
- **基于采样的加速方法**（如 [18], [12]）：聚焦于仅在感兴趣表面附近采样以加速渲染，但通常需要准确的表面几何先验。HollowNeRF 专注于在特征存储层面进行压缩，与这些采样策略正交，可结合使用。
- **Plenoxels / DVG**：无 MLP 的纯体素/网格方法，渲染速度快但参数量大。HollowNeRF 保持在哈希网格+MLP 的范式内，通过剪枝在保持轻量 MLP 优势的同时降低参数量。

## 局限性与未来方向
- **场景稀疏性假设**：HollowNeRF 的压缩增益依赖于场景具有空间稀疏性（大部分为空或被遮挡）。对于烟雾、火焰、云朵等不透明或半透明体积场景，性能会退化至基线水平。
- **反射表面建模**：与 Instant-NGP 类似，当前版本在处理具有强反射表面的物体时存在困难。
- **未来方向**：计划将 HollowNeRF 的理念扩展至支持更好反射表面建模的新框架（如 Ref-NeRF 等）。

## 研究启发与可借鉴点
- **可迁移的“学习稀疏性”范式**：提出通过训练一个低分辨率的 3D 网格来隐式学习场景的显著性/占据分布，而非依赖硬阈值或外部先验，这一思路可迁移到其他基于体素或网格的 3D 表示学习任务中。
- **ADMM 用于神经网络剪枝**：将 ADMM 优化框架应用于特征网格的稀疏化约束，通过可学习的对偶变量自适应控制剪枝强度，避免了手动调参的困难，为模型压缩中的约束优化提供了新思路。
- **门控机制保稀疏性**：在解码阶段引入可微的软门控，确保经过剪枝的稀疏特征能够正确地映射回零输出，这一技巧可用于保证其他稀疏化架构（如稀疏卷积、稀疏 Transformer）的功能完整性。
- **实验设计借鉴**：详细的成本-精度权衡分析（不同哈希网格大小、显著性网格分辨率、稀疏约束阈值）以及全面的消融实验，为评估压缩类方法提供了规范的实验框架。

## 关键术语表
- **Hashgrid（哈希网格）**：Instant-NGP 中使用的一种高效特征编码结构，通过哈希函数将高分辨率体素坐标映射到较小的特征表，以加速查询和训练。
- **Hash Collision（哈希碰撞）**：当两个不同的 3D 坐标被哈希函数映射到同一个桶（bucket）时发生，导致其特征相互干扰，是哈希编码精度的主要瓶颈。
- **Saliency Grid（显著性网格）**：HollowNeRF 提出的一个低分辨率可训练 3D 张量，用于学习每个空间区域特征的“重要性”或“可见性”概率。
- **ADMM（Alternating Direction Method of Multipliers）**：一种用于求解带约束优化问题的迭代算法，通过引入对偶变量将约束问题转化为无约束问题，常于分布式优化和压缩感知。
- **Zero-Skipping Gate（零跳过门控）**：HollowNeRF 中引入的门控函数，确保当输入特征向量为零时，MLP 的输出密度也为零，从而维持稀疏性。
- **PSNR / LPIPS**：PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）衡量图像重建的像素级误差；LPIPS（Learned Perceptual Image Patch Similarity）基于深度学习特征衡量图像的感知相似度。

## 可复现要素
- **数据集**：NeRF Synthetic Dataset [15]、Tanks and Temples Dataset [11]。NeRF Synthetic 数据集公开可用；Tanks and Temples 数据集公开可用。
- **代码/权重**：论文基于 `torch-ngp` [25] 实现，但该开源仓库是 Instant-NGP 的 PyTorch 复现，**论文未明确声明提供 HollowNeRF 的官方开源代码或预训练权重**。
- **关键超参**：显著性网格分辨率 $T=64$，稀疏约束 $C=0.04$，哈希网格级别数 16，特征维度 2，MLP 层数 2 隐藏层，每层宽度 64，训练步数 300,000，优化器 Adam，初始学习率 $1 \times 10^{-2}$，门控参数 $\alpha$ 调度值（前期 $10^4$，后期 $10^5$）。
