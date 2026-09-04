---
title: "HollowNeRF-Pruning-Hashgrid-Based-NeRFs-with-Trainable-Colli"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_HollowNeRF_Pruning_Hashgrid-Based_NeRFs_with_Trainable_Collision_Mitigation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:16:01"
field: "神经辐射场高效表示与压缩"
keywords: ["NeRF压缩", "哈希网格", "ADMM剪枝", "显著性网格", "零跳过门", "哈希冲突缓解"]
innovations: ["可训练3D显著性网格自适应学习稀疏分布指导特征剪枝", "ADMM约束优化框架自动学习稀疏化程度", "软零跳过门确保剪枝特征正确映射为零密度"]
benchmarks: ["NeRF Synthetic", "Tanks and Temples"]
---

# 论文速读：HollowNeRF-Pruning-Hashgrid-Based-NeRFs-with-Trainable-Collision-Mitigation

## 一句话总结
HollowNeRF 提出一种可训练哈希碰撞缓解机制，通过自适应学习三维显著性网格指导特征剪枝，在显著减少参数量的同时提升渲染质量，实现了比 Instant-NGP 更优的成本-精度权衡。

## 研究问题与动机
- **哈希冲突导致的精度瓶颈**：Instant-NGP 等轻量哈希网格方法将大量特征压缩进小规模 hash table，导致不同体素的特征在哈希桶中冲突，共享同一存储位置，降低了渲染精度。
- **现有剪枝方法依赖外部先验**：已有方法（如 DONeRF）依赖显式表面几何先验（由 shape-from-silhouette 或深度预测 MLP 提供）来识别可剪枝区域，但粗糙的表面估计会损害渲染质量，精确估计又增加复杂性。
- **大量空间资源被无效区域占用**：典型 3D 场景中，空区域或被遮挡的内部体素并不对最终渲染有贡献，但仍占用哈希网格的空间容量，造成资源浪费。
- **训练效率与模型紧凑性的矛盾**：如何在保持快速训练的同时进一步压缩模型体积，同时不牺牲渲染质量，仍是 NeRF 领域的开放问题。

## 核心贡献（创新点）
- **可训练三维显著性网格（Trainable 3D Saliency Grid）**：直接学习场景的稀疏分布来引导特征压缩，无需任何表面几何先验，与 DONeRF 等依赖显式几何预测的方法形成本质区别。
- **ADMM 剪枝器（ADMM Pruner）**：将显著性网格的稀疏化转化为带约束的优化问题，通过交替方向乘子法自动学习最优剪枝比例，相比简单 L1 正则化具有更稳定的稀疏控制能力。
- **软零跳过门（Soft Zero-Skipping Gate）**：在 MLP 解码器中引入可微分的门控机制 $\hat{g}(\boldsymbol{v}) = \tanh(\alpha \|\boldsymbol{v}\|_2)$，确保被剪枝至零的特征向量正确映射为零密度，解决了硬门梯度消失导致剪枝不充分的问题。
- **更优的成本-精度权衡**：在 NeRF Synthetic 数据集上，HollowNeRF（6.48M 参数）相比 Instant-NGP（11.49M 参数）提升约 1dB PSNR，参数量仅为对方的 56%。

## 方法详解
- **整体流程**：给定三维坐标 $\boldsymbol{x} = (x, y, z)$，从多层哈希网格中提取特征向量 $\boldsymbol{f}$；通过可训练显著性网格 $\mathcal{G}$（分辨率 $T \times T \times T$）以三线性插值 + sigmoid 计算显著性权重 $p$，得到加权特征 $\boldsymbol{v} = p \cdot \boldsymbol{f}$；加权特征经带软零跳过门的 MLP 解码为密度 $\sigma$ 和颜色；通过体积渲染计算损失并与 GT 对比。
- **显著性网格的学习语义**：显著性权重 $p$ 可解释为坐标 $\boldsymbol{x}$ 处特征非零的概率，$\boldsymbol{v} = \mathcal{P}(\boldsymbol{x})\boldsymbol{f}$ 即为该位置特征的期望值；训练过程中，可见/重要体素的 $p$ 趋近于 1，不可见/内部体素的 $p$ 趋近于 0，实现"空心化"。
- **哈希冲突重分布原理**：当两个体素哈希冲突共享桶 $f_H$ 时，Instant-NGP 强制两者获得相同特征值；HollowNeRF 通过显著性权重实现 $f_1^* = p(\boldsymbol{x}_1) f_H$ 和 $f_2^* = p(\boldsymbol{x}_2) f_H$，将不重要体素的权重压至 0 后，冲突桶可完全拟合重要体素的特征。
- **软零跳过门设计**：$\hat{\mathcal{M}}(\boldsymbol{v}) = \hat{g}(\boldsymbol{v}) \cdot \mathcal{M}(\boldsymbol{v})$，其中 $\hat{g}(\boldsymbol{v}) = \tanh(\alpha \|\boldsymbol{v}\|_2)$；$\alpha$ 在训练前 1000 epoch 设为 $10^4$，之后增大至 $10^5$，逐步"硬化"门控，避免扰动微调过程。
- **ADMM 剪枝器**：将稀疏化建模为约束优化 $\min L(\mathcal{W}, \mathcal{H}, \mathcal{G})$ s.t. $\|S_{sig}(\mathcal{G})\|_1 < C$，通过增广拉格朗日方法转化为无约束极小极大问题，引入可训练对偶变量 $\gamma$ 替代固定正则化系数 $\lambda$；每步先用梯度下降更新 $\mathcal{G}, \mathcal{H}, \mathcal{W}$，再用梯度上升更新 $\gamma^{(t+1)} = [\gamma^{(t)} + \rho_\gamma (\|S_{sig}(\mathcal{G})\|_1 - C)]_+$。

## 实验与结果
- **数据集**：NeRF Synthetic 数据集（含 Chair、Lego、Hotdog 等场景）和 Tanks and Temples 数据集。
- **评估指标**：PSNR（越高越好）和 LPIPS（越低越好，AlexNet 后端）。
- **硬件与实现**：基于 torch-ngp（PyTorch CUDA 扩展）实现，训练使用 NVIDIA Tesla A100 GPU，Adam 优化器，初始学习率 $1 \times 10^{-2}$，训练 300,000 步。
- **关键结果**：
  - HollowNeRF（$T=64$，哈希大小 $2^{18}$，6.48M 参数）在 Chair 场景达到 **34.85 dB PSNR**，较 Instant-NGP（11.49M 参数，33.86 dB）提升约 **1 dB**，参数仅用 56%。
  - 在全部测试的 10 个 NeRF Synthetic 场景中，HollowNeRF 在所有哈希尺寸下均优于 Instant-NGP；7 个场景中最小哈希配置（$2^{17}$，3.34M 参数）的 PSNR 即超过 Instant-NGP 最大哈希配置（$2^{19}$）。
  - 平均 PSNR 达 **32.53 dB**，模型大小仅 **14.0 MB**，成本-精度权衡优于 CC-NeRF、TensoRF 等对比方法。
  - 消融实验验证显著性网格、零跳过门、ADMM 剪枝器三个组件各自对性能有正向贡献，且组合效果最佳。

## 相关工作脉络
- **Instant-NGP**：本文的基础框架，采用多层哈希网格编码 + 轻量 MLP 解码；HollowNeRF 在此基础上引入可训练显著性网格和 ADMM 剪枝器，解决其哈希冲突均匀分布导致的精度损失。
- **CC-NeRF / TensoRF**：基于张量分解的低秩压缩方法；HollowNeRF 按特征对渲染精度的实际影响进行剪枝，而非按原始信息熵，二者压缩策略本质不同。
- **DONeRF**：使用深度预测网络估计表面占据空间以引导采样；HollowNeRF 不依赖任何显式表面几何先验，通过端到端训练的显著性网格自主学习稀疏分布。
- **Plenoxels**：无 MLP 的稀疏体素网格方案，速度快但空间复杂度高；HollowNeRF 在保持轻量参数的同时实现更好的 PSNR-LPIPS 平衡。
- **Mip-NeRF / Mip-NeRF 360**：通过抗 aliasing 的多尺度表示提升渲染质量；HollowNeRF 侧重于参数压缩和哈希冲突缓解，两者正交。

## 局限性与未来方向
- 方法假设场景具有稀疏性（大部分为空白或被遮挡区域），对于烟雾、火焰、云层等半透明/弥漫场景，剪枝效果会退化至基线水平。
- 当前版本与 Instant-NGP 一样，对高反射表面的建模能力有限，尚未支持反射渲染。
- 未来计划将 HollowNeRF 的思想扩展到支持反射表面的 NeRF 框架（如 Ref-NeRF、NeRFren 等）。
- 显著性网格分辨率 $T$ 的选择需在精度与参数开销之间权衡，过小会导致收敛不稳定，过大则收益边际递减。

## 研究启发与可借鉴点
- **可训练稀疏掩码的学习范式**：通过附加轻量可学习网格引导主干特征压缩的思路，可迁移至其他基于哈希/网格的神经表示方法（如 3D Gaussian Splatting 的稀疏化）。
- **ADMM 约束优化的适配**：将正则化系数转化为可训练对偶变量的思路，避免了手动调参的困难，适用于各类稀疏约束下的模型压缩任务。
- **零映射保障机制**：软零跳过门确保"输入为零则输出为零"的结构性约束，这一设计对稀疏特征处理、特征解耦等场景具有通用参考价值。
- **与团队方向结合机会**：可将显著性网格思想应用于大场景 NeRF 的跨设备部署，或在压缩版 3DGS 中引入类似的碰撞缓解机制。
- **消融设计的系统性**：论文逐组件消融并报告收敛速度，值得借鉴为后续工作提供完整评估基准。

## 关键术语表
**Hashgrid / 哈希网格**：将连续三维空间离散化为多层级哈希表的结构，通过快速哈希索引查询特征向量，是 Instant-NGP 的核心加速组件。
**Saliency Grid / 显著性网格**：低分辨率可训练三维张量，存储各空间区域的显著性权重，用于指导特征压缩与哈希冲突缓解。
**ADMM Pruner / ADMM 剪枝器**：基于交替方向乘子法的优化器，将显著性网格稀疏化建模为带 $L_1$ 约束的优化问题，实现自动化的特征剪枝。
**Zero-Skipping Gate / 零跳过门**：嵌入 MLP 解码器的可微分门控模块，确保被剪枝至零的特征向量映射为零密度输出。
**Hash Collision / 哈希冲突**：不同三维体素的特征被哈希到同一存储桶，导致特征混淆和渲染精度下降。
**Volume Rendering / 体积渲染**：沿相机光线对三维空间的密度和颜色进行累加积分，生成二维图像的渲染方法。

## 可复现要素
- **数据集**：NeRF Synthetic 数据集（公开）、Tanks and Temples 数据集（公开）。
- **代码**：基于 torch-ngp 实现，torch-ngp 为开源项目（GitHub: ashawkey/torch-ngp）。
- **关键超参**：显著性网格分辨率 $T = 64$，稀疏约束 $C = 0.04$，哈希网格 16 层，特征维度 2，MLP 2 层隐层 × 64 神经元，$\alpha$ 调度（前 1000 epoch 为 $10^4$，之后为 $10^5$），Adam 初始学习率 $1 \times 10^{-2}$，训练 300,000 步。
