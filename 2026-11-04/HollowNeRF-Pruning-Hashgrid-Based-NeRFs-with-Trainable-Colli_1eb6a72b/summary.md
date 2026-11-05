---
title: "HollowNeRF-Pruning-Hashgrid-Based-NeRFs-with-Trainable-Colli"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_HollowNeRF_Pruning_Hashgrid-Based_NeRFs_with_Trainable_Collision_Mitigation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:43:25"
field: "神经辐射场轻量化与压缩"
keywords: ["NeRF压缩", "hashgrid", "model pruning", "saliency grid", "ADMM", "hash collision mitigation", "lightweight neural rendering"]
innovations: ["可训练3D显著性网格引导hash碰撞优先级重分配", "ADMM对偶变量驱动的端到端稀疏剪枝", "Soft zero-skipping gate保证剪零特征与零密度一致"]
benchmarks: ["NeRF Synthetic", "Tanks and Temples"]
---

# 论文速读：HollowNeRF - Pruning Hashgrid-Based NeRFs with Trainable Collision Mitigation

## 一句话总结
HollowNeRF 提出一种基于可训练 3D 显著性网格（saliency grid）和 ADMM 剪枝器的 NeRF 压缩方法，在训练阶段自动学习场景稀疏分布并剪除不可见/内部体素的特征，从而在减少 hash 碰撞的同时显著降低参数量；实验表明其在仅使用 56% 参数时可获得最高 1dB 的 PSNR 增益，或仅用 31% 参数达到与 Instant-NGP 相当的渲染质量。

## 研究问题与动机
- **Hashgrid NeRF 的碰撞精度瓶颈**：Instant-NGP 使用轻量级多分辨率 hashgrid 编码场景特征，但碰撞均匀散布在所有非零密度体素上，导致特征混合、精度次优。
- **现有稀疏利用方法的不足**：已有方法依赖先验几何信息（如 silhouette-based 表面重建或额外 MLP 预测深度分布），粗估计降低渲染质量，精估计增加复杂度抵消加速收益。
- **内部不可见体素浪费资源**：即使有 occupancy mask，被遮挡的内部体素仍占用 hashbucket 容量，引发不必要碰撞。
- **缺乏端到端可学习的稀疏诱导机制**：传统 L1 正则化难以同时兼顾训练稳定性和稀疏目标控制，固定超参 λ 易导致过度剪枝或剪枝不足。

## 核心贡献（创新点）
- **可训练 3D 显著性网格引导特征优先分配**：通过 $T \times T \times T$ 可微 saliency grid 学习空间可视概率分布 $\mathcal{P}(\boldsymbol{x})$，使可见体素特征获得更高权重并主导共享 hashbucket，与 Instant-NGP 均匀碰撞分配形成本质区别。
- **ADMM 剪枝器实现精确零值强制**：将稀疏约束转化为带可训练对偶变量 $\gamma$ 的增广 Lagrangian 优化问题，端到端地将不可见/空区域特征精确推至零，优于简单 L1 正则化的软压缩效果。
- **Soft zero-skipping gate 保持密度稀疏一致性**：引入可微 $\tanh(\alpha\|\boldsymbol{v}\|_2)$ gate 替代 hard gate，确保 MLP 输出密度满足 $\hat{\mathcal{M}}(\boldsymbol{0}) = 0$，避免剪枝特征被 MLP 还原为非零密度，同时保留梯度通路供 ADMM 继续优化。
- **端到端轻量级压缩 pipeline**：无需任何表面先验或离线几何估计，saliency grid + pruner + zero-skipping gate 三个模块均在标准 NeRF 训练循环中联合优化。

## 方法详解
**整体流程（图2）**：输入坐标 $\boldsymbol{x}$ → 从多分辨率 hashgrid $H$ 查询特征 $f$ → 从 saliency grid $\mathcal{G}$ 双线性插值得权重 $p$ → 加权特征 $\boldsymbol{v} = p\boldsymbol{f}$ → MLP（含 zero-skipping gate）输出密度 $\sigma$ 和颜色 → 体积渲染 → 与 GT 计算损失 $\mathcal{L}$ → ADMM pruner 同步更新 $\mathcal{G}$。

**3.1 可训练 3D 显著性网格**：将空间划分为 $T \times T \times T$ 粗粒度网格，$\mathcal{G} \in \mathbb{R}^{T \times T \times T}$ 存储可训练标量。对任意坐标 $\boldsymbol{x}$，找到周围 8 个 grid voxel，经 trilinear interpolation 得 $p \in (0,1)$，再经 sigmoid 投影。物理含义：$p = \mathcal{P}(\boldsymbol{x})$ 表示该位置特征非零的概率，期望特征 $\boldsymbol{v} = p\boldsymbol{f} + (1-p)\cdot\boldsymbol{0} = p\boldsymbol{f}$。高 $p$ 值体素在反向传播中累积更多梯度，主导共享 bucket；$p \to 0$ 时该特征对碰撞无干扰。

**3.2 Soft zero-skipping gate**：为防止 MLP 将剪至零的特征 $\boldsymbol{v}=\boldsymbol{0}$ 映射为非零密度，定义：
$$\hat{\mathcal{M}}(\boldsymbol{v}) = \hat{g}(\boldsymbol{v}) \cdot \mathcal{M}(\boldsymbol{v}), \quad \hat{g}(\boldsymbol{v}) = \tanh(\alpha \|\boldsymbol{v}\|_2)$$
其中 $\alpha$ 从 $10^4$（前 1000 epoch）逐步增至 $10^5$，使 gate 由软变硬。相比不可微 hard gate，软 gate 保证梯度可传回 saliency grid，使 ADMM 能继续压小 $p$。

**3.3 ADMM 剪枝器**：核心优化问题：
$$\min_{\mathcal{W},\mathcal{H},\mathcal{G}} \mathcal{L}(\mathcal{W},\mathcal{H},\mathcal{G}) \quad \text{s.t.} \quad \|S_{\mathrm{sig}}(\mathcal{G})\|_1 < C$$
转化为增广 Lagrangian 无约束形式：
$$\min_{\mathcal{W},\mathcal{H},\mathcal{G}} \max_{\gamma \geq 0} \; \mathcal{L} + \frac{\rho_\gamma}{2}[\|S_{\mathrm{sig}}(\mathcal{G})\|_1 - C]_+^2 + \gamma(\|S_{\mathrm{sig}}(\mathcal{G})\|_1 - C)$$
对偶变量 $\gamma$ 替代固定 $\lambda$，随训练自适应调整稀疏压力，避免 L1 正则化的超参敏感问题。每步：先用 SGD 更新 $(\mathcal{W},\mathcal{H},\mathcal{G})$，再用梯度上升更新 $\gamma^{(t+1)} = [\gamma^{(t)} + \rho_\gamma(\|S_{\mathrm{sig}}(\mathcal{G})\|_1 - C)]_+$。

## 实验与结果
- **数据集**：NeRF Synthetic（标准场景）、Tanks and Temples；评估指标 PSNR（↑）和 LPIPS AlexNet（↓）。
- **基线**：Instant-NGP（同 torch-ngp 实现，16 级 hashgrid，feature dim=2，MLP 2层×64），以及 CC-NeRF、TensoRF、DONeRF 等。
- **超参**：$T=64$、$C=0.04$、hashgrid size $2^{17}/2^{18}/2^{19}$、训练 300k steps、Adam LR=$1\times10^{-2}$、单卡 A100。
- **核心结果**：
  - **最省参比**：HollowNeRF $T=64$ + hashgrid $2^{19}$（6.48M 参数）vs Instant-NGP（11.49M 参数）→ **PSNR 高出 ~1dB**（Chair 场景 34.85 vs 33.86 dB）。
  - **等参比**：HollowNeRF 用 **31% 参数**（$2^{17}$, 3.34M）即可达到 Instant-NGP $2^{19}$（11.49M）相当质量。
  - **平均 PSNR**：HollowNeRF-H19（$2^{19}$）在 NeRF Synthetic 上达 **32.53 dB / 14.0 MB**，成本-精度权衡最优。
  - **鲁棒性**：5 次重复实验标准差更低，收敛稳定性优于 Instant-NGP。
  - **Ablation**：逐组件移除均无灾难性退化，完整方法始终优于更小 Instant-NGP。

## 相关工作脉络
- **Instant-NGP [16]**：本文基线，hashgrid + 2层 MLP，碰撞均匀分布；HollowNeRF 在同架构上引入 saliency 引导优先级重分配。
- **DONeRF [18]**：使用辅助 MLP 预测深度分布以加速采样；HollowNeRF 不需要额外几何网络，salience 与渲染 jointly 学习。
- **TensoRF [4] / CC-NeRF [26]**：低秩张量分解压缩；HollowNeRF 基于空间可视性而非信息熵裁剪，保留对渲染最有影响力的特征。
- **Plenoxels [22] / DVF [23]**：无 MLP 的稀疏体素/网格表示；HollowNeRF 属于 hashgrid+MLP 路线的轻量化改进，保留 MLP 的非线性表达能力。
- **Mip-NeRF 系列 [1,2]**：抗混叠多尺度编码；与 HollowNeRF 正交，可结合用于更大场景高质量渲染。
- **Ref-NeRF [28] / Nerfren [8]**：反射/环境光建模；本文明确讨论将 HollowNeRF 推广至此类框架作为未来方向。

## 局限性与未来方向
- **场景稀疏性假设**：对烟雾、火焰、云层等全空/半透明场景，性能退化为 baseline（无可剪枝区域）。
- **反射面建模不足**：与 Instant-NGP 同源，对高光/镜面反射处理有限。
- **未集成抗混叠编码**：当前未结合 Mip-NeRF 等多尺度滤波，在高频细节场景可能有提升空间。
- **未来方向**：扩展至含反射/透射的 NeRF 框架；探索更大规模室外场景适配；与 Mip-NeRF 360 等结合。

## 研究启发与可借鉴点
- **Saliency 加权特征替代硬性 mask**：用连续概率 $p \in (0,1)$ 做特征缩放而非二值剔除，兼顾可微性和渐进稀疏化，可迁移至其他 hash-based 特征编码系统（如 3D Gaussian Splatting 的 attribute pruning）。
- **ADMM 对偶变量替代固定正则化超参**：用可训练 $\gamma$ 自动追踪稀疏预算 $C$，避免 L1 调参痛苦；适合任何需要动态稀疏约束的视觉/图形学习管线。
- **Zero-skipping gate 设计**：通过 $\tanh(\alpha\|\boldsymbol{v}\|_2)$ 将非线性门控融入 MLP 前向，既保证 $\boldsymbol{v}=\boldsymbol{0} \Rightarrow \sigma=0$ 的物理一致性，又保留梯度；可推广至任何"特征稀疏→输出稀疏"需对齐的解码器设计。
- **Cost-accuracy 三维扫描实验设计**：同时 Sweep hashgrid size × saliency resolution × sparsity constraint，给出 Pareto 前沿图（Fig.4/5/9），为后续工作提供直接对比基准。
- **团队可结合方向**：将 HollowNeRF 的 saliency 机制嫁接至 Instant-NGP 的扩展版（如带 Reflection 的 Ref-NGP、或 3DGS 的 attribute pruning），或用 ADMM 风格对偶优化替代 L0/L1 正则化研究其他神经表示压缩。

## 关键术语表
**Instant-NGP**：基于多分辨率 hashgrid + 轻量 MLP 的实时 NeRF 加速方案，本文核心基线与架构起点。
**Hash collision（Hash 碰撞）**：不同 3D 体素映射到同一 hashbucket 导致特征混杂，是 hashgrid 精度瓶颈的根本原因。
**Saliency grid（显著性网格）**：$T \times T \times T$ 可训练 3D 张量，通过插值生成空间可视概率分布 $\mathcal{P}(\boldsymbol{x})$，引导特征优先级分配。
**ADMM pruner（ADMM 剪枝器）**：基于增广 Lagrangian 的稀疏约束优化器，用可训练对偶变量 $\gamma$ 将 saliency grid 中无效特征精确推至 0。
**Soft zero-skipping gate**：$\tanh(\alpha\|\boldsymbol{v}\|_2)$ 门控，嵌入 MLP 保证剪零特征对应零密度输出，同时维持梯度连通性。
**Sparsity constraint $C$**：saliency grid sigmoid 输出之和的上界，控制非零体素比例；本文最优 $C=0.04$（即 4% 体素保留）。
**Volume rendering（体积渲染）**：沿相机光线积分密度与颜色得到像素值的 NeRF 标准合成过程。
**torch-ngp**：PyTorch 实现的 Instant-NGP，本文 HollowNeRF 的上游代码基础。

## 可复现要素
- **数据集**：NeRF Synthetic（公开）、Tanks and Temples（公开）。
- **代码**：基于 torch-ngp（开源，见引用 [25]）扩展；论文未提供独立仓库链接。
- **关键超参**：$T=64$、$C=0.04$、16 级 hashgrid（feature dim=2，coarse=16，fine=1024）、MLP 2层×64、Adam LR=$1\times10^{-2}$、300k steps、$\alpha$  schedule（前 1000 epoch $10^4$，之后 $10^5$）、$\rho_\gamma$ 未明确（论文未提及具体数值）。
- **硬件**：单卡 NVIDIA A100。
