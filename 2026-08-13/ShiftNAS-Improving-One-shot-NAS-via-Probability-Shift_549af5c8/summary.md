---
title: "ShiftNAS-Improving-One-shot-NAS-via-Probability-Shift"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ShiftNAS_Improving_One-shot_NAS_via_Probability_Shift_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:31:04"
field: "神经架构搜索(NAS)"
keywords: ["One-shot NAS", "神经架构搜索", "概率偏移", "架构生成器", "权重纠缠", "Vision Transformer", "CNN搜索", "ImageNet"]
innovations: ["提出基于训练充分性梯度的概率偏移采样策略，动态调整不同FLOPs区域的采样概率", "设计LSTM架构生成器结合矩阵映射技术，实现端到端可微分的任意FLOPs子网生成"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：ShiftNAS-Improving-One-shot-NAS-via-Probability-Shift

## 一句话总结
ShiftNAS提出了一种端到端的One-shot NAS训练框架，通过动态学习基于计算资源（FLOPs）的采样概率分布（概率偏移）和LSTM架构生成器，解决均匀采样导致极端复杂度子网训练不足的问题，在CNN和ViT上实现无需重训即可直接继承最优权重的SOTA级搜索性能。

## 研究问题与动机
1. **均匀采样的结构性偏差**：One-shot NAS传统采用均匀采样策略，由于每个操作的计算资源独立服从均匀分布，根据Irwin-Hall定理，子网总计算资源近似正态分布，导致中间复杂度子网被过度训练，而大/小FLOPs子网训练不充分。
2. **不同复杂度子网需要差异化训练策略**：不同参数量的子网收敛速度不同（如1.0 GFLOPs可能30000次迭代收敛，2.0 GFLOPs需50000次），统一采样使极端复杂度子网在相同训练步数下训练不足，影响权重质量和排名准确性。
3. **继承权重性能与重训存在差距**：One-shot NAS的核心假设是子网可直接继承supernet权重，但实际训练中训练不足区域的子网权重质量差，部署时需额外fine-tune或retrain，抵消了搜索效率优势。
4. **现有改进方法仍有局限**：AttentiveNAS通过Pareto感知采样、FocusFormer通过架构采样器聚焦帕累托前沿，但未从"训练充分性"角度系统性解决资源分配不均问题。

## 核心贡献（创新点）
1. **可学习的概率偏移采样策略**：通过计算不同FLOPs区域的子网验证集梯度，识别训练不足区域并动态提升其采样概率，与AttentiveNAS/FocusFormer等基于帕累托前沿的方法本质区别在于直接以"训练梯度充分性"作为资源分配的代理信号。
2. **LSTM架构生成器（Architecture Generator, AG）**：将NAS建模为序列生成问题，AG接收目标FLOPs约束并输出满足该约束的子网one-hot策略序列，与DARTS/DNAS等搜索方法区别在于端到端可微分训练且无需外部搜索器（RL/进化算法）。
3. **矩阵映射技术适配权重纠缠搜索空间**：针对BigNAS等权重纠缠搜索空间（低索引操作权重优先保留），设计掩码矩阵将one-hot策略映射为可微分子网掩码，使梯度可从supernet反向传播至AG。
4. **模型无关的SOTA搜索性能**：在CNN（ImageNet分类）和ViT两个架构类型上均实现SOTA或竞争力结果，且无需任何额外搜索或重训成本（ShiftCNN比BigNAS节省2×+ GPU时间，Top-1精度提升0.7%）。

## 方法详解
**整体框架**：ShiftNAS采用两阶段联合训练范式——supernet训练阶段通过概率偏移动态调整FLOPs采样分布，AG生成满足目标约束的子网；搜索阶段AG直接根据目标FLOPs生成最优子网架构，权重直接从训练好的supernet继承。

**概率偏移（Probability Shift）**：
- 将连续FLOPs范围离散化为K个区间（步长0.1 GFLOPs），初始化可学习分布向量B（1×K）
- 每个训练步通过Gumbel Softmax从B采样一个目标FLOPs b
- 通过二阶梯度信息评估该FLOPs区域的训练充分性：当验证损失梯度∇w接近零时子网收敛，反之训练不足
- 利用有限差分近似更新B（类比DARTS的二阶优化思想），每100次迭代更新一次分布
- 优化目标：$\arg\min_B E_{b\sim B}[\nabla_w L_{val}(w, \alpha|b)]$

**架构生成器（AG）**：
- LSTM结构（64个隐藏单元）逐位置生成子网的one-hot操作策略序列
- 通过Gumbel Softmax实现可微分采样
- 矩阵映射技巧：对于权重纠缠搜索空间，设计掩码矩阵M（如ViT多头注意力场景m⁰=[1,0,0], m¹=[1,1,0], m²=[1,1,1]），使one-hot策略p通过p' = M·p^T转换为可微分掩码参与前向/反向传播
- 联合优化目标：$\mathcal{L} = L_{task} + \lambda L_{RC}$，其中$L_{RC} = (\sum_{i=1}^D\sum_{j=1}^n b_j^i p_j'^i - C)^2$为资源约束损失，C为目标FLOPs

**搜索与部署**：
- 训练完成后，AG对任意给定FLOPs约束直接输出对应子网架构
- 子网权重直接从supernet继承，无需微调或重训
- 训练策略：前50个epoch仅优化supernet权重和AG，之后每100步更新采样分布

## 实验与结果
**数据集与基准**：ImageNet分类任务（1.2M训练图，50K验证图），评估指标为Top-1/Top-5精度和FLOPs。

**主要结果（ImageNet）**：
- **ViT-tiny (1.3 GFLOPs)**：ShiftFormer-T达到76.0% Top-1，超越AutoFormer-T（74.7%，+1.3%）和FocusFormer-T（75.1%，+0.9%）
- **ViT-small (5.0 GFLOPs)**：ShiftFormer-S达到82.2%，超越AutoFormer-S（81.4%）和FocusFormer-S（81.6%）
- **ViT-base (11.0 GFLOPs)**：ShiftFormer-B达到82.8%，超越AutoFormer-Base（81.4%）
- **CNN-S (0.24 GFLOPs)**：ShiftCNN-S达到77.2%，超越BigNAS-S（76.5%，+0.7%）
- **CNN-B (0.42 GFLOPs)**：ShiftCNN-B达到79.6%，超越BigNAS-M（78.9%，+0.7%）
- **效率对比**：ShiftCNN与BigNAS共享相同搜索空间，但BigNAS因使用sandwich rule训练需112 GPU天，ShiftCNN仅需32 GPU天（节省约2.5×）

**消融实验**：
- **概率偏移有效性**：图4显示ShiftNAS（含概率偏移）的FLOPs/Accuracy权衡曲线显著优于无概率偏移的baseline，证明动态采样分布能覆盖更广的性能区间
- **AG排名相关性**：图6显示继承权重Top-1 Acc. > 65%的子网，其继承性能与微调后性能高度相关，验证supernet的排名预测能力
- **无需重训**：表2显示Fine-tune/Re-train对ShiftNAS模型精度影响极小（±0.1~0.2%），证明继承权重已足够优质
- **最优采样策略**：图5显示学习到的分布倾向于为更大FLOPs子网分配更高采样概率（如ViT-tiny中1.6~1.9 GFLOPs区域概率高于1.3~1.4 GFLOPs），符合"更大子网需要更多训练资源"的直觉

## 相关工作脉络
1. **均匀采样One-shot NAS**（SNAS [7], Single-Path NAS）：假设所有候选架构同等重要，均匀采样操作。ShiftNAS指出该策略导致计算资源正态分布偏差，提出概率偏移纠正。
2. **Pareto感知采样**（AttentiveNAS [29]）：通过提取等大小子网并筛选最优/最差架构训练来贴近部署场景。ShiftNAS与它的区别在于：不依赖显式Pareto筛选，而是以训练梯度充分性为信号动态调整FLOPs分布。
3. **强公平性采样**（FairNAS [5]）：要求每个操作选项在各阶段获得相同更新次数。ShiftNAS关注的是不同FLOPs区域的训练充分性差异，而非操作选项的公平性。
4. **资源聚焦采样**（FocusFormer [19]）：通过架构采样器聚焦帕累托前沿架构。ShiftNAS通过LSTM AG直接生成目标FLOPs子网，避免"反复采样直到满足约束"的低效过程。
5. **权重纠缠搜索空间**（BigNAS [33], AutoFormer [4]）：采用矩阵映射确保低索引操作权重优先保留。ShiftNAS在此基础上设计可微分的AG训练机制，填补了权重纠缠空间高效搜索的空白。
6. **DARTS类可微分搜索**（DARTS [18], DNAS [11]）：使用连续松弛和梯度优化。ShiftNAS选用DNAS而非RL/进化算法的原因是其收敛更快，且AG的序列生成方式天然适配权重纠缠空间的掩码映射。

## 局限性与未来方向
1. **仅适用于权重纠缠搜索空间**：论文明确指出ShiftNAS不能应用于DARTS-like搜索空间（连续松弛），因矩阵映射技巧依赖于"低索引操作权重优先保留"的离散权重共享假设。
2. **仅验证ImageNet分类任务**：实验集中在标准图像分类基准，未扩展到检测、分割等下游任务，也未在真实硬件延迟上进行评测。
3. **资源离散化粒度依赖超参**：FLOPs区间的划分（步长0.1 GFLOPs）和分布更新频率（每100步）需手动设定，可能影响不同搜索空间下的泛化性。
4. **AG训练稳定性**：AG需在50 epoch后才能有效生成子网，前期训练阶段可能存在浪费；未来可探索更好的预热策略或课程学习方案。
5. **未讨论极端FLOPs约束**：实验主要覆盖中低FLOPs范围（0.2~17.6 GFLOPs），对于极低或极高端点资源的搜索能力未充分验证。

## 研究启发与可借鉴点
1. **"训练充分性"作为资源分配信号**：概率偏移以验证梯度大小衡量子网训练状态，这一信号可迁移至其他NAS框架（如进化搜索、RL搜索）的资源调度策略设计中，替代纯基于准确性的筛选逻辑。
2. **LSTM架构生成器+矩阵映射的端到端范式**：将序列生成与权重纠缠搜索空间结合的设计值得借鉴，可推广至其他具有优先级依赖的离散搜索空间（如多尺度特征融合网络、动态路由网络）。
3. **有限差分近似二阶梯度的优化技巧**：类比DARTS的二阶优化思想，用两次训练步近似计算分布梯度，避免昂贵矩阵运算，这一工程技巧在资源受限的NAS训练中具有通用价值。
4. **无需重训的"即搜即用"目标**：ShiftNAS以"子网权重可直接继承"为核心设计目标，而非追求supernet绝对高精度；这一目标函数视角可启发后续工作重新审视One-shot NAS的成功标准。
5. **模型无关性的统一框架**：同一套方法在CNN和ViT两个差异巨大的架构类型上均有效，表明概率偏移和AG的设计具有良好的架构泛化性，可作为通用组件嵌入不同NAS管线。

## 关键术语表
**One-shot NAS**：通过训练一次超网络（supernet），所有子架构共享权重，从而大幅降低NAS搜索成本的神经架构搜索方法。
**Probability Shift（概率偏移）**：ShiftNAS核心机制，通过梯度信号动态调整不同FLOPs区域的采样概率，使训练资源向训练不足的区域倾斜。
**Architecture Generator (AG)**：基于LSTM的序列生成器，接收目标FLOPs约束并端到端生成满足该约束的子网架构one-hot策略。
**Weight Entanglement（权重纠缠）**：搜索空间中不同候选操作共享权重，且低索引操作权重被高索引操作包含的权重共享机制（如BigNAS、AutoFormer采用）。
**Matrix Mapping（矩阵映射）**：将one-hot策略向量转换为可微分掩码矩阵的技术，使AG的梯度可通过supernet反向传播，适配权重纠缠搜索空间。
**Gumbel Softmax**：用于离散采样重参数化的技术，将from categorical distribution的采样过程转化为可微分操作，支持端到端训练。
**Irwin-Hall Distribution**：D个独立Uniform(0,1)变量之和的分布；论文用于理论解释均匀采样导致子网FLOPs近似正态分布的原因。
**Finite Difference Approximation**：用两次训练步的损失差值近似梯度，避免计算二阶导数矩阵，降低概率偏移的优化开销。

## 可复现要素
- **数据集**：ImageNet（公开），论文声明使用50K图片进行验证
- **代码开源**：论文末尾声明"Source codes are available at GitHub"（具体仓库链接未在正文提供，需查找原始论文仓库）
- **超参数**：LSTM隐藏单元数64；Adam优化器学习率1e-3；前50 epoch仅优化supernet和AG；采样分布每100步更新一次；FLOPs区间步长0.1 GFLOPs
- **硬件**：8×Nvidia Tesla A100 GPU
- **搜索空间**：CNN（遵循BigNAS设计）、ViT（遵循AutoFormer设计）
- **论文未提及**：具体的λ系数值、LSTM层数细节（仅提及4层FC for ViT/10层FC for CNN）、Gumbel Softmax温度参数
