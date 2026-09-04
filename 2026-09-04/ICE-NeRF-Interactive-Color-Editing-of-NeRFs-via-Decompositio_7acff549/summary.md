---
title: "ICE-NeRF-Interactive-Color-Editing-of-NeRFs-via-Decompositio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_ICE-NeRF_Interactive_Color_Editing_of_NeRFs_via_Decomposition-Aware_Weight_Optimization_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:16:36"
field: "3D场景重建与编辑"
keywords: ["NeRF", "颜色编辑", "3D场景编辑", "体积渲染", "部分微调", "激活场正则化"]
innovations: ["提出基于激活场的权重正则化（AFR）解决NeRF隐式表示纠缠问题", "设计单掩码多视图渲染（SMR）实现单视图掩码下的多视图一致编辑"]
benchmarks: ["NeRF Blender", "LLFF", "Mip-NeRF360"]
---

# 论文速读：ICE-NeRF-Interactive-Color-Editing-of-NeRFs-via-Decompositio

## 一句话总结
ICE-NeRF 提出了一种基于预训练 NeRF 和单视图粗略掩码的交互式颜色编辑框架，通过部分微调策略在不到一分钟内完成 NeRF 场景的颜色重绘，同时利用激活场正则化（AFR）和单掩码多视图渲染（SMR）技术解决隐式表示纠缠和多视图一致性问题。

## 研究问题与动机
- **NeRF 编辑的核心挑战**：准确分解 3D 空间中期望编辑的区域并保持多视图一致性极为困难。
- **现有方法不足**：PaletteNeRF 和 RecolorNeRF 采用调色板基编辑方法，当场景中存在多个颜色相似的对象时无法选择性重绘；且 PaletteNeRF 需要 1-2 小时的训练时间，限制实用价值。
- **隐式表示纠缠问题**：NeRF 的颜色信息缠绕在整个网络权重中，简单微调会导致非目标区域的颜色意外变化。
- **单/少视图训练的视角依赖性**：NeRF 输出依赖观察方向，仅用少量视图掩码训练会导致新视图合成时目标区域颜色突变。

## 核心贡献（创新点）
- **ICE-NeRF 框架**：提出基于部分微调的交互式颜色编辑方法，通过冻结大部分权重仅微调 MLP 最后三层的部分权重实现快速编辑，与已有工作相比避免了引入额外模块或参数。
- **激活场正则化（AFR）**：首次将 GAN Dissection 思想引入 NeRF，通过渲染激活场计算惩罚矩阵来区分不同权重对目标/非目标区域的影响程度，从本质上解决权重纠缠问题。
- **单掩码多视图渲染（SMR）**：利用 NeRF 输出的深度图将单视图 2D 掩码反投影到 3D 空间并渲染多视图掩码，以 DIBR 方式保证多视图一致性，无需用户提供多视图标注。

## 方法详解
- **整体流程**：用户通过预训练 NeRF 渲染任意视角图像后绘制变化掩码 M^C 和冻结掩码 M^F，选择目标颜色，仅微调 MLP 最后三层的部分权重（original NeRF 为倒数第二层，Instant-NGP/TensoRF 为最后两层）。
- **激活场（Activation Field）**：将体积渲染公式应用于网络各层的激活值，得到 A_l(r) = Σ ω_i · a_l(x_i, d)，其中 a_l 为第 l 层的激活，ω_i 为体积渲染权重，以此获得每个区域在输入/输出侧的平均激活 a^C_IN, a^C_OUT, a^F_IN, a^F_OUT。
- **AFR 权重正则化**：惩罚矩阵 R_ij 由四部分组成：Norm(Rank(a^C_IN, desc))_j + Norm(Rank(a^C_OUT, desc))_i + Norm(Rank(a^F_IN, ascend))_j + Norm(Rank(a^F_OUT, ascend))_i，其中 Rank+Normalize 返回归一化排名向量；目标区域高激活对应的权重允许更大变化，非目标区域高激活对应的权重受到更强正则化约束，总损失 L_AFR = |W_l - W*_l| ⊙ R。
- **SMR 多视图掩码生成**：通过 D = Σ ω_i · t_i 提取深度图，将 2D 掩码反投影为 3D 点云，再从其他视角渲染时通过对比点云深度与新视图渲染深度来判断可见性，生成多视图掩码。
- **总损失函数**：L_Total = L_R + L_I + L_F + L_AFR，其中 L_R = ||Mask(M^C, I) - C_Target||_2（重绘损失），L_I = ||Int(Mask(M^C, I)) - Int(Mask(M^C, I*))||_2（强度保持损失），L_F = ||Mask(M^F, I) - Mask(M^F, I*)||_2（冻结损失）。

## 实验与结果
- **数据集**：NeRF Blender 数据集（Lego, Drums, Hotdog, Chair, Ship）、LLFF 数据集（Horns, Flower, Fortress）、Mip-NeRF360 数据集（Kitchen, Bonsai）。
- **基线方法**：PaletteNeRF [14]、RecolorNeRF [10]、PosterNeRF [27]。
- **分解性能定量结果**：在 Horns/fortress/Flower 三个场景中评估背景区域 MSE，ICE-NeRF 平均 MSE 为 0.0075，显著优于 PaletteNeRF（0.0277）和 RecolorNeRF（0.0627）；其中 Horns 场景（多个相似颜色恐龙骨骼）提升最为显著，从 PaletteNeRF 的 0.0818 降至 0.0213。
- **时间效率**：ICE-NeRF 约 25 秒完成编辑，RecolorNeRF 需 160 秒，PaletteNeRF 需 49 分钟（RTX 3090 GPU）。
- **消融实验**：AFR 有效抑制粗糙掩码导致的非目标区域颜色变化；强度损失保证重绘区域亮度与原始场景一致；即使掩码极度稀疏也能有效重绘。

## 相关工作脉络
- **PaletteNeRF [14]**：基于调色板的 NeRF 外观编辑方法，通过提取基础颜色并加权求和实现颜色编辑，但无法区分颜色相似的不同对象，且训练耗时 1-2 小时，本文在其基础上提出基于掩码的部分微调方案实现更快的交互式编辑。
- **RecolorNeRF [10]**：通过层分解辐射场实现颜色编辑，同样基于调色板方法，在复杂颜色场景下分解性能较差，本文方法在 Horns 等难场景下 MSE 降低约 3-6 倍。
- **PosterNeRF [27]**：基于可见性加权调色板提取的海报化编辑方法，对单一颜色对象的效果有限，本文在 Lego 场景中展示了更强的对象级编辑能力。
- **Instruct-NeRF2NeRF [11]**：基于扩散模型 InstructPix2Pix 的文本指令驱动 NeRF 编辑，无需用户掩码但需要生成模型支持，本文方法更轻量且不需要额外生成模型。
- **GAN Dissection [3]**：分析 GAN 网络中各神经元的语义嵌入，本文为该方法在 NeRF 中的首次应用，将激活场概念引入辐射场编辑。
- **DIBR [8]**：深度图像渲染技术，本文将其用于 NeRF 掩码的多视图重投影，解决单视图掩码导致的多视图不一致问题。

## 局限性与未来方向
- **360° 无界场景的伪影**：在 Mip-NeRF360 数据集的 360° 场景中观察到非目标区域的 unwanted color changes，推测原因是模型需要在更广阔复杂的 3D 空间中进行学习。
- **复杂缠绕物体的掩码困难**：对于场景中充满复杂缠绕结构（如复杂树枝）的对象，用户手动掩码可能难以精确指定目标区域。
- **未来方向**：探索通过权重优化实现场景几何编辑；将方法扩展至动态 NeRF 场景。

## 研究启发与可借鉴点
- **激活场分析的可迁移性**：将 GAN Dissection 思想迁移到 NeRF 权重分析的思路可推广至其他 3D 编辑任务（如几何编辑、材质编辑），作为理解 NeRF 隐式表示的工具。
- **部分微调策略的工程价值**：仅微调 MLP 最后三层而非全参数重训练的思路，为 NeRF 高效编辑提供了一个通用范式，可结合本团队研究方向探索其他属性的快速适配。
- **SMR 的掩码传递机制**：基于深度反投影生成多视图掩码的方法可借鉴于其他需要多视图一致性的 NeRF 编辑任务，减少对多视图人工标注的依赖。
- **惩罚矩阵设计的灵活性**：AFR 中基于激活值排名的权重惩罚矩阵设计思路可推广到其他需要区分不同权重重要性的场景优化问题。

## 关键术语表
**NeRF (Neural Radiance Fields)**：通过 MLP 学习 3D 空间中点的体积密度和方向相关辐射率的隐式 3D 表示方法。
**Volumetric Rendering（体积渲染）**：沿相机光线采样多个 3D 点，通过累积透明度和颜色权重合成 2D 像素颜色的渲染过程。
**Activation Field（激活场）**：将体积渲染公式应用于网络各层激活值后得到的场，反映不同空间区域与网络内部激活的对应关系。
**AFR (Activation Field-based Regularization)**：基于激活场计算权重惩罚矩阵的正则化技术，区分不同权重对目标/非目标区域的贡献程度。
**SMR (Single-mask Multi-view Rendering)**：利用深度图将单视图 2D 掩码反投影到 3D 空间并从多视角重新渲染，生成多视图一致掩码的技术。
**DIBR (Depth-Image-Based Rendering)**：基于深度图像将 2D 内容重投影到 3D 空间并从不同视角渲染的渲染方法。
**Implicit Representation Entanglement（隐式表示纠缠）**：NeRF 中特定区域的颜色信息缠绕在整个网络权重中而非集中在少数权重的现象。
**Decomposition Performance（分解性能）**：编辑方法将目标区域与非目标区域准确分离、避免非目标区域颜色意外变化的能力。

## 可复现要素
- **数据集**：NeRF Blender、LLFF、Mip-NeRF360（均为公开数据集）。
- **代码开源**：基于 nerf-pytorch [33]、torch-ngp [25, 26] 实现，使用 PyTorch 框架。
- **预训练权重**：使用官方预训练权重，论文未提供 ICE-NeRF 编辑后的权重公开声明。
- **关键超参**：优化器 Adam，学习率 0.01，mini-batch size 1024，迭代次数 N=10（重绘+保持）+ M=90（仅保持）= 100 次以内，训练时间 < 1 分钟（RTX 3090）。
- **模型适配**：original NeRF 微调倒数第二层，Instant-NGP 和 TensoRF 微调最后两层。
