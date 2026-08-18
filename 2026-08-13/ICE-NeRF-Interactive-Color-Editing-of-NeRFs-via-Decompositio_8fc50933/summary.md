---
title: "ICE-NeRF-Interactive-Color-Editing-of-NeRFs-via-Decompositio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_ICE-NeRF_Interactive_Color_Editing_of_NeRFs_via_Decomposition-Aware_Weight_Optimization_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:10:32"
---

# 论文速读：ICE-NeRF-Interactive-Color-Editing-of-NeRFs-via-Decompositio

## 一句话总结
本文提出 ICE-NeRF 框架，一种基于预训练 NeRF 的交互式颜色编辑方法，仅需用户单视图的粗略遮罩和少数轮次部分权重微调，即可在不到一分钟内完成高质量、多视角一致且精确分割的目标物体颜色重绘，显著优于现有方法的分解性能与效率。

## 研究问题与动机
1. **隐式表示的权重纠缠问题**：NeRF 的颜色信息分散在整个网络的权重中，直接对黑盒权重进行微调会导致非目标区域产生意外的颜色变化，难以实现精确的 3D 空间分割。
2. **单视图遮罩导致多视角不一致**：NeRF 输出依赖于视角方向，仅使用单视图或少数视图遮罩进行训练时，在视角变换过程中会出现突兀的意外颜色变化，破坏多视角渲染一致性。
3. **现有调色板方法的局限性**：SOTA 方法 PaletteNeRF 和 RecolorNeRF 采用调色板方式，当场景中存在多个颜色相似的物体时无法选择性编辑；同时训练耗时长达 1–2 小时（RTX 3090），缺乏交互实时性。
4. **时间与效率瓶颈**：现有方法难以满足用户对交互式、直觉化 NeRF 编辑的需求，需要更高效、更精确的解决方案。

## 核心贡献（创新点）
1. **提出 ICE-NeRF 交互式颜色编辑框架**：仅通过预训练模型和用户单视图粗糙遮罩，选择性微调负责颜色计算的 MLP 最后三层中部分权重，在不到一分钟内完成高质量颜色重绘；与 PaletteNeRF/RecolorNeRF 的本质区别在于无需额外模块、数据或调色板分解，直接以权重优化为核心。
2. **提出基于激活场的正则化技术（AFR）**：通过计算目标区域和冻结区域在特定层输入/输出侧的平均激活值，构建惩罚矩阵对权重变化进行正则化，有效缓解隐式表示纠缠导致的非目标区域意外变色；与已有工作的本质区别在于首次将激活场概念引入 NeRF 权重优化，通过"高激活→允许变化/低激活→严格约束"的双向策略实现区域解耦。
3. **提出单遮罩多视角渲染技术（SMR）**：利用深度图将用户 2D 遮罩逆投影到 3D 空间构建点云，再结合可见性检测渲染多个视角的遮罩，仅需单视图遮罩即可获得多视角一致的训练信号；与已有工作的本质区别在于无需用户多视图标注，通过 DIBR 思想低成本模拟多视图监督。

## 方法详解
**整体流程**：用户输入预训练 NeRF 模型、单视图遮罩（$\mathbf{M}^{\mathrm{C}}$ 目标变色区域、$\mathbf{M}^{\mathrm{F}}$ 冻结保护区域）和目标颜色；冻结大部分权重，仅微调 MLP 最后三层中的部分层；结合 AFR 和 SMR 技术进行优化，在不到 100 次迭代内完成训练。

**AFR（Activation Field-based Regularization）**：
- 将体积渲染公式应用于激活场，计算射线上的加权激活值：
$$\mathbf{A}_{l}(\mathbf{r}(t)) = \sum_{i=1}^{N} \omega_{i} \cdot \mathbf{a}_{l}(\mathbf{x}_{i}, \mathbf{d})$$
- 对目标区域（C）和冻结区域（F）分别计算平均激活值：
$$\bar{\mathbf{a}}^{\mathrm{C}} = \text{Avg}(\text{Mask}(\mathbf{M}^{\mathrm{C}}, \mathbf{A})), \quad \bar{\mathbf{a}}^{\mathrm{F}} = \text{Avg}(\text{Mask}(\mathbf{M}^{\mathrm{F}}, \mathbf{A}))$$
- 在特定层的输入/输出两侧均计算上述值（$\bar{\mathbf{a}}^{\mathrm{C_{IN}}}$, $\bar{\mathbf{a}}^{\mathrm{C_{OUT}}}$, $\bar{\mathbf{a}}^{\mathrm{F_{IN}}}$, $\bar{\mathbf{a}}^{\mathrm{F_{OUT}}}$）
- 构建惩罚矩阵 R：
$$\mathbf{R}_{ij} = \text{Normalize}(\text{Rank}(\bar{\mathbf{a}}^{\mathrm{C_{IN}}}, \text{desc}))_j + \text{Normalize}(\text{Rank}(\bar{\mathbf{a}}^{\mathrm{C_{OUT}}}, \text{desc}))_i + \text{Normalize}(\text{Rank}(\bar{\mathbf{a}}^{\mathrm{F_{IN}}}, \text{asc}))_j + \text{Normalize}(\text{Rank}(\bar{\mathbf{a}}^{\mathrm{F_{OUT}}}, \text{asc}))_i$$
- AFR 损失函数：
$$L_{\mathrm{AFR}} = |\mathbf{W}_{l} - \mathbf{W}_{l}^{*}| \odot \mathbf{R}$$
- 核心思想：目标区域中高激活值的权重连接对颜色影响大，允许较大变化；冻结区域中高激活值的权重连接需严格约束。

**SMR（Single-mask Multi-view Rendering）**：
- 利用深度图将用户遮罩逆投影到 3D 空间：
$$\mathbf{D}(\mathbf{r}(t)) = \sum_{i=1}^{N} \omega_{i} \cdot t_{i}$$
- 构建点云后从不同视角渲染，并通过比较像素级深度判断可见性以处理遮挡关系
- 实现仅用单视图遮罩即可达到多视角一致的训练效果。

**优化策略**：
- 重绘损失：$L_{\mathrm{R}} = \|\text{Mask}(\mathbf{M}^{\mathrm{C}}, \mathbf{I}) - \mathbf{C}^{\mathrm{Target}}\|_2$
- 强度保持损失：$L_{\mathrm{I}} = \|\text{Int}(\text{Mask}(\mathbf{M}^{\mathrm{C}}, \mathbf{I})) - \text{Int}(\text{Mask}(\mathbf{M}^{\mathrm{C}}, \mathbf{I}^{*}))\|_2$
- 冻结损失：$L_{\mathrm{F}} = \|\text{Mask}(\mathbf{M}^{\mathrm{F}}, \mathbf{I}) - \text{Mask}(\mathbf{M}^{\mathrm{F}}, \mathbf{I}^{*})\|_2$
- 总损失：$L_{\mathrm{Total}} = L_{\mathrm{R}} + L_{\mathrm{I}} + L_{\mathrm{F}} + L_{\mathrm{AFR}}$
- 训练策略：N=10 次迭代同时学习重绘目标和冻结区域，M=90 次迭代仅学习冻结区域，交替进行。

## 实验与结果
**数据集**：NeRF Blender（Lego, Drums, Hotdog, Chair, Ship）、LLFF（Horns, Flower, Fortress）、Mip-NeRF360（Kitchen, Bonsai）

**基线方法**：PaletteNeRF、RecolorNeRF（均公平适配遮罩支持进行比较）

**主要结果**：
- **分解性能量化对比（MSE，背景区域颜色变化越小越好）**：
  - Horns 场景：Ours **0.0213** vs PaletteNeRF 0.0818 vs RecolorNeRF 0.1867
  - Fortress 场景：Ours **0.0010** vs PaletteNeRF 0.0013 vs RecolorNeRF 0.0013
  - Flower 场景：三者均为 0.0003
  - 平均：**Ours 0.0075** vs PaletteNeRF 0.0277 vs RecolorNeRF 0.0627
  - **提升幅度**：相比 PaletteNeRF 提升约 **3.7 倍**，相比 RecolorNeRF 提升约 **8.3 倍**
- **时间效率**：
  - ICE-NeRF：**25 秒**（至保存首帧）
  - RecolorNeRF：160 秒
  - PaletteNeRF：49 分钟
  - **提升幅度**：比 PaletteNeRF 快约 **117 倍**，比 RecolorNeRF 快约 **6.4 倍**
- **消融实验**：
  - AFR 有效抑制了遮罩粗糙时的非目标区域颜色变化
  - 强度损失保持了原始场景的光照强度
  - 即使遮罩极度稀疏（点、线、涂鸦）仍能完成有效重绘
- **多模型泛化**：成功应用于原始 NeRF、Instant-NGP、TensoRF 三种结构，证明方法通用性
- **最强结果**：Horns 场景中 ICE-NeRF 的 MSE 最低（0.0213），在多色相似物体分离任务上表现最优。

## 相关工作脉络
1. **NeRF 基础工作**[16]：Mildenhall 等人提出神经辐射场隐式 3D 表示，本文在其基础上进行颜色编辑扩展。
2. **PaletteNeRF**[14,30]：基于调色板的 NeRF 外观编辑方法，通过提取少量基础颜色加权求和实现重绘，无法处理相似颜色多物体选择，训练耗时 1–2 小时；本文方法通过权重优化实现更精确的区域分解和更快推理。
3. **RecolorNeRF**[10]：层分解辐射场方法，同样依赖调色板机制，在 Horns 等相似颜色场景下分解性能不足；本文通过 AFR 正则化克服此局限。
4. **风格迁移方法**[36]：ARF 等方法将参考图像风格迁移到整个 NeRF 场景，不局限于特定区域，定位为全局风格编辑而非局部精确编辑。
5. **文本/指令驱动编辑**：Instruct-NeRF2NeRF[11] 利用 InstructPix2Pix 根据文本指令修改 NeRF，Clip-NeRF[29] 支持文本和图像驱动；本文方法依赖用户遮罩，定位更精确的几何级别编辑。
6. **几何编辑方法**[20]：CageNeRF 等通过笼状结构修改场景几何；本文专注于颜色编辑，两者在编辑维度上互补。

## 局限性与未来方向
1. **360° 场景的意外颜色变化**：在 Mip-NeRF360 数据集上观察到部分区域的非期望颜色变化，推测是由于 360° 场景需要学习更广泛复杂的 3D 空间，当前方法对大规模开放场景的泛化能力有限。
2. **复杂纠缠对象的限制**：对于场景级复杂纠缠对象（如密集树枝），基于用户指定遮罩的方法实用性受限，可结合传统分割算法或现成分割模型（如 SAM）提升易用性。
3. **未来方向**：探索通过权重优化实现场景几何编辑；将方法扩展至动态 NeRF 的编辑应用。

## 研究启发与可借鉴点
1. **激活场概念的可迁移性**：将激活场与体积渲染结合分析 NeRF 权重-区域对应关系的思路，可迁移至 NeRF 的几何编辑、材质编辑等任务，作为权重选择与正则化的通用分析工具。
2. **基于激活排名构建惩罚矩阵的模式**：AFR 中"高激活→宽松约束/低激活→严格约束"的双向策略，可推广至其他神经隐式表示（如 SDF、Gaussian Splatting）的编辑任务中，作为解耦区域影响的正则化范式。
3. **单遮罩多视角渲染的 SMR 思想**：通过深度逆投影低成本模拟多视角监督的策略，可应用于仅需少量用户标注的多视角一致编辑任务（如 NeRF 补全、缺失区域重建）。
4. **部分权重微调而非全参数 Fine-tuning**：仅微调最后几层而非全参数优化的策略大幅降低计算成本，可推广至 NeRF 的其他编辑任务（如亮度、饱和度、纹理调整）。
5. **消融实验对遮罩稀疏度的系统性测试**：论文对点、线、曲线、涂鸦等多种遮罩类型的鲁棒性评估，为后续研究提供了可复用的评估基准和方法验证范式。

## 关键术语表
**NeRF (Neural Radiance Fields)**：神经辐射场，一种利用多层感知机学习 3D 场景密度和视角相关辐射率的隐式 3D 表示方法，通过体积渲染生成高质量新视角图像。
**体积渲染 (Volumetric Rendering)**：沿射线采样空间点并加权叠加其辐射率和密度，计算最终像素颜色的渲染技术，是 NeRF 的核心渲染机制。
**Activation Field (激活场)**：将模型各层神经元激活值视为 3D 空间函数，通过体积渲染获得激活分布，用于分析不同权重对特定空间区域的贡献度。
**DIBR (Depth-Image-Based Rendering)**：基于深度图像的渲染技术，通过深度图将 2D 内容重投影到 3D 空间并从新视角重新渲染，SMR 技术基于此实现多视角遮罩生成。
**MSE (Mean Squared Error)**：均方误差，本文用于量化背景区域颜色变化的指标，值越小表示分解性能（非目标区域颜色保持）越好。

## 可复现要素
- **数据集**：NeRF Blender 数据集、LLFF 数据集、Mip-NeRF360 数据集（均已公开）
- **代码开源**：论文未明确提及 ICE-NeRF 代码开源；使用了 ner
