---
title: "Vox-E-Text-guided-Voxel-Editing-of-3D-Objects"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Sella_Vox-E_Text-Guided_Voxel_Editing_of_3D_Objects_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:41:21"
field: "3D 生成与编辑"
keywords: ["3D editing", "text-guided generation", "voxel representation", "score distillation sampling", "cross-attention", "volumetric regularization"]
innovations: ["3D体素空间正则化实现文本引导编辑与输入结构保持的平衡", "基于2D cross-attention提升为3D概率网格的体素分割定位技术", "耦合双体素网格架构支持局部和全局几何/外观编辑"]
benchmarks: ["CLIP Similarity", "CLIP Direction Similarity", "FID"]
---

# 论文速读：Vox-E: Text-Guided Voxel Editing of 3D Objects

## 一句话总结
Vox-E 是一种基于文本提示的 3D 物体编辑方法，利用预训练 2D 扩散模型的 SDS 损失配合 novel 3D 体素正则化与 cross-attention 空间定位，实现对输入 3D 物体的局部或全局几何与外观编辑，同时保持原始物体的高保真度。

## 研究问题与动机
1. **现有 3D 编辑方法的局限**：现有工作大多仅支持外观修改（风格迁移、纹理编辑）或基于显式网格的几何编辑（需手动放置控制点），无法自由添加新结构或大幅调整几何形态。
2. **神经场表示的优化困难**：NeRF 等神经场方法将结构与外观耦合在 2D 投影中，当同时优化文本引导目标和输入保持目标时，易出现冲突，难以平衡两个目标。
3. **2D 图像编辑无法保证 3D 一致性**：直接在多视角图像上应用 2D 编辑方法会产生视角间不一致的编辑结果，无法保证跨视角的几何一致性。
4. **无监督/弱监督 3D 编辑的需求**：非专家用户希望仅通过自然语言描述即可编辑 3D 物体，降低 3D 内容创作的门槛。

## 核心贡献（创新点）
1. **耦合体素表示与 3D 体积正则化**：提出双网格耦合架构，在 3D 空间中对密度特征施加相关性正则化损失 $\mathcal{L}_{reg3D}$，使编辑网格在遵循文本提示的同时保持与输入物体的几何结构一致，区别于以往仅在图像空间做 L2/L1 正则化的方法。
2. **基于 3D Cross-Attention 的体素分割技术**：将 2D cross-attention map 提升为 3D 概率网格，再通过能量最小化（unary + smoothness term + graph cuts）得到二值体素掩码，精确界定编辑区域与非编辑区域，实现局部编辑的空间定位。
3. **支持多种编辑类型（局部/全局、几何/外观）**：实验表明方法可生成此前工作无法实现的编辑（如给动物加配饰、整体风格转换），在 CLIP 相似度指标上显著优于基线。

## 方法详解
**体素表示（Section 3.1）**：
- 采用基于 DVGO/ReLU Fields 的体素网格 $G$，每个体素存储 4D 特征向量：1 通道用于密度（经 ReLU 激活），3 通道用于颜色（经 Sigmoid 映射到 RGB）。
- 不使用 positional encoding，直接对网格采样插值；不建模 view-dependent appearance，避免扩散模型引导下的伪影。
- 通过 volumetric rendering 从多视角图像优化初始网格 $G_i$，使用 L1 渲染损失。

**文本引导编辑（Section 3.2）**：
- 从 $G_i$ 初始化编辑网格 $G_e$，联合优化两个目标：
  - **SDS 损失**（生成性目标）：基于 Latent Diffusion Model，在随机时间步 $t$ 向渲染图像添加噪声 $\epsilon_t$，计算 score distillation gradient：
    $$\nabla_x \mathcal{L}_{SDS} = w(t)(\epsilon_t - \epsilon_\phi(x_t, t, s))$$
    其中 $s$ 为目标文本 prompt，采用 annealed SDS（逐步降低最大时间步）以先塑形后细化。
  - **3D 体素正则化损失**：
    $$\mathcal{L}_{reg3D} = 1 - \frac{Cov(f_i^\sigma, f_e^\sigma)}{\sqrt{Var(f_i^\sigma)Var(f_e^\sigma)}}$$
    鼓励编辑网格与初始网格的密度特征保持相关性，解耦结构与外观。

**空间细化（Section 3.3）**：
- 利用 2D cross-attention map 监督训练 3D cross-attention 网格 $A_e$（编辑 token）和 $A_{obj}$（物体 token）。
- 将标签概率定义为两个 attention grid 的 element-wise softmax。
- 构建能量函数 $E = \sum_p U(p) + \lambda \sum_{p,q} w_{pq} V(p,q)$，其中 smoothness weight $w_{pq} = \exp\left(\frac{-(c_p - c_q)^2}{2\sigma^2}\right)$，通过 graph cuts 求解二值体素掩码 $M$。
- 最终输出网格：$G_r = M \cdot G_e + (1 - M) \cdot G_i$。

## 实验与结果
**数据集**：自建合成数据集，从互联网获取 mesh，在 Blender 中以 100 视角渲染，搭配局部编辑 prompt（如 "wearing sunglasses"）和全局编辑 prompt（如 "a yarn doll of..."）。

**评估指标**：
- $\mathrm{CLIP}_{Sim}$：输出图像与目标文本的语义相似度（越高越好）
- $\mathrm{CLIP}_{Dir}$：编辑方向与 prompt 方向的余弦相似度
- $\mathrm{FID}_{Input}$ / $\mathrm{FID}_{Rec}$：输出与输入、输出与初始重建的视觉差异

**定量结果（Table 1）**：
| 编辑类型 | 方法 | $\mathrm{CLIP}_{Sim}$ | $\mathrm{CLIP}_{Dir}$ |
|---------|------|---------------------|---------------------|
| 局部编辑 | Ours | **0.36** | **0.07** |
| 局部编辑 | DFF+CN | 0.34* | 0.05* |
| 局部编辑 | Text2Mesh | 0.34* | 0.03* |
| 局部编辑 | Latent-NeRF (Sketch/Paint) | 0.36*/0.32 | 0.08*/0.01 |
| 全局编辑 | Ours | **0.34** | **0.02** |
| 全局编辑 | Latent-NeRF (Sketch/Paint) | 0.30/0.31 | 0.01/0.01 |

- Vox-E 在 CLIP 相似度上全面优于所有 3D 编辑基线。
- 注：Text2Mesh 和 DFF+CN 显式优化 CLIP loss，其指标不完全具可比性。

**消融实验（Table 2）**：
| $\mathcal{L}_{reg3D}$ | SR | $\mathrm{CLIP}_{Sim}$ | $\mathrm{CLIP}_{Dir}$ | $\mathrm{FID}_{Rec}$ | $\mathrm{FID}_{Input}$ |
|---|---|---|---|---|---|
| × | × | 0.29 | 0.05 | 367.53 | 384.55 |
| √ | × | 0.37 | 0.08 | 240.37 | 288.26 |
| √ | √ | 0.36 | 0.06 | **119.44** | **236.32** |

- 去掉 $\mathcal{L}_{reg3D}$ 导致严重噪点和编辑失败；去掉 SR 模块导致输入保真度下降（FID 升高）。

**运行时**：单张 RTX A5000 GPU，编辑阶段约 50 分钟，细化阶段约 15 分钟。

**真实场景**：在 360° Real Scenes 数据集上同样有效，可适配 DVGO 或 ReLU Fields 底层表示。

## 相关工作脉络
1. **DreamFusion / Magic3D**：无条件文本驱动 3D 生成，使用 SDS 损失优化辐射场；本文将其思想迁移至条件编辑场景，通过 3D 正则化保持输入一致性。
2. **Text2Mesh**：基于 CLIP 的风格化网格编辑，仅支持法向位移，无法添加新结构；本文可生成显著几何变化。
3. **Latent-NeRF**：支持草图/颜色引导的 3D 生成，SketchShape 可改几何但无法保持输入几何；本文在保持输入的同时实现编辑。
4. **DFF + CLIP-NeRF**：将 2D 特征蒸馏到 3D 场再局部编辑；本文不依赖 2D 特征蒸馏，直接通过 cross-attention 定位编辑区域。
5. **InstructPix2Pix / SDEdit**（2D 编辑）：直接编辑多视角图像产生 3D 不一致结果；本文直接优化 3D 体素网格保证视角一致性。
6. **Instruct-NeRF2NeRF**（同期工作）：迭代编辑 2D 投影再重建 3D；本文直接优化 3D 表示，无需迭代重渲染。

## 局限性与未来方向
1. **属性绑定错误**：模型可能将属性绑定到错误的部位（如马鼻子变成猪鼻子），这是扩散模型常见问题。
2. **跨视角不一致**：优化多个视角时，同一物体在不同视角可能被编辑到不同位置（如独角兽长两个角）。
3. **过度正则化**：3D 正则化可能导致某些区域未被充分编辑（如地毯只出现在马身而非下方）。
4. **体素表示分辨率限制**：复杂真实场景的编辑质量受限于体素网格的表达能力；可结合 Mip-NeRF 360 的场景收缩等技术改进。
5. **未来方向**：接入更大规模/更高质量的扩散模型（如 SDXL）；结合更精细的体素表示；探索更多编辑类型（动画、材质物理属性等）。

## 研究启发与可借鉴点
1. **3D 空间正则化优于 2D 图像空间正则化**：在 3D 体素空间中直接约束结构相关性，能有效解耦几何与外观，这一思路可迁移到其他 3D 生成/编辑任务（如 3D 补全、3D 变形）。
2. **2D Attention → 3D Attention 的提升策略**：利用 2D cross-attention map 监督训练 3D 概率网格，再通过图割细化，解决了 2D attention 视角不一致的问题，该策略可用于 3D 分割、3D 属性定位等下游任务。
3. **耦合双网格设计**：通过 $\mathcal{L}_{reg3D}$ 耦合初始网格与编辑网格，为"生成-保持"双重目标的优化提供了简洁有效的范式，可推广到其他条件生成任务。
4. **不含 view-dependent 的显式体素表示**：放弃 view-dependent appearance 避免了扩散模型引导下的视角伪影，在文本引导编辑场景中是务实的选择，值得在类似任务中验证。
5. **annealed SDS 策略**：逐步降低扩散时间步以先粗后细地生成编辑，可有效提升输出质量，可作为 SDS-based 方法的通用技巧。

## 关键术语表
**SDS (Score Distillation Sampling)**：利用预训练 2D 扩散模型的对生成图像的噪声预测梯度，将 2D 生成知识蒸馏到 3D 表示优化中的技术。

**ReLU Fields**：基于体素网格的 3D 场景表示，每个体素存储特征向量，通过 ReLU 和 Sigmoid 激活函数分别解码密度和颜色，无需神经网络即可渲染。

**Cross-Attention Map**：在文本条件扩散模型中，解码器的 cross-attention 层输出反映了图像各区域与文本 token 的空间关联概率分布。

**Volumetric Regularization ($\mathcal{L}_{reg3D}$)**：在 3D 体素空间中计算初始网格与编辑网格密度特征的相关性损失，鼓励编辑前后几何结构保持一致。

**Energy Minimization Segmentation**：通过构建包含数据项和光滑项的能量函数，利用 graph cuts 算法求解最优二值体素分割掩码。

**CLIP Direction Similarity ($\mathrm{CLIP}_{Dir}$)**：衡量编辑前后图像在 CLIP 空间中变化方向与 prompt 变化方向之间余弦相似度的指标。

## 可复现要素
- **数据集**：自建合成数据集（互联网免费 mesh + Blender 100 视角渲染），**未公开**；使用了 360° Real Scenes（Mildenhall et al.）作为真实场景测试集。
- **代码/权重**：论文未明确声明开源状态（需进一步确认 project page）。
- **关键超参**：$\sigma = 0.1$（smoothness weight 计算）、$\lambda = 5$（能量函数中光滑项权重）；annealed SDS 的时间步调度策略。
- **训练硬件**：单张 RTX A5000 GPU（24GB VRAM）。
- **训练时长**：编辑阶段约 50 分钟，细化阶段约 15 分钟。
