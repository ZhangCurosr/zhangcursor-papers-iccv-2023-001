---
title: "EverLight-Indoor-Outdoor-Editable-HDR-Lighting-Estimation"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Dastjerdi_EverLight_Indoor-Outdoor_Editable_HDR_Lighting_Estimation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:14:19"
---

# 论文速读：EverLight-Indoor-Outdoor-Editable-HDR-Lighting-Estimation

## 一句话总结
EverLight提出了一种跨室内/室外的统一HDR光照估计框架，通过 novel 的照明共调制机制将可编辑的参数化球形高斯光源注入360° GAN生成器，在单次前向传播中同步输出可用于渲染引擎的高质量HDR环境贴图与直观可改的光照参数，显著优于现有域专用方法。

## 研究问题与动机
- **域割裂严重**：现有光照估计方法通常针对室内或室外单独设计，缺乏能够无缝处理两类场景的统一架构。
- **表现力与可编辑性难以兼得**：参数化模型（如球形高斯）能准确建模阴影且易于编辑，但缺乏真实反射纹理；GAN-based方法能生成高保真环境贴图，但输出不可编辑或需耗时优化。
- **编辑效率低下**：支持编辑的方法（如StyleLight）依赖昂贵的GAN inversion迭代优化，难以满足AR/虚拟资产合成等实时应用需求。
- **HDR信息缺失**：多数外推方法仅生成LDR全景图，无法提供驱动物理渲染所需的高动态范围辐射度信息。

## 核心贡献（创新点）
1. **室内室外统一的HDR光照估计框架**：单一模型直接接受任意室内或室外单图输入，打破传统按场景域划分专家模型的设计范式。
2. **可编辑照明共调制（Editable Lighting Co-modulation）机制**：将用户编辑后的参数化光源重新渲染为全景图，通过独立编码器注入StyleGAN的共调制风格向量，使背景纹理生成过程与编辑后的光照条件严格耦合。
3. **参数化光源与稠密环境贴图的联合输出**：同时提供可用于渲染引擎的360° HDR环境贴图（保障反射质量）与可交互编辑的球形高斯参数集（保障阴影与光照控制），二者在同一前向推理中协同生成。
4. **毫秒级高效编辑推理**：摒弃GAN inversion，仅凭单次网络前向传递完成光照估计与编辑，推理速度较StyleLight提升约800倍（0.07s vs 58s）。

## 方法详解
- **坐标映射与全景输入**：根据已知相机参数（FOV、俯仰角、滚动角）将输入图像 $I$ 通过针孔模型投影为等距柱状全景图 $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$（$W=2H$），作为后续所有网络模块的统一输入。
- **光照预测网络 $\mathcal{L}$**：采用带fixup初始化的UNet结构，输入 $\mathbf{X}$ 后输出HDR光照贴图 $\hat{\mathbf{E}}$。训练时在log域计算损失并使用余弦模糊滤波器稳定收敛。
- **球形高斯参数拟合**：对 $\hat{\mathbf{E}}$ 进行阈值分割（98.5百分位）与连通分量提取，以各分量质心初始化位置 $\xi_k$、最大强度初始化 $\mathbf{c}_k$，固定带宽初值 $\sigma=0.45$。通过SGD最小化L2重建误差（公式4）优化参数集 $\hat{\mathbf{p}}=\{\mathbf{c}_k, \xi_k, \sigma_k\}_{k=1}^K$，并采用非极大值抑制融合重叠光斑。
- **用户编辑与重渲染**：用户修改 $\hat{\mathbf{p}}$ 得到 $\hat{\mathbf{p}}_e$ 后，按公式3渲染回编辑全景图 $\hat{\mathbf{E}}_e$，送入光照编码器 $\mathcal{E}_l$。
- **照明共调制生成**：在原有 co-modulation 公式 $\mathbf{w}' = A(\mathcal{E}_i(\mathbf{X}), \mathcal{M}(\mathbf{z}))$ 基础上扩展为 $\mathbf{w}' = A(\mathcal{E}_i(\mathbf{X}), \mathcal{M}(\mathbf{z}), \mathcal{E}_l(\hat{\mathbf{E}}_e))$，将条件特征、随机风格与编辑光照特征融合后驱动生成器 $\mathcal{G}$。最终将观测区域 $\mathbf{X}$ 与生成器输出 $\hat{\mathbf{Y}}'$ 合成，得到完整HDR全景图。

## 实验与结果
- **数据集**：训练使用360cities购买的360°全景图（239,064/1,000/1,000划分），并用Zhang et al. [59]方法扩展为HDR。室内评测基于Laval Indoor HDR Dataset（224张全景图抽提2,240张LDR图像）；室外评测基于Cheng et al. [5]户外集（839张全景图按0°/120°/240°抽
