---
title: "Texture-Generation-on-3D-Meshes-with-Point-UV-Diffusion"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Texture_Generation_on_3D_Meshes_with_Point-UV_Diffusion_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:33:25"
field: "3D 几何与纹理生成"
keywords: ["texture generation", "diffusion model", "UV mapping", "3D mesh", "coarse-to-fine", "style guidance"]
innovations: ["首个面向 mesh 纹理生成的专用点-UV 两阶段扩散模型框架", "引入 PCA+K-means 风格引导解决 ShapeNet 颜色分布偏差", "Hybrid Condition + Condition-Truncated Sampling 弥合级联训练/推理条件失配"]
benchmarks: ["ShapeNet chair/table/car/bench", "FID", "KID", "LPIPS"]
---

# 论文速读：Texture-Generation-on-3D-Meshes-with-Point-UV-Diffusion

## 一句话总结
本文提出 **Point-UV Diffusion**，一种"由粗到细"的两阶段扩散框架：先用 3D 点扩散模型生成全局一致的低频颜色（加风格引导解决颜色偏差），再将粗糙纹理图投影到 UV 空间，通过带混合条件（hybrid condition）的 2D UV 扩散模型生成高保真纹理；该方法可处理任意拓扑 mesh，在 ShapeNet 上取得 SOTA。

## 研究问题与动机
1. **3D Mesh 纹理生成的质量与效率矛盾**：Voxel/Point-Cloud 等方法受显存和复杂度限制只能生成低分辨率纹理；Texture Fields 虽能出高分辨率，但结果过平滑。
2. **UV 展开直接结合 2D 扩散模型的断裂问题**：UV 映射将连续 3D 表面切割成孤立 2D patch，导致 3D 邻接关系丢失，直接训 2D 扩散会产生明显拼接不连贯（Figure 2(c)）。
3. **已有方法无法兼顾拓扑任意性与几何细节保留**：部分 UV-based 方法依赖球面参数化或部件分割，仅适用于低 genus；Texturify 使用四面体参数化会破坏原始 mesh 的几何细节（Figure 2(b)）。
4. **扩散模型在纹理生成任务中尚未被充分探索**：现有工作多用 GAN/VAE；首个直接面向 mesh 纹理生成的专用扩散模型仍未出现。

## 核心贡献（创新点）
1. **提出首个专为 mesh 纹理生成训练的扩散模型框架（Point-UV Diffusion）**：两阶段"由粗到细"策略，兼顾 2D 表示的高效性与 3D 一致性；区别于以往仅用 GAN/VAE 或隐式场（Texture Fields）的方法。
2. **设计 Point Diffusion + Style Guidance 解决低频颜色生成与颜色分布偏差**：先对 mesh 上 FPS 采样的 4096 个点做 3D 扩散颜色预测，并以 K-means 聚类的风格标签作为额外条件；区别于纯无条件扩散，缓解 ShapeNet 数据集中颜色偏向（如 chair 偏白/木色）导致的多样性不足。
3. **UV Diffusion + Hybrid Condition 机制**：fine stage 将粗纹理图 $\mathbf{x}_{\mathrm{coarse}}$ 与 smooth 版本 $\mathbf{x}_{\mathrm{smooth}}$ 以概率 $p_{\mathrm{hybrid}}=0.3$ 混合输入；并探索 condition-truncated sampling（$t_c=0.4$）；区别于单条件级联或纯 coarse 引导的方式，弥合 train/test 条件分布失配。
4. **支持无条件/文本/单视图图像三类条件生成，且推理速度快**：文本引导用 CLIP 嵌入接 MLP；对比 Text2Mesh 需 10 min/实例，本方法仅 30 s，且能输出低分辨率 mesh 上的高频细节。

## 方法详解
- **Coarse Stage（Point Diffusion）**：
  - 在 mesh 表面做 Farthest Point Sampling（FPS）得 $K=4096$ 个点，点集由坐标 $\mathbf{z}_{\mathrm{coold}}$ 和颜色 $\mathbf{z}_0$ 定义。
  - 构造 shape map $\mathbf{x}_{\mathrm{shape}} = [\mathbf{x}_{\mathrm{normal}}, \mathbf{x}_{\mathrm{mask}}, \mathbf{x}_{\mathrm{coor d}}]$（法向量+掩码+坐标）。
  - 轻量 shape encoder $E_\phi$ 提取全局 embedding $f_g = E_\phi(\mathbf{x}_{\mathrm{shape}})$；denoising network $G^1_{\theta_1}$（基于 PVCNN）输出 $\hat{\mathbf{z}}_0 = G^1_{\theta_1}([\mathbf{z}_{\mathrm{coor d}}, \mathbf{z}_t, f_g, t])$。
  - **Style Guidance**：将每个 $\mathbf{z}_0$ reshape 后做 PCA 降维、K-means 聚类得到风格标签 $z_{\mathrm{style}}$，作为额外条件：$\hat{\mathbf{z}}_0 = G^1_{\theta_1}([\mathbf{z}_{\mathrm{coor d}}, \mathbf{z}_t, f_g, t, z_{\mathrm{style}}])$。
  - 投影到 UV 空间后，以 3D 坐标为基准的 KNN 插值填充未着色像素，得到粗纹理图 $\mathbf{x}_{\mathrm{coarse}}$。
- **Fine Stage（UV Diffusion）**：
  - 2D U-Net $G^2_{\theta_2}$ + self-attention 预测 $\hat{\mathbf{x}}_0 = G^2_{\theta_2}([\mathbf{x}_{\mathrm{shape}}, \mathbf{x}_t, \mathbf{x}_{\mathrm{coarse}}, t])$。
  - **Hybrid Condition**：对 mask map 做四连通分割后在每个 region 内做平均颜色 pooling，得到平滑版本 $\mathbf{x}_{\mathrm{smooth}}$；训练时以 $p_{\mathrm{hybrid}}=0.3$ 概率用 $\mathbf{x}_{\mathrm{smooth}}$ 替代 $\mathbf{x}_{\mathrm{coarse}}$ 作条件，使 fine stage 在 inference 时也能适应退化条件。
  - **Condition-Truncated Sampling**：推理时前 $t_c=40\%$ 步同时使用两条件图生成低频，后半段仅用 $\mathbf{x}_{\mathrm{smooth}}$ 专注高频细节。
  - 额外引入渲染损失：随机 4 视角渲染 $1024\times1024$ 图像，裁剪 $224\times224$ patch 计算 $L_1$ loss。
- **训练设置**：cosine noise schedule（0.0001→0.02，1024 steps）；预测 clean signal 而非 noise；finestage 采用 noise scaling [7]。

## 实验与结果
- **数据集**：ShapeNet（chair/table/car/bench 四类），UV 展开工具 xatlas [46]。
- **基线**：Texture Fields [29]、Texturify [38]、PVD-Tex（PVD [49] 改造为 RGB 学习）。
- **评估指标**：FID [15]、KID [2]（渲染 4 视角 $512\times512$ 图像评测）；多样性评测用 LPIPS [48] + 用户偏好。
- **主表（Table 1）核心数字**：
  | 方法 | Chair FID | Chair KID | Car FID | Table FID | Bench FID |
  |---|---|---|---|---|---|
  | Texture Fields | 24.24 | 1.07 | 156.38 | 68.96 | 62.71 |
  | Texturify | 27.80 | 1.32 | 73.16 | — | — |
  | PVD-Tex | 15.52 | 0.62 | 59.47 | 16.12 | 28.94 |
  | **Ours** | **9.88** | **0.22** | **26.89** | **9.63** | **23.09** |
  Ours 在四类上均显著领先：Chair FID 9.88 vs PVD-Tex 15.52（↓36%），Car FID 26.89 vs PVD-Tex 59.47（↓55%）。
- **消融（Table 2）**：完整模型 Chair FID=9.88/KID=0.22；w/o coarse stage 为 15.11/0.49；w/o fine stage 为 17.88/0.76；仅 coarse 条件为 14.93/0.56；$p_{\mathrm{hybrid}}=1.0$ 为 15.25/0.59。
- **多样性（Table 3）**：LPIPS Ours=0.083 vs PVD-Tex=0.029、Texture Fields=0.005；用户偏好 Ours 获 49.8% 首选。
- **条件生成**：文本/图像引导均可行；推理耗时约 30 s/实例（vs Text2Mesh 10 min）。

## 相关工作脉络
1. **Texture Fields [29]**：隐式 texture field 表征，本论文认为其生成结果"过平滑"（Figure 2(a)），且仅用 GAN/VAE；本文用扩散模型 + UV 两阶段弥补细节。
2. **Texturify [38]**：四面体 mesh 卷积生成纹理，但参数化会破坏原始几何细节（Figure 2(b)）；本文保留原始 mesh 拓扑，不做四面体化。
3. **PVD [49]**（Point-Voxel Diffusion）：原为点云生成设计，本文将其扩展为 PVD-Tex 作为 baseline，证明专用 UV 扩散优于直接在点空间学 RGB。
4. **Text2Mesh [24]** / **DreamFusion [31]** / **Magic3D [21]**：test-time optimization 或 score distillation 路线，耗时长且需修改几何；本文面向给定 mesh 直接生成颜色，无需变形几何。
5. **CLIP-mesh [26]** / **GramGAN [32]**：依赖 2D exemplar 或 CLIP 嵌入；本文除支持文本/图像条件外，强调无条件生成的 SOTA 能力。
6. **Style-based GAN [20]** / **Cascaded Diffusion [17]**：风格引导思想受 StyleGAN 启发；hybrid condition 借鉴 cascaded diffusion 的 blur augmentation 策略。

## 局限性与未来方向
1. **受限于 3D 数据集规模与多样性**：无法达到 2D 图像合成那样丰富的深度效果；需更大更 Diversified 的 3D texture 数据集。
2. **对 UV 映射质量敏感**：UV 展开产生过多碎片化切割时（如 car 类别，Figure 15），会产生不连贯 artifacts；更好的 UV 参数化工具可缓解。
3. **style guidance 聚类不平衡**（Figure 5）：不同风格 cluster 样本量差异大，可能影响无偏合成；需更鲁棒的多模态风格建模。
4. **推理速度仍慢于 GAN**：虽然比 Text2Mesh 快很多，但扩散模型本身的采样开销仍可进一步优化（如 latent diffusion、加速采样器）。

## 研究启发与可借鉴点
1. **"粗-细两阶段 + 混合条件"范式可迁移**：hybrid condition + condition-truncated sampling 的设计思路可推广至其他 3D→2D 投影生成任务（法线贴图、法向 AO、material map 等）。
2. **风格引导解决数据分布偏差**：PCA + K-means 聚类作 style token 的思路简单有效，可复用至其他存在类别颜色偏置的 3D 生成任务。
3. **渲染损失辅助 UV 扩散**：fine stage 加入多视角渲染 $L_1$ loss 提升几何一致性，可作为 3D 纹理任务的通用正则项。
4. **任意 genus 的通用性**：方法兼容任意拓扑 mesh，为后续扩展至 Human/Animal 等非刚性 mesh 纹理生成提供基座。
5. **条件生成扩展便捷**：CLIP embedding 经 MLP 注入 diffusion condition 的架构可快速迁移到 text-to-texture、image-to-texture、甚至 multi-view conditioned 场景。

## 关键术语表
**Point-UV Diffusion**：本文提出的两阶段纹理生成框架，先点扩散再 UV 扩散。
**Farthest Point Sampling (FPS)**：从 mesh 表面均匀采样 K 个点的采样策略。
**Style Guidance**：利用 PCA+K-means 对颜色分布聚类，以风格标签作为额外条件提升多样性。
**Hybrid Condition**：fine stage 训练中以概率 $p_{\mathrm{hybrid}}$ 随机用平滑 version $\mathbf{x}_{\mathrm{smooth}}$ 替换粗纹理图 $\mathbf{x}_{\mathrm{coarse}}$ 作条件。
**Condition-Truncated Sampling**：推理时在采样前半段同时用两条件图、后半段只用 smooth 图的策略。
**Texture Fields [29]**：基于隐式函数学习的 3D 纹理表征方法，本文主要对比基线。
**PVD-Tex**：将 PVD [49] 点云扩散模型改造为学习 RGB 颜色的 baseline。
**xatlas [46]**：开源 UV 展开工具，本文用于预处理 dataset 生成 UV map。

## 可复现要素
- **数据集**：ShapeNet [4]（chair/table/car/bench 四类），公开。
- **代码**：已开源，https://cvmilab.github.io/Point-UV-Diffusion。
- **权重**：论文未单独声明权重下载链接，代码仓库内应包含。
- **关键超参**：$K=4096$ 采样点数；UV 纹理分辨率 $512\times512$；noise schedule cosine（0.0001→0.02，1024 steps）；$p_{\mathrm{hybrid}}=0.3$；condition-truncated $t_c=0.4$；渲染视角数 4，patch 大小 $224\times224$。
