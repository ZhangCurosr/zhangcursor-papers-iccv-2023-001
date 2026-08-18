---
title: "Rethinking-Amodal-Video-Segmentation-from-Learning-Supervise"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Fan_Rethinking_Amodal_Video_Segmentation_from_Learning_Supervised_Signals_with_Object-centric_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:28:31"
field: "视频无模分割"
keywords: ["amodal segmentation", "video segmentation", "object-centric learning", "BEV", "occlusion completion"]
innovations: ["首次将视频无模分割转为监督学习，利用VOS掩码作为监督信号", "提出BEV翻译模块引入3D无遮挡视角信息", "设计基于object slots的多视角融合时序模块，替代光流对齐"]
benchmarks: ["Movi-B", "Movi-D", "KITTI"]
---

# 论文速读：Rethinking-Amodal-Video-Segmentation-from-Learning-Supervised

## 一句话总结
本文首次将视频无模分割任务从自监督设定转向监督设定，提出**EoRaS**框架，通过结合**形状先验**（监督信号）与**视角先验**（多视角特征融合 + BEV 3D信息），利用基于object slots的多视角融合模块实现遮挡物体的完整掩码预测，在合成与真实数据集上均达到SOTA。

## 研究问题与动机
- **现有方法局限**：视频无模分割的近期工作（如SaVos）依赖光流进行帧间信息对齐，在相机运动场景下会产生变形信号；图像级方法则过度依赖形状先验，泛化能力差。
- **监督信号的缺失**：视频对象分割（VOS）已能产生高精度对象掩码，但其作为监督信号的价值尚未被充分利用于无模分割任务。
- **多视角信息的价值**：被遮挡物体的一部分可能在其他帧中出现，不同视角的特征可有效辅助当前帧的完整形状推断。
- **BEV空间的潜力**：BEV（鸟瞰图）空间中几乎不存在遮挡（除非物体堆叠），可作为额外的"无遮挡视角"补充正面视角的特征。

## 核心贡献（创新点）
1. **首次将视频无模分割 formulated 为监督学习任务**，与已有自监督方法形成本质区别，摆脱了对光流的依赖。
2. **提出基于多视角融合层的时序编码器（Multi-view Fusion Layer）**，利用object slots作为信息容器，通过非递归的attention机制高效融合不同视角特征，区别于传统ConvGRU/BiLSTM的高计算成本。
3. **首次在无模分割任务中引入BEV特征翻译模块**，通过相机内参将前视特征投影到BEV空间，引入3D几何信息，弥补纯2D方法的不足。
4. **设计了形状提供者（SP）与形状接收者（SR）的交互范式**，在ObjAttention中实现视图间形状信息的定向传递与融合。

## 方法详解
**整体架构**由四个模块组成：

1. **特征编码模块**：使用预训练的FPN50提取输入帧的前视特征 $f_t^k$。

2. **BEV翻译网络**：
   - 在相机坐标系中构建3D体素特征 $V_{3D} \in \mathbb{R}^{c \times m \times n \times h}$
   - 通过内参矩阵K将3D点投影到图像平面：$\lambda u, \lambda v, \lambda)^T = K(x, y, z)^T$
   - 双线性插值获取对应特征值
   - 经轻量CNN压缩高度维度得到BEV特征 $b_t^k$

3. **基于多视角融合层的时序编码器**：
   - 初始化可学习的object slots $S_0 \in \mathbb{R}^{n_s \times d}$
   - 每个ObjAttention层包含自注意力、交叉注意力和FFN：
     - $\hat{SR} = SR + \text{Attention}(SR, SR, SR)$
     - $\widetilde{SR} = \hat{SR} + \text{Attention}(SP, \hat{SR}, SP)$
     - $\text{output} = \text{MLP}(\widetilde{SR})$
   - 信息流：$S_{t-1} \xrightarrow{\text{BEV特征}} S'_t \xrightarrow{\text{前视特征}} S_t \xrightarrow{\text{更新前视特征}} \hat{f}_t^k$
   - 双向预测解决冷启动问题

4. **反卷积网络**：对更新后的前视特征 $\hat{f}_t^k$ 进行全掩码 $\hat{M}_t^k$ 和可见掩码 $\hat{V}_t^k$ 的联合预测。

**损失函数**：
$$\mathcal{L} = \mathcal{L}_{full} + \lambda \cdot \mathcal{L}_{vis} = \sum \text{Focal}(\hat{M}, M) + \lambda \sum \text{Focal}(\hat{V}, V)$$

## 实验与结果
**数据集**：
- **Movi-B**：合成数据，CLEVR规则形状物体，较简单遮挡
- **Movi-D**：合成数据，Google Scanned Objects真实形状，更复杂遮挡
- **KITTI**：真实自动驾驶数据集，弱监督场景，仅标注car类别

**基线方法**：VM、Convex、PCNET、AISFormer、SaVos-sup.（修改为监督版本）、BiLSTM

**主要结果**（Table 1）：

| 数据集 | 方法 | mIoU_full | mIoU_occ |
|--------|------|-----------|----------|
| Movi-B | SaVos-sup. | 77.93 | 46.21 |
| Movi-B | **EoRaS** | **79.22** | **47.89** |
| Movi-D | SaVos-sup. | 68.43 | 36.00 |
| Movi-D | **EoRaS** | **69.44** | **36.96** |
| KITTI | SaVos-sup. | 86.68 | 49.95 |
| KITTI | **EoRaS** | **87.07** | **52.00** |

**提升幅度**：
- Movi-B：mIoU_occ提升**14.28%** vs SaVos-sup.
- Movi-D：mIoU_occ提升**14.32%** vs SaVos-sup.
- KITTI：mIoU_occ提升**~15%** vs SaVos-sup.
- 全面超越图像级SOTA方法AISFormer

**消融实验**（Table 2）：
- 时序模块带来~2.3%提升
- BEV模块带来~1.06%提升
- 双向预测带来~1.38%提升

**鲁棒性分析**：slot数量在8-256范围内变化时性能几乎不变（Table 3）；可见掩码损失权重λ的选择不敏感（Table 4）。

**开放集泛化**（Table 5）：在Movi-B预训练后直接迁移到Movi-D和KITTI，EoRaS优于AISFormer约2%。

## 相关工作脉络
1. **图像级无模分割**（[22, 30, 32]）：依赖形状先验（多层编码、VAE、形状码本等），在复杂真实场景中分布偏移导致性能受限。
2. **SaVos [35]**：首个视频无模分割工作，利用自监督光流对齐帧间信息；本文在其监督版本基础上对比，指出光流在相机运动下的变形问题。
3. **Object-centric learning**（[3, 19, 25]）：Slot Attention用于无监督场景分解；本文借鉴其监督版本思路，将slots作为信息容器。
4. **BEV特征生成**（[6, 17, 23, 24, 26, 27]）：自动驾驶领域广泛使用；本文首次将其引入无模分割任务，作为补充视角。
5. **VOS监督信号**（[6, 13, 21]）：高精度视频对象分割掩码可作为形状先验；本文首次系统性利用这一信号。

## 局限性与未来方向
- **BEV生成的假设**：依赖相机内参和水平放置的相机假设，在非标准配置下可能受限。
- **slot数量虽不敏感但需预设**：虽然实验显示8-256范围内性能稳定，但实际应用中需根据场景复杂度选择。
- **真实数据的标注依赖**：KITTI使用弱监督设置，仅car类别有标注，扩展到其他类别需更多标注。
- **未来方向**：可探索更复杂的3D几何建模、端到端联合优化BEV生成与分割、扩展到多物体交互场景。

## 研究启发与可借鉴点
1. **监督信号引入的范式**：将VOS的高精度掩码作为监督信号，为自监督→监督的范式转换提供了成功范例，可迁移到其他遮挡推理任务。
2. **BEV作为"无遮挡视角"**：利用BEV空间遮挡少的特性补充正面视角，这一思路可推广至图像级无模分割或多视角补全任务。
3. **ObjAttention的SP/SR设计**：形状提供者/接收者的定向信息流设计比双向对称attention更高效，可借鉴于其他特征融合场景。
4. **non-recurrent时序建模**：用attention替代ConvGRU/BiLSTM处理视频时序信息，在保持效果的同时提升效率，适用于长视频处理。

## 关键术语表
- **Amodal Segmentation（无模分割）**：预测被遮挡物体的完整掩码，而非仅可见部分。
- **Object-centric Representation（对象中心表示）**：将场景分解为独立对象实体及其属性的表示方式。
- **Object Slots（对象槽）**：可学习的query向量，作为对象信息的容器，通过attention机制与特征交互。
- **BEV（Bird's-Eye View，鸟瞰图）**：从正上方俯视场景的投影视角，常用于自动驾驶感知。
- **Shape Prior（形状先验）**：利用物体的已知形状结构引导补全被遮挡部分。
- **View Prior（视角先验）**：利用不同视角观察到物体的互补信息进行补全。
- **Focal Loss**：针对类别不平衡设计的损失函数，通过调节难样本权重改善训练。
- **mIoU_occ**：仅在被遮挡区域计算的交并比，专门评估补全能力。

## 可复现要素
- **数据集**：Movi-B、Movi-D（Kubric合成）、KITTI（公开）；训练/测试split遵循SaVos设定
- **代码**：已开源，https://github.com/kfan21/EoRaS
- **超参数**：batch_size=4，epochs=50，lr=1e-5（Movi）/1e-4（KITTI），decay=0.95，weight_decay=5e-4，γ=2（Focal Loss），λ=1，n_s=8，N=2
- **硬件**：4× Tesla T4 GPU，PyTorch实现
