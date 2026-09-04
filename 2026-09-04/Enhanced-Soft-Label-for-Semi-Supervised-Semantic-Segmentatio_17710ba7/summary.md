---
title: "Enhanced-Soft-Label-for-Semi-Supervised-Semantic-Segmentatio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Enhanced_Soft_Label_for_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:13:43"
field: "半监督视觉分割"
keywords: ["半监督语义分割", "伪标签", "对比学习", "软标签", "自训练", "物体部分分组"]
innovations: ["Dynamic Soft Label模块通过累积概率动态维护主导类别，避免固定阈值的刚性筛选", "Pixel-to-Part对比学习结合无监督物体部分分组，在保留类内多样性的同时缓解软标签的类别模糊问题"]
benchmarks: ["Pascal VOC 2012", "Cityscapes"]
---

# 论文速读：Enhanced-Soft-Label-for-Semi-Supervised-Semantic-Segmentatio

## 一句话总结
本文提出 **Enhanced Soft Label (ESL)** 框架，通过 **Dynamic Soft Label (DSL)** 保留高熵预测中的高概率"主导类别"信息，并结合 **pixel-to-part对比学习** 解决软标签带来的类别边界模糊问题，在半监督语义分割任务上显著优于现有方法。

---

## 研究问题与动机

1. **自训练范式中的阈值依赖问题**：现有半监督语义分割方法使用固定阈值筛选伪标签像素，无法适配不同类别的学习难度差异及模型不同训练阶段的状态变化。
2. **低置信度像素的信息浪费**：当前方法直接丢弃低置信度（高熵）伪标签，但这些样本往往包含重要监督信号（如物体边界区域）。
3. **软标签引入的类别模糊问题**：若将高熵预测转为soft label保留所有高概率类别，会导致主导类别之间产生歧义，模糊分类边界。
4. **现有对比学习方法的技术局限**：pixel-to-pixel对比学习受伪标签噪声干扰存在采样误差；pixel-to-region方法过于粗糙，忽略类内多样性（如强制将猫眼像素与整只猫相似）。

---

## 核心贡献（创新点）

1. **Dynamic Soft Label (DSL) 模块**：动态维护每个像素的高概率类别集合，将硬标签转化为soft标签以充分利用高熵预测；与以往方法的本质区别在于不依赖固定阈值，而是根据累积概率动态确定主导类别范围。
2. **Pixel-to-Part对比学习机制**：结合无监督物体部分分组，将像素级对比学习从"像素-区域"升级为"像素-部件"粒度；本质区别在于保留类内多样性，避免将局部特征强制对齐到整个类别区域。
3. **完整的ESL半监督分割框架**：Teacher-Student架构下联合优化交叉熵损失、软交叉熵损失和对比损失；相比先前工作，同时解决了信息浪费和类别模糊两个问题。

---

## 方法详解

### 整体框架（Section 3.1）
- 采用 Teacher-Student 双网络架构，共享 ResNet-101 + DeepLabv3+ 结构
- 教师网络 $G^{\mathbf{t}}$ 生成软伪标签监督学生网络 $G^{\mathbf{s}}$，参数通过指数移动平均（EMA）更新
- 数据增强：弱增强 $\hat{x}$（随机翻转+缩放），强增强 $\tilde{x}$（CutMix）

### Dynamic Soft Label (DSL)（Section 3.2）
- **主导类别定义**：对像素 $j$，主导类别集合 $\mathbf{C}_j = \{c | p_{c,j}^{\mathbf{t}} \geq \delta_j\}$，其中动态阈值 $\delta_j$ 由累积概率不小于 $\eta$ 的最小类别数决定
- **软标签计算**：对主导类别对应的概率归一化得到 $y_j^* = (h_j \cdot p_j^{\mathbf{t}}) / |h_j \cdot p_j^{\mathbf{t}}|$
- **软交叉熵损失**：$\mathcal{L}_{\mathrm{SCE}} = -\frac{1}{N}\sum_{c,j} y_{c,j}^* \log(p_{c,j}^{\mathbf{s}})$

### Pixel-to-Part Contrastive Learning（Section 3.3）
- **无监督物体部分分组**：
  - 预训练阶段：收集标注数据的编码器特征，按类别划分后用 K-means 聚类得到 $K$ 个子类原型 $\mathbf{P} \in \mathbb{R}^{C \times K \times L'}$
  - 训练阶段：通过余弦相似度为每个像素分配最近子类 $k^*$，构建 object-part mask $\mathbf{M}$
- **Mask Average Pooling (MAP)**：对每个子类 $(c,k)$ 计算平均特征 $\mu_{c,k} = \frac{\sum_j \mathbf{M}_{j,c,k} \cdot z_j^{\mathbf{t}}}{\sum_j \mathbf{M}_{j,c,k}}$，存入 FIFO 记忆库
- **InfoNCE 对比损失**：
  - Anchor 采样：从置信度 $> \zeta=0.95$ 的像素中随机选取10%作为 anchor
  - Positive：同类同子类的记忆库样本；Negative：不同子类样本（$N_s=255$）
  - $\mathcal{L}_{\mathrm{CTR}}^c = -\frac{1}{|\mathcal{A}_c|}\sum_j \log\frac{e^{\cos(v_j,v_j^+)/\sigma}}{e^{\cos(v_j,v_j^+)/\sigma} + \sum_{v_j^-} e^{\cos(v_j,v_j^-)/\sigma}}$，温度 $\sigma=0.5$

### 总损失函数（Section 3.4）
$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_1 \mathcal{L}_{\mathrm{SCE}} + \lambda_2 \mathcal{L}_{\mathrm{CTR}}, \quad \lambda_1=0.2, \lambda_2=0.1$$

---

## 实验与结果

### 数据集与协议
- **Pascal VOC 2012**：classic（1464标注）和 blender（5291混合质量标注），测试 1/16、1/8、1/4、1/2、Full 划分
- **Cityscapes**：2975训练图像，测试相同划分协议
- 评估指标：mIoU， backbone 为 ResNet-101

### 主要结果

**Pascal VOC 2012 (classic)**：
| 方法 | 1/16 | 1/8 | 1/4 | 1/2 | Full |
|------|------|-----|-----|-----|------|
| ESL | **70.97** | **74.06** | **78.14** | **79.53** | **81.77** |
| 次优 | GTA-Seg 70.02 | CPS 73.16 | PSMT 76.57 | ST++ 77.30 | PSMT 80.01 |
| 提升 | +0.95% | +0.90% | +1.57% | +1.11% | +1.30% |

**Pascal VOC 2012 (blender)**：
- ESL 在 1/16(76.36)、1/8(78.57)、1/4(79.02)、Full(79.98) 均获最佳

**Cityscapes**：
| 方法 | 1/16 | 1/8 | 1/4 | 1/2 |
|------|------|-----|-----|-----|
| ESL | **75.12** | **77.15** | **78.93** | **80.46** |
| 次优 | U²PL 74.90 | U²PL 76.48 | U²PL 78.51 | U²PL 79.12 |
| 提升 | +0.22% | +0.26% | +0.42% | +1.34% |

### 关键消融结论
- DSL vs Hard label（Full划分）：**+2.09%**（80.41 vs 78.32）
- 加入对比损失后达到 **81.77%**，较纯监督基线（72.50%）提升 **+9.27%**
- DSL 在低标注量下提升更显著（1/8划分提升5.53%），说明对数据稀缺场景更有价值
- K=5 为最优子类数，过多反而性能下降

---

## 相关工作脉络

1. **U²PL [43]**：将不可靠像素视为最不可能类别的负样本；本文指出其忽略了主导类别中的其他有益类别，而DSL通过软标签保留全部高概率信息。
2. **PC²Seg [49]**：像素级对比学习；本文认为其面临伪标签噪声导致的采样误差问题，提出pixel-to-part而非pixel-to-pixel范式缓解该问题。
3. **ReCo [26] / RegionContrast [19]**：区域级对比学习；本文指出这类方法过于粗糙，忽略类内多样性（如猫眼像素被强制对齐到整只猫的特征），pixel-to-part保留了细粒度类内结构。
4. **ST++ [45] / PSMT [27]**：自训练+强数据增强方法；本文与其定位差异在于不依赖固定阈值筛选，而是通过软标签充分利用所有预测信息。
5. **CPS [7] / CCT [30]**：一致性正则化方法；本文属于自训练范式下的软标签增强路线，与一致性方法形成互补视角。

---

## 局限性与未来方向

1. **K值依赖与计算开销**：子类数K需人工调优（实验表明K>5后性能下降），且预处理阶段需对所有标注图像进行K-means聚类并维护C×K个记忆队列，增加训练复杂度。
2. **仅验证于两个数据集**：实验仅在Pascal VOC 2012和Cityscapes上验证，未扩展到更复杂场景（如ADE20K等大规模场景分割）。
3. **主导类别数量自适应的潜在限制**：动态阈值机制依赖超参η，虽比固定阈值灵活，但不同图像/场景可能需要更细粒度的自适应策略。
4. **未来方向**：可探索自动学习最优K值、将方法迁移至更广泛的半监督视觉任务（检测、分割一体化）、结合更先进的对比学习变体（如BYOL式无负样本对比）进一步降低计算开销。

---

## 研究启发与可借鉴点

1. **"软伪标签"设计思路可迁移**：DSL将"丢弃低置信度样本"转为"保留高概率类别集"的思路，可迁移至半监督目标检测、实例分割等任务，避免 hard threshold 的刚性筛选。
2. **Pixel-to-Part对比学习范式具有通用价值**：将对比学习粒度从像素/区域细化到物体部件，能更好保留类内多样性；此设计可应用于其他需要细粒度区分的视觉任务（如细粒度分类、开放词汇分割）。
3. **无监督子类原型的构建方式值得借鉴**：基于预训练特征的K-means聚类生成子类原型，无需额外标注即可建立类内结构先验；该机制可与原型学习（Prototype Learning）结合，用于少样本分割等场景。
4. **实验设计中的"标注质量vs数量"分析**：论文发现更多标注数据不一定带来更好性能（annotation quality > quantity），这一洞察对半监督学习的数据效率评估有参考价值。

---

## 关键术语表

**Dominant Classes（主导类别）**：对某像素预测概率较高的类别集合，通常包含语义相似或空间相邻的类别。

**Dynamic Soft Label (DSL)**：动态软标签，根据累积概率自适应确定每个像素的主导类别并归一化为soft伪标签。

**Pixel-to-Part Contrastive Learning**：像素到部件的对比学习，通过无监督物体部分分组实现更细粒度的正负样本对齐。

**Object-Part Grouping（物体部分分组）**：基于K-means子类原型的无监督分组机制，将同一物体的不同部分区分开来。

**Mask Average Pooling (MAP)**：基于object-part mask的平均池化操作，提取每个子类的代表性特征。

**Confirmation Bias（确认偏差）**：半监督学习中模型过度拟合错误伪标签导致性能退化的问题。

**Mean Teacher / EMA Update**：教师网络通过学生网络参数的指数移动平均更新，增强伪标签稳定性。

---

## 可复现要素

- **数据集**：Pascal VOC 2012（公开）、Cityscapes（公开）、SBD（公开）
- **代码**：论文未明确提供开源链接（需查阅项目主页或联系作者）
- **权重**：使用 ImageNet 预训练的 ResNet-101 backbone（公开）
- **关键超参**：η=0.95（主导类别累积概率阈值）、K=5（子类数）、ζ=0.95（anchor采样置信度阈值）、σ=0.5（温度系数）、λ₁=0.2、λ₂=0.1、θ=0.99（EMA动量）、Batch size=16（8 labeled + 8 unlabeled）、训练轮数：Pascal VOC 80 epochs / Cityscapes 200 epochs
- **骨干网络**：ResNet-101 + DeepLabv3+ decoder
- **输入尺寸**：Pascal VOC 513²，Cityscapes 769²

---
