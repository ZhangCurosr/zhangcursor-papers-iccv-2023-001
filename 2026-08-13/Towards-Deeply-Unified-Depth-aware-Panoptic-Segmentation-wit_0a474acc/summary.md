---
title: "Towards-Deeply-Unified-Depth-aware-Panoptic-Segmentation-wit"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/He_Towards_Deeply_Unified_Depth-aware_Panoptic_Segmentation_with_Bi-directional_Guidance_Learning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:31:29"
field: "多任务视觉感知"
keywords: ["Depth-aware Panoptic Segmentation", "Multi-task Learning", "Cross-modality Guidance", "Geometric Query Enhancement", "Bi-directional Learning"]
innovations: ["提出深度统一架构，通过几何查询增强将场景几何信息融入统一 per-segment queries", "设计双向指导学习机制，语义引导深度对比学习与深度引导语义连续性学习相互协同", "引入 backup query 补全低置信度 mask 导致的深度空白区域，提升深度估计鲁棒性"]
benchmarks: ["Cityscapes-DVPS", "SemKITTI-DVPS"]
---

# 论文速读：Towards-Deeply-Unified-Depth-aware-Panoptic-Segmentation-wit

## 一句话总结
本文提出了一种深度统一架构用于深度感知全景分割（DPS），通过在统一查询中融入几何增强并利用双向指导学习机制促进语义与深度特征互相辅助，在 Cityscapes-DVPS 和 SemKITTI-DVPS 数据集上均达到 SOTA。

## 研究问题与动机
- 深度感知全景分割需要同时预测实例级语义分割和深度图，早期方法简单地在分割模型上附加密集深度预测头，忽略了两个子任务的互相关系。
- 近年提出的统一架构（如 PolyphonicFormer）虽然使用统一架构输出，但学习过程仍各自独立，仅依赖任务特定的损失函数，缺乏跨模态知识的学习。
- 单向引导方法（如 SGT）只利用语义引导深度表示学习，未探索深度对语义的反向引导关系。
- 不完整监督（仅有部分标签）条件下的多任务学习性能下降严重，需要设计更强的跨任务互学机制。

## 核心贡献（创新点）
- 提出了深度统一架构：使用统一的 per-segment queries 以 per-segment 方式同时完成分割和深度估计，相比分离式查询设计更深层地统一了两任务。
- 设计了几何查询增强（Geometric Query Enhancement）：通过固定大小的 latent representation 作为中介，融合多尺度深度特征，将场景几何信息注入到统一查询中。
- 提出了双向指导学习（Bi-directional Guidance Learning）：包含语义到深度的对比学习（最大化类内/最小化类间深度特征距离）和深度到语义的连续性引导，实现跨模态特征的双向优化。
- 引入 backup query 机制：解决低置信度 mask 过滤导致的深度空白区域问题，通过全局深度估计补全缺陷 mask 区域的深度值。
- 实验验证了该方法在不完整监督标签下仍能显著缓解性能下降，展示了跨任务互学的泛化价值。

## 方法详解
- **共享编码器结构**：使用共享 backbone 提取特征，生成语义特征金字塔 $\mathcal{F}$ 和深度特征金字塔 ${\mathcal{F}}_d$（分辨率分别为 $\times 1/8, \times 1/16, \times 1/32$），以及像素嵌入 $\mathcal{E}_{pixel}$ 和深度嵌入 $\mathcal{E}_{depth}$。
- **统一查询与分割**：采用 Transformer Decoder（9层）处理 per-segment queries，通过 masked attention 与多尺度语义特征交互，输出统一查询 $X_o^l$；分类通过 MLP+Softmax 实现，mask 预测通过 dot product $M = \sigma(\text{MLP}(X_o^l) \otimes \mathcal{E}_{pixel})$ 得到。
- **几何查询增强**：引入固定大小 latent representation $\mathcal{R}^0$ 作为几何信息中介；先对 masked depth features 与 $\mathcal{R}$ 做 cross-attention，再通过 self-attention 更新 $\mathcal{R}^l$；最后用 $X_o^l$ 与 $\mathcal{R}^l$ 做 cross-attention 生成几何增强查询 $X_d^l$。
- **逐实例深度预测**：通过 $d = D_{max} \times \sigma(\psi(X_d^l) \otimes \mathcal{E}_{depth})$ 生成 per-segment depth map，再按 mask 聚合为最终深度图 $D(u,v) = \sum_{i \in \mathcal{H}} d_i(u,v) \cdot \mathbb{1}[M_i(u,v) > 0.5]$。
- **Backup Query**：无 mask 约束地通过 cross-attention 查询多尺度深度特征，生成全局深度图补充空白区域。
- **Semantic-to-Depth Guidance**：在 $K \times K$ 局部 patch 内，计算同类内最大特征距离 $d_{max}^+$ 和跨实例最小距离 $d_{min}^-$，通过 triplet loss $\mathcal{L}_{sg} = \frac{1}{N}\sum_i \max(0, \alpha + d_{max}^+(i) - d_{min}^-(i))$ 约束。
- **Depth-to-Semantic Guidance**：在 patch 内以深度连续性和语义特征距离的联合分布为约束，定义 loss $\mathcal{L}_{dg}(i,j) = -\frac{1}{N}\sum_i\sum_j e^{-\|\hat{d}_i-\hat{d}_j\|/\tau} \cdot e^{-\|\mathcal{F}_i-\mathcal{F}_j\|_2}$。
- **总损失**：$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{sg}\mathcal{L}_{sg} + \lambda_{dg}\mathcal{L}_{dg}$，权重分别为 2, 5, 2.5, 0.1, 0.1。

## 实验与结果
- **数据集**：Cityscapes-DVPS（3000 标注帧，8 thing + 11 stuff 类）和 SemKITTI-DVPS（19130 训练图像，稀疏标注）。
- **评估指标**：DPQ（Depth-aware Panoptic Quality），在 $\lambda=\{0.1, 0.25, 0.5\}$ 下平均；辅以 PQ、深度估计指标（abs rel, RMSE, σ<1.25等）。
- **Cityscapes-DVPS**（ResNet-50）：DPQ 达到 63.0，超越 PanopticDepth（52.3）10.7 个百分点、PolyphonicFormer（59.8）3.2 个百分点；PQ 69.5，abs rel 0.063。
- **Cityscapes-DVPS**（Swin-B）：DPQ 达到 64.3，超越 ViP-DeepLab（61.9）2.4 个百分点。
- **SemKITTI-DVPS**（ResNet-50）：DPQ 47.9，超越所有现有方法；（Swin-B）DPQ 52.8，同样 SOTA。
- **Ablation**：完整方法较 baseline（Variant A）提升约 2.7% DPQ；双向指导在稀疏标签下可缩小与全监督差距约 1.2%。

## 相关工作脉络
- **Panoptic-DeepLab / ViP-DeepLab**：ViP-DeepLab 首次提出 DPS 任务及数据集，采用附加深度头的多分支架构；本文在其基础上实现架构级深度统一。
- **PanopticDepth**：使用动态 instance-specific kernels 实现联合预测，但仍将两任务视为独立分支；本文通过共享查询和双向指导实现更深层次统一。
- **PolyphonicFormer**：采用 query-based learning 统一架构，但通过 grouping 选择 salient features 后分别更新查询；本文用 latent representation 直接融合几何信息，无需分组操作。
- **SGT (Jung et al.)**：利用语义引导深度表示学习的 triplet loss；本文扩展为双向指导，增加了深度对语义的反向约束。
- **MaskFormer / Mask2Former**：基于 mask classification 的统一分割框架；本文在此基础上扩展深度估计分支并设计几何增强机制。
- **SDC-Depth**：分解为 category-specific depth 估计；本文直接在统一框架内实现 per-segment 深度预测，无需预处理分解。

## 局限性与未来方向
- 不准确的 mask 预测会直接影响深度图质量，说明模型对分割预训练质量有较强依赖，可能影响对未见物体的泛化能力。
- 当前方法主要在城市驾驶场景验证，对复杂自然场景或域偏移的鲁棒性有待探索。
- 未来希望向完全统一框架演进，进一步消除两任务间的隐式边界，并扩展到其他几何理解任务。

## 研究启发与可借鉴点
- **Latent Representation 作为跨模态中介**：使用固定大小的 latent tokens 桥接不同任务特征的设计可迁移至其他多任务学习场景（如 3D 检测 + 分割）。
- **双向对比学习思想**：不仅用 A 引导 B，也用 B 反向约束 A，这种对称式跨模态监督可用于视觉-语言、点云-图像等任务对齐。
- **Backup Query 处理缺失区域**：针对低置信度过滤导致的空洞，引入全局 fallback query 的思路可推广至其他实例级预测任务中的异常/缺失补偿。
- **局部 Patch 内对比约束**：将对比学习限制在含边界的局部 patch 而非全图，既降低计算量又提高边界敏感度，值得在 dense prediction 任务中借鉴。
- **不完整监督下的鲁棒性验证**：论文设计了三类子集实验（仅分割/仅深度/两者），为半监督多任务学习提供了可复现的评估范式。

## 关键术语表
- **Depth-aware Panoptic Segmentation (DPS)**：结合语义分割与单目深度估计的任务，输出每个像素的类别标签和深度值，形成 3D 实例级语义标注。
- **Per-segment Query**：基于 MaskFormer 理念的实例级查询向量，每个 query 对应一个预测 mask 及其所属类别。
- **Geometric Query Enhancement**：通过 latent representation 将多尺度深度特征信息注入统一查询，使查询向量同时携带几何先验。
- **Bi-directional Guidance Learning**：包含语义到深度（contrastive）和深度到语义（smoothness continuity）的双向跨模态监督学习。
- **DPQ (Depth-aware Panoptic Quality)**：DPS 任务的核心评估指标，在指定深度误差阈值 $\lambda$ 下计算 panoptic quality。
- **Backup Query**：无 mask 约束的全局深度查询，用于补全因低置信度过滤而产生的深度空白区域。
- **Latent Representation**：固定大小的可学习向量，作为几何信息与 per-segment queries 之间的通信媒介。
- **Masked Attention**：在 Transformer decoder 中使用预测 mask 进行注意力屏蔽，使 query 只关注对应区域特征。

## 可复现要素
- **数据集**：Cityscapes-DVPS 和 SemKITTI-DVPS，公开可用。
- **代码/权重**：已开源，仓库地址 https://github.com/jwh97nn/DeepDPS。
- **关键超参**：深度最大值 $D_{max}=80$；损失权重 $\lambda_{cls}=2, \lambda_{mask}=5, \lambda_{depth}=2.5, \lambda_{sg}=\lambda_{dg}=0.1$；depth guidance 温度系数 $\tau=10$；patch 大小 $K$ 同 [23]；训练分两阶段：50 epoch 纯分割预训练 + 10 epoch 联合微调。
- **Backbone**：ResNet-50 和 Swin-B，Detectron2 实现，多尺度 deformable attention Transformer decoder。
