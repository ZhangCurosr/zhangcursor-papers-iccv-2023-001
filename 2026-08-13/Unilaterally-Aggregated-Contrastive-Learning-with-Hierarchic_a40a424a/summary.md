---
title: "Unilaterally-Aggregated-Contrastive-Learning-with-Hierarchic"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Unilaterally_Aggregated_Contrastive_Learning_with_Hierarchical_Augmentation_for_Anomaly_Detection_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:35:21"
field: "异常检测"
keywords: ["Anomaly Detection", "Contrastive Learning", "Self-Supervised Learning", "Virtual Outlier", "Soft Aggregation", "Hierarchical Augmentation"]
innovations: ["UniCLR单侧聚合对比损失，同时优化inlier紧凑聚集与virtual outlier分散分布", "Soft Aggregation软聚合机制，基于偏离程度加权抑制增强引入的伪异常", "Hierarchical Augmentation层次化增强策略，按网络深度递增增强强度并分布式聚合"]
benchmarks: ["CIFAR-10 One-class", "CIFAR-100", "ImageNet-30", "MVTec-AD", "Labeled CIFAR-10", "Unlabeled Multi-class CIFAR-10/ImageNet-30"]
---

# 论文速读：Unilaterally-Aggregated-Contrastive-Learning-with-Hierarchic

## 一句话总结
本文提出 **UniCon-HA**，一种基于对比学习的新颖异常检测框架，通过非对称的对比损失函数同时实现正常样本（inliers）的紧凑聚集与虚拟异常样本（virtual outliers）的分散分布；结合软聚合机制（Soft Aggregation）和易到难的层次化数据增强（Hierarchical Augmentation），在三种典型异常检测设定下均超越现有最优方法。

## 研究问题与动机
1. **现有对比学习方法未同时满足AD的两个核心要求**：良好的异常检测表示分布需同时满足（a）inliers 紧凑聚集，（b）outliers 分散分布；但已有工作如 RotNet 只保证可区分性、DROC 虽分散 outliers 却推散了 inliers、CSI 的聚类程度仍不足。
2. **标准数据增强可能引入"伪异常"**：随机裁剪等强增强操作会扭曲语义，产生 outlier-like 样本，将其与正常样本聚合会损害 inlier 紧凑性。
3. **纯对比学习框架缺乏对异常检测目标的忠实适配**：传统 instance discrimination 在全量数据（含 inliers 和 virtual outliers）上一视同仁地进行区分，违背了"正常集中、异常分散"的 AD 先验。
4. **模型从头训练而非依赖预训练**：聚焦严格设定下仅用正常样本训练的场景，排除大规模预训练模型的影响，更具通用性。

## 核心贡献（创新点）
1. **UniCLR 单侧聚合对比损失**：将所有 inliers 视为一类进行聚集（监督对比），将每个 virtual outlier 作为独立类进行分散（无监督对比），与 CSI/DROC 在全量数据上做统一 instance discrimination 形成本质区别。
2. **Soft Aggregation（SA）软聚合机制**：通过计算各增强视图与其他 inliers 的平均相似度来赋予权重，抑制因数据增强引入的偏离样本的影响，实现"净化的"紧凑分布——区别于简单限制增强强度的做法。
3. **Hierarchical Augmentation（HA）层次化增强策略**：受课程学习启发，浅层使用弱增强、深层使用强增强，并在不同网络深度执行对比聚合，比单阶段聚合能获得更高程度的 inlier 紧凑性。
4. **无需辅助变换预测分支**：与 CSI 的旋转分类头不同，本文方法不需要学习区分具体变换类型，简化了结构。
5. **统一支持三种 AD 设定**：无标签单类、无标签多类、有标签多类均可适用，且在 OE（Outlier Exposure）辅助下性能进一步提升。

## 方法详解
**整体流程**：给定训练正常样本集 $\mathcal{D}_{\mathrm{in}}$，通过分布偏移变换 $S$（如旋转 $\{90°, 180°, 270°\}$）生成虚拟异常样本集 $\mathcal{D}_{\mathrm{vout}}$；再对 $\mathcal{D}_{\mathrm{in}} \cup \mathcal{D}_{\mathrm{vout}}$ 分别施加保真增强 $\mathcal{T}$ 构造正样本对，组成 mini-batch 进行训练。

**UniCLR 损失（核心）**：对 inlier 样本 $\tilde{x}_i \in \tilde{\mathcal{D}}_{\mathrm{in}}$，采用**监督对比损失**：正样本集合为整个 $\tilde{\mathcal{D}}_{\mathrm{in}} \setminus \{\tilde{x}_i\}$，负样本集合为 $\tilde{\mathcal{D}}_{\mathrm{vout}}$；对 outlier 样本 $\tilde{x}_i \in \tilde{\mathcal{D}}_{\mathrm{vout}}$，采用**无监督对比损失**（SimCLR 形式）：正样本仅为自身增强视图，负样本为 batch 内其余所有样本。实现"inliers 互相拉拢、outliers 互相推开"的非对称目标。

**Soft Aggregation（SA）**：仅在 inlier 上应用，将标准对比损失中的正样本对加权重：$w_{x_i} = \frac{\sum_{x_j \in D_{x_i}^+ \setminus \{x_i\}} e^{z(x_i)^T z(x_j)/\tau_\omega}}{\sum_{x_k \in D_{x_i}^+}\sum_{x_j \in D_{x_i}^+ \setminus \{x_k\}} e^{z(x_k)^T z(x_j)/\tau_\omega}}$，距离 inlier 分布越远的样本权重越低，抑制语义漂移样本对紧凑性的破坏。论文实验中将 SA 应用于最深层 $res_4$。

**Hierarchical Augmentation（HA）**：在 ResNet 的四个 residual 阶段 $res_1 \sim res_4$ 上分别应用递增强度的增强 $T_1 \sim T_4$（相同类型但强度递增），每阶段附加投影头 $g_i$ 提取特征，在各阶段独立计算 UniCLR 损失并加权求和：$\mathcal{L}_{\mathrm{all}} = \frac{1}{4}\sum_{i=1}^{4}\lambda_i \mathcal{L}_{\mathrm{UniCLR}}(\cdot; T_i)$。

**推理**：移除投影头，检测分数为测试样本与训练样本中最接近的 inlier 的余弦相似度：$s_i(x_i) = \max_m \mathrm{cosine}(f(x_i), f(x_m))$。

## 实验与结果
**数据集与设定**：
- 无标签单类：CIFAR-10（逐类）、CIFAR-100（20 super-classes）、ImageNet-30
- 无标签多类：CIFAR-10 作 inlier + SVHN/LSUN/ImageNet/CIFAR-100/Interp. 作 outlier；ImageNet-30 作 inlier + CUB-200/Dogs/Pets/Flowers/Food-101/Places-365/Caltech-256/DTD 作 outlier
- 有标签多类：CIFAR-10 和 ImageNet-30
- 真实数据集：MVTec-AD（patch 级 32×32）

**最强结果**：
| 数据集 | 方法 | AUROC | 相比 CSI 提升 |
|--------|------|-------|---------------|
| One-class CIFAR-10 | UniCon-HA + OE | **96.9** | +2.6 |
| One-class CIFAR-10 | UniCon-HA | **95.4** | +1.1 |
| One-class CIFAR-100 | UniCon-HA | **92.4** | +2.8 |
| One-class ImageNet-30 | UniCon-HA | **93.2** | +1.6 |
| Labeled CIFAR-10 | UniCon-HA | **95.4** | +2.1 |
| MVTec-AD (Image) | UniCon-HA | 89.8 | +3.3 |

**消融结论**：SA 与 HA 均带来稳定增益；HA 在多阶段（2-3-4）优于单阶段；去除旋转分类头后方法依然有效；旋转是最有效的 shifting transformation。

## 相关工作脉络
1. **RotNet [26]**：通过旋转预测构建 pretext task，将数据分成 4 个旋转角度簇，但未能充分紧凑 inliers——本文明确聚合全部 inliers 为单一类，与 RotNet 的"可区分但松散"形成对比。
2. **DROC [49]**：对比学习建模 inliers 与其旋转的统一分布，导致 inliers 被推散——本文用单侧聚合修正此问题。
3. **CSI [51]**：在 DROC 基础上增加旋转分类辅助头，将 inliers 限制在子区域，但仍受限于分类器的约束强度——本文无需辅助分支即能实现更高紧凑度。
4. **SupCLR [29]**：监督对比学习的基线框架，按类别标签聚合正样本——本文借鉴其思想但将所有 inliers 视为同一类，而非多类别。
5. **CutPaste [33]**：针对工业异常检测的特化方法，不适合通用 AD 设定——本文方法在通用数据集上显著优于 CutPaste（如 CIFAR-10 上 95.4 vs 69.4）。
6. **OE [25]（Outlier Exposure）**：传统认为对对比学习方法有害，本文证明配合 UniCLR 后 OE 可有效提升性能。

## 局限性与未来方向
1. **虚拟异常生成方式较单一**：主要依赖旋转，虽验证了其他变换（CutPerm/Blur/Noise/Sobel），但未系统探索更丰富的分布偏移策略。
2. **层次化增强增加计算开销**：四阶段聚合需额外投影头和多次前向计算，推理时虽移除但训练成本上升。
3. **未结合预训练模型**：论文强调从 scratch 训练，未利用 ImageNet 等预训练权重，在更大规模数据集上可能受限。
4. **HA 增强强度 schedule 依赖人工设计**：四个阶段的增强强度为预设递增，未探索自适应策略。
5. **未来方向暗示**：可探索更多 shifting transformations、结合 OE 的进一步优化、迁移到视频异常检测等时序场景。

## 研究启发与可借鉴点
1. **"单侧聚合"思想可迁移**：将"聚集一类、分散另一类"的非对称对比学习范式推广到其他需要紧凑-分散分离的任务，如域适应、半监督学习中的类别边界学习。
2. **软聚合机制的设计值得复用**：基于相似度赋予样本权重的思想可用于任何存在"增强引入噪声"的对比学习场景，是一个通用的去噪聚合技巧。
3. **层次化增强策略可与 deep supervision 思路结合**：在不同网络深度应用不同强度的数据增强是一种新的正则化手段，可在 self-supervised representation learning 中探索。
4. **去除辅助预测头的简化思路**：证明在合理损失设计下，辅助分类头并非必需，减少了模型复杂度和超参数量。
5. **OE 与对比学习的兼容性重估**：此前认为 OE 对对比学习有害，本文展示了在正确框架下 OE 可以显著提升性能，提示团队重新审视已有方法中的"常识性"限制。

## 关键术语表
- **UniCLR (Unilaterally Aggregated Contrastive Loss)**：非对称对比损失，将所有正常样本作为单一正类聚集、将每个虚拟异常样本作为独立负类分散。
- **Soft Aggregation (SA)**：基于样本与正常分布的偏离程度动态加权聚合的机制，降低增强引入的伪异常样本的影响力。
- **Hierarchical Augmentation (HA)**：按网络深度递增增强强度的层次化数据增强策略，浅层弱增强捕捉低级特征、深层强增强捕捉高级语义。
- **Virtual Outlier**：通过对正常样本施加分布偏移变换（如旋转）人为生成的异常样本，用于替代现实中难以获取的真实异常。
- **Outlier Exposure (OE)**：在训练中引入外部异常数据集以提升模型对异常敏感性的辅助训练策略。
- **AUROC**：ROC 曲线下面积，异常检测任务的主流评估指标，值越高表示检测性能越好。
- **Instance Discrimination**：对比学习中的核心范式，将 batch 内每个样本视为独立类别，通过区分不同样本学习判别性表示。
- **Inlier**：来自训练分布的正常样本，与 outlier 相对。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、ImageNet-30、MVTec-AD 均为公开数据集；80 Million Tiny Images 用于 OE 亦公开。
- **代码/权重**：论文未提及开源状态（ICCV 2023）。
- **关键超参**：训练 epoch = 2048；学习率 = 0.01（cosine decay）；温度 $\tau = 0.5$，$\tau_\omega = 0.5$；inliers : virtual outliers 比例 = 1:3；ResNet-18  backbone；旋转集合 $\{90°, 180°, 270°\}$；SGD 优化器。详细增强配置见 supplementary material。
- **训练设备**：论文未提及。
