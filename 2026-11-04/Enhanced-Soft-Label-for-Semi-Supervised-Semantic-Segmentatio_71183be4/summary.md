---
title: "Enhanced-Soft-Label-for-Semi-Supervised-Semantic-Segmentatio"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Enhanced_Soft_Label_for_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:37:43"
field: "半监督语义分割"
keywords: ["semi-supervised semantic segmentation", "pseudo labeling", "contrastive learning", "soft label", "curriculum learning", "object part grouping"]
innovations: ["动态软标签(DSL)：基于累积概率动态确定主类集合并生成软伪标签，充分利用高熵预测的监督信号", "像素到部件对比学习：结合无监督部件分组机制，以部件粒度进行对比学习，兼顾类内多样性与抗噪声能力"]
benchmarks: ["Pascal VOC 2012", "Cityscapes"]
---

# 论文速读：Enhanced-Soft-Label-for-Semi-Supervised-Semantic-Segmentatio

## 一句话总结
本文提出了增强软标签（ESL）框架，通过动态软标签（DSL）充分利用高熵伪标签中的监督信号，并结合像素到部件对比学习（pixel-to-part contrastive learning）解决多主类间的边界模糊问题，在 Pascal VOC 2012 和 Cityscapes 半监督语义分割任务上取得了 SOTA 性能。

## 研究问题与动机
1. **现有自训练方法依赖固定阈值筛选低置信度像素**：主流自训练半监督语义分割方法用预定义的常数阈值过滤伪标签，无法适配不同类别的学习难度差异以及模型不同训练阶段的学习状态，导致大量有价值的高熵样本被丢弃。
2. **高熵样本蕴含重要监督信号却被忽略**：边界等高熵像素虽然难以精确分类，但模型对其属于某个"主类集合"（dominant classes）具有较高的确定性，直接丢弃会造成信息损失并陷入次优解。
3. **软标签会引入主类间歧义**：若仅将高概率类保留为软标签，会导致主类之间边界模糊，缺乏区分不同主类的能力。
4. **现有对比学习方法存在采样误差或粒度粗的问题**：像素级对比易受伪标签噪声影响产生采样误差；区域级对比忽略了类内多样性，导致不准确的样本分配（如强制猫眼像素与整只猫相似）。

## 核心贡献（创新点）
1. **提出动态软标签（DSL）模块**：通过累积概率动态确定每个像素的主类集合，将伪标签保持为"软"形式以充分利用高熵预测；与已有工作本质区别在于自适应感知样本难度（高/低熵）并动态调整主类数量，而非固定阈值一刀切。
2. **提出像素到部件对比学习（pixel-to-part contrastive learning）**：结合无监督物体部件分组机制，利用多个原型表示类的不同子模式，实现更精细的类内多样性和更忠实的正负样本分配；与已有像素级/区域级对比的本质区别在于以"部件"为粒度，兼顾细粒度类内差异与抗噪声能力。
3. **构建端到端的 ESL 半监督语义分割框架**：将 DSL 与 pixel-to-part 对比学习统一于 teacher-student 架构中，在 Pascal VOC 2012 和 Cityscapes 上取得领先性能，尤其在少标注数据场景下提升显著。

## 方法详解

**整体架构**：采用 teacher-student 框架，两个同构 FCN（编码器 E + 解码器 D + 投影头 P + 分割头 S），参数不共享。Teacher 通过 EMA 从 Student 更新。输入经增强模块 A 生成弱增强版本（随机翻转+缩放）和强增强版本（CutMix）。Teacher 对弱增强无标签图生成伪标签和部件特征，Student 对强增强无标签图接收监督。

**动态软标签（DSL）**：
- 对教师网络的预测概率图 $p^{\mathbf{t}}$，按概率降序累积，设定阈值 $\eta$，找到累积概率首次 ≥ η 的分类数 $C^*$，其对应概率值即为动态阈值 $\delta_j$。
- 像素 $j$ 的主类集合定义为 ${\bf C}_j = \{c | p_{c,j}^{\mathbf{t}} \ge \delta_j\}$，对应的主类掩码 $h$ 在 ${\bf C}_j$ 中元素为 1。
- DSL 通过将掩码后的概率做 $l_1$ 归一化得到软标签 $y^* = (h_j \cdot p_j^{\mathbf{t}}) / |h_j \cdot p_j^{\mathbf{t}}|$，再用软交叉熵 $\mathcal{L}_{\mathrm{SCE}}$ 监督 Student。
- 当 $\eta \le \max_c p_{c,j}^{\mathbf{t}}$ 时 DSL 退化为一-hot 硬标签，兼容传统方法。默认 $\eta = 0.95$。

**无监督物体部件分组机制**：
- 训练前，用 ImageNet 预训练的 Encoder $E^{\mathbf{t}}$ 提取整个标注数据集的特征，按 GT 标签分为 C 个类，对每类特征做 K-means（$K=5$），得到 $C \times K$ 个子类原型矩阵 $\mathbf{P} \in \mathbb{R}^{C \times K \times L'}$。
- 训练过程中，每个像素根据与对应类原型的余弦相似度分配到最近子类，生成 one-hot 部件掩码 $\mathbf{M}$。
- 对投影头特征 $z^{\mathbf{t}}$ 按部件掩码做掩码平均池化（MAP），得到每个子类 $(c,k)$ 的均值特征 $\mu_{c,k}$，存入 FIFO 记忆库（每队列大小 $q$），并实时更新原型矩阵 $\mathbf{P}$（动量系数 $\theta=0.99$）。

**对比损失**：
- 锚点选取：对标注图取 GT 为正的像素；对无标签图取教师预测概率 > $\zeta$（$\zeta=0.95$）的像素，再从合格像素中随机采样 10% 作为锚点集。
- 正样本：锚点对应子类 $k^*$ 所在的记忆库队列中随机采样一个。
- 负样本：从其他子类队列中随机采样 $N_s=255$ 个。
- 采用 InfoNCE 损失（温度 $\sigma=0.5$），对所有类别取平均得到 $\mathcal{L}_{\mathrm{CTR}}$。

**总损失**：$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + 0.2 \cdot \mathcal{L}_{\mathrm{SCE}} + 0.1 \cdot \mathcal{L}_{\mathrm{CTR}}$。

## 实验与结果

**数据集**：Pascal VOC 2012（classic 与 blender 两个版本）、Cityscapes。在 1/16、1/8、1/4、1/2 及 full（完整标注）划分协议下评测，指标为 mIoU。

**主要结果**：
- **Pascal VOC 2012（classic）**：Full 划分下 ESL 达到 **81.77%** mIoU，相比第二名 GTA-Seg（80.47%）提升 **1.30%**；1/16 划分下 70.97% vs 第二名 70.02%，提升 **0.95%**。DSL 相比硬标签在少标注（92 张）时提升 5.53%，标注多（1464 张）时提升 2.09%。
- **Pascal VOC 2012（blender）**：1/2 划分下 ESL 达 **79.98%**，优于第二名 U²PL（79.94%，实际为修正后结果）；1/16 划分下 76.36% vs 第二名 75.50%，提升约 **0.86%**。
- **Cityscapes**：1/2 划分下 ESL 达 **80.46%**，相比第二名 U²PL（79.12%）提升 **1.34%**；1/16 划分下 75.12% vs 第二名 74.90%，提升 **0.22%**。
- **消融结论**：加入 DSL 比硬标签提升 2.09%（Table 2）；加入 pixel-to-part 对比学习进一步提升 1.36%（80.41%→81.77%）；$K=5$ 为最优部件数（Table 4）；$\eta=0.95$ 为最优累积阈值（Table 3）。

## 相关工作脉络
1. **自训练基线方法（MT、CPS、ST++、U²PL、PSMT 等）**：均依赖固定阈值丢弃低置信度伪标签像素；本文 DSL 动态维护主类集合，避免高熵信息的丢失。
2. **自适应阈值方法（Freematch [42]）**：虽自适应阈值但针对 classification 域，未处理多主类软标签与边界模糊问题；本文 DSL 同时解决主类确定与软标签利用。
3. **像素级对比学习（PC²Seg [49]）**：利用像素级对比但面临伪标签噪声导致的采样误差；本文 pixel-to-part 以部件均值特征作为对比单元，减轻采样误差同时保留类内多样性。
4. **区域级对比学习（ReCo [26]、RegionContrast [19]）**：通过区域平均缓解噪声但忽略类内多样性；本文部件分组细化到子类粒度，实现更忠实的样本分配。
5. **利用不可靠伪标签的方法（AEL [20]、U²PL [43]）**：AEL 对可信样本加权，U²PL 将不可靠像素作为最不可能类的负样本；两者均忽略主类集中其他有益类别；本文 DSL 将所有主类纳入监督信号。
6. **一致性正则化方法（CCT [30]、GCT [23]）**：通过扰动不变性利用无标签数据；本文从伪标签质量角度切入，聚焦于如何更好利用低置信度预测。

## 局限性与未来方向
1. **部件分组依赖初始原型**：K-means 初始化基于标注数据的 encoder 特征，若标注量极少（如 1/16 划分），原型质量可能受限，影响部件分组的准确性。
2. **超参数敏感性**：累积阈值 $\eta$、部件数 $K$、对比温度 $\sigma$ 等需手动调优；文中表明 $\eta$ 过小会退化到硬标签，$K>5$ 性能不再提升，说明存在较窄的最优区间。
3. **长尾分布场景表现有限**：Cityscapes 实验讨论中指出，极少量标注数据无法覆盖长尾类别的所有模式，性能提升相对 smaller。
4. **计算开销**：FIFO 记忆库、部件分组和对比采样增加了训练复杂度，在高分辨率图像（如 Cityscapes）上可能带来显存压力。
5. **未来方向**：可扩展至 fewer-shot 或 zero-shot 场景下的半监督分割；探索更鲁棒的自动超参数搜索策略；结合 Transformer  backbone 验证方法的通用性。

## 研究启发与可借鉴点
1. **动态软标签设计思路可迁移**：基于累积概率的动态阈值思想可迁移到其他下游任务（如检测、分割、关键点），解决"哪些低置信度样本值得利用"的通用问题。
2. **部件级对比学习范式有价值**：pixel-to-part 将对比学习粒度从像素→区域→部件逐步细化，兼顾抗噪与类内多样性，这一思想可应用于其他需要细粒度区分的视觉任务。
3. **实验设计值得借鉴**：对比"硬标签 vs DSL"在不同标注比例下的性能增益差异（少标注时增益更大），直观揭示了方法的核心价值所在；类别分布可视化（Fig. 7b）有力论证了软标签的类别平衡优势。
4. **可结合团队方向创新**：若团队关注医学图像分割（标注稀缺），ESL 的 DSL 机制可有效利用低置信度边界像素；若关注视频语义分割，可将 temporal 一致性融入部件分组。

## 关键术语表
**Dynamic Soft Label (DSL)**：基于累积概率动态确定每个像素的主类集合，并将主类概率归一化为软标签，以充分利用高熵预测的监督信号。
**Dominant Classes**：某像素预测概率高于动态阈值的类别集合，通常包含语义相近或空间相邻的类别。
**Pixel-to-Part Contrastive Learning**：以部件（subclass）为粒度的对比学习，通过平均池化部件特征获得对比样本，兼顾类内多样性与抗伪标签噪声能力。
**Unsupervised Object-Part Grouping**：利用 K-means 在每类特征中聚类出 $K$ 个子类原型，训练时通过余弦相似度将像素分配到最近原型所属部件。
**Teacher-Student Framework with EMA**：教师网络参数通过学生参数的指数移动平均（EMA）更新，生成更稳定的伪标签。
**Masked Average Pooling (MAP)**：根据部件掩码对特征图做掩码平均池化，得到每个部件的均值特征表示。
**InfoNCE Loss**：对比学习常用的信息噪声对比估计损失，通过 softmax 形式最大化正样本对相似度、最小化负样本对相似度。
**Curriculum Learning**：本文 DSL 配合对比学习渐进式利用高熵样本的思路，属于 curriculum learning 的一种体现。

## 可复现要素
- **数据集**：Pascal VOC 2012（classic + blender，SBD 增强）和 Cityscapes，均公开可获取。
- **代码开源**：论文未明确提及开源声明，但通常 ICCV 论文会提供代码；需查阅官方项目页面确认。
- **权重开源**：论文未提及。
- **关键超参数**：$\eta=0.95$（DSL 累积阈值），$K=5$（子类数），$\zeta=0.95$（对比采样置信度阈值），$\sigma=0.5$（InfoNCE 温度），$\lambda_1=0.2$、$\lambda_2=0.1$（损失权重），$N_s=255$（负样本数），$\theta=0.99$（EMA 动量系数），backbone=ResNet-101，decoder=DeepLabV3+，batch=8 labeled + 8 unlabeled，resolution=513²（VOC）/ 769²（Cityscapes），epochs=80（VOC）/ 200（Cityscapes），lr_backbone=$10^{-3}$（VOC）/$10^{-2}$（Cityscapes）。
