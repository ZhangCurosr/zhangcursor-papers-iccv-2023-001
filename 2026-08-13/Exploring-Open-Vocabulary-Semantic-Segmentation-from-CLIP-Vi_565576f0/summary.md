---
title: "Exploring-Open-Vocabulary-Semantic-Segmentation-from-CLIP-Vi"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:21:08"
field: "开放词汇语义分割"
keywords: ["open-vocabulary semantic segmentation", "zero-shot segmentation", "CLIP distillation", "masked autoencoder", "segment tokens", "multi-scale feature distillation"]
innovations: ["仅蒸馏 CLIP 视觉编码器知识实现零标注开放词汇语义分割", "提出多尺度特征蒸馏损失与段匹配损失实现视觉知识向 pixel-level 任务迁移", "以 1/20 训练数据和 1/9 计算成本达到与 GroupViT 相当甚至更优的零样本分割性能"]
benchmarks: ["PASCAL VOC 2012", "PASCAL Context", "COCO", "Conceptual Caption (open-vocabulary human study)"]
---

# 论文速读：Exploring-Open-Vocabulary-Semantic-Segmentation-from-CLIP-Vi

## 一句话总结
本文提出 **ZeroSeg**，一种仅通过蒸馏预训练 CLIP 视觉编码器知识即可实现开放词汇零样本语义分割的新方法，全程无需任何像素级标注或文本标注，在 ImageNet-1k 上训练即达到与使用 26M 图像-文本对训练的 GroupViT 相当甚至更优的性能。

## 研究问题与动机
- **标注成本瓶颈**：语义分割严重依赖昂贵的人工像素级标注，难以扩展到大规模无标注数据集或开放词汇场景。
- **CLIP 的像素级适配难题**：CLIP 等 VL 模型虽学习了丰富的视觉概念，但仅在图像级别训练，缺乏细粒度空间信息，直接迁移到像素级分割效果不佳。
- **已有零样本方法的局限**：GroupViT、MaskCLIP、SegCLIP 等工作仍需依赖大量图像-文本对（20M–26M）或伪像素级标注，且存在文本偏见（对复合词/子词理解差）。
- **训练效率低下**：现有 SOTA 方法（如 GroupViT）需 ~768 GPU 小时，资源开销巨大。

## 核心贡献（创新点）
1. **提出 ZeroSeg 框架**：仅蒸馏 CLIP 视觉编码器知识进行开放词汇零样本语义分割，无需任何标注；与 GroupViT/MaskCLIP 等依赖文本或伪标签的方法本质不同。
2. **设计多尺度特征蒸馏损失（Multi-scale Feature Distillation Loss）**：将 CLIP 多尺度局部特征蒸馏到全局表示 z，使模型捕获不同粒度的对象语义；区别于仅使用全图特征的传统做法。
3. **引入段匹配损失（Segment Matching Loss）**：强制每个 segment token 与最近的局部区域特征对齐，避免语义不一致的分割区域；该设计在零样本设定下首次有效解决 token-区域映射问题。
4. **验证高数据效率与计算效率**：仅在 1.3M ImageNet-1k 上训练（约为 GroupViT 数据的 1/20），VOC mIoU 达 40.8；训练仅需 ~84 GPU 小时（GroupViT 的 1/9）。
5. **开放词汇定性/定量优势**：在 1000 类开放词汇人类评测中 ZeroSeg 获 677/1000 票（GroupViT 仅 323/1000），且在含子词/复合词类别（如 ground, bedclothes）上平均提升 18.07% mIoU。

## 方法详解
**整体架构**：基于 MAE（Masked Autoencoder）的 ViT-base 不对称编码器-解码器结构。编码器 12 层 Transformer，仅输入 40% 可见 patch；解码器分两支：重建解码器（8 层 Transformer，目标像素级 MSE 重构）与分割头（5 层 Transformer + 两级分组层，分别使用 32 和 8 个可学习 segment token）。

**多尺度图像特征提取**：将输入图像划分为 2×2、3×3、4×4 等非重叠网格，各视图 resize 至 224×224 后输入预训练 CLIP-L 视觉编码器，得到多尺度图像特征集合 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$（$\mathbf{v}_1$ 为整图特征，其余为局部视图特征）。所有多尺度特征可预先缓存以提升训练效率。

**多尺度特征蒸馏损失**：对 segment tokens 经 Transformer 编码、平均池化与 MLP 后得到全局表示 $\mathbf{z}$，对每个 $\mathbf{v}_i$ 计算 L1 蒸馏损失：
$$\mathcal{L}_{distill} = \sum_{i} |\mathbf{z} - \mathbf{v}_i|_1$$
迫使全局表示捕获 diverse regional semantic representation。

**段匹配损失（Segment Matching Loss）**：对每个 segment token $\mathbf{g}_i$，在所有局部区域特征（排除 $\mathbf{v}_1$ 整图特征）中寻找最近邻 $\mathbf{v}_j$，最小化 L1 距离：
$$\mathcal{L}_{match} = \sum_{i=1}^{m} \min_{j} |\mathbf{g}_i - \mathbf{v}_j|_1$$
鼓励每个 token 聚焦于特定对象-centric 语义区域，避免 mask 像素泄漏到邻域对象。

**总损失**：$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{distill} + \mathcal{L}_{match}$，其中 $\mathcal{L}_{recon}$ 为 MAE 像素级重构损失。

**分组层级**：借鉴 GroupViT 的分层 grouping block，逐步将小 token 组聚合为大 token 组，最终输出固定数量（40 个）的 disjoint 图像区域 segment tokens。

## 实验与结果
**数据集**：PASCAL VOC 2012（20 类）、PASCAL Context（59 类）、COCO（80 类）；训练数据为 ImageNet-1k（1.3M 图像），消融实验使用 CC3M+COCO（3.4M）及 CC12M+IN-1K。

**评估协议**：推理时用预定义 prompt（如 "a photo of the {class}"）生成 CLIP 文本嵌入，计算 segment token 与 class embedding 的 cosine similarity，阈值过滤背景后取最近邻分类；VOC/Context/COCO 阈值分别为 0.95 / 0.05 / 0.35，短边 resize 至 448。

**主要结果（mIoU）**：
| 模型 | 训练数据 | VOC | Context | COCO |
|------|---------|-----|---------|------|
| ZeroSeg (Ours) IN-1k | 1.3M 纯图像 | **40.8** | 20.4 | **20.2** |
| GroupViT | 26M 图像-文本 | 52.3 | 22.4 | 24.3 |
| ZeroSeg (Ours) CC3M+COCO | 3.4M 图像 | 37.3 | 19.7 | 17.8 |
| MaskCLIP | 20M 图像-文本 | — | 17.7 | 11.8 |
| SegCLIP | 3.4M 图像-文本 | 33.3 | 19.1 | 15.2 |

- **相对 GroupViT**：在相近数据量（CC3M+COCO 3.4M）下 VOC +9.2%，COCO +8.4；即便仅用 1.3M 图像也优于 MaskCLIP（20M 数据）的 Context +2.7 / COCO +8.4。
- **计算效率**：ZeroSeg ~84 GPU 小时 vs GroupViT ~768 GPU 小时（**1/9**）。
- **开放词汇人类评测**：ZeroSeg 获 677/1000 票（68%），显著优于 GroupViT（323/1000，32%）。
- **子词/复合词鲁棒性**：ZeroSeg 在 ground/bedclothes/keyboard/motorbike 等类别上平均 27.85 mIoU，GroupViT 仅 9.78（+18.07%）。
- **扩展性**：CC12M+IN-1K 训练时 VOC 达 42.9，超过 GroupViT(CC12M) 的 41.1。

**Ablation 关键结论**：
- 多尺度特征：仅 1×1 全图特征 VOC=21.1；加入 3×3 后 40.2；全组合 40.8（几乎翻倍）。
- 段匹配损失：Base 21.1 → Base+Multi-scale 28.5 → Base+Segment matching 38.6 → 完整 40.8；segment matching 单独贡献 +17.5（相对于 Base）。
- Mask ratio：60% 最优（40.8 mIoU，+36% 加速），高于 MAE 的 75%（因像素级任务需更多可见像素）。

## 相关工作脉络
- **GroupViT (CVPR'22)**：首个通过文本监督学习开放词汇分割的方法，依赖 26M 图像-文本对；ZeroSeg 以 1/20 数据和零标注实现可比性能，且消除文本偏见。
- **MaskCLIP / MaskCLIP+ (arXiv'22)**：基于 CLIP 视觉编码器分类 mask segments，需预训练 mask generator 并在目标数据集上做 adaptation；ZeroSeg 完全 zero-shot 无适配。
- **SegCLIP (arXiv'22)**：使用 CC3M+COCO 文本+CLIP 文本编码器双重监督；ZeroSeg 仅用 CLIP 视觉编码器，不依赖任何文本信号。
- **DenseCLIP / OpenSeg**：利用伪像素级标注或大规模图文数据；ZeroSeg 无需任何形式的 dense label 或 image-text pair。
- **CLIP 直接应用**：仅图像级训练，缺乏像素级空间表征；本文证明通过蒸馏+多尺度+segment matching 可有效 bridge 这一 gap。
- **MAE (CVPR'23)**：本文架构基础，但将 MAE 从自监督重建延伸至视觉知识蒸馏到 segmentation head。

## 局限性与未来方向
- **分割边界较粗糙**：相比像素级监督方法（如 SegFormer/Mask2Former），ZeroSeg 的 boundary 精度仍不足，限制其在需精细分割的实际场景部署。
- **依赖 CLIP 视觉编码器质量**：蒸馏上限受 teacher 能力约束；作者建议可探索更强 VL 基础模型（如 GPT-4V）作为蒸馏教师。
- **未利用文本编码器**：仅蒸馏视觉侧知识，文本编码器的语义对齐能力未被充分挖掘。
- **可能继承 CLIP 偏见**：训练数据（ImageNet/web images）存在的 social bias 会传递给下游模型，需配合数据过滤等缓解手段。
- **未来方向**：① 结合少量像素级微调进一步提升边界质量；② 探索多模态 teacher（视觉+文本联合蒸馏）；③ 扩展到实例分割、全景分割等更广义任务；④ 与 GPT-4 等新一代 VL 模型结合实现更强 open-vocabulary 能力。

## 研究启发与可借鉴点
1. **视觉编码器蒸馏范式**：将预训练 VL 模型的视觉知识"无标注"迁移至下游像素级任务，可复用于实例分割、深度估计、法向量预测等任务，作为低资源预训练的通用范式。
2. **多尺度特征提取策略**：将单图划分为多个 grid 视图并各自过 teacher encoder 提取特征，是一种简单有效的局部语义捕获手段，无需额外标注即可补充空间信息。
3. **Segment Matching Loss 设计**：通过 nearest-neighbor L1 距离将可学习 token 与局部区域对齐，避免语义混淆，思路可迁移至任意基于 token 的分组/聚类任务（如设码向量量化、open-set recognition）。
4. **预计算缓存多尺度特征**：CLIP 特征可离线缓存，大幅降低训练时间；该技巧适用于任何使用固定 teacher 做蒸馏的场景。
5. **文本无关分割的鲁棒性优势**：摆脱文本监督可避免 word-level bias（如复合词/子词误解），启示团队在标注敏感任务中考虑纯视觉蒸馏路径以降低语言偏见风险。

## 关键术语表
**ZeroSeg**：本文提出的开放词汇零样本语义分割模型，仅通过蒸馏 CLIP 视觉编码器知识训练，无需像素级或文本标注。
**CLIP Vision Encoder**：OpenAI 预训练的视觉-语言模型中的视觉分支（ViT-L/14），输出与文本空间对齐的图像级表示，本文作为蒸馏 teacher。
**Segment Tokens**：可学习的固定数量 token（本文 40 个），经分组层级聚合后各自对应图像中一个 disjoint 语义区域。
**Multi-scale Image Feature Distillation**：将图像切分为 2×2/3×3/4×4 等多尺度视图，经 CLIP 提取局部特征后蒸馏到全局表示 z 的损失机制。
**Segment Matching Loss**：强制每个 segment token 与其最近的局部区域特征最小化 L1 距离，促进 token-区域语义对齐的损失函数。
**MAE (Masked Autoencoder)**：被 Kaiming He 等提出的掩码自编码器，本文以其不对称 encoder-decoder 架构为基础，并调整 mask ratio 至 60%。
**GroupViT**：Xu et al. (CVPR'22) 提出的通过文本监督学习语义分组的方法，本文在其分组层级设计上有所借鉴但完全去除文本依赖。
**Open-vocabulary Segmentation**：在训练阶段未见类别上仍能进行语义分割的任务设定，本文通过 1000 类 ImageNet 词汇模拟评估。

## 可复现要素
- **数据集**：训练用 ImageNet-1k（公开）；评测用 PASCAL VOC 2012、PASCAL Context、COCO（均公开）；消融用 CC3M、CC12M（部分公开）。
- **代码**：https://github.com/facebookresearch/ZeroSeg（GitHub，已开源）。
- **权重**：论文未提及公开权重地址，代码仓库应包含模型 checkpoints。
- **关键超参**：ViT-base（12 层 encoder，8 层 recon decoder，5 层 seg decoder）；分组 token 数 32+8；学习率 1.5e-4；batch size 4096；80 epochs（前 20 warmup）；AdamW；mask ratio 60%；输入 resize 224×224；中心裁剪；CLIP-L vision encoder；推理短边 resize 448；阈值 VOC=0.95 / Context=0.05 / COCO=0.35。
