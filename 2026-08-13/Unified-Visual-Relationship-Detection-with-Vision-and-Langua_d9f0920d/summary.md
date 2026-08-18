---
title: "Unified-Visual-Relationship-Detection-with-Vision-and-Langua"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Unified_Visual_Relationship_Detection_with_Vision_and_Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:35:01"
field: "视觉关系检测"
keywords: ["视觉关系检测", "视觉语言模型", "统一标签空间", "人-物交互", "场景图生成", "VLM", "多数据集训练"]
innovations: ["利用 VLM 文本嵌入统一异构数据集标签空间", "级联训练架构结合索引预测实现高效关系解码", "首次证明统一 VRD 模型可比肩专用模型并具可扩展性"]
benchmarks: ["HICO-DET", "V-COCO", "Visual Genome"]
---

# 论文速读：Unified-Visual-Relationship-Detection-with-Vision-and-Langua

## 一句话总结
本文提出 UniVRD，一种基于视觉-语言模型（VLM）的统一视觉关系检测框架，通过语言嵌入空间对齐异构数据集的标签，实现单个模型在多数据集上的联合训练。该方法在 HICO-DET 上达到 38.07 mAP 的最强性能，同时证明统一模型可与数据集专用模型相当，且在模型缩放时进一步提升长尾类别性能。

## 研究问题与动机
- **多数据集标签异构性**：现有 VRD 模型通常仅使用单一数据源训练，不同数据集的物体类别和关系谓词存在不一致（如同义词、上下位词、重叠词），手动合并标签空间极其困难。
- **二阶语义复杂性**：视觉关系涉及主体-谓词-客体三元组，组合复杂度高于单体检测，加剧了标签统一挑战。
- **统一模型的泛化瓶颈**： prior work 训练统一模型时普遍出现性能下降，无法达到专用模型水平。
- **VLM 语义对齐潜力未发掘**：预训练 VLM（如 CLIP、LiT）已具备图像-文本语义对齐能力，但此前无人将其用于 VRD 的标签统一。

## 核心贡献（创新点）
1. **首次利用 VLM 实现 VRD 多数据集统一**：提出 UniVRD 框架，通过 VLM 的文本嵌入空间统一异构标签，相比 prior work 无需人工合并标签体系。
2. **自底向上级联训练设计**：编码器-only 的 ViT 检测器 + Transformer 关系解码器，避免冗余预测且支持直接复用预训练知识，无需知识蒸馏或检测专用预训练。
3. **语言定义标签空间**：用自然语言提示（如 "a person riding a horse"）替代类别整数，使语义相似关系在嵌入空间中邻近，支持开放词汇推理。
4. **统一模型可比肩专用模型并具可扩展性**：首次在 VRD 任务上证明统一模型 mAP 可与数据集专用模型相当，且在缩放模型时进一步带来提升（HICO-DET 提升 4.49 mAP on Rare categories）。

## 方法详解
**架构设计**：
- **对象检测器**：采用 encoder-only ViT 架构，移除 pooling 和最终投影层，每个输出 token 经线性层映射为实例嵌入 $z_i$，再经 FFN 预测边界框坐标 $b_i$。
- **关系解码器**：追加 Transformer decoder，输入为可学习的关系查询和检测器的实例嵌入，输出关系嵌入 $r_j$。使用类似 Perceiver Resampler 的键值拼接策略提升效率。
- **索引预测机制**：关系嵌入通过 FFN 分别投影为主体/客体嵌入，通过余弦相似度匹配实例嵌入集合 $\mathcal{Z}$，获得主体/客体框索引（非坐标），避免冗余预测。

**分类策略**：
- 对象分类：将类别名转换为提示模板（如 "a photo of a {object}"），经文本编码器生成文本查询，与实例嵌入计算 cosine similarity 后使用 focal sigmoid cross-entropy 损失。
- 关系分类：三元组转为提示（如 "a {subject} {predicate}-ing a {object}"），使用关系文本查询集 $\mathcal{T}_{rel}$ 进行分类。

**损失函数**：
- 检测器匈牙利损失：$\mathcal{L}_{OD} = \frac{1}{N}\sum_{i=1}^{N}[\mathcal{L}_{cls}(z_i, \mathcal{T}_{obj}') + \mathcal{L}_{box}(b_i)]$
- 关系解码器匈牙利损失：$\mathcal{L}_{VRD} = \frac{1}{M}\sum_{j=1}^{M}[\mathcal{L}_{cls}(r_j, \mathcal{T}_{rel}') + \mathcal{L}_{ind}(r_j;\hat{s}_j, \hat{o}_j)]$，其中索引损失用 focal softmax cross-entropy。

**训练策略**：
- 级联训练：第一阶段用边界框数据微调检测器，第二阶段训练关系解码器；小数据量时冻结检测器防过拟合，大数据时微调可进一步提升。
- 数据增强：Mosaic（概率 0.4/0.3/0.3 for 1×1, 2×2, 3×3）、随机裁剪/翻转、CLIP prompt 采样。

## 实验与结果
**数据集与基线**：
- HICO-DET（HOI 检测）：37,536 训练图，600 HOI 三元组
- V-COCO（HOI 检测）：2,533 训练图
- Visual Genome（SGG）：108,077 图像，使用 top-150 物体/top-50 谓词
- 辅助物体检测数据集：COCO、Objects365

**主要结果**：
- **HICO-DET**：UniVRD (LiT: ViT-H/14) 达到 **38.07 mAP**，超越 prior bottom-up 方法 14.26 mAP（60% 相对提升），超越 single-stage 方法如 CDN (31.44 mAP) 和 GEN-VLKT (33.75 mAP)。
- **VG SGG**：mR@50=12.6，mR@100=14.5，与专用 SGG 模型 competitive。
- **V-COCO**：统一模型显著优于专用模型（S#1: +5.21 mAP, S#2: +5.73 mAP）。

**扩展性分析**：
- ViT-B/32 → R26+B/1：大幅提升，GFLOPs 略增
- ViT-L/14 → ViT-H/14：mAP 继续提升 0.7，证明良好可扩展性

**消融实验关键发现**：
- 去掉 Mosaic 降 1.76 mAP，去掉 CLIP prompts 降 1.85 mAP
- 级联训练 vs 端到端训练：+5.05 mAP
- 冻结 vs 微调检测器：数据量大时微调更优

## 相关工作脉络
1. **传统 VRD 方法**：Motif、VCTree、GPS-Net 等通过消息传递或图网络生成场景图，依赖单一数据集且需手动设计损失。
2. **HOI 检测 baseline**：InteractNet、GPNN、iCAN、HOTR、QPIC、CDN、RLIP、GEN-VLKT 等，多聚焦于改进关联建模或噪声过滤。
3. **多数据集统一检测**：MSeg、Universal Object Detection 等工作尝试合并标签空间，但统一模型普遍出现性能下降。
4. **VLM for VRD**：RLIP 利用关系语言-图像预训练做 HOI 检测，PEVL/CPT 探索 prompt tuning，但均未利用 VLM 嵌入进行标签统一。
5. **语言先验 VRD**：DRG、FCMNet、ACP、PD-Net 使用 word embeddings 捕捉语义关系，但局限于固定小词表，不如 VLM 嵌入灵活强大。

## 局限性与未来方向
- **长尾类别处理不足**：当前方法未专门处理极端偏斜/长尾关系类别，可结合数据转移或重采样策略改进。
- **关系层次结构缺失**：对象和谓词在同一层级推理，未建模层次关系，未来可引入更强 VQA-VLM（如 PaLI）支持层次推理。
- **SGG 召回略降**：统一训练在 HOI 领域提升的同时，VG 非 HOI 关系召回略有下降。
- **无背景类设计**：虽避免了对未充分标注样本的惩罚，但可能在某些场景漏检。

## 研究启发与可借鉴点
1. **VLM 嵌入统一标签空间**：利用预训练 VLM 的文本嵌入对齐异构类别，无需人工合并标签，可迁移至多任务/跨域检测。
2. **索引预测替代坐标回归**：关系解码器预测对象索引而非坐标，减少冗余预测，设计简洁高效，适用于 pair-wise 推理任务。
3. **级联训练稳定性**：两阶段训练（检测→关系）结合冻结/微调策略，对小/大数据量自适应，可作为训练复杂级联系统的通用经验。
4. **Mosaic 增强跨数据集融合**：Mosaic 拼接有效融合不同数据集样本，缓解分布差异，适用于多源训练场景。
5. **图像条件关系检索**：用图像嵌入替代文本嵌入作为查询，实现跨图像关系检索，扩展模型应用能力。

## 关键术语表
**Visual Relationship Detection (VRD)**：检测图像中两个对象之间的关系，输出主体-谓词-客体三元组及对应边界框。
**Vision and Language Models (VLMs)**：如 CLIP、LiT，在大规模图像-文本对上预训练，学习对齐的视觉和文本嵌入空间。
**Bottom-up 方法**：先检测实例，再基于实例预测关系，与 single-stage 直接预测三元组形成对比。
**Text Queries**：将类别名或关系三元组转为自然语言提示，经文本编码器生成嵌入，作为分类查询。
**Hungarian Loss**：基于二分图匹配的匈牙利算法，将预测与 ground-truth 最优配对后计算损失。
**Per-class PNMS**：在每个关系类别内执行非极大值抑制，过滤高度重叠的预测结果。
**Cascade Training**：分阶段训练策略，先训练检测器再训练关系解码器，比端到端训练更稳定。
**Focal Loss**：针对难例加权的损失函数，处理非均衡标注数据时比标准交叉熵更有效。

## 可复现要素
- **数据集**：HICO-DET、V-COCO、Visual Genome（公开）、COCO（公开）、Objects365（公开）
- **代码**：论文声明代码将在 GitHub 公开（论文未提供链接）
- **预训练模型**：CLIP (ViT-B/32, ViT-B/16, ViT-L/14)、LiT (ViT-B/32, R26+B/1, ViT-H/14)
- **关键超参**：关系查询数=100，PNMS 阈值=0.7，学习率=1.0×10⁻⁴，batch size=64
- **硬件**：TPUv3
- **实现框架**：JAX + Scenic library
