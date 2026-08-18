---
title: "SegGPT-Towards-Segmenting-Everything-In-Context"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:31:32"
field: "视觉通用模型"
keywords: ["in-context learning", "segmentation", "generalist model", "random coloring", "feature ensemble"]
innovations: ["随机着色训练方案避免多任务坍缩", "特征集成策略聚合多示例上下文信息", "上下文提示调优实现零参数适配"]
benchmarks: ["COCO-20i", "PASCAL-5i", "FSS-1000", "YouTube-VOS 2018", "DAVIS 2017", "MOSE", "ADE20K", "COCO Panoptic"]
---

# 论文速读：SegGPT: Towards Segmenting Everything In Context

## 一句话总结
本文提出 SegGPT，一个通用的 in-context 分割模型，将各种分割任务统一为相同的图像格式，通过随机着色训练方案实现单一模型对实例分割、语义分割、视频对象分割等多种任务的零样本/少样本推理能力。

## 研究问题与动机
- **专用模型可扩展性差**：现有分割模型针对特定任务、类别、粒度或数据类型设计，切换任务或适应新场景需重新训练，成本高昂。
- **多任务训练易坍缩**：传统多任务学习方法容易让模型依赖固定的输出空间（如预定义颜色），而非真正学习上下文推理能力。
- **缺乏统一框架**：不同分割数据（part、semantic、instance、panoptic、视频等）格式各异，难以直接统一训练。
- **out-of-domain 泛化需求**：需要在未见过的类别或场景上实现灵活分割，而现有方法难以满足这一需求。

## 核心贡献（创新点）
1. **首个通用 in-context 分割模型**：首次展示单一模型可自动执行多种分割任务（实例、stuff、part、轮廓、文本等）。
2. **随机着色训练方案**：提出 in-context coloring with random color mapping，强制模型依赖上下文信息而非固定颜色来完成任务，避免多任务学习坍缩。
3. **两种上下文集成策略**：提出空间集成（Spatial Ensemble）和特征集成（Feature Ensemble），有效利用多示例提示提升推理性能。
4. **上下文提示调优（In-Context Tuning）**：在不更新模型参数的情况下，通过优化可学习图像张量实现特定数据集/场景的自适应，兼具通用性与定制化能力。

## 方法详解
- **基础架构**：基于 Painter 框架，使用 vanilla ViT-L（307M 参数），输出空间定义为"图像"，将各类分割任务统一为图像修复（inpainting）问题。
- **随机着色方案**：训练时，从与输入图像共享相似上下文（如同一语义类别或实例）的图像中随机采样颜色映射，每个样本使用不同的随机配色，使模型学会根据上下文完成分割而非记忆固定颜色。
- **上下文集成**：
  - **Spatial Ensemble**：将多个示例拼接为 n×n 网格后下采样到单示例尺寸，适合低分辨率数据。
  - **Feature Ensemble**：在 batch 维度合并多个示例，查询图像的特征在每个 attention 层后取平均，充分聚合多示例信息，适合高分辨率数据。
- **In-Context Tuning**：冻结整个预训练模型，仅优化一个可学习的图像张量作为输入上下文，用于适配特定数据集（如 ADE20K）或场景。
- **损失函数**：沿用 Painter 的 smooth-$\ell_1$ 损失。
- **训练配置**：AdamW 优化器，base LR 1e-4，batch size 2048，训练 9K 次迭代（1.8K warm-up），输入分辨率 448×448，使用随机裁剪、颜色抖动、水平翻转等增强。

## 实验与结果
- **数据集**：ADE20K、COCO、PASCAL VOC、Cityscapes、LIP、PACO、视网膜血管分割（CHASE DB1、DRIVE、HRF、STARE）、航拍图像（iSAID、loveDA）。
- **Few-shot 语义分割**：
  - COCO-20^i：SegGPT one-shot 56.1，优于 specialist FPTrans（56.5*）；few-shot 67.9，远超 FPTrans（65.5*）。
  - PASCAL-5^i：one-shot 83.2，few-shot 89.8，均显著优于所有 baseline。
  - FSS-1000（out-of-domain）：one-shot 85.6，few-shot 89.3，未训练过该数据集但具有竞争力。
- **视频对象分割**（未在训练中使用视频数据）：
  - YouTube-VOS 2018：G=74.7，J_u=67.4，F_u=75.9，优于Painter（G=24.1）；在 MOSE 上与 SOTA RDE 相当（J&F=45.1 vs 48.8）。
  - DAVIS 2017：J&F=75.6（8 frames Feature Ensemble）。
- **Abaltion**：Feature Ensemble 在高分辨率 DAVIS 上显著优于 Spatial Ensemble；最优帧数为 8 帧。
- **In-Context Tuning**：ADE20K mIoU=39.6（比 Painter 降 10.3）；COCO panoptic PQ=34.4（比 Painter 降 9.0），说明随机着色使 in-domain 任务优化难度增加。

## 相关工作脉络
- **Painter [50]**：本文基础框架，将视觉任务统一为 in-context coloring，但 Painter 使用预定义颜色空间，易坍缩为多任务学习；SegGPT 引入随机着色解决此问题。
- **Visual Prompting [3]**：Bar et al. 通过图像修复实现视觉提示，但未统一多种分割任务，且为任务特定设计。
- **UViM [27]**：统一像素级标注任务（panoptic、depth、colorization），但为每个任务训练独立模型；SegGPT 为单一通用模型。
- **Pix2Seq [8, 9]**：将视觉任务输出定义为离散 token 序列，自回归生成；SegGPT 保持像素级连续输出，更适合密集预测任务。
- **OFA [49] / Unified-IO [37]**：统一视觉-语言跨模态任务，但依赖特殊 token 指示任务类型，泛化到新任务困难；SegGPT 通过 in-context 示例灵活定义任务。
- **HODOR [1]**：无需视频数据训练的视频分割 specialist，但仅适用于特定任务；SegGPT 为通用模型。

## 局限性与未来方向
- **In-domain 性能下降**：随机着色使模型在拥有充足训练数据的数据集（如 ADE20K、COCO panoptic）上性能低于 Painter 和 specialist 方法。
- **模型规模限制**：当前使用 ViT-L（307M 参数），未进行大规模缩放实验。
- **训练数据规模**：需要更多数据来支撑更大模型的训练。
- **未来方向**：扩大模型规模、探索自监督学习获取更多数据、扩展到更广泛的应用场景。

## 研究启发与可借鉴点
1. **随机着色避免多任务坍缩**：该思路可迁移到其他 in-context 视觉学习任务，防止模型依赖固定的输出空间分布。
2. **Feature Ensemble 策略**：在 attention 层中间聚合多示例特征的方法可用于其他多示例推理场景（如检测、captioning）。
3. **In-Context Tuning 范式**：冻结大模型、仅优化可学习图像张量的方式，为视觉任务的 prompt tuning 提供了新思路，无需微调参数即可适配特定场景。
4. **统一格式转换**：将不同分割数据转换为相同图像格式的思路，可推广至其他多任务视觉学习场景，简化数据管线。
5. **无需视频数据的视频分割**：通过静态图像的 in-context 示例实现视频分割，为数据稀缺场景提供了新思路。

## 关键术语表
- **In-Context Learning**：通过在输入中提供示例（context）来引导模型完成特定任务，无需更新模型参数。
- **Random Coloring**：为每个训练样本随机映射目标颜色，迫使模型依赖上下文关系而非固定颜色进行推理。
- **Feature Ensemble**：在推理时将多个示例的查询特征在 batch 维度合并，并在每个 attention 层后取平均，实现多示例信息聚合。
- **Spatial Ensemble**：将多个示例图像拼接为网格并下采样到输入分辨率，以空间方式融合上下文信息。
- **In-Context Tuning**：冻结预训练模型，仅优化可学习的图像张量作为 context prompt，适配特定数据集或场景。
- **Smooth-$\ell_1$ Loss**：结合 L1 和 L2 优势的损失函数，对小误差敏感、对异常值鲁棒。

## 可复现要素
- **数据集**：ADE20K、COCO、PASCAL VOC、Cityscapes、LIP、PACO、CHASE DB1、DRIVE、HRF、STARE、iSAID、loveDA（均为公开数据集）。
- **代码**：已开源，见 https://github.com/baaivision/Painter。
- **权重**：使用 Painter [50] 预训练 checkpoint 初始化。
- **关键超参**：ViT-L（307M），LR=1e-4，batch size=2048，9K iterations（1.8K warm-up），分辨率 448×448，AdamW，weight decay=0.05，cosine scheduler。
