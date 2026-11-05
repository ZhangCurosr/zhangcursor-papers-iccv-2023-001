---
title: "Distilling-Large-Vision-Language-Model-with-Out-of-Distribut"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:36:04"
field: "多模态表示学习"
keywords: ["模型蒸馏", "视觉-语言模型", "分布外泛化", "开放词汇", "知识蒸馏", "少样本学习"]
innovations: ["提出相对视觉空间对齐损失Im-cst替代MSE", "设计VLProx损失保留教师视觉-语言对齐结构", "利用LLM生成细粒度语义描述增强语言表示"]
benchmarks: ["Caltech-Birds", "StanfordCars", "Flower102", "Food101", "SUN397", "tiered-ImageNet"]
---

# 论文速读：Distilling-Large-Vision-Language-Model-with-Out-of-Distribut

## 一句话总结
本文研究将大型视觉-语言模型（VLM）的表示空间蒸馏到轻量级学生模型中，重点解决开放词汇分布外（OOD）泛化问题。通过改进视觉表示空间对齐和丰富语言表示，显著提升学生在零样本和少样本OOD分类任务上的性能。

## 研究问题与动机
1. **核心问题**：大型VLM（如CLIP）虽然泛化能力强，但计算成本高，难以部署到资源受限设备；如何将大型VLM的表示能力蒸馏到小型学生模型并保留OOD泛化能力？
2. **现有方法不足**：现有模型蒸馏研究主要关注分布内性能，很少考虑开放场景中OOD泛化能力；直接最小化教师-学生视觉特征距离（MSE损失）在实践中效果不佳，因为高精度匹配在高维空间中 inherently challenging。
3. **小/中等规模数据集的实用性**：互联网规模数据虽好，但小/中规模数据集更灵活、开发周期更快、资源依赖更少，适合机器人等应用场景。
4. **视觉-语言对齐的缺失**：传统蒸馏只关注视觉表示，忽视了视觉-语言对齐结构的一致性，而这对于开放词汇泛化至关重要。

## 核心贡献（创新点）
1. **提出评估教师-学生视觉空间一致性指标**：设计 $\mathcal{M}_{rel}$（相对特征关系保持度）和 $\mathcal{M}_{neigh}$（局部邻域结构相似度）两个指标，量化学生模型对教师视觉表示空间的保留程度。
   - *区别*：不同于传统蒸馏仅关注绝对距离，本文强调相对关系和局部结构的保持对OOD泛化更重要。

2. **改进教师-学生视觉空间对齐策略**：提出对比损失 $\mathcal{L}_{im-cst}$ 替代MSE损失，"软化"匹配教师视觉特征，保留相对距离关系而非绝对位置。
   - *区别*：发现更低MSE不一定带来更好泛化，相对结构一致性比绝对距离更重要。

3. **显式增强视觉-语言对齐一致性**：设计 $\mathcal{L}_{vl-prox}$ 损失，过滤教师误对齐样本后，保留教师前k个最相似语言特征的相对顺序。
   - *区别*：传统蒸馏直接蒸馏logits，本文针对VLM特性设计，并考虑教师可能存在的噪声对齐。

4. **语言表示丰富化策略**：使用LLM（ChatGPT）生成细粒度语义描述替代简单类别名，并使用OFA生成辅助图像描述。
   - *区别*：首次系统研究语言表示质量对蒸馏效果的影响，发现LLM生成的语义描述显著优于辅助caption。

## 方法详解

### 4.1 视觉空间对齐
** baseline（仅分类损失）**：
$$\mathcal{L}_{cls}(x,y) = \sum_{y'} -\mathbf{1}_{y'=y}\log P_S(y'|x)$$
其中 $P_S(y|x)$ 基于学生视觉特征与教师文本特征的余弦相似度。

**MSE损失（直接拟合）**：
$$\mathcal{L}_{mse}(x) = ||S(x) - T_{img}(x)||_2^2$$
实验发现学生难以精确复现教师特征（Table 2显示MSE仍较大），且低MSE不代表好泛化。

**改进的对比损失**：
$$\mathcal{L}_{im-cst}(x) = \frac{\exp(-||S(x)-T_{img}(x)||_2^2/\tau)}{\sum_{x'}\exp(-||S(x)-T_{img}(x')||_2^2/\tau)}$$
鼓励学生特征与其对应的教师特征更接近，而非所有其他教师特征。

**评估指标**：
- $\mathcal{M}_{rel}$：学生特征最近邻是否为对应教师特征的比例（公式4）
- $\mathcal{M}_{neigh}$：学生与教师k近邻的重叠度（公式5）

### 4.2 视觉-语言对齐一致性
**VLProx损失**：
$$\mathcal{L}_{vl-prox}(x,k) = I(x) \cdot D_{KL}(P_{T,topk}(\cdot|x) || P_{S,topk}(\cdot|x))$$
其中 $I(x)$ 过滤教师预测错误的样本，$P_{topk}$ 基于教师top-k最相似语言特征的概率分布。
- 关键超参：$k=256$（越大越好）

**评估指标**：
- $\mathcal{M}_{vl-align}$：衡量学生与教师在语言特征排序上的一致性（逆序对数量，越低越好）

### 5. 语言表示丰富化
**LLM增强描述**：
使用ChatGPT生成细粒度外观描述：
> "Use a single sentence to describe the appearance and shape of {cls}. Only describe the shape and appearance"

例如"trumbone"生成："large, round, flat leaves; tall, slender stems; delicate petals in shades of pink, white, or yellow"

**辅助caption**：
使用OFA为每张图片生成描述，添加辅助损失：
$$\mathcal{L}_{cap}(x) = \frac{\exp(\cos(S(x), T_{txt}(cap(x)))/\tau)}{\sum_{x':y'\neq y}\exp(\cos(S(x), T_{txt}(cap(x')))/\tau)}$$

**Prompt设计比较**（Table 6）：
- More Succinct/Detailed/Distinct 三种prompt效果不同，并非越详细越好
- 原因：LLM生成的细节未grounded到具体图像，且CLIP文本嵌入对细粒度细节不敏感

## 实验与结果

**数据集**：Caltech-Birds, StanfordCars, Flower102, Food101, SUN397, tiered-ImageNet（训练/ID/OOD标签各半）

**评估设置**：
- $X_{id}$: 分布内分类（训练标签）
- Zero-shot $X_{ood}$: 未见类别零样本
- 5-shot $X_{ood}$: 未见类别少样本微调后评估

**主要结果（Table 1，ResNet18学生 + CLIP ViT-L/14教师）**：

| 方法 | Caltech-Birds | StanfordCars | Flower102 | Food101 | SUN397 | tiered-ImageNet | 平均 |
|------|---------------|--------------|-----------|---------|--------|-----------------|------|
| $\mathcal{L}_{cls}$ | 61.0/14.2/34.9 | 56.3/14.7/20.0 | 81.2/4.5/46.2 | 72.2/16.1/24.5 | 57.5/13.6/28.6 | 64.4/13.9/27.5 | **65.4/12.8/30.3** |
| $\mathcal{L}_{cls}+\mathcal{L}_{im-cst}$ | 60.9/20.4/37.6 | 59.6/18.3/31.2 | 82.4/12.7/52.5 | 74.0/30.5/42.0 | 62.5/18.8/35.2 | 64.4/18.0/33.5 | **67.3/19.8/38.7** |
| $\mathcal{L}_{cls}+\mathcal{L}_{im-cst}+\mathcal{L}_{vl-prox}$ | 62.3/21.6/39.0 | 63.9/19.8/38.5 | 82.7/14.6/52.0 | 74.3/32.0/43.2 | 61.7/21.5/34.7 | 67.5/20.5/35.3 | **68.7/21.7/40.5** |

**最强结果**：完整方法在tiered-ImageNet上达到67.5/20.5/35.3（ID/0-shot/5-shot），平均提升约+5.4%/+8.9%/+10.2%相对baseline。

**语言丰富化效果（Table 5）**：
- LLM语义描述：显著提升零样本OOD（如tiered-ImageNet从20.5→23.5）
- 辅助caption：对零样本帮助有限
- 组合最佳：Semantic+Caption（平均69.4/24.1/41.5）

**ViT-B/32学生实验（Table 9）**：
发现结论一致，证明方法对不同架构有效。

**少样本策略（Table 10）**：
微调（5-shot Finetune）显著优于无训练检索（5-shot Retrieval）：Caltech-Birds 40.36 vs 24.39。

**机器人应用（Table 11）**：
Pick-Clutter抓取可行性预测任务中，完整方法zero-shot准确率84.3/92.0（ID/OOD）。

## 相关工作脉络
1. **VLM蒸馏**：传统蒸馏聚焦纯视觉骨干（CNN/ViT），本文首次系统研究VLM的多模态蒸馏，特别是OOD泛化。
2. **对比表示蒸馏**：如CRTD（Tian et al., 2019）关注视觉表示，本文扩展到视觉-语言对齐结构。
3. **提示学习**：Prompt-based方法（Zhou et al., 2022; Liu et al., 2022）优化输入prompt，本文优化蒸馏目标函数。
4. **少样本学习**：TipAdapter（Zhang et al., 2021）提出无训练检索，本文发现微调更适合小规模蒸馏学生。
5. **LLM辅助生成**：利用LLM增强标签描述，相关工作如HowMuchCanCLIPBenefit等未深入此方向。

## 局限性与未来方向
1. **数据集规模限制**：当前工作在小/中规模数据集上验证，未测试互联网规模数据的效果。
2. **仅关注分类任务**：扩展到检测、分割、3D几何等任务需进一步验证。
3. **LLM依赖**：语言丰富化依赖外部LLM（ChatGPT），存在成本和隐私问题。
4. **教师网络固定**：假设教师表示已很好对齐，未研究教师选择的影响。
5. **未来方向**：扩展到更多模态（3D）、更大规模数据、更高效的语言生成策略。

## 研究启发与可借鉴点
1. **相对结构比绝对距离重要**：蒸馏时应优先保持表示空间的拓扑结构（相对距离、邻域关系），而非强行匹配绝对特征值。
2. **评估指标设计**：$\mathcal{M}_{rel}$、$\mathcal{M}_{neigh}$、$\mathcal{M}_{vl-align}$ 提供了可量化的分析工具，可用于诊断蒸馏效果。
3. **语言表示质量是关键**：丰富的语义描述对OOD泛化影响显著，未来可探索自动化细粒度描述生成。
4. **教师噪声过滤**：$\mathcal{L}_{vl-prox}$ 中的 $I(x)$ 过滤机制值得借鉴，可避免蒸馏教师错误对齐样本。
5. **结合微调与蒸馏**：对于小模型，少样本微调比无训练检索更有效，实验设计值得参考。

## 关键术语表
**Out-of-Distribution (OOD)**：模型在训练时未见过的新类别或新场景上的泛化能力。
**Vision-Language Alignment**：图像特征与文本特征在多模态空间中的对应关系。
**Open-Vocabulary**：模型能识别训练时未见过的类别，只需提供类别名称即可。
**Knowledge Distillation**：将大型教师模型的知识迁移到小型学生模型的过程。
**Contrastive Loss**：通过拉近正样本对、推远负样本对来学习表示的 Loss。
**Few-shot Learning**：仅用少量标注样本（如5个）进行微调的学习范式。
**Teacher-Student Framework**：蒸馏中的经典框架，教师提供指导信号，学生学习。
**Singular-Value Decomposition (SVD)**：矩阵分解技术，用于分析特征空间的结构。

## 可复现要素
- **数据集**：Caltech-Birds, StanfordCars, Flower102, Food101, SUN397, tiered-ImageNet（公开可用）
- **代码**：论文声明代码已开源（"Code released at this link"，具体链接见原文）
- **权重**：教师使用CLIP ViT-L/14（公开），学生为ResNet18/ViT-B/32（从头训练）
- **关键超参**：温度τ（未明确，标准设置）、$\mathcal{L}_{vl-prox}$ 中k=256、训练批次平衡（最多一半few-shot数据）
- **辅助模型**：ChatGPT（OpenAI API）、OFA（生成caption）
