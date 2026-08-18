---
title: "Distilling-Large-Vision-Language-Model-with-Out-of-Distribut"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:36:02"
field: "视觉语言模型蒸馏与泛化"
keywords: ["vision-language distillation", "out-of-distribution generalization", "knowledge distillation", "open-vocabulary classification", "representation learning"]
innovations: ["提出视觉空间相对对齐损失保留教师表征结构的相对关系", "设计视觉-语言对齐保持损失结合LLM语义增强提升OOD泛化"]
benchmarks: ["Caltech-Birds", "StanfordCars", "Flower102", "SUN397", "tiered-ImageNet"]
---

# 论文速读：Distilling-Large-Vision-Language-Model-with-Out-of-Distribution-Generalizability

## 一句话总结
本文研究如何将大型视觉语言模型（VLM）的视觉表征空间蒸馏到轻量级学生模型中，通过改进教师-学生视觉空间对齐和视觉-语言对齐一致性，并结合大语言模型丰富的语义描述，显著提升学生模型在开放词汇分布外（OOD）场景下的零样本和少样本泛化能力。

## 研究问题与动机
1. **大规模VLM难以部署**：CLIP、GLIP等大型VLM虽具有强大的OOD泛化能力，但模型规模大、计算成本高，无法在移动端、IoT设备和机器人控制等实时场景中部署。
2. **现有蒸馏方法忽视OOD泛化**：当前VLM蒸馏工作主要关注下游任务性能提升，很少研究如何通过蒸馏增强小型学生模型的OOD泛化能力。
3. **视觉空间精确匹配存在固有挑战**：学生模型难以精确复现教师的高维视觉特征空间，且绝对距离最小化并不能保证保留教师视觉空间的相对关系和局部结构。
4. **语言表征质量影响蒸馏效果**：简单使用类别名称作为语言描述会忽略细粒度语义属性，限制了学生对不同类别的有效区分能力。

## 核心贡献（创新点）
1. **提出视觉空间相对对齐损失 $\mathcal{L}_{im-cst}$**：通过对比学习"软化地"匹配教师视觉特征，保留教师视觉空间的相对距离关系和局部结构，而非追求绝对距离最小化。
2. **设计视觉-语言对齐保持损失 $\mathcal{L}_{vlprox}$**：通过KL散度保留教师对每个图像最相似的Top-K语言特征的排序结构，同时过滤教师自身的误对齐样本。
3. **引入LLM增强语言表征策略**：使用ChatGPT为类别生成包含形状、颜色、纹理等细粒度语义属性的描述，并通过SVD分析证明丰富表征能更好地分离不同类别。
4. **提出多种评估度量**：定义 $\mathcal{M}_{rel}$（相对特征关系保持度）、$\mathcal{M}_{neigh}$（局部邻域结构一致性）、$\mathcal{M}_{vllign}$（视觉-语言对齐一致性）等度量来量化分析蒸馏效果。

## 方法详解

**问题设置**：将大型VLM教师（如CLIP ViT-L/14）蒸馏到仅含视觉编码器的小型学生模型（如ResNet18），使用小/中规模数据集进行蒸馏，目标是在ID类和OOD类上均取得良好性能。

**核心损失函数设计**：

1. **分类损失 $\mathcal{L}_{cls}$**：基础对比损失，将学生视觉特征与教师文本特征进行匹配：
$$\mathcal{L}_{cls}(\mathbf{x}, y) = \sum_{y'} -\mathbf{1}_{y'=y} \log P_S(y'|\mathbf{x})$$
其中 $P_S(y|\mathbf{x})$ 基于学生视觉特征与教师文本特征的余弦相似度计算。

2. **视觉空间对齐损失 $\mathcal{L}_{im-cst}$**：对比损失，鼓励学生视觉特征在其对应教师特征附近：
$$\mathcal{L}_{im-cst}(\mathbf{x}) = \frac{\exp(-||S(\mathbf{x}) - T_{img}(\mathbf{x})||_2^2/\tau)}{\sum_{\mathbf{x}'} \exp(-||S(\mathbf{x}) - T_{img}(\mathbf{x}')||_2^2/\tau)}$$
相比MSE损失，该损失关注相对距离而非绝对距离。

3. **视觉-语言对齐保持损失 $\mathcal{L}_{vlprox}$**：保留教师对每个图像最相似语言特征的排序：
$$\mathcal{L}_{vlprox}(\mathbf{x}, k) = I(\mathbf{x}) \cdot \mathcal{D}_{KL}(P_{T,topk}(\cdot|\mathbf{x}) || P_{S,topk}(\cdot|\mathbf{x}))$$
其中 $I(\cdot)$ 过滤教师自身误对齐的样本，$k$ 控制Top-K语言特征数量（实验中取$k=256$）。

4. **辅助Caption损失 $\mathcal{L}_{cap}$**：利用OFA为每张图像生成描述，增强语言特征多样性：
$$\mathcal{L}_{cap}(\mathbf{x}) = \frac{\exp(\cos(S(\mathbf{x}), T_{txt}(cap(\mathbf{x})))/\tau)}{\sum_{(\mathbf{x}', y'): y' \neq y} \exp(\cos(S(\mathbf{x}), T_{txt}(cap(\mathbf{x}')))/\tau)}$$

5. **语言表征增强**：使用ChatGPT生成细粒度类别描述，提示词为："Use a single sentence to describe the appearance and shape of {cls}. Only describe the shape and appearance"。

**评估度量**：
- $\mathcal{M}_{rel}$：衡量学生特征最近邻是否为对应教师特征的比例
- $\mathcal{M}_{neigh}(k)$：衡量学生与教师k近邻的重叠程度
- $\mathcal{M}_{vllign}$：衡量教师与学生到相同语言特征距离排序的一致性（逆序对数量）

## 实验与结果

**数据集**：Caltech-Birds、StanfordCars、Flower102、Food101、SUN397、tiered-ImageNet，每个数据集划分为等大小的ID类和OOD类。

**主要结果**（ResNet18学生，CLIP ViT-L/14教师）：

| 方法 | Caltech-Birds | StanfordCars | Flower102 | Food101 | SUN397 | tiered-ImageNet | 平均 |
|------|---------------|--------------|-----------|---------|--------|-----------------|------|
| $\mathcal{L}_{cls}$ | 61.0 / 14.2 / 34.9 | 56.3 / 14.7 / 20.0 | 81.2 / 4.5 / 46.2 | 72.2 / 16.1 / 24.5 | 57.5 / 13.6 / 28.6 | 64.4 / 13.9 / 27.5 | 65.4 / 12.8 / 30.3 |
| $\mathcal{L}_{cls}$ + $\mathcal{L}_{im-cst}$ | 60.9 / 20.4 / 37.6 | 59.6 / 18.3 / 31.2 | 82.4 / 12.7 / 52.5 | 74.0 / 30.5 / 42.0 | 62.5 / 18.8 / 35.2 | 64.4 / 18.0 / 33.5 | 67.3 / 19.8 / 38.7 |
| $\mathcal{L}_{cls}$ + $\mathcal{L}_{im-cst}$ + $\mathcal{L}_{vlprox}$ | 62.3 / 21.6 / 39.0 | 63.9 / 19.8 / 38.5 | 82.7 / 14.6 / 52.0 | 74.3 / 32.0 / 43.2 | 61.7 / 21.5 / 34.7 | 67.5 / 20.5 / 35.3 | 68.7 / 21.7 / 40.5 |
| + 语义增强 | 62.0 / 22.7 / 39.8 | 64.9 / 20.4 / 39.7 | 83.7 / 18.2 / 53.4 | 75.6 / 35.7 / 42.9 | 61.0 / 24.0 / 37.5 | 68.9 / 23.6 / 35.8 | 69.4 / 24.1 / 41.5 |

**关键发现**：
- 零样本OOD泛化平均提升约**8.8%**（从12.8%到21.7%，再到24.1%）
- 5-shot OOD泛化平均提升约**11.2%**（从30.3%到40.5%，再到41.5%）
- $\mathcal{L}_{im-cst}$比$\mathcal{L}_{mse}$更有效，尽管后者视觉特征绝对距离更小
- LLM语义增强对零样本OOD提升显著，辅助caption效果有限
- 更详细的语义描述并非总是更好，适当简洁的描述反而更有效

**ViT-B/32学生实验**：结论与ResNet18一致，验证了方法的通用性。

**少样本学习策略**：微调学生视觉骨干网络显著优于免训练检索策略。

## 相关工作脉络
1. **传统模型蒸馏**：Hinton等人提出的logits蒸馏框架主要针对同架构教师-学生，本文聚焦跨模态VLM蒸馏到纯视觉学生模型。
2. **视觉表征蒸馏**：如CRD、RKD等工作研究CNN/ViT的视觉表征蒸馏，但未考虑视觉-语言对齐结构。
3. **VLM蒸馏下游任务**：如Clip-td、Open-vocabulary detection等工作将VLM知识蒸馏到检测/分割模型，关注特定任务而非OOD泛化。
4. **CLIP微调/适配**：如Tip-Adapter、Coop等工作通过prompt tuning或adapter提升CLIP性能，本文从蒸馏视角研究小模型获取OOD能力。
5. **对比表示蒸馏**：如CPCD等工作研究对比学习表征的蒸馏，本文特别关注视觉-语言对齐结构的保持。

## 局限性与未来方向
1. **学生模型仅含视觉编码器**：蒸馏后学生为纯视觉模型，虽保留了OOD泛化能力，但推理时仍需依赖冻结的教师文本编码器。
2. **未探索跨架构蒸馏**：主要验证ResNet到ViT的蒸馏，对于其他架构组合（如MobileNet到EfficientNet）的效果未深入研究。
3. **语言增强依赖外部模型**：使用ChatGPT和OFA生成描述需要额外API调用，在实际部署中可能增加复杂度。
4. **扩展到3D和其他模态待研究**：论文提及未来可扩展到3D几何表征蒸馏，但尚未验证。
5. **数据规模限制**：使用小/中规模数据集，在大额数据上的泛化能力需进一步验证。

## 研究启发与可借鉴点
1. **相对对齐优于绝对匹配**：当学生难以精确复现教师特征时，保持相对距离关系和局部结构比最小化绝对距离更重要，这一原则可迁移到其他蒸馏场景。
2. **度量设计指导方法改进**：通过定义$\mathcal{M}_{rel}$、$\mathcal{M}_{neigh}$、$\mathcal{M}_{vllign}$等度量，系统性地分析各技术的贡献，这种"度量驱动设计"的研究范式值得借鉴。
3. **语言表征质量的关键作用**：LLM增强的细粒度描述显著提升OOD泛化，提示我们在多模态学习中应重视语言侧的质量而非仅优化视觉侧。
4. **过滤噪声对齐的有效性**：$\mathcal{L}_{vlprox}$中过滤教师自身误对齐样本的设计，展示了在蒸馏过程中识别和排除噪声的重要性。
5. **少样本微调整优于免训练检索**：在小规模数据上蒸馏的学生模型更适合微调而非免训练检索，这一发现对后续工作有指导意义。

## 关键术语表
**Out-of-Distribution (OOD)**：分布外样本，指与训练数据分布不同的测试样本，模型需要在此类样本上展现泛化能力。
**Vision-Language Model (VLM)**：视觉语言模型，如CLIP，同时学习视觉和语言表征并建立两者间的对齐关系。
**Knowledge Distillation**：知识蒸馏，将大型教师模型的知识转移到小型学生模型的过程。
**Open-vocabulary**：开放词汇，指模型能够识别训练期间未见过的类别标签。
**Visual Representation Space**：视觉表征空间，模型将图像映射到高维向量的空间结构。
**Vision-Language Alignment**：视觉-语言对齐，图像特征与对应文本特征在共享空间中的一致性。
**Zero-shot Classification**：零样本分类，模型直接在未见过的类别上进行分类，无需额外训练。
**Few-shot Learning**：少样本学习，仅使用少量标注样本快速适应新类别的学习范式。

## 可复现要素
- **数据集**：Caltech-Birds、StanfordCars、Flower102、Food101、SUN397、tiered-ImageNet均为公开数据集
- **代码开源**：论文声明代码已发布（原文："Code released at this link"，但链接在解析文本中未完整保留）
- **教师模型**：CLIP ViT-L/14、CLIP RN50（公开权重）
- **学生模型**：ResNet18、ViT-B/32（公开权重）
- **关键超参**：温度参数τ、$\mathcal{L}_{vlprox}$的k=256、训练批次平衡策略（few-shot数据不超过一半）
- **辅助模型**：ChatGPT（用于语义增强）、OFA（用于caption生成）
