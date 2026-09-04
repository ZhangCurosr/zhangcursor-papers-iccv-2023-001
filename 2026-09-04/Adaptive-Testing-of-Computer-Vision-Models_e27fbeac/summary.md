---
title: "Adaptive-Testing-of-Computer-Vision-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Adaptive_Testing_of_Computer_Vision_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:13:06"
field: "视觉模型评测与鲁棒性"
keywords: ["human-in-the-loop testing", "vision model evaluation", "adaptive testing", "error discovery", "CLIP retrieval", "out-of-distribution generalization"]
innovations: ["提出ADAVISION人机协同自适应测试框架，通过CLIP插值检索实现爬山式高错误率区域探索", "设计测试生成与选题生成双循环结构，结合GPT-3辅助选题生成", "验证发现的语义相干失败组可通过微调修复且不影响分布内性能"]
benchmarks: ["ImageNet", "ImageNet-V2", "ImageNet-A", "ImageNet-Sketch", "ImageNet-R", "ObjectNet", "COCO Captions", "LAION-5B"]
---

# 论文速读：Adaptive-Testing-of-Computer-Vision-Models

## 一句话总结
本文提出 ADAVISION，一个面向视觉模型的**人机协同自适应测试框架**，通过自然语言驱动、CLIP检索、小样本人工标注反馈的"爬山"机制，高效发现视觉模型中语义相干的系统性失败模式，并可指导针对性微调修复。

## 研究问题与动机
- **核心问题**：视觉模型在整体指标上表现优异，但在特定语义子群（如罕见物体、特殊场景）上存在系统性失败，识别这些"相干失败组"对安全部署与针对性修复至关重要。
- **现有自动方法的不足**：主流做法（如 DoMINO）在验证集上聚类错误并自动生成标题，但聚类结果往往缺乏语义连贯性，且容易过拟合小规模验证集，难以泛化到未见数据。
- **静态评测集的局限**：传统评测依赖预定义数据集（如 ImageNet），覆盖范围有限，随着模型在基准上趋于饱和，其发现新失败模式的能力大幅下降。
- **视觉领域缺乏成熟的开放测试框架**：NLP 领域已有大量人机交互开放测试工作（如 Dynabench、Adaptive Testing for NLP），但计算机视觉尚无成体系的开放测试框架，视觉测试多局限于固定数据增强或3D仿真引擎。

## 核心贡献（创新点）
1. **提出 ADAVISION 人机协同测试框架**：将 NLP 领域的自适应测试范式引入计算机视觉，支持分类、目标检测、图像描述等多任务。
2. **自适应测试生成循环（Test Generation Loop）**：通过用户少量标注 + CLIP 嵌入插值实现"爬山式"检索，快速收敛至高错误率语义子群；与仅基于文本检索的基线相比，发现失败数接近 **2×**。
3. **LLM 驱动的选题生成循环（Topic Generation Loop）**：利用 GPT-3 基于用户已探索的高失败率主题生成新测试选题建议，降低用户创造性负担；与 DoMINO 等自动切片发现方法相比，ADAVISION 发现的失败率是其 **2–3 倍**。
4. **验证了发现失败后可通过微调修复且不损害分布内性能**：在 ViT-H/14 上微调 ADAVISION 发现的 30 个主题数据，处理主题准确率从 72.6% 提升至 91.2%，ImageNet 整体准确率不变，OOD 平均准确率从 78.0% 提升至 84.0%。

## 方法详解
**整体框架**包含两个循环：测试生成循环（Test Generation Loop）和选题生成循环（Topic Generation Loop）。

**测试生成循环**（逐轮迭代）：
- **初始检索**：用户使用自然语言描述主题 t（如"stop sign"），ADAVISION 通过 CLIP ViT-L/14 将文本嵌入 $q_t$，在 LAION-5B（50亿图文对）中检索最近邻图片作为首轮测试集。
- **用户标注**：用户运行目标模型 $m$ 获取预测，标注少量图片为 pass / fail / off-topic（检索误差或不符合下游应用场景）。
- **自适应插值检索**：对每轮收集到的失败图片，采样最多3张，将其图像嵌入 $q_i$ 通过随机凸组合合并，再与主题文本嵌入 $q_t$ 进行球形线性插值（spherical interpolation），生成新检索查询，实现向高错误率区域的"爬山"。
- **轻量分类器自动排序**：对用户标注的 pass/fail 标签，训练支持向量分类器（SVC）在拼接的 CLIP 输入-输出嵌入上预测标签，并据此对检索结果重排序（失败靠前、无关靠后），降低用户标注负担。

**选题生成循环**：
- 使用 GPT-3（text-davinci-002）配合模板（如"List some conditions a {LABEL} could be in that would make it hard to see"）生成候选主题。
- 将以有高失败率的既有主题融入 few-shot prompt，使 GPT-3 倾向于生成高失败主题名；用户选择感兴趣的主题进入测试生成循环。

## 实验与结果
**数据集**：LAION-5B（检索源）、ImageNet（分布内评估）、ImageNet V2/A/Sketch/R、ObjectNet（OOD 评估）、COCO Captions（captioning 训练）、OpenImages（检测模型参考指标）。

**评估基线**：
- **NONADAPTIVE**：相同 CLIP 后端但不使用插值检索和自动标注，仅基于主题名称检索。
- **DoMINO（BERT）/ DoMINO（OFA）**：自动切片发现方法，在 CLIP 潜在空间用误差感知高斯混合模型聚类错误并用 BERT/OFA 生成标题。

**主要结果**：
| 比较 | 关键数字 |
|---|---|
| ADAVISION vs NONADAPTIVE（用户研究） | ADAVISION 发现失败数约为非自适应基线的 **~2×**（分类 $d=0.588$，检测 $d=0.882$，captioning $d=0.967$，均 $p<0.05$） |
| ADAVISION vs DoMINO（ViT-H/14） | 平均失败率：**ADAVISION 28.47%** vs DoMINO(BERT) 8.6% vs DoMINO(OFA) 7.33% vs 通用主题 1.33% |
| ADAVISION vs DoMINO（ResNet-50，跨模型迁移） | 平均失败率：**ADAVISION 56.93%** vs DoMINO(BERT) 20.44% vs DoMINO(OFA) 25.45% |
| 微调修复效果（ViT-H/14） | 处理主题准确率：72.6% → 91.2%（+18.6pp）；ImageNet 整体准确率不变（88.4%）；OOD 处理类平均准确率：78.0% → 84.0%（+6pp） |
| 用户主观评价 | 84.6% 用户认为"无法用现有工具发现这些 bug"；ADAVISION 平均认知难度 3.05 vs NONADAPTIVE 4.10（$p<0.001$） |

**最强结果**：ADAVISION 在 ResNet-50 上达到的平均失败率 **56.93%**，是 DoMINO(OFA) 的 **2.2 倍**、通用主题基线的 **3.6 倍**。

## 相关工作脉络
- **DoMINO（Eyuboglu et al., 2022）**：在 CLIP 空间中用误差感知 GMM 聚类验证集错误并自动生成标题的自动切片发现方法。本文定位差异：ADAVISION 依赖人工交互迭代，发现的主题更相干、泛化到未见数据的失败率显著更高（DoMINO 61.6%/33.3% 的主题标题无意义）。
- **Wiles et al.（2022，arXiv:2208.08831）**：利用文本到图像生成模型 + 自动切片发现进行开放视觉测试。本文定位差异：ADAVISION 为纯人机交互、无需生成模型，发现的失败率高出数量级。
- **Dynabench / Red Teaming for NLP（Ganguli et al., 2022；Kiela et al., 2021）**：NLP 领域的人机开放测试实践。本文定位差异：首次将此范式系统化迁移到多任务视觉领域，并设计了专门的自适应检索机制。
- **Vision Checklist / DeepTest**：面向视觉模型的静态测试套件，局限于预定义数据增强轴或固定场景，本文定位差异：ADAVISION 沿不受限的语义轴动态检索真实图片。
- **Adaptive Testing for NLP（Ribeiro & Lundberg, 2022）**：NLP 领域的自适应测试框架。本文定位差异：将"测试生成+选题生成"双循环结构迁移到视觉任务，解决视觉检索（CLIP + LAION-5B）与自动标注（SVC 排序）的适配问题。
- **Maximum Discrepancy Competition（Ma et al., 2018; Wang et al., 2020）**：通过动态选样比较模型的自动方法。本文定位差异：ADAVISION 以人为主导、聚焦语义相干失败组的发现与修复，而非纯模型间对比。

## 局限性与未来方向
- **检索覆盖局限**：LAION-5B 对日常场景覆盖良好，但对生物医学、卫星影像等专业领域适用性差；CLIP 在复杂不对称关系主题上的检索质量下降，off-topic 图片仍会干扰测试。
- **实验规模有限**：微调实验仅针对 ViT-H/14 的有限类别、单轮测试进行；多轮迭代测试-微调尚未充分验证；较小模型可能面临灾难性遗忘风险。
- **非分类任务的修复难点**：检测/描述任务的修复需要额外标注正确边界框或描述，当前仅收集 pass/fail 标签，需引入负标签损失函数或补充标注步骤。
- **GPT-3 选题质量依赖模板设计**：选题生成循环的召回率高但精确度有限，需人工筛选。

## 研究启发与可借鉴点
- **"插值式"自适应检索策略**：将失败样本嵌入与主题文本嵌入做球形插值，是一种轻量且有效的"爬山"机制，可迁移至任何基于 embedding 的检索评测场景。
- **轻量 SVC 自动排序降低人工成本**：在每一轮后快速训练分类器对检索结果重排序，是平衡人机协作效率的实用设计，值得在其他 human-in-the-loop 系统中借鉴。
- **双循环架构（测试生成 + 选题生成）的模块化迁移**：该架构解耦了"深度测试一个主题"与"探索新主题"两个认知负担，可迁移到 NLP 安全测试、多模态模型评测等领域。
- **LLM 辅助选题的 few-shot 提示工程**：用历史高失败率主题作为先验引导 GPT-3 生成主题，这一"成功先例放大"策略在创意生成任务中具有普遍价值。
- **跨模型主题迁移验证**：在 ViT-H/14 上发现的失败主题直接迁移到 ResNet-50 仍保持高失败率，表明发现的失败模式具有语义层面的普适性，可作为后续模型对比评测的新基准构建思路。

## 关键术语表
- **ADAVISION**：自适应视觉模型测试框架，通过人机交互+CLIP检索+LLM选题发现视觉模型相干失败组。
- **Coherent failure group（相干失败组）**：由人类可理解语义概念统一、共享预期行为的测试集合，其失败率远高于模型基线失败率。
- **Test generation loop（测试生成循环）**：用户在给定主题下通过迭代标注+自适应检索逐步收敛到高错误率区域的交互过程。
- **Topic generation loop（选题生成循环）**：利用 GPT-3 基于已有高失败率主题生成新候选主题的交互式探索过程。
- **Spherical interpolation（球形插值）**：在嵌入空间中沿球面路径混合文本主题嵌入与失败样本嵌入，以生成新的检索查询。
- **Automatic slice discovery（自动切片发现）**：在模型潜空间中聚类错误样本并自动生成标题描述的方法（如 DoMINO）。
- **In-distribution / Out-of-distribution（ID / OOD）**：分别指与模型训练分布相同和不同的数据；本文在 ImageNet 上评估 ID 性能，在 ImageNet-V2/A/Sketch/R 和 ObjectNet 上评估 OOD 性能。
- **Catastrophic forgetting（灾难性遗忘）**：模型在微调修复特定失败模式时，对原有分布性能产生显著退化的现象。

## 可复现要素
- **数据集**：LAION-5B（公开）、ImageNet（公开）、ImageNet V2/A/Sketch/R（公开）、ObjectNet（公开）、COCO Captions（公开）。
- **代码**：论文声明已开源，仓库地址 https://github.com/i-gao/adavision。
- **模型权重**：CLIP ViT-L/14（公开）、ViT-H/14（公开）、ResNet-50（公开）、OFA-Huge（公开）、GPT-3 text-davinci-002（API 访问）。
- **关键超参**：每轮采样最多3张失败图片嵌入合并；SVC 每轮重新训练；测试时长 15–20 分钟/轮；每主题发现 8–10 个失败后切换；微调每主题采样 20 张，共 600 张。
