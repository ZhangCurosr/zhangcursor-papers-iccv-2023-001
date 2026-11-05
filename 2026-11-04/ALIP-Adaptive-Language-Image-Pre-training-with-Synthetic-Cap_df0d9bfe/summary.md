---
title: "ALIP-Adaptive-Language-Image-Pre-training-with-Synthetic-Cap"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_ALIP_Adaptive_Language-Image_Pre-Training_with_Synthetic_Caption_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:34:15"
field: "多模态预训练与对比学习"
keywords: ["对比学习", "语言-图像预训练", "噪声鲁棒性", "合成描述", "自适应门控", "CLIP", "多模态表示学习"]
innovations: ["利用离线 OFA 合成描述构建双路径对比学习框架，避免在线 captioner 的额外开销", "语言一致性门控（LCG）和描述一致性门控（DCG）动态调整样本和成对权重以抑制噪声影响", "自适应对比损失在保持全部训练样本的同时实现对低质量数据的有效降权"]
benchmarks: ["Flickr30K", "MSCOCO", "CIFAR10/100", "ImageNet", "Food101", "Oxford Pets", "Flowers102", "SUN397", "Stanford Cars", "DTD", "Caltech101", "FGVC Aircraft"]
---

# 论文速读：ALIP-Adaptive-Language-Image-Pre-training-with-Synthetic-Caption

## 一句话总结
本文提出 ALIP，利用离线预生成的合成描述（synthetic caption）与原始网络文本构建双路径对比学习框架，通过语言一致性门控（LCG）和描述一致性门控（DCG）动态调整样本及图像-文本/描述对的权重，有效缓解大规模噪声数据对 CLIP 类预训练的负面影响，在零样本图像-文本检索和线性探测任务上达到 SOTA。

## 研究问题与动机
1. **大规模噪声数据的负面影响**：Web 爬取的图像-文本对（如 YFCC100M、LAION-400M/5B）中存在大量噪声和低对齐样本（如图 1 中"Leisure Sunday"与图像内容不匹配），会损害对比表征学习的效果。
2. **现有去噪方法计算成本高**：BLIP 需要在线 captioner 和 filter 模型（需先在 COCO 上微调）；ALBEF/PSD 采用动量模型生成软标签，均带来额外的计算开销和显存消耗。
3. **直接离线过滤会丢失大量训练数据**：LAION-5B 的原始 50 亿数据经过 CLIP 过滤后损失巨大，且离线过滤方法本身存在偏差（受限于过滤模型的代表能力）。
4. **粗粒度合成描述仍具补充价值**：OFA 生成的合成描述虽粒度较粗，但能提供物体、属性、动作和空间关系等额外信息，可作为可靠的辅助监督信号。

## 核心贡献（创新点）
1. **双路径（bi-path）对比预训练框架**：同时利用原始文本和 OFA 生成的合成描述构建 triplet (image, raw text, synthetic caption)，与 BLIP 在线 captioner/filter 相比无需额外在线推理开销。
2. **语言一致性门控（LCG）**：基于 t-c 相似度和历史平均阈值动态判断样本质量并输出样本权重 $W^s$，与 ALBEF/PSD 的动量模型软标签机制本质不同——LCG 无需维护移动平均模型，计算更轻量。
3. **描述一致性门控（DCG）**：在低质量样本中仍可识别高质量的 image-text 或 image-caption 对，输出成对权重 $W^t$ 和 $W^c$（$W^s=1$ 时回退到权重 1），弥补了 BLIP 直接过滤掉低质量样本而丢失可用信息的不足。
4. **自适应对比损失函数**：将 $W^s$、$W^t$、$W^c$ 乘入 InfoNCE 损失，实现对噪声数据的自适应降权，而非丢弃或固定权重处理。

## 方法详解
- **数据构造**：使用预训练的 $\mathrm{OFA}_{base}$ 模型，以 prompt "What does the image describe?" 为条件，离线生成每张图像的 synthetic caption，构建 triplet 数据集 $D = \{(X_i, T_i, C_i)\}_{i=1}^{N}$。
- **编码器结构**：双编码器 $\Phi = \{\Phi_{image}, \Phi_{text/caption}\}$，其中图像编码器和文本/描述共享编码器采用与 CLIP-ViT-B/32 相同的架构；对图像 $X$、文本 $T$、描述 $C$ 分别得到 $l_2$ 归一化嵌入 $\mathbf{x}$、$\mathbf{t}$、$\mathbf{c}$。
- **三种相似度**：$S_{tc} = \mathbf{t}^\top \mathbf{c}$（t-c 相似度）、$S_{xt} = \mathbf{x}^\top \mathbf{t}$（图像-文本相似度）、$S_{xc} = \mathbf{x}^\top \mathbf{c}$（图像-描述相似度）。
- **LCG（公式 1-2）**：维护历史平均相似度 $H_{tc} = m \cdot H_{tc} + (1-m) \cdot \bar{S}_{tc}$，当 $S_{tc} > H_{tc}$ 时 $W^s=1$（高质量样本），否则 $W^s = \exp((S_{tc} - H_{tc}) \cdot \gamma_s) \in (0,1)$，对低质量样本降权。
- **DCG（公式 3-4）**：独立维护 $H_{xt}$ 和 $H_{xc}$，根据 $W^s$ 是否等于 1 来区分处理：当 $W^s < 1$ 时，$W^t$ 和 $W^c$ 分别由 $S_{xt} - H_{xt}$ 和 $S_{xc} - H_{xc}$ 决定（可大于 1），使得低质量样本中仍与图像高度对齐的那一侧描述能被充分利用。
- **自适应对比损失（公式 6-7）**：$L_{xt} = -\sum_i W_i^s W_i^t [\log \frac{e^{\mathbf{x}_i^\top \mathbf{t}_i/\tau}}{\sum_j e^{\mathbf{x}_i^\top \mathbf{t}_j/\tau}} + \log \frac{e^{\mathbf{x}_i^\top \mathbf{t}_i/\tau}}{\sum_j e^{\mathbf{x}_j^\top \mathbf{t}_i/\tau}}]$，$L_{xc}$ 同理；总损失 $L_{ALIP} = L_{xt} + L_{xc}$。超参：$\gamma_s=2$，$\gamma_p=2$，$\tau=0.07$。

## 实验与结果
- **数据集**：预训练使用 YFCC15M（DeCLIP 过滤的子集），并扩展至 LAION400M 的 10M 和 30M 随机子集。
- **下游评测**：Flickr30K 和 MSCOCO 零样本图像-文本检索（I2T/T2I Recall@1/5/10）；10 数据集线性探测；11 数据集零样本分类（CIFAR10/100, Food101, Oxford Pets, Flowers102, SUN397, Stanford Cars, DTD, Caltech101, FGVC Aircraft, ImageNet）。
- **关键数字**：
  - **MSCOCO I2T R@1 = 46.8%，T2I R@1 = 29.3%**（全指标 SOTA），相比 HiDeCLIP 提升 8.1%/5.4%，相比 HiCLIP 提升 12.6%/8.7%。
  - **Flickr30K I2T R@1 = 70.5%，T2I R@1 = 48.9%**，分别较 HiCLIP 提升 18.2% 和 35.6%。
  - **线性探测平均准确率**较 HiCLIP 提升 1.4%~9.2%，较 HiDeCLIP 在 CIFAR10/CIFAR100/Aircraft 上分别提升 6.2%/7.1%/3.5%。
  - 零样本分类因合成描述粒度较粗，在 CIFAR10/100 上有提升，但在多数数据集上略逊于 HiDeCLIP。
- **鲁棒性**：在 LAION10M 和 LAION30M 上，ALIP-ViT-B/32 在 23/24 个数据集上超越 CLIP-ViT-B/32；$\mathrm{OFA}_{large}$ 仅带来边际提升（线性探测 +0.1%，零样本 +0.4%）。

## 相关工作脉络
1. **CLIP（ICML 2021）**：对比语言-图像预训练的开山之作，采用 InfoNCE 损失在 400M 噪声数据上训练；本文在其基础上针对噪声问题引入自适应权重机制。
2. **SLIP（ECCV 2022）**：结合自监督视觉学习与对比预训练；本文聚焦于利用多路文本监督增强对比对齐，而非引入额外自监督信号。
3. **HiCLIP（ICLR 2023）**：引入层次注意力机制；本文方法与之正交，ALIP 在零样本检索上大幅超越 HiCLIP，但在细粒度零样本分类上不如 HiDeCLIP。
4. **HiDeCLIP**：层次感知去噪 CLIP，在零样本分类上表现最优；本文与其定位不同，ALIP 更侧重利用合成描述作为可靠辅助信号而非仅去噪。
5. **BLIP（ICML 2022）**：在线 captioner + filter 去噪；本文的离线预生成方案避免了在线推理开销和额外模型微调成本。
6. **ALBEF / PSD**：使用动量模型的软标签进行去噪；本文的 LCG/DCG 无需维护额外模型，显存和计算开销更低。

## 局限性与未来方向
1. **合成描述粒度较粗**：OFA 只能识别大类物体（如"花"而非具体品种），导致零样本分类性能提升有限，难以捕捉细粒度层次语义。
2. **离线生成依赖单一 captioner**：当前仅使用 OFA 模型，captioner 的生成质量和多样性直接影响预训练效果。
3. **门控权重设计的可解释性**：$W^t, W^c$ 可大于 1 的机制在极端情况下可能导致对某些噪声对的过度放大，尚缺充分分析。
4. **未探索更多类型的辅助描述**：如 region-level caption、object detection 结果、属性标注等更丰富的辅助监督信号。
5. **训练效率**：双路径对比损失使有效 batch 内数据维度翻倍，实际 GPU 利用率需进一步优化。

## 研究启发与可借鉴点
1. **离线预生成 + 在线自适应加权**的思路可迁移到其他多模态预训练场景（如视频-文本、3D-语言），避免在线生成带来的计算开销。
2. **LCG/DCG 的双层门控设计**（先判样本质量，再判描述质量）是一种通用的噪声鲁棒训练范式，可推广至其他存在多源标签噪声的任务。
3. **合成描述的统计特性分析**（图 7：token 数分布、相似度分布）提供了评估辅助监督数据质量的量化方法，值得借鉴。
4. **对同一训练样本的多路监督充分利用**（不因样本整体质量差而丢弃，而是通过 DCG 保留优质对）是提升数据效率的关键设计。
5. 本团队可在**细粒度描述生成**（如引入 GPT-4V 或区域级 captioning）上进一步探索，解决当前合成描述粒度粗的问题，并与 LCG/DCG 框架结合。

## 关键术语表
**InfoNCE Loss**：对比学习的标准损失函数，通过最大化正样本对相似度、最小化负样本对相似度来学习对齐的跨模态表征。
**Language Consistency Gate (LCG)**：基于原始文本与合成描述之间的嵌入相似度，动态输出样本级权重 $W^s \in (0,1]$ 的门控模块，用于识别高质量样本。
**Description Consistency Gate (DCG)**：基于图像与文本/描述的相似度，在低质量样本中进一步筛选出仍高质量的对，输出成对权重 $W^t$ 和 $W^c$。
**Synthetic Caption**：由预训练模型（本文使用 OFA）基于图像内容生成的描述文本，与 Web 原始文本相比语义更准确、粒度更结构化。
**Triplet (Image, Text, Caption)**：ALIP 的基本训练单元，包含一张图像及其原始文本描述和合成描述，用于计算三种两两相似度。
**Adaptive Contrastive Loss**：将 LCG 和 DCG 输出的权重乘入 InfoNCE 损失的自适应对比损失，实现对噪声数据的动态降权。
**Momentum-based Soft Label**：ALBEF/PSD 等方法使用移动平均模型生成软标签的思路，本文将其替换为基于相似度阈值的高效门控机制。

## 可复现要素
- **数据集**：YFCC15M（来自 YFCC100M 经 DeCLIP 过滤），LAION400M 的 10M 和 30M 子集；**已公开**。
- **代码**：https://github.com/deepglint/ALIP（论文声明已开源）。
- **预训练模型**：论文声明已开源释放。
- **Captioner**：$\mathrm{OFA}_{base}$（$\mathrm{OFA}_{large}$ 为 470M 参数）；OFA 为公开模型。
- **关键超参**：$\gamma_s = 2$，$\gamma_p = 2$，温度 $\tau = 0.07$，batch size = 4096，32 epochs，lr = 0.001，weight decay = 0.2，AdamW，图像 224×224，文本长度 77。
- **硬件**：16 张 NVIDIA V100 GPU。
