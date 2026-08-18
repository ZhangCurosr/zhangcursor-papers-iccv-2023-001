---
title: "Self-supervised-Cross-view-Representation-Reconstruction-for"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Tu_Self-supervised_Cross-view_Representation_Reconstruction_for_Change_Captioning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:30:34"
---

# 论文速读：Self-supervised-Cross-view-Representation-Reconstruction-for

## 一句话总结
本文针对变化描述任务中视角偏移引发的“伪变化”干扰问题，提出自监督跨视图表示重构网络（SCORER）。该方法通过多头逐token匹配与跨视图对比对齐学习视角不变表示，重构不变对象特征以提取稳定的差异表示，并结合跨模态逆向推理模块自监督约束生成质量，在四个公开数据集上全面达到SOTA。

## 研究问题与动机
1. **核心挑战**：变化描述（Change Captioning）需同时理解成对相似图像并生成自然语言描述差异，但相机视角变化会导致对象尺度、位置发生偏移（伪变化），使直接特征匹配或图像相减难以捕捉真实变化。
2. **现有方法瓶颈**：主流方法（如DUDA、MCCFormers-D等）直接在两图特征间做跨注意力匹配，视角变化引起的特征偏移会淹没局部真实差异；直接相减则易引入配准噪声并丢失参照物信息。
3. **关键观察**：跨不同图像对的比较能有效凸显真实特征差异，不受单一图像对的视角偏移主导；伪变化本质是对象形变而非语义相似度改变，因此可通过对比学习强制对齐相似对。
4. **解决路径**：先利用自监督对比学习消除视角偏差、获得不变表示，再从中蒸馏共有特征重构不变对象，最后通过残差融合隐式推断差异表示用于描述生成。

## 核心贡献（创新点）
1. **提出SCORER自监督表示重构框架**：通过学习视角不变的双视图表示并重构不变对象特征，从根本上将“变化建模”与“视角偏移”解耦，避免直接相减的信息损失。
2. **设计多头逐Token匹配（MTM）**：在多个线性投影子空间内并行计算双向逐token最大相似度，实现细粒度特征子空间交互，突破全局池化或单头匹配在弱局部变化上的表达瓶颈。
3. **发明跨模态逆向推理（CBR）模块**：利用生成caption与“before”图像反向构建“幻觉”视觉表征，并通过InfoNCE对比损失强制其逼近真实“after”表征，以自监督方式确保caption充分描述变化及参照物。
4. **端到端联合训练范式**：将对比学习损失与语言生成负对数似然损失无缝融合，无需额外预训练或强化学习阶段，在CLEVR-Change、CLEVR-DC、IER和Spot-the-Diff四个数据集上均取得最优结果。

## 方法详解
方法由四个协同模块构成，全程端到端优化：
- **视图编码**：预训练 ResNet-101 提取网格特征 $X \in \mathbb{R}^{C \times H \times W}$，经 2D卷积降维并叠加可学习位置编码得到 $\tilde{X} \in \mathbb{R}^{D \times H \times W}$，展平为 $N \times D$ 的token序列。
- **多头逐Token匹配（MTM）**：单head相似度计算为 $\text{TM}(Q,K) = [\frac{1}{N}\sum_i \max_j e_{ij} + \frac{1}{N}\sum_j \max_i e_{ij}]/2$，其中 $e_{ij}=q_i^\top k_j$。多头版本在不同投影子空间 $W^Q_{i'}, W^K_{i'}$ 上并行执行后拼接，实现细粒度子空间交互。
- **视角不变表示学习**：在batch内同对bef/aft为正样本，其余为负样本，采用InfoNCE损失 $\mathcal{L}_{cv}$ 最大化跨视图对比对齐，使 $\tilde{X}_{bef}$ 与 $\tilde{X}_{aft}$ 对伪变化保持不变。
- **跨视图表示重构**：利用多头跨注意力（MHCA）从对方图像挖掘共有特征，重构不变表示 $\tilde{X}^u$；通过LayerNorm融合 $\tilde{X}^c = \text{LN}(\tilde{X} + \tilde{X}^u)$ 突出不变区域并隐式提示差异；拼接bef/aft经ReLU全连接层得到差异表示 $\tilde{X}_c$。
- **Transformer解码器**：以 $\tilde{X}_c$ 为视觉上下文，结合词嵌入自注意力与交叉注意力自回归生成caption。
- **跨模态逆向推理（CBR）**：对解码器隐藏状态 $\tilde{H}$ 均值池化得 $\tilde{T}$，与 $\tilde{X}_{bef}$ 拼接后经卷积保持空间维度，再经MHSA生成“幻觉”表示 $\tilde{X}_{hal}$；再次使用InfoNCE损失 $\mathcal{L}_{cm}$ 将 $\tilde{X}_{hal}$ 与 $\tilde{X}_{aft}$ 对齐。
- **联合损失**：$\mathcal{L} = \mathcal{L}_{cap} + \lambda_v \mathcal{L}_{cv} + \lambda_m \mathcal{L}_{cm}$，全程联合优化。

## 实验与结果
- **数据集**：CLEVR-Change（中度视角变化，5类变化）、CLEVR-DC（极端视角变化）、Image Editing Request (IER，对齐编辑)、Spot-the-Diff（监控场景单变化设置）。
- **评估指标**：BLEU-4 (B)、METEOR (M)、ROUGE-L (R)、CIDEr (C)、SPICE (S)。
- **主要结果**：
  - **CLEVR-Change**：SCORER+CBR在Total性能上取得 B=41.2, M=56.3, R=74.5, C=126.8, S=33.3；语义变化评估 B=54.4, C=122.4。在最具挑战的“Move”类别上较 R³Net+SSP 取得 4.7% 相对提升。
  - **CLEVR-DC**：SCORER在多数指标领先；加入CBR后较 VACC 在 CIDEr 上提升 16.7%，验证了CBR对极端视角的校准作用。
  - **IER**：SCORER+CBR 在 BLEU-4 上较最新方法 NCT 取得 23.5% 相对提升（B=10.0），对微小编辑变化识别能力突出。
  - **Spot-the-Diff**：SCORER取得最佳（B=10.2, C=38.9）；CBR模块因该数据集常含多处变化，单变化假设导致METEOR/SPICE轻微下降，作者指出此为数据集设定与模块假设不匹配所致。
- **消融结论**：表示重构（RR）显著优于直接相减；MTM优于全局max/mean-pooling及单
