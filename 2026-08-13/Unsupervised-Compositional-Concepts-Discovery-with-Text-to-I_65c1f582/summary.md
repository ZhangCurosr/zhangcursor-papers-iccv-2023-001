---
title: "Unsupervised-Compositional-Concepts-Discovery-with-Text-to-I"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Unsupervised_Compositional_Concepts_Discovery_with_Text-to-Image_Generative_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:35:36"
field: "生成模型与视觉表示学习"
keywords: ["无监督概念发现", "组合生成", "扩散模型", "文本到图像", "可组合扩散", "能量基模型"]
innovations: ["提出无监督组合概念发现框架，同时发现多个语义概念并支持重组生成", "利用预训练扩散模型的语义空间作为概念参数化基座，实现小样本高效学习", "统一分解-重组-表征学习三功能，验证跨域通用性"]
benchmarks: ["ImageNet", "ADE20K", "Artistic Paintings"]
---

# 论文速读：Unsupervised-Compositional-Concepts-Discovery-with-Text-to-I

## 一句话总结
本文提出一种无监督方法来从无标签图像集合中发现可组合的生成概念（如物体类别、场景元素、艺术风格），利用预训练的文本到图像扩散模型作为语义基座，将图像分解为多个独立概率分布的乘积，并支持概念重组生成新图像及下游分类任务。

## 研究问题与动机
- **核心问题**：给定一组无标签图像，如何自动发现能够代表每张图片内容的生成概念（包括全局概念如光照、局部概念如物体）？
- **现有方法不足**：
  - 大多数概念发现方法需要监督标签来指定每个概念
  - 部分无监督方法仅能发现物体概念，无法处理更复杂的场景分解
  - COMET（2021）虽能分解全局和局部概念，但仅适用于简单数据集，无法生成复杂高分辨率图像

## 核心贡献（创新点）
1. **可扩展的无监督组合概念发现框架**：利用预训练扩散模型作为能量函数的表示基座，从少量无标签图像中同时发现多个语义概念；与COMET的本质区别在于使用扩散模型替代EBM，支持高分辨率复杂图像生成。
2. **跨域通用性验证**：在ImageNet物体分类、ADE20K厨房场景分解、艺术家画作风格发现三个不同领域均取得SOTA性能；与COMET的本质区别在于方法可扩展至真实场景而非仅合成数据集。
3. **概念可组合性与下游应用**：发现的概念可通过组合算子（AND）生成新图像，且学习到的权重向量可作为有效表征用于分类；与已有工作的区别在于同时实现"分解-重组-表征"三者统一。

## 方法详解
- **核心思想**：将每张图像的概率分布分解为K个独立概念的乘积：$p_{decomp}(x_i) \propto p(x_i) \prod_{k=1}^{K} \frac{p(x_i|c^k)}{p(x_i)}$
- **可组合扩散模型**：利用diffusion model作为EBM的score function估计器，通过hybrid denoising score $\epsilon_{comb} = \epsilon_{c_1} + \epsilon_{c_2} - \epsilon_\phi$实现概念组合采样
- **无监督发现机制**：
  - 共享概念库 $\{c^1, ..., c^K\}$ 初始化为随机word embedding
  - 每张图像 $x_i$ 分配权重向量 $\boldsymbol{w}_i \in \mathbb{R}^K$（$\sum_k w_i^k = 1$）
  - 最终score prediction: $\epsilon_{unsup} = \epsilon_\phi + \sum_{k=1}^{K} w_i^k(\epsilon_k - \epsilon_\phi)$
  - 联合优化目标：$\mathcal{L}_{MSE} = ||\epsilon - \epsilon_{unsup}||^2$，同时对 $\boldsymbol{w}_i$ 和 $\{c^k\}$ 进行梯度更新
- **参数化技巧**：将概念参数化为text-to-image模型中的word embedding，利用预训练扩散模型的丰富语义空间降低学习难度

## 实验与结果
- **数据集**：ImageNet（4组各5类，每类5张）、ADE20K（25张厨房图）、艺术画作（Van Gogh 5张、Monet 7张、Picasso 5张）
- **评估指标**：Classification Accuracy（ResNet-50/CLIP）、KL Divergence、Representation Accuracy
- **主要结果**：
  - ImageNet分类准确率（平均）：**55.70%**（ResNet）/ **38.60%**（CLIP），优于Textual Inversion variants
  - KL散度（平均）：**0.1538**（ResNet）/ **0.1337**（CLIP），显著优于baseline
  - 表示学习（分类）：**62.75%**，优于K-means（32.25%）和Textual Inversion（24.25%），仅次于CLIP-based K-means（73.50%）
- **最强结果**：ImageNet S4分类准确率达82.81%，KL散度低至0.0285

## 相关工作脉络
- **COMET [11]**：首个无监督场景分解工作，使用EBM组合分解图像为全局/局部概念；本文扩展其思想但用扩散模型实现，支持更复杂高分辨率图像
- **Textual Inversion [15]**：监督方法，通过优化单个word embedding将相似图像映射为一词；本文无需标签同时发现多个概念
- **Composable Diffusion [32]**：提出基于EBM的概念组合算子；本文在其理论基础上实现无监督概念发现
- **DAAM [50]**：可视化注意力机制；本文用于理解学到的概念与图像内容的对应关系
- **CLIP [38]**：多模态编码器；本文用作评估工具验证生成图像质量

## 局限性与未来方向
- **小数据集依赖**：实验中使用极少训练样本（ImageNet每类仅5张），在大数据场景下的可扩展性需验证
- **概念数量固定**：K值需预先设定，缺乏自动确定最优概念数的机制
- **one-hot约束限制**：权重向量为概率分布可能限制某些复杂场景分解
- **未来方向**：扩展到视频/3D生成、自动概念数选择、结合结构化先验知识

## 研究启发与可借鉴点
1. **扩散模型EBM解释的利用**：将diffusion model视为unnormalized density estimator，通过score function加减实现概念组合，这一视角可迁移至其他生成模型
2. **低维语义参数化策略**：用word embedding而非直接优化网络参数来表示概念，大幅降低样本效率要求，可用于其他小样本生成任务
3. **组合分解统一框架**：同时实现"分解发现-重组生成-表征学习"三个功能，为多任务生成模型设计提供新思路

## 关键术语表
- **Compositional Generation**：通过组合多个独立概念或条件生成复杂图像的技术
- **Energy-Based Model (EBM)**：用能量函数定义未归一化概率分布的生成模型
- **Diffusion Attentive Attribution Map (DDAM)**：利用cross-attention层可视化扩散模型中文本条件与图像区域的关系
- **Classifier-free Guidance**：不依赖额外分类器而通过条件/无条件预测的差值引导生成的技术
- **KL Divergence**：衡量生成图像分布与真实类别分布差异的指标，值越小表示概念区分度越高

## 可复现要素
- **数据集**：ImageNet（公开）、ADE20K（公开）、Artistic Paintings（论文提供来源描述）
- **代码**：网站 https://energy-based-model.github.io/unsupervised-concept-discovery/ 注明有额外材料
- **权重**：使用 pretrained Stable Diffusion v2.1
- **关键超参**：K个共享概念、每图像50步DDIM采样、classifier-free guidance、学习率λ未明确提及
