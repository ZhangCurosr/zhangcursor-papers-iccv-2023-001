---
title: "ITI-GEN-Inclusive-Text-to-Image-Generation"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ITI-GEN_Inclusive_Text-to-Image_Generation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:10:12"
field: "生成模型的公平性与包容性"
keywords: ["text-to-image generation", "inclusive generation", "bias mitigation", "prompt tuning", "CLIP", "diffusion models"]
innovations: ["利用参考图像在CLIP联合嵌入空间中学习direction-aligned inclusive tokens，实现冻结模型下的属性均匀分布生成", "提出方向对齐损失与语义一致性损失联合优化，避免语言漂移并提升生成质量", "首创train-once-for-all plug-and-play范式，inclusive tokens可跨prompt和下游模型(如ControlNet/IP2P)复用"]
benchmarks: ["CelebA", "FairFace", "FAIR benchmark", "LHQ"]
---

# 论文速读：ITI-GEN: Inclusive Text-to-Image Generation

## 一句话总结
本文提出 ITI-GEN，一种无需微调预训练文本到图像模型的包容性生成框架，利用少量参考图像在 CLIP 联合嵌入空间中学习 discriminative inclusive tokens，使生成图像在目标属性上实现均匀分布。

## 研究问题与动机
- **现有模型偏见严重**：Stable Diffusion 等模型继承训练数据偏差，例如输入"a headshot of a person"时生成约 92% 无眼镜人物、仅 8% 有眼镜人物。
- **微调方案不可行**：重新收集大规模平衡数据集极不现实，且 diffusion 模型训练计算成本高昂。
- **硬提示搜索受限**：直接在 prompt 中指定属性（如加"with eyeglasses"）受语言歧义和模型误表示影响，无法稳定生成多样样本。
- **图像比文字表达更精准**：某些属性（如肤色分类）难以用自然语言精确描述，但可通过参考图像直观表达——"一张图片胜过千言万语"。

## 核心贡献（创新点）
1. **提出 ITI-GEN 框架**：首次利用参考图像指导 inclusive prompt 学习，实现冻结模型下的包容性生成；与个性化方法不同，本文学习的是属性类别 token 而非特定主体 token。
2. **方向对齐损失（Direction Alignment Loss）**：在 CLIP 联合嵌入空间中将对齐"提示方向"与"图像方向"，将视觉属性差异翻译为语言差异；与图像编辑方法中基于文本变化修改源图的方向相反。
3. **语义一致性损失（Semantic Consistency Loss）**：防止 prompt 在优化过程中发生语言漂移，确保生成图像忠实于原始 prompt；该损失阈值 λ 可基于先验语言知识（硬提示与原始 prompt 的 CLIP 相似度均值约 0.8）鲁棒设定。
4. **Plug-and-Play 兼容性**：仅更新 ~10KB 的 inclusive tokens（length q=3），不修改模型参数，可直接与 ControlNet、InstructPix2Pix 等下游模型组合使用；支持"in-domain generation"和"train-once-for-all"两种模式。

## 方法详解
**整体流程**：给定输入 prompt T 和属性集合 A={A₁,...,Aₘ}，为每属性每类别注入 q 个可学习 token Sₖᵐ，构造 inclusive prompt Pₖᵐ=[T;Sₖᵐ]；多属性组合通过逐元素求和聚合：S_{o₁...oₘ}=∑S_{oₘ}ᵐ。

**方向对齐损失**（公式 4-6）：
- 对属性 Aₘ 的每对类别 (i,j)，计算类别平均图像嵌入差 Δ_I^m(i,j)=𝔛_i^m−𝔛_j^m 和类别平均 prompt 嵌入差 Δ_P^m(i,j)=𝔓_i^m−𝔓_j^m
- 最大化两者余弦相似度：L_dir=1−⟨Δ_I,Δ_P⟩

**语义一致性损失**（公式 7）：
- L_sem=max(0,λ−⟨E_text(P),E_text(T)⟩)，约束 learning prompt 不偏离原始 prompt 语义

**总损失**（公式 9）：
- L_total=∑_m∑_{i<j}L_pair^m(S_i^m,S_j^m)，每次迭代只更新单一属性的 tokens，交替遍历所有属性

## 实验与结果
- **数据集**：CelebA（40 二元属性）、FairFace（2 性别×9 年龄）、FAIR benchmark（6 类肤色）、LHQ（场景质量/抽象属性）
- **评估指标**：KL 散度衡量分布均匀性（越低越好）、FID 衡量图像质量
- **主要结果**（Table 1）：
  - 单属性：ITI-GEN 在所有 6 个属性上 KL 接近 0（如 eyeglass=0，mustache=2×10⁻⁴），远超 SD/EI/HPS/PD/CD
  - 多属性：male×young×eyeglass×smile 四属性组合 KL=0.094，显著优于 SD(1.406) 和 HPS(0.476)
  - 多类别：可处理 perceived gender×age、gender×skin tone，对极端少见类别（<10岁/>50岁）亦有改善
  - 场景域：成功生成不同 colorfulness/brightness/scary 级别的景观图像
- **最佳结果**：在 CELEbA 全部 40 个二元属性平均上，ITI-GEN KL 均低于 10⁻³，接近完美均匀；FID=51.83（优于 SD 的 67.40），使用 FAIR 合成数据略优于真实数据

## 相关工作脉络
- **Hard Prompt Searching（HPS, Ding et al., 2021）**：直接添加属性词到 prompt；本文与之对比证明语言歧义下 HPS 在多类别/多属性场景失效。
- **Ethical Intervention（EI, Bansal et al., 2022）**：手动编辑 prompt 加入伦理干预词；不透明且劳动密集，ITI-GEN 无需人工设计。
- **Prompts Debiasing（PD, Chuang et al., 2023）**：校准 text embedding 偏差；仅适用于预定义类别名，ITI-GEN 支持连续嵌入空间学习。
- **Custom Diffusion（CD, Kumari et al., 2022）**：基于 Textual Inversion 微调扩散模型；本文完全不修改模型参数，训练快且通用性强。
- **StyleGAN-NADA（Gal et al., 2022）**：CLIP 引导方向损失用于图像编辑；本文逆向使用——从图像差异学习 prompt 差异，目标为 inclusive generation 而非单图编辑。
- **Prompt Tuning（Lester et al., 2021；Jia et al., 2022）**：PT 通常在下游任务上 fine-tune 少量 token；本文将其迁移至连续嵌入空间，聚焦属性区分而非任务适配。

## 局限性与未来方向
- **细微属性效果有限**：对非常细微的面部属性（Appendix F.2）或高度纠缠的属性组合（Appendix F.3），ITI-GEN 无法达到最优。
- **仍需参考图像**：每类别需数十张参考图，参考图本身可能引入偏差或不准；可通过结合 InstructPix2Pix 等强控制模型缓解。
- **未见完全解耦**：引入的 token 可能隐式纠缠参考图中的其他属性（如服装风格）， disentangling 非本文主要目标。
- **未来方向**：探索更少参考样本下的鲁棒学习、与 instruction-based editing 深度结合、推广至视频/3D 生成任务。

## 研究启发与可借鉴点
1. **"图像→提示"方向学习范式**：可将 direction alignment loss 迁移至其他需要跨模态对齐的任务（如多语言 prompt 学习、跨域风格迁移）。
2. **语义一致性正则化**：λ 基于先验知识（硬提示与原始 prompt 的 CLIP 相似度均值）设定的策略值得推广到 prompt tuning 的稳定性保障。
3. **Marginal distribution 独立学习**：各属性仅需自身 marginal 参考集即可联合训练，避免收集昂贵 multi-attribute joint distribution 数据，可扩展至高维属性场景。
4. **Plug-and-Play token 设计**：冻结模型 + 极低参数量（~10KB）的 inclusive tokens，可封装为即插即用模块供社区复用。
5. **合成数据有效性**：FAIR benchmark 合成数据在肤色生成上优于真实数据（背景噪声干扰小），提示未来可用渲染引擎 bootstrap 包容性数据生成管线。

## 关键术语表
- **ITI-GEN**：Inclusive Text-to-Image GENeration，本文提出的基于参考图像的包容性文本到图像生成框架。
- **Inclusive Token**：注入到原始 prompt 后的可学习 token 序列，用于编码特定属性类别的语义方向。
- **Direction Alignment Loss**：促使 prompt 嵌入差方向与参考图像嵌入差方向对齐的损失函数，实现视觉差异→语言差异翻译。
- **Semantic Consistency Loss**：防止 prompt 在优化过程中偏离原始语义的正则化损失，约束 learning prompt 与原始 prompt 的 CLIP 相似度不低于 λ。
- **Train-once-for-all**：用单一中性 prompt（如"a headshot of a person"）训练 inclusive tokens 后，可泛化至任意 out-of-domain prompt 的 plug-and-play 模式。
- **Distribution Discrepancy (D_KL)**：使用 CLIP/分类器预测生成图像的属性分布与目标均匀分布之间的 KL 散度，越低表示包容性越好。
- **FAIR Benchmark**：合成人脸数据集，提供 6 类 Fitzpatrick 肤色的 ground-truth albedo 标注，用于肤色包容性研究。

## 可复现要素
- **数据集**：CelebA（公开）、FairFace（公开）、FAIR benchmark（公开）、LHQ（公开）；论文未声明自有数据集公开。
- **代码/权重**：论文未提及代码是否开源（注：作者单位为 CMU + Google，需另行查证 GitHub）。
- **关键超参**：inclusive token 长度 q=3；embedding 维度 e=768（SD v1-4）；训练 30 epochs，batch size=16，lr=0.01；语义一致性阈值 λ=0.8；参考图像每类别 N=25（消融实验检验 5~50 范围）。
- **基础模型**：Stable Diffusion v1-4（sdv1-4），CLIP ViT-L/14。
