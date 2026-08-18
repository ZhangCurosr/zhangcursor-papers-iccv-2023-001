---
title: "VL-PET-Vision-and-Language-Parameter-Efficient-Tuning-via-Gr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_VL-PET_Vision-and-Language_Parameter-Efficient_Tuning_via_Granularity_Control_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:32:03"
field: "多模态参数高效微调"
keywords: ["parameter-efficient tuning", "vision-and-language", "granularity control", "encoder-decoder", "adapter", "LoRA", "multimodal learning"]
innovations: ["粒度控制机制有效约束PET模块修改幅度", "针对编码器-解码器功能差异的轻量级差异化适配策略", "模型无关的VL-PET框架可实例化为多种参数复杂度模块"]
benchmarks: ["VQAv2", "GQA", "NLVR²", "MSCOCO", "TVQA", "How2QA", "TVC", "YC2C"]
---

# 论文速读：VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control

## 一句话总结
本文提出 VL-PET 框架，通过粒度控制机制（granularity-controlled mechanism）有效约束 PET 模块的输出幅度，并结合轻量级模块设计策略区分编码器和解码器的适配需求，在极低参数开销下显著优于 VL-Adapter、LoRA 等现有方法，在图像-文本和视频-文本任务上均取得 SOTA 级别性能。

## 研究问题与动机
- **现有 VL PET 技术缺乏对模块化修改的有效控制**：直接将 Adapter/LoRA 等重修改嵌入 PLM 会导致中间输出不稳定，引发性能退化；VL-Adapter 等现有方法未对修改幅度施加约束。
- **忽略了编码器-解码器之间的功能差异**：编码器专注于 VL 对齐与表征建模，解码器专注于自回归文本生成；现有方法对两者采用完全相同的模块修改策略，未考虑各自的独特能力。
- **VL PET 研究起步较晚且多局限于判别任务**：多数 VL PET 方法仅关注图像-文本检索等判别任务，缺乏对生成式任务（如图注生成、视频描述）的有效探索。
- **直接迁移 NLP/CV PET 技术到 VL 场景效果不佳**：在更大规模骨干（T5-base）上，部分现有 PET 方法（如 VL-Adapter、LoRA）不仅未能进一步提升，甚至出现性能退化。

## 核心贡献（创新点）
- **提出粒度控制机制（Granularity-Controlled Mechanism）**：生成粒度控制矩阵 G 对 PET 模块输出进行逐元素缩放，从而有效控制模块修改幅度；与 VL-Adapter 等直接叠加模块修改的做法本质不同，引入了显式的强度控制。
- **设计轻量级 PET 模块集成策略（Lightweight PET Module Designs）**：编码器中将 VL-PET 模块集成到 self-attention 和 feed-forward 以增强 VL 对齐，解码器中仅集成到 cross-attention 的 value 矩阵以保留预训练文本生成能力；与现有方法在 encoder-decoder 所有模块中均匀插入修改的做法存在本质区别。
- **提出多头模块化修改（Multi-head Modular Modification）**：将 PET 模块扩展为多头形式以提升表征能力；与 LoRA/Adapter 的单头设计本质不同，增强了对复杂 VL 模式的学习。
- **构建模型无关的 VL-PET 框架**：可从同一框架实例化 small/middleX/middleY/large 四种参数复杂度的模块，在效率-效果之间灵活权衡；与特定模型绑定的方法（如仅支持 BART）本质不同，具备良好的泛化性。
- **验证 VL-PET 设计对已有 PET 方法的增益**：将粒度控制和轻量化设计迁移到 Compacter、VL-Adapter 上，显著提升其性能并减少参数量。

## 方法详解
- **统一公式**：PET 模块的更新形式化为 H ← G ⊙ (H + ΔH)，其中 H 为 PLM 模块的中间隐藏状态，ΔH 为模块修改，G ∈ ℝ^(N×d) 为粒度控制矩阵（⊙ 表示逐元素乘积）。
- **四种粒度控制级别**：
  - **Large**：G_large = s·σ(φ(X·W_down)·W_up)，参数复杂度 O(dr)，通过瓶颈结构生成完整 N×d 矩阵。
  - **MiddleX**：G_middleX = Ḡ_middleX · 1_(1×d)，其中 Ḡ_middleX = s·σ((X+H)·W_down)，基于输入和输出组合生成，参数复杂度 O(d)。
  - **MiddleY**：G_middleY = 1_(N×1)·Ḡ_middleY，其中 Ḡ_middleY = s·(σ(z)+1_(1×d))，z 为可训练向量，参数复杂度 O(d)。
  - **Small**：G_small = 1_(N×1)·Ḡ_small·1_(1×d)，其中 Ḡ_small = s·ψ(σ(Concat(X,H)·W_down))，ψ 为沿序列维度平均池化，参数复杂度 O(d)。
- **多头模块化修改**：ΔH' = φ(Concat(X'·W_down^(1), ..., X'·W_down^(Nh)))·W_up，将输入分头投影后拼接再升维，增强表征能力。
- **轻量级集成策略**：
  - **编码器**：将 VL-PET 模块集成到 self-attention 和 feed-forward 模块，使用 G_large 等较大粒度进行充分适配。
  - **解码器**：仅将 VL-PET 模块集成到 cross-attention 的 value 矩阵（而非整个 cross-attention 模块），使用轻量级 G（如 G_small 或 G=1）进行精细控制，避免破坏预训练文本生成能力。
- **层归一化策略**：编码器 LN 可训练、解码器 LN 冻结，与 VL-Adapter 全量可训练策略不同，实验表明该策略更有效。
- **多任务学习**：采用统一文本生成范式，共享 PET 模块完成多任务微调，跨任务知识有效促进 NLVR² 等推理任务。

## 实验与结果
- **数据集**：图像-文本任务含 VQAv2、GQA、NLVR²、MSCOCO（4项）；视频-文本任务含 TVQA、How2QA、TVC、YC2C（4项，VALUE benchmark）。
- **骨干模型**：BART-base 和 T5-base。
- **基线方法**：Full Fine-tuning、BitFit、Prompt Tuning、Compacter、HyperFormer、LoRA、VL-Adapter、LST。
- **主要结果（BART-base，图像-文本）**：
  - **VL-PET_large** 在四任务平均得分达 **79.18%**，显著提升 **VL-Adapter（76.93%，+2.92%）** 和 **LoRA（76.60%，+3.37%）**，可训练参数占比仅为 4.16%（VL-Adapter 为 4.18%，LoRA 为 5.93%）。
  - **VL-PET_small/middleX/middleY** 仅占 2.98% 参数，仍超越 VL-Adapter 约 1.6%~1.9%。
  - COCO 图像描述任务提升最为显著（122.03 CIDEr vs. VL-Adapter 114.61），证明轻量级解码器设计对生成任务的重要性。
- **主要结果（T5-base，图像-文本）**：
  - **VL-PET_large** 平均得分 **79.52%**，超越 **VL-Adapter（76.90%，+3.41%）** 和 **LoRA（74.30%，+7.03%）**。
  - 在更大模型上 VL-PET 持续受益，而其他 PET 方法（如 LoRA）反而出现性能下降。
- **主要结果（BART-base，视频-文本）**：
  - **VL-PET_large** 平均得分 **88.63%**，超越 **VL-Adapter（87.95%，+0.77%）** 和 **LoRA（83.77%，+5.80%）**。
- **消融实验关键结论**：
  - 粒度控制机制不可或缺（无 G 版本性能下降约 0.33%）。
  - 解码器中仅用 cross-attention 的 value 矩阵（而非 Key 或整个 Cross）效果最佳。
  - 多头设计（最优 4 头）显著优于单头。
  - 多任务学习在 NLVR² 上提升约 23 个百分点（50.36% → 73.43%）。

## 相关工作脉络
- **VL-Adapter** [49]：当前 SOTA 的 VL PET 方法，直接将 NLP 的 Adapter 移植到 encoder-decoder PLM；本文定位为其直接竞争者，通过在粒度和集成策略上的 VL 特定设计实现超越。
- **LoRA** [16]：NLP 领域经典的低秩适应方法；本文指出其在 VL 任务上直接应用时存在过度修改问题，VL-PET 通过粒度控制和轻量化设计将其性能提升 3~7 个百分点。
- **Compacter** [21]：基于超复数分解的高效 Adapter；本文将其作为验证对象，证明粒度控制和轻量化设计可同样增益 Compacter（+2.56%）。
- **LST（Ladder Side-Tuning）** [48]：另一项针对 encoder-decoder 的 VL PET 方法；本文在 T5-base 上以更少参数实现更高性能，证明轻量化设计的优越性。
- **BitFit / Prompt Tuning**：参数最少的基线方法，但在 VL 生成任务上表现明显落后，凸显了模块修改类方法的必要性。
- **MaPLE / UniAdapter**：多模态 prompt 学习和统一适配器；本文强调其在生成任务和 VL 特定适配设计上的局限性。

## 局限性与未来方向
- **仅在 BART-base 和 T5-base 上验证**：尚未在更大规模模型（如 BART-large、T5-large）或新兴架构（如 LLaMA-based VL 模型）上测试泛化性。
- **仅针对 encoder-decoder 架构**：decoder-only（如 LLaMA）和 encoder-only（如 CLIP）架构的适用性有待探索。
- **粒度控制矩阵的生成涉及额外计算**：尤其是 G_large 级别（需前向计算），在实际推理中可能引入延迟，论文未讨论推理效率。
- **多任务学习虽有效但需要所有任务标签**：对于仅能访问单一任务数据的场景，单任务微调的潜力未被充分评估。
- **视觉编码器固定为离线特征提取**：未来可探索端到端训练中同时适配视觉和语言部分。

## 研究启发与可借鉴点
- **粒度控制机制可迁移到其他 PET 技术**：VL-PET 的 G 矩阵思想可作为后处理方法集成到 LoRA、Adapter、Compacter 等任何模块化修改中，形成"控制+修改"的通用范式。
- **编码器-解码器差异化适配策略具有普适价值**：对编码器进行充分适配、对解码器进行轻量/精细适配的设计思路，可推广到其他领域（如 NLP 的多任务学习、CV 的检测/分割）。
- **多任务统一文本生成范式的效率优势显著**：共享 PET 模块完成多任务微调，参数减少约 4 倍且跨任务知识 Transfer 效果显著（NLVR² 提升巨大），适合资源受限的多任务 VL 研究。
- **跨 Attention 模块的 value 矩阵适配**：仅修改 cross-attention 的 value 矩阵即可达到最优效果，这一发现降低了解码器适配的复杂度，值得在视频-语言等更长序列任务中复用。
- **可结合本团队方向探索**：团队若关注低资源 VL 任务，VL-PET 的轻量化设计可直接用于 Few-shot/Zero-shot VL 微调；若关注推理效率，可进一步压缩 G_large 的计算开销。

## 关键术语表
- **Parameter-Efficient Tuning (PET)**：冻结预训练模型主干，仅训练少量额外参数或子模块的低成本微调方法，避免全量微调的资源开销。
- **Granularity-Controlled Mechanism**：通过生成粒度控制矩阵 G 对模块修改输出进行逐元素缩放，动态控制适配强度的核心机制。
- **VL-Adapter**：将 NLP 领域的 Adapter 模块直接移植到 encoder-decoder PLM 的 VL PET 方法，是本文最主要的对比基线。
- **Lightweight PET Module Designs**：针对编码器和解码器的不同功能设计差异化适配策略（编码器充分适配、解码器轻量适配）的集成方案。
- **Multi-head Modular Modification**：将 PET 模块的投影操作扩展为多头形式，提升对复杂 VL 模式的学习能力。
- **Cross-Attention Value Matrix Adaptation**：在解码器中仅对 cross-attention 的 value 矩阵进行模块化修改，而非整个注意力模块，以最小化对预训练知识的破坏。
- **Unified Text Generation**：将多种 VL 任务统一为文本生成任务（如 "vqa: [Q]"），通过共享 PET 模块实现多任务学习。

## 可复现要素
- **数据集**：VQAv2、GQA、NLVR²、MSCOCO、TVQA、How2QA、TVC、YC2C（VALUE benchmark）；均为公开数据集。
- **代码开源**：https://github.com/HenryHZY/VL-PET
- **权重开源**：论文未明确说明，仅提供了代码链接。
- **关键超参**：
  - 骨干模型：BART-base（146M 可训练参数）和 T5-base（241M 可训练参数）；
  - 缩放因子 s：针对不同 PLM 单独设定（论文附录有说明）；
  - 投影维度 r：论文未明确给出具体数值；
  - 多头数：编码器最优为 4 头，VL-Adapter 对比时为 8 头；
  - 视觉编码器：使用离线特征提取（论文附录含实现细节）。
