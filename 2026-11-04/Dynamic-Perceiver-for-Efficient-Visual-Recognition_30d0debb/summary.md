---
title: "Dynamic-Perceiver-for-Efficient-Visual-Recognition"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:36:36"
field: "高效视觉识别"
keywords: ["dynamic early exiting", "visual recognition", "dual-branch architecture", "latent code", "cross-attention", "efficient inference"]
innovations: ["双分支架构显式解耦特征提取与早期分类", "对称交叉注意力(X2Z/Z2X)实现特征与隐码双向信息流动", "FKT模块与自蒸馏训练策略提升多出口分类器性能"]
benchmarks: ["ImageNet", "Something-Something V1", "COCO"]
---

# 论文速读：Dynamic Perceiver for Efficient Visual Recognition

## 一句话总结
本文提出 **Dynamic Perceiver（Dyn-Perceiver）**，通过双分支架构显式解耦特征提取与早期分类任务，将早期分类器仅置于分类分支，从而在不损害最终分类器性能的前提下实现高效的动态早期退出推理。

## 研究问题与动机
- **核心问题**：现有早期退出（early-exiting）网络在中间层特征上构建线性分类器，迫使低层特征承载高层语义，导致分类器相互干扰，显著降低最终分类器的性能。
- **现有方法不足**：传统多退出网络（如 MSDNet、RANet）的早期分类器直接附加在中间特征上，特征提取与分类任务紧密耦合；部分工作通过多尺度结构或稠密连接缓解此问题，但设计复杂且不具通用性。
- **设计理想**：存在一个独立于特征提取的隐码（latent code），直接承载语义用于分类；特征提取与早期分类应完全解耦，保证深层特征的无损提取。
- **Perceiver 启发的局限**：Perceiver 使用隐码查询原始输入，计算开销巨大，难以直接应用于高分辨率图像视觉识别任务。

## 核心贡献（创新点）
- **双分支解耦架构**：设计特征分支与分类分支，早期分类器仅置于分类分支，从根本上避免早期分类对特征提取过程的干扰，且最终出口精度甚至有所提升。
- **对称交叉注意力机制**：提出 X2Z（特征→隐码）与 Z2X（隐码→特征）双向交叉注意力，使分类分支的隐码能逐步聚合语义信息，同时特征分支也能吸收分类任务的语义反馈。
- **轻量 Token Mixer 设计**：在每个阶段之间对隐码进行 token 下采样与通道扩展，显著减少原始 Perceiver 的计算开销，使其适用于视觉识别任务。
- **FKT 模块与自蒸馏训练策略**：提出 Forward Knowledge Transfer（FKT）模块将早期分类器知识传递至深层分类器，并结合最后一层分类器的自蒸馏损失，在端到端训练下同时提升各出口性能。
- **通用框架与多任务验证**：Dyn-Perceiver 可作为即插即用模块构建于任意主流视觉骨干网之上，并在图像分类、动作识别和目标检测任务上均取得最优精度-效率权衡。

## 方法详解

**整体架构**：Dyn-Perceiver 由两个并行分支组成，共 4 个阶段（stage）。

1. **特征分支（Feature Branch）**
   - 采用标准 CNN 骨干网（ResNet-50、RegNet-Y、MobileNet-v3）。
   - 输入图像经过各阶段卷积操作生成特征图 $\mathbf{X}_0, \cdots, \mathbf{X}_4$，自低层向高层逐级提取特征。

2. **分类分支（Classification Branch）**
   - 输入为可训练隐码 $\mathbf{Z}_0$（随机初始化），表征待学习的分类语义。
   - 每阶段内部执行 X2Z 交叉注意力 → Transformer 自注意力块 → Token Mixer。
   - 公式（Stage i）：
     $$\mathbf{Z}_i = \psi_i(f_i^{\text{att}}(g_i(\mathbf{Z}_{i-1}, \mathbf{X}_{i-1})))$$
     其中 $g_i$ 为 X2Z 交叉注意力，$f_i^{\text{att}}$ 为 Transformer 自注意力块，$\psi_i$ 为 Token Mixer。

3. **X2Z 交叉注意力**
   - 隐码 $\mathbf{Z}$ 作为 Query，图像特征 $\mathbf{X}$ 作为 Key 和 Value。
   - 对早期高维特征图先用深度可分离卷积（DWC）增强局部特征，再下采样至 $7 \times 7$，以降低计算量。
   - 引入相对位置偏置（RPB）编码位置信息。

4. **Z2X 交叉注意力**
   - 特征分支卷积输出 $\tilde{\mathbf{X}}_i$ 作为 Query，隐码 $\mathbf{Z}_i$ 作为 Key 和 Value。
   - 将分类分支学到的语义信息反馈给特征分支：
     $$\mathbf{X}_i = h_i(f_i^{\text{conv}}(\mathbf{X}_{i-1}), \mathbf{Z}_i)$$

5. **Token Mixer**
   - 使用两个线性层调节隐码：减少 token 数量并扩展通道维度，对齐相邻阶段的特征维度。

6. **分类器与 FKT 模块**
   - 早期分类器仅附加在最后两个阶段的隐码输出上（池化后接线性分类头）。
   - FKT（Forward Knowledge Transfer）：早期分类器输出经线性层后，与下一阶段池化隐码拼接再输入分类器，形成分类器间"捷径"。
   - 最终分类器融合两分支输出后进行分类。

7. **动态推理与训练策略**
   - 推理时按分类置信度（Softmax 最大值）判断是否提前退出；前两个阶段两分支可并行执行，后两阶段串行以最小化早退计算。
   - 训练损失：
     $$\mathcal{L}_k = \alpha \mathcal{L}_k^{\text{CE}} + (1-\alpha) \mathcal{L}_k^{\text{KD}}, \quad k=1,\cdots,K-1$$
     $$\mathcal{L}_K = \mathcal{L}_K^{\text{CE}}$$
     其中 $\mathcal{L}^{\text{KD}}$ 为与最终分类器的 KL 散度，$\alpha=0.5$，总损失 $\mathcal{L} = \sum_{k=1}^{K} \mathcal{L}_k$。

## 实验与结果

**数据集**：ImageNet（1000类，1.2M训练/50K验证）、Something-Something V1（98k视频）、COCO（80类目标检测）。

**评估指标**：Top-1 准确率、平均 FLOPs、实际推理延迟。

**主要结果**：
- **ImageNet 分类（ResNet 系列）**：Dyn-Perceiver 显著优于各类动态网络（Layer/Channel/Spatial skipping），且单模型可通过调整阈值适应不同计算预算，无需重新训练。
- **ImageNet 分类（RegNet-Y 系列）**：在 400M–3.2G FLOPs 范围内持续超越静态骨干网；相比 RegNet-Y-4GF 减少 4.8× 计算量且精度持平；相比 Swin-Transformer 和 DAT 分别减少 1.8× 和 1.4× 计算量。
- **对比早期退出方法**：在同等计算预算下，全面超越 MSDNet、RANet、GFNet、DVT、CF-ViT 等 SOTA 早期退出网络。
- **MobileNet-v3 系列**：在 0.2–0.8 GFLOPs 预算内，相比 MobileNet-v3 减少约 1.2–1.4× 计算量并保持相同精度。
- **动作识别（Something-Something V1）**：嵌入 TSM 框架后取得良好的精度-效率权衡。
- **目标检测（COCO）**：基于 RegNet-Y-1.6GF 的 Dyn-Perceiver 在 RetinaNet 框架下 mAP 达 40.2%，优于 RegNet-X-3.2GF（39.0%）和 RegNet-Y-3.2GF*（39.3%），计算量更低。
- **实际硬件加速**：在 TX2 移动端、桌面 CPU 和 A100 GPU 上均验证了理论效率到实际延迟的有效转化，且两分支并行设计优于 Mobile-Former 等序列化结构。

**消融实验关键结论**：
- 早期分类器置于特征分支会降低最终出口精度（↓0.2% 甚至更多），而置于分类分支可提升最终出口精度（↑0.7%）。
- 移除 Token Mixer 导致显著性能下降（Exit 1: 42.3% vs 58.0%），证明其必要性。
- FKT 和自蒸馏训练均对提升早期分类器性能有正向贡献。

## 相关工作脉络

- **Perceiver / Perceiver IO**（Jaegle et al., ICML 2021 / ICLR 2022）：通用感知架构，使用隐码迭代查询输入；本文受其启发但引入特征分支和对称交叉注意力，大幅降低计算开销并支持动态推理。
- **MS-DNet**（Huang et al., ICLR 2018）：多尺度密集网络，在中间层附加分类器；本文指出其仍基于中间特征做分类，设计复杂，而 Dyn-Perceiver 以双分支解耦实现更简洁的通用方案。
- **RANet**（Yang et al., CVPR 2020）：分辨率自适应网络，通过密集连接缓解早期退出性能退化；本文通过架构解耦而非结构增广解决问题。
- **Mobile-Former**（Chen et al., CVPR 2022）：结合 CNN 与 Transformer 的轻量静态网络；本文与之区别在于面向动态早期退出设计，且双分支可并行计算，实际延迟更低。
- **动态视觉 Transformer（DVT）**（Wang et al., NeurIPS 2021）：自适应序列长度的动态 ViT；本文聚焦特征提取与分类解耦，提供与 backbone 类型无关的通用框架。
- **CF-ViT**（Chen et al., AAAI 2023）：近期 coarse-to-fine ViT；本文在其基础上展示更优的精度-效率曲线，且支持对象检测等下游任务。

## 局限性与未来方向

- **隐码初始化的敏感性**：可训练隐码的初始化和规模（token 数量取 {128, 192, 256}）需要经验调优，未提供系统性的初始化策略分析。
- **仅验证三种 CNN 骨干**：特征分支选用 ResNet、RegNet-Y 和 MobileNet-v3，未探索 Vision Transformer 或其他架构作为特征分支的可能性。
- **目标检测任务未使用动态推理**：COCO 检测实验仅验证通用性，未在检测框架中启用 early exiting，动态推理在检测中的潜力有待探索。
- **固定阈值策略**：推理时置信度阈值的确定依赖验证集划分，未讨论自适应阈值学习机制。
- **计算假设简化**：FLOPs 评估基于理论估算，实际硬件加速受内存访问模式等因素影响，本文仅在最简单设置（batch=1/128）下验证。

## 研究启发与可借鉴点

- **解耦思想的可迁移性**：将"特征提取"与"分类/决策"显式解耦的双分支范式，可推广至分割、检测等密集预测任务，避免中间层监督对主干特征的污染。
- **对称交叉注意力的通用设计**：X2Z 与 Z2X 的双向信息流动机制，可作为通用的"任务分支↔特征分支"交互模块，嵌入其他动态网络架构。
- **FKT 模块的普适价值**：分类器间的前向知识传递是一种轻量级辅助优化手段，可替代或部分替代"预训练-微调"两阶段策略，适用于多出口网络的端到端训练。
- **实际硬件验证的价值**：论文在 TX2、CPU、A100 多平台上验证实际延迟，而非仅依赖 FLOPs，这一实验设计值得在后续效率优化工作中借鉴。
- **即插即用框架设计**：Dyn-Perceiver 无需修改骨干网内部结构即可附加，这种"外挂式"设计降低了对已有模型的改造成本，适合在团队现有 pipeline 中快速集成验证。

## 关键术语表

- **Early Exiting（早期退出）**：在深度网络中间层设置分类器，高置信度样本提前输出，跳过后续层计算以节省推理开销。
- **Latent Code（隐码）**：可训练的独立表征向量，从输入或特征中查询语义信息，用于下游分类任务而不干扰特征提取。
- **Cross-Attention（交叉注意力）**：不同模态或分支的特征之间建立的注意力机制，Query 来自一个分支，Key/Value 来自另一个分支。
- **Token Mixer（Token 混洗器）**：通过线性变换调节隐码的 token 数量和通道维度，实现跨阶段的信息压缩与对齐。
- **FKT（Forward Knowledge Transfer）**：将早期分类器输出经线性层后传递至深层分类器的知识迁移模块，形成分类器间的捷径连接。
- **Self-Distillation（自蒸馏）**：利用最终（最深）分类器的软概率分布指导早期分类器训练，以提升浅层出口的分类质量。
- **Dynamic Network（动态网络）**：根据输入样本复杂度自适应调整计算深度的网络，"简单"样本消耗更少计算资源。
- **Relative Position Bias（RPB，相对位置偏置）**：Transformer 注意力中编码相对位置信息的可学习偏置项，增强空间位置感知。

## 可复现要素

- **数据集**：ImageNet、Something-Something V1、COCO — 均公开可用。
- **代码**：已开源，地址 https://www.github.com/LeapLabTHU/Dynamic_Perceiver。
- **骨干网**：ResNet-50、RegNet-Y（400M–3.2G FLOPs）、MobileNet-v3（0.75×–1.5×），均可从官方或 MMDetection 获取预训练权重。
- **隐码初始 token 数**：L ∈ {128, 192, 256}。
- **自注意力头数**：Stage i 的头数为 $2^{i-1}$（i=1,2,3,4）。
- **交叉注意力头数**：所有交叉注意力层均使用 1 个头。
- **蒸馏权重**：$\alpha = 0.5$。
- **其他超参**：详见论文 Appendix A（模型配置）和 Appendix B（训练细节）。
