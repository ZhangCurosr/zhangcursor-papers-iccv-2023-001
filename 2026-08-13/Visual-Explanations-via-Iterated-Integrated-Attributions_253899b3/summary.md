---
title: "Visual-Explanations-via-Iterated-Integrated-Attributions"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Barkan_Visual_Explanations_via_Iterated_Integrated_Attributions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:32:52"
---

# 论文速读：Visual-Explanations-via-Iterated-Integrated-Attributions

## 一句话总结
本文提出了迭代积分归因（IIA）方法，通过在输入图像与网络多层中间表示（CNN激活图/ViT注意力矩阵）及其梯度上进行迭代线性插值并累积梯度，生成精确且聚焦的视觉解释热力图。该方法统一适用于 CNN 与 ViT 架构，并在多项客观解释指标与分割基准上全面超越现有 SOTA 方法。

## 研究问题与动机
- **深度视觉模型黑盒化严重**：CNN 与新兴的 ViT 在分类、检测等任务取得突破的同时，缺乏对预测决策过程的直观解释，阻碍了高风险场景下的可信部署。
- **现有方法局限于单一空间或单层**：Grad-CAM 类方法仅利用最后一层激活与梯度，丢失了深层特征聚合的全局信息；路径积分方法（如 IG）仅在输入像素空间插值，忽略内部表征的梯度结构，易产生稀疏/分散的热力图。
- **ViT 解释方法缺乏通用性与梯度引导**：Rollout、T-Attr 等方法依赖原始注意力分数的简单累积或特定的泰勒分解，难以有效融合跨层信息流，且多数仅针对 Transformer 设计，无法统一覆盖 CNN。
- **计算效率与解释质量的权衡难题**：多层积分理论上会带来组合爆炸，如何在不显著增加推理开销的前提下实现多层协同归因，仍是待解难题。

## 核心贡献（创新点）
1. **提出通用迭代积分归因（IIA）统一框架**，可同时处理 CNN 激活图与 ViT 注意力矩阵的跨层插值积分。*与已有工作的本质区别在于打破了现有方法仅在输入空间或单一网络层进行积分的限制，首次将“输入-中间表示-梯度”三者纳入同一迭代积分范式。*
2. **设计 Gradient Rollout (GR) 变体**，将传统 Rollout 中的注意力矩阵相乘替换为“插值后注意力矩阵与其梯度的 Hadamard 积”。*区别于 T-Attr/GAE 的直接相关性传播，GR 通过梯度加权机制显著增强目标类别的 attribution 聚焦度，避免无关 token 的干扰。*
3. **构建双积分（IIA2）与三积分（IIA3）可控变体**，分别激活输入+末层、输入+倒数第二层+末层，实现低层空间细节与高层语义聚合的互补。*与 IG 等单一路径积分不同，IIA 的多层联合插值能更完整地追踪决策路径，且在相同空间分辨率下末层比倒数第二层提供更优的特征聚合。*
4. **给出严谨的计算复杂度分析并验证 GPU 批处理可行性**，证明 IIA 的实际运行开销可与 GC/IG 相当。*突破了“迭代积分必然昂贵”的刻板印象，为 XAI 方法的工程落地扫清障碍。*

## 方法详解
- **问题设定**：输入 $\mathbf{x} \in \mathbb{R}^{c_0 \times p_0 \times q_0}$，网络含 $L$ 个中间层 $h^l$，目标生成与输入空间对齐的 2D 归因图 $\mathbf{m}$。
- **广义插值构造**：对第 $l$ 层提取感兴趣表示 $\mathbf{u}^l = u^l(\mathbf{h}^{l-1})$，定义参考表示 $\mathbf{r}^l = \min(\mathbf{u}^l)$（CNN 逐通道取最小值，ViT 注意力因 softmax 输出全正故设 $\mathbf{r}^l=\mathbf{0}$）。插值变量为 $\mathbf{v}^l = \mathbf{r}^l + (a_l)^{b_l}(\mathbf{u}^l - \mathbf{r}^l)$，其中 $b_l \in \{0,1\}$ 控制该层是否参与积分，$a_l \in [0,1]$ 为插值步长。
- **IIA 通用积分公式**：
  $\mathbf{m}_{\mathbf{b}}^{l} = \int_0^1 \cdots \int_0^1 (\mathbf{u}^l - \mathbf{r}^l) \circ \left( \int_0^1 \mathbf{q}^l d a_l \right) \cdots d a_0$
  其中 $\mathbf{q}^l$ 为前 $l$ 层表示及其梯度的任意函数。当仅对输入插值（$\mathbf{b}=[1,0,\dots,0]$）且 $\mathbf{q}^0 = \partial \phi_y / \partial \mathbf{v}^0$ 时，公式严格退化为 Integrated Gradients (IG)。
- **CNN 实例化**：所有卷积块输出激活图参与插值，被积函数设为 $\mathbf{q}^l = \mathbf{v}^l \circ \frac{\partial f(\mathbf{h}^L)}{\partial \mathbf{v}^l}$（激活与对应类别梯度的 Hadamard 积），最后沿通道取均值并 resize 至输入尺寸。
- **ViT 实例化**：对每个注意力头的注意力矩阵进行独立插值，被积函数采用 **GR**：将传统 Rollout 的矩阵连乘替换为逐元素的 “注意力矩阵 $\circ$ 梯度矩阵”，跨头跨层累积后得到 token 级权重，resize 为 2D 热力图。
- **数值近似**：采用嵌套求和离散化，$\mathbf{m}_{\mathbf{b}}^{l} \approx \frac{1}{n^{\beta}} \sum \cdots \sum (\mathbf{u}^l - \mathbf{r}^l) \circ \sum \mathbf{q}^l$，$\beta = \sum b_i$，实验中统一取插值步数 $n=10$。
- **复杂度优化**：理论复杂度 $\mathcal
