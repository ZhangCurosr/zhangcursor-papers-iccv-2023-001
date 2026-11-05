---
title: "DiffuMask-Synthesizing-Images-with-Pixel-level-Annotations-f"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_DiffuMask_Synthesizing_Images_with_Pixel-level_Annotations_for_Semantic_Segmentation_Using_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:36:08"
---

# 论文速读：DiffuMask-Synthesizing-Images-with-Pixel-level-Annotations-f

## 一句话总结
本文提出 DiffuMask，通过挖掘预训练文本到图像扩散模型（Stable Diffusion）的文本-图像跨注意力图，无需任何人工像素级标注即可自动生成高质量语义掩码；基于该合成数据训练的分割模型在 VOC 2012 和 Cityscapes 上性能逼近纯真实数据，并在开放词汇（zero-shot）分割任务上刷新 SOTA。

## 研究问题与动机
- 像素级语义分割标注成本极高（Cityscapes 单图标注可达 60 分钟），且受隐私、版权与场景稀缺性限制难以持续采集。
- 弱监督分割（图像级标签、点、边界框等）虽降低标注成本，但精度天花板低、训练策略复杂且仍需额外手工标注。
- 传统合成数据方法（DatasetGAN、BigDatasetGAN）依赖少量人工掩码校准生成器，掩码精度不足且难以泛化至未见类别。
- 大尺度语言-图像扩散模型已具备强大的零样本语义组合生成能力，但其内部跨注意力机制与像素级语义定位的映射关系尚未被系统挖掘，存在“有图无标”的资源浪费。

## 核心贡献（创新点）
1. **首次系统论证扩散模型跨注意力图可直接作为像素级语义掩码**：与 DatasetGAN/BigDatasetGAN 依赖人工掩码微调解码器不同，DiffuMask 完全免人工定位，直接利用 SD 内部注意力实现零成本标注。
2. **提出基于语义亲和力的自适应阈值二值化策略**：突破固定阈值（γ）无法适配异构类别的局限，引入 AffinityNet 匹配相似度空间自动寻优，显著优于静态截断方法。
3. **设计噪声学习（Noise Learning）联合剪流水线**：针对生成掩码中普遍存在的低质量样本，通过交叉验证统计置信度分布并按类剔除尾部噪声，释放注意力图的潜在精度。
4. **构建提示工程与域适配数据增强的完整合成数据 Pipeline**：通过子类别枚举与 CLIP 检索双管齐下提升语义多样性，配合 Splicing、模糊、遮挡、透视变换系统性压缩合成-真实域偏移。

## 方法详解
- **跨注意力图提取与多尺度聚合**：在 Stable Diffusion U-Net 的不同层（输出 8×8、16×16、32×32、64×64 四档分辨率）与各扩散步 $t$ 提取每个文本 token 的注意力图 $\mathcal{A}_j^{s,t}$，经逐样本 max 归一化后对层与时间步取平均：$\hat{\mathcal{A}}_j = \frac{1}{S\cdot T}\sum_{s,t}\frac{\mathcal{A}_j^{s,t}}{\max(\mathcal{A}_j^{s,t})}$，得到每个类别的概率图。
- **自适应阈值二值化**：固定 $\gamma$ 对各类别效果差异显著（如 Bottle 最优约 0.48，Dog 约 0.39）。利用 AffinityNet 预测粗粒度语义亲和力图 $\hat{B}$，在搜索空间 $\Omega=\{\gamma_i\}$ 中最大化 IoU 匹配代价 $\hat{\gamma}=\arg\max_{\gamma\in\Omega}\sum\mathcal{L}_{\text{match}}(\hat{B},B_\gamma)$，再经 Dense CRF 空间细化得到最终掩码 $B$。
- **噪声学习（NL）剪枝**：对生成数据集执行 3-fold 交叉验证，统计模型预测与伪真值间的 IoU 联合分布，按类别剔除自置信度最低的 $\alpha\%$ 样本（VOC 设 $\alpha=0.7$，Cityscapes 每子类保留 10k），有效过滤掩码边缘粘连、漏标等噪声。
- **提示工程**：(1) 子类别提示：从 Wiki 选取 K=100 个细分子类构造模板 “Photo of a [sub-class] object”；(2) 检索式提示：用 CLIP-retrieval 在 Laion5B 上检索 top N=10 真实图像 caption 替换手搓模板，共生成 $K\times N$ 条 prompt 池，推理时随机采样。
- **域适配数据增强**：Splicing（1×2 至 8×8 多尺度拼贴模拟目标尺度变化）、Gaussian Blur（Kernel 6~22）、Occlusion（类 CutMix 比例遮挡）、Perspective Transform，针对性缓解合成图目标居中/尺寸统一/纹理单一等问题。

## 实验与结果
- **数据集与设置**：VOC 2012（20类）、Cityscapes（Human/Vehicle）、ADE20K；基线 Mask2Former（R50/Swin-B/Swin-L）；每类生成 10k 图像，NL 后 VOC 保留 60k，Cityscapes 每子类保留 10k（共 100k）。
- **VOC 2012 语义分割**：DiffuMask (Swin-B, 纯合成) 取得 70.6% mIoU，较纯真实数据（84.3%）差距可控；bird、horse、sheep 等类仅落后 3%~5%。微调 5k 真实数据后达 84.9%，反超全量真实数据（83.4%）。
- **Cityscapes**：纯合成 R50 得 78.0% mIoU；加入 1.5k 真实数据微调后 R50 达 90.1%、Swin-B 达 91.4%，逼近全量
