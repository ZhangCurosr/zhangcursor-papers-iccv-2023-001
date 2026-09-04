---
title: "EQ-Net-Elastic-Quantization-Neural-Networks"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_EQ-Net_Elastic_Quantization_Neural_Networks_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:21:58"
field: "模型压缩与高效推理"
keywords: ["模型量化", "一次性网络", "混合精度量化", "弹性量化空间", "超网训练"]
innovations: ["提出涵盖位宽/粒度/对称性的弹性量化空间，实现一套权重适配多种量化形式", "设计WDR-Loss和GPG-Loss解决超网负梯度抑制问题，提升多形态量化鲁棒性", "提出条件化精度预测器CQAP结合遗传算法实现高效的混合精度搜索"]
benchmarks: ["ImageNet", "CIFAR-10"]
---

# 论文速读：EQ-Net: Elastic Quantization Neural Networks

## 一句话总结
本文提出 **EQ-Net**，一种一次性权重共享量化超网方法，通过设计涵盖位宽、粒度和对称性的弹性量化空间，训练一个鲁棒的统一超网，无需重复优化即可快速搜索并部署多种量化形式的模型，达到或接近静态量化方法的性能。

## 研究问题与动机
- **硬件量化形式异构导致重复优化**：不同 AI 加速器支持的量化方案不同（如 NVIDIA TensorRT 采用 per-channel 对称量化，Qualcomm SNPE 采用 per-tensor 非对称量化），传统 QAT 方法需针对不同硬件重新训练，部署效率极低。
- **已有 B-OFA 方法灵活性不足**：现有 bit-width One-For-All 方法仅支持统一切换位宽，未将粒度（per-tensor/per-channel）和对称性（对称/非对称）纳入搜索空间，无法适应多样化的硬件需求。
- **超网训练存在负梯度抑制问题**：不同量化配置之间预测不一致会互为"负样本"，相互抵消梯度，导致超网收敛缓慢、低比特子网性能下降。
- **混合精度搜索成本高昂**：传统方法搜索时需对大量候选子网执行 BN 校准和验证，计算开销大，需要高效的代理评估器。

## 核心贡献（创新点）
1. **弹性量化空间设计**：首次将位宽（bit-width）、粒度（granularity）和对称性（symmetry）统一纳入一个可参数分裂切换的弹性量化空间，实现一套权重适配多种主流量化形式。
   - 与 B-OFA 方法的本质区别：不仅支持位宽弹性，还同时支持粒度和对称性的弹性切换，覆盖更多硬件部署场景。

2. **WDR-Loss（Weight Distribution Regularization Loss）**：引入偏度（skewness）和峰度（kurtosis）正则化，约束共享权重的分布形状趋于均匀，提升权重在弹性量化空间中的鲁棒性。
   - 与 RobustQuant 的本质区别：RobustQuant 仅针对单一固定位宽做峰度正则，本文将其扩展至弹性多形态场景，同时控制偏度和峰度以适配位宽/对称性的动态切换。

3. **GPG-Loss（Group Progressive Guidance Loss）**：基于三明治规则采样高/中/低比特子网组，利用高比特子网的软标签对低比特子网进行渐进式知识蒸馏，减少组间负梯度冲突。
   - 与 MultiQuant 的本质区别：MultiQuant 使用自适应软标签缓解高低比特竞争，本文通过分组渐进引导（H→R→L）建立输出 logits 分布的一致性，训练效率更高。

4. **CQAP（Conditional Quantization-Aware Accuracy Predictor）+ 遗传算法混合精度搜索**：提出以量化对称性和粒度为条件的精度预测器，结合遗传算法快速搜索 Pareto 最优的混合精度配置。
   - 与 HAQ/DARTS 等方法的本质区别：预测器显式将粒度和对称性作为条件输入，支持任意弹性量化配置的精度估计，搜索空间更灵活。

## 方法详解

### 3.1 弹性量化空间
将量化形式分解为三个维度：
- **弹性位宽**：2/3/4/5/6/7/8-bit（轻量模型从 3-bit 起），权重共享，仅分离存储不同位宽对应的步长 $s$ 和零点 $z$。
- **弹性对称性**：通过对 $z$ 的动态设置实现切换——对称量化时固定 $z=0$，非对称量化时 $z \in \mathbb{Z}$ 可学习。
- **弹性粒度**：per-channel 量化为每个 kernel 独立学习步长（$s \in \mathbb{R}_+^{1\times C}$）；per-tensor 量化为整层共享单一步长（$s \in \mathbb{R}_+$），per-tensor 参数可独立学习或通过 per-channel 启发式推导。激活始终为 per-tensor。

### 3.2 WDR-Loss
$$\mathcal{L}_{\text{WDR}} = \frac{1}{L}\sum_{i=1}^{L}\left(|\text{Skew}[\boldsymbol{w}_i]|^2 + |\text{Kurt}[\boldsymbol{w}_i] - \mathcal{K}_T|^2\right)$$
其中 $\mathcal{K}_T = 1.8$ 为理想均匀分布的峰度目标值（引自 RobustQuant 实验）。偏度正则抑制分布倾斜，峰度正则抑制分布尖峰，使权重分布更接近均匀分布，增强量化鲁棒性。

### 3.3 GPG-Loss
每步训练按三明治规则采样最高位宽组 $H$、最低位宽组 $L$、随机位宽组 $R$，构建渐进蒸馏损失：
$$\left\{\begin{array}{l}\mathcal{L}_H = \mathcal{L}_{\text{CE}}(\mathbb{Y}_H, \boldsymbol{y})\\\mathcal{L}_R = \lambda \cdot \mathcal{L}_{\text{KL}}(\mathbb{Y}_R, \mathbb{Y}_H) + (1-\lambda)\cdot\mathcal{L}_{\text{CE}}(\mathbb{Y}_R, \boldsymbol{y})\\\mathcal{L}_L = \lambda \cdot \mathcal{L}_{\text{KL}}(\mathbb{Y}_L, \mathbb{Y}_R) + (1-\lambda)\cdot\mathcal{L}_{\text{CE}}(\mathbb{Y}_L, \boldsymbol{y})\end{array}\right.$$
聚合所有组梯度后更新超网权重，实现从高比特到低比特的渐进一致性。

### 3.4 CQAP + 遗传算法搜索
CQAP 以二进制编码的量化配置为输入，条件项为每层的粒度 $G_w$、对称性 $S_w$（权重）和 $S_a$（激活），位宽项为 $B_w$ 和 $B_a$，输出预测精度：
$$\text{acc} = \text{MLP}(G_w, S_w, S_a, B_w, B_a)$$
遗传算法初始化满足约束的候选种群后，以 CQAP 预测精度为适应度，通过选择-交叉-变异迭代 500 代，搜索满足平均位宽目标的 Pareto 最优混合精度配置。

## 实验与结果
- **数据集**：ImageNet（主实验）、CIFAR-10（消融实验）
- **模型**：ResNet18/50、MobileNetV2、EfficientNetB0
- **基线对比**：LSQ、LSQ+、EdMIPS、RobustQuant、CoQuant、AnyPrecision、MultiQuant、HAWQ-V2、HAQ

**关键结果**：
- **ResNet18 3-bit 固定位宽**：69.3%（↓0.5%），超越 MultiQuant（67.5%，↓2.3%）约 1.8%，接近 FP32（69.8%）；3-bit MPQ 达到 69.8%（↓0.0%，与全精度持平）。
- **ResNet18 2-bit**：69.3% vs RobustQuant 57.3%，提升约 12%。
- **ResNet50 3-bit MPQ**：75.1%（↓1.0%），优于 MultiQuant 75.4%（↓0.7%）但略低于其最优配置；2-bit 达 72.5%（↓3.6%）。
- **MobileNetV2 4-bit**：71.2%（↓0.7%），超越 RobustQuant（+11.4%）和 MultiQuant（+1.1%），MPQ 超越 HAQ（+4.2%）。
- **EfficientNetB0 4-bit**：对称 74.1%，非对称 75.1%（↓2.6% vs FP32 77.7%），体现非对称对含负值激活函数的优势。

**消融结论**：
- WDR：2-bit 提升近 1%，2/4/8-bit 平均提升 0.5%。
- GPG：在 2-bit 训练过程中稳定优于 hard label 和 label smoothing，且 8-bit 最终精度不下降。
- CQAP 秩相关：ResNet18/MobileNetV2 Pearson > 0.90，Kendall > 0.80；EfficientNetB0 略低（Kendall 0.71）因对称/非对称精度差异较大。
- Per-tensor 参数学习：learnable 优于 min/mean/max 启发式方法（2/4/8-bit 分别 +0.2%/+0.6%/+0.3%）。

## 相关工作脉络
1. **OFA（Cai et al., ICLR 2020）**：一次性架构搜索思想奠基者，通过权重共享实现多尺度子网。EQ-Net 借鉴此范式但将搜索空间从网络结构切换为量化配置切换，且完全共享权重无结构差异。
2. **RobustQuant（Shkolnik et al., NeurIPS 2020）**：提出峰度正则化提升量化鲁棒性，仅支持 per-tensor 对称量化。EQ-Net 将其扩展至弹性多形态场景，同时控制偏度和峰度，并引入对称性/粒度弹性。
3. **MultiQuant（Xu et al., IJCAI 2022）**：同一团队前期工作，支持多比特但仅限固定精度策略，无法切换粒度和对称性。EQ-Net 在此基础上扩展为 BGS-OFA，并新增 WDR 和 CQAP。
4. **AnyPrecision（Yu et al., AAAI 2021）**：训练时模拟多比特量化并保存浮点权重，推理时截断低位实现任意位宽。EQ-Net 与之不同：不仅支持位宽弹性，还统一支持粒度和对称性切换，且搜索空间更丰富。
5. **HAQ（Wang et al., CVPR 2019）**：硬件感知的混合精度量化搜索方法。EQ-Net 的 CQAP+GA 搜索策略与之目标相似，但 HAQ 无弹性对称性和粒度支持，且需对每个候选执行 BN 校准，搜索成本更高。
6. **HAWQ-V2（Dong et al., NeurIPS 2020）**：基于 Hessian 信息的混合精度量化，采用 per-channel 对称量化。EQ-Net 的弹性空间可覆盖 per-channel 对称场景，但额外支持 per-tensor 和非对称形式，适应性更强。

## 局限性与未来方向
- **大模型 per-channel 量化稳定性待提升**：作者自述 ResNet50 上 3-bit per-channel 因不稳定性导致性能略低于 MultiQuant，弹性粒度在大模型中的鲁棒性需进一步优化。
- **CQAP 在精度差异大的网络中秩相关下降**：EfficientNetB0 的 Kendall 系数较低，表明当对称/非对称量化精度差距较大时，预测器的泛化能力受限。
- **未探索更深层次的硬件协同优化**：当前仅覆盖位宽/粒度/对称性三个维度，未考虑硬件特有的量化分组（group-wise）、整数量化格式等细节。
- **未来可扩展至 Activation 弹性粒度**：当前激活固定为 per-tensor，未来可探索激活的 per-channel 弹性量化以进一步提升精度。

## 研究启发与
可借鉴点
1. **弹性量化空间的设计范式**：将量化形式分解为位宽/粒度/对称性三个正交维度并统一建模，可作为通用框架迁移到其他量化压缩任务（如稀疏化、剪枝联合优化）。
2. **WDR-Loss 的分布正则思路**：偏度+峰度双正则使权重分布趋近均匀，这一分布整形策略可迁移至权重稀疏化、低秩分解等对权重分布敏感的任务中。
3. **GPG 渐进蒸馏的分组策略**：按三明治规则将子网分为高/中/低三组进行渐进知识传递，比 MultiQuant 的全局软标签更高效，可用于其他超网训练中的负梯度抑制场景。
4. **CQAP 的条件化精度预测**：将结构化先验（粒度、对称性）作为条件输入 MLP 预测精度，比纯位宽预测更贴近实际部署需求，可为混合精度 NAS 提供精度评估新思路。
5. **learnable per-tensor 参数设计**：从 per-channel 参数启发式推导 per-tensor 精度不如独立学习，这一发现提示在超网设计中应优先为低粒度形式分配独立可学习参数，而非强行共享。

## 关键术语表
**EQ-Net（Elastic Quantization Neural Network）**：本文提出的一次性权重共享量化超网，支持弹性位宽/粒度/对称性切换。
**BGS-OFA（Bit-width, Granularity, Symmetry One-For-All）**：同时支持位宽、粒度、对称性三种弹性切换的一次性量化范式。
**WDR-Loss（Weight Distribution Regularization Loss）**：通过偏度和峰度正则约束共享权重分布形状，提升多形态量化的鲁棒性。
**GPG-Loss（Group Progressive Guidance Loss）**：基于三明治采样的分组渐进蒸馏损失，缓解超网中不同比特子网间的负梯度冲突。
**CQAP（Conditional Quantization-Aware Accuracy Predictor）**：以量化粒度和对称性为条件的混合精度精度预测器，用于加速遗传算法搜索。
**弹性量化空间**：由弹性位宽、弹性粒度和弹性对称性三部分构成的量化配置搜索空间。
**负梯度抑制（Negative Gradient Suppression）**：超网中不同量化配置子网因预测不一致而相互抵消梯度的现象。
**三明治规则（Sandwich Rule）**：在超网训练中同时保留最大、最小和随机深度子网的训练策略，以保持各尺度子网的性能。

## 可复现要素
- **数据集**：ImageNet（公开）、CIFAR-10（公开）
- **代码开源**：是，https://github.com/xuke225/EQ-Net
- **预训练权重**：论文未明确说明是否提供预训练超网权重，代码仓库中可能包含
- **关键超参**：
  - 训练 epoch：120
  - 优化器：Adam，base lr=0.001，cosine decay
  - CQAP 训练：SGD，lr=0.0004，weight decay=0.0001，100 epochs
  - 遗传算法：种群大小 100，迭代 500 代
  - 采样子网数：8000 个用于 CQAP 训练
  - 峰度目标值 $\mathcal{K}_T$：1.8
  - GPG 蒸馏权重 $\lambda$：论文未明确给出具体数值
