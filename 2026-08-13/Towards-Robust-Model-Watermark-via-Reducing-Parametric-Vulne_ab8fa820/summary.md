---
title: "Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gan_Towards_Robust_Model_Watermark_via_Reducing_Parametric_Vulnerability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:34:32"
field: "模型版权保护/水印鲁棒性"
keywords: ["model watermarking", "adversarial robustness", "backdoor", "copyright protection", "parameter space", "batch normalization"]
innovations: ["提出minimax对抗参数扰动框架在参数空间寻找并恢复水印已移除的邻域模型", "提出c-BN解决防御与攻击之间的domain shift问题提升水印鲁棒性", "揭示水印模型参数空间中的脆弱性分布规律"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet-subset"]
---

# 论文速读：Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne

## 一句话总结
论文针对基于backdoor的DNN模型水印易被移除攻击（如微调）破坏的脆弱性问题，通过在参数空间中寻找水印已移除的邻域模型并恢复其水印行为，结合clean-sample-based BatchNorm缓解domain shift，显著提升了模型水印对抗各类移除攻击的鲁棒性。

## 研究问题与动机
- **现有水印方法对抗移除攻击脆弱**：主流backdoor-based水印方法（Vanilla、EW、CW）在面对fine-tuning、fine-pruning、ANP、NAD、MCR、NNL等攻击时，水印成功率（WSR）显著下降，无法有效保护模型版权。
- **参数空间存在大量"水印已移除"邻域模型**：作者在参数空间中观察到，在原水印模型附近存在大量低WSR但高BA的模型，攻击者仅需微小参数偏移即可找到此类模型从而绕过水印检测。
- **防御与攻击之间存在domain shift**：vanilla方法在水印样本域上计算BatchNorm统计量，而攻击者使用clean数据移除水印，两者分布差异削弱了防御效果。

## 核心贡献（创新点）
- **揭示参数空间中的水印脆弱性分布规律**：首次系统分析水印模型在参数空间邻域内的性能变化，发现fine-tuning方向之外存在对抗方向可快速降低WSR，解释了现有方法为何易受移除攻击。
- **提出APP（Adversarial Parametric Perturbation）minimax框架**：通过maximization在参数空间中寻找最坏情况的水印已移除邻居（即最低WSR的邻域模型），再通过minimization恢复其水印行为，从根本上加固参数空间的脆弱性。
- **提出c-BN（Clean-sample-based BatchNorm）缓解domain shift**：在使用watermark样本进行前向传播时，改用clean batch的统计量进行BatchNorm，使防御与攻击在同一域上操作，显著提升鲁棒性。

## 方法详解
- **威胁模型**：黑盒验证场景，防御者仅能获取可疑模型的预测输出，攻击者可通过微调或参数修改移除水印。
- **APP核心minimax公式**：$\min_{\theta} [\mathcal{L}(\theta, \mathcal{D}_c) + \alpha \cdot \max_{\delta \in \mathcal{B}} \mathcal{L}(\theta + \delta, \mathcal{D}_w)]$，其中$\mathcal{B} = \{\delta \mid \|\delta\|_2 \leq \epsilon \|\theta\|_2\}$为相对扰动预算。
- **参数扰动计算**：采用单步近似$\delta = \epsilon \|\theta\|_2 \cdot \frac{\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)}{\|\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)\|_2}$，通过梯度方向搜索最坏情况邻域。
- **c-BN设计**：在forward过程中，对watermark batch使用clean batch $B_c$的running mean和variance进行归一化，而clean batch保持原有BatchNorm不变，有效缩小域间差异。
- **整体训练流程**：每个训练步中，先用clean batch计算基础梯度，再用c-BN估计的统计量计算APP扰动，最后结合watermark数据的梯度更新参数。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100、ImageNet-subset（100类，训练5万张，测试5千张）。
- **评估基线**：Vanilla水印、EW（指数加权）、CW（认证水印）。
- **移除攻击**：FT（微调）、FP（细剪枝）、ANP（对抗神经元扰动）、NAD（神经注意力蒸馏）、MCR（模式连通性修复）、NNL（神经网络清洗）。
- **核心指标**：WSR（水印成功率）和BA（良性准确率）。
- **主要结果（CIFAR-10，Content水印）**：Ours初始WSR达99.87%，受6种攻击后AvgDrop仅↓10.10%，而Vanilla↓59.15%、EW↓51.20%、CW↓68.36%；最强攻击NNL下Ours仍保留68.58% WSR。
- **主要结果（CIFAR-10，Unrelated水印）**：Ours AvgDrop仅↓6.20%，其他方法最低↓50.90%。
- **主要结果（ImageNet-subset，Unrelated水印）**：Ours AvgDrop仅↓7.42%，在93.98%-99.99%的WSR范围内显著优于基线。
- **最强提升**：在Unrelated水印类型下，Ours相比次优方法提升约45%的平均WSR保持率。

## 相关工作脉络
- **Adi et al. [1]**：首次将backdoor用于DNN水印，定义"TEST"内容水印范式，是本文Vanilla基线的核心参考。
- **Namba & Sakuma [37]**：提出EW方法，通过对权重指数加权降低水印对参数扰动的敏感度，本文在其基础上进一步考虑参数空间全局脆弱性。
- **Bansal et al. [3]**：提出CW方法，基于randomized smoothing提供认证水印保证，但需白盒访问；本文聚焦黑盒验证场景，与之公平对比。
- **Zhao et al. [56]**：提出MCR方法，通过修复mode connectivity路径移除水印；本文实验表明MCR是最强攻击之一，但仍无法突破Ours防御。
- **Uchida et al. [45]**：早期模型水印工作，通过嵌入特定权重实现版权保护，是vanilla方法的理论基础。
- **Lukas et al. [35]**：使用adversarial examples作为水印样本，本文扩展分析三种水印类型（Content、Noise、Unrelated）的鲁棒性差异。

## 局限性与未来方向
- **威胁模型简化**：当前方法限制参数扰动大小，而实际攻击只需保持BA不显著下降，约束更宽松，本文防御是更强威胁模型下的必要前提。
- **BA与参数关系难以显式建模**：由于无法精确描述BA与参数间的映射关系，难以直接优化BA约束下的鲁棒性，限制了方法的理论完备性。
- **扩展性待验证**：实验主要聚焦图像分类模型，对生成模型（如GAN、Diffusion）或其他模态的扩展需进一步探索。

## 研究启发与可借鉴点
- **minimax对抗训练思想可迁移**：将minimax框架应用于其他模型保护场景（如联邦学习隐私保护、模型共享授权），有望构建更通用的鲁棒性保障机制。
- **c-BN的domain shift缓解策略对backdoor防御有价值**：在backdoor触发器检测或净化任务中，类似的域对齐策略可提升防御模型在不同数据分布下的稳定性。
- **参数空间邻域分析提供可复用的评估范式**：通过可视化WSR/BA在参数空间的分布，可快速诊断任意水印方法的安全边界，为后续研究提供评估工具。
- **超参ε的选取具有攻击无关性**：实验表明最优ε范围与具体攻击类型无关，便于实际应用中的调参决策。

## 关键术语表
- **WSR（Watermark Success Rate）**：水印样本被模型预测为目标标签的比例，衡量水印有效性。
- **BA（Benign Accuracy）**：模型在干净数据上的分类准确率，衡量模型实用性能。
- **APP（Adversarial Parametric Perturbation）**：对抗性参数扰动，通过最大化watermark loss在参数空间寻找最坏情况的邻域模型。
- **c-BN（Clean-sample-based BatchNorm）**：使用clean batch统计量对watermark样本进行BatchNorm，缓解防御与攻击间的domain shift。
- **MCR（Mode Connectivity Repair）**：通过修复loss landscape中的mode connectivity路径来移除水印的攻击方法。
- **NNL（Neural Network Laundering）**：系统性地通过微调、剪枝等操作清洗backdoor行为的攻击方法。

## 可复现要素
- **代码**：已开源，地址为 https://github.com/GuanhaoGan/robust-model-watermarking
- **数据集**：CIFAR-10、CIFAR-100、ImageNet-subset（100类）
- **模型架构**：ResNet-18（主要实验）、MobileNetV2、VGG16、ResNet-50
- **关键超参数**：α=0.01（watermark loss系数）、ε=0.02（CIFAR）、ε=0.01（ImageNet）、初始学习率0.1、weight decay 5×10⁻⁴、训练100 epoch
- **攻击设置**：FT使用30 epoch、初始学习率0.05、momentum 0.9
