---
title: "Tracing-the-Origin-of-Adversarial-Attack-for-Forensic-Invest"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Tracing_the_Origin_of_Adversarial_Attack_for_Forensic_Investigation_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:34:22"
field: "对抗攻击与模型安全"
keywords: ["adversarial attack", "forensic investigation", "black-box attack", "model tracing", "buyers-seller setting"]
innovations: ["提出并行网络结构在主分类器中嵌入独特tracer以实现对抗样本溯源", "设计VAE-based tracer训练方法与排列置换差异化策略", "基于tracer logits差异(DOL)的无白盒溯源机制"]
benchmarks: ["CIFAR10-ResNet18", "GTSRB-ResNet18", "Mini-ImageNet-ResNet50"]
---

# 论文速读：Tracing-the-Origin-of-Adversarial-Attack-for-Forensic-Invest

## 一句话总结
本文针对对抗攻击溯源问题，在买家-卖家场景中提出两阶段框架：通过并行网络结构为每个分布式模型注入独特tracer，使其生成的对抗样本携带可追踪特征，最终基于tracer输出logits差异实现源模型识别。

## 研究问题与动机
1. **核心问题**：在黑盒攻击设置下，如何从已生成的对抗样中追溯其来源于哪一模型副本，服务于取证调查与威慑。
2. **现有防御不足**：现有对抗防御方法仅能缓解攻击，无法追溯攻击来源；即使防御失败，也无事后追责能力。
3. **买家-卖家攻击场景**：多个买家分别持有相同任务的模型副本，恶意买家利用己方副本生成对抗样本攻击其他买家，取证者需识别源模型。
4. **技术挑战**：如何在保持分类精度前提下，使不同副本决策边界显著差异化，且使黑盒攻击过程自然引入可追踪的唯一特征。

## 核心贡献（创新点）
1. **提出对抗攻击溯源的新防御视角**：区别于传统防御思路，从被动抵御转向主动追溯攻击来源，为取证调查提供技术手段。
2. **设计并行网络结构实现模型分离**：将唯一tracer与主分类器并联，通过加权组合使不同副本获得差异化决策边界。
3. **提出VAE-based tracer训练方法**：利用捕捉损失（L_Tr、ap）确保tracer易被攻击且与主分类器类别不重叠，结合排列置换生成多个差异化tracer。
4. **设计logits差异（DOL）溯源机制**：基于攻击标签与真实标签对应的tracer输出logits差值最大化原则识别源模型，无需白盒访问。

## 方法详解
**框架概览**：两阶段流程——模型分离（Model Separation）与溯源追踪（Origin Tracing）。

**1) 模型分离**
- **并行结构**：$\mathcal{M}_i(x) = \mathcal{C}(x) + \alpha \times \mathcal{T}_i(x)$，其中$\mathcal{C}$为共享主分类器（训练一次），$\mathcal{T}_i$为第i个独特tracer，$\alpha$控制tracer参与权重。
- **$\mathcal{T}_0$生成**：基于VAE架构，编码器即$\mathcal{T}_0$，解码器用于重建输入。训练损失为：
  $\mathcal{L} = \lambda\mathcal{L}_{VAE} + \mathcal{L}_{Trap}$，其中$\mathcal{L}_{VAE} = \|\mathcal{V}(X)-X\|_2 + KL(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,1))$，$\mathcal{L}_{Trap} = \frac{\mathcal{T}_0(x)\otimes\mathcal{T}_0(x_\xi)}{\|\mathcal{T}_0(x)\|_2\|\mathcal{T}_0(x_\xi)\|_2} - \cos(\theta)$，通过控制输出与加噪输出之间的余弦相似度，使tracer对微小扰动敏感。
- **$\mathcal{T}_i$分离**：对$\mathcal{T}_0$的输出类别进行排列置换：$\mathcal{T}_i(x) = \pi_i(\mathcal{T}_0(x))$，要求任意两个排列重叠元素数不超过u（实验取u=1）。

**2) 溯源追踪**
- 输入对抗样本$x_{adv}$，对所有候选模型$i\in[1,m]$计算：
  - $\mathcal{M}_i(x_{adv})$的logits，取攻击标签att和正确标签cln对应位置。
  - $\mathcal{T}_i(x_{adv})[att]$与$\mathcal{T}_i(x_{adv})[cln]$的差值（DOL）。
- 源模型判定：$s = \arg\max_i(\mathcal{T}_i(x_{adv})[att] - \mathcal{T}_i(x_{adv})[cln])$，即DOL最大的tracer对应的模型为源。

## 实验与结果
- **数据集与架构**：CIFAR10（10类）、GTSRB（43类）、Mini-ImageNet（100类）；ResNet18/VGG16/ResNet50/VGG19。
- **攻击基线**：Boundary、HSJA、QEBA、SurFree四种黑盒攻击。
- **最佳结果**：ResNet18-CIFAR10任务，5个分布式模型，QEBA攻击+α=0.15时，追溯准确率达**98.0%**；ResNet18-CIFAR10用Boundary攻击+α=0.15也达98.0%。
- **提升幅度**：与α=0（基线）相比，引入tracer后在仅降低分类精度不足1%的前提下实现>94%追溯率；相较前作买家-卖家分散模型方法（arXiv:2111.15160），本文从可迁移性更强的对抗样本中仍能高精度溯源。
- **分布验证**：Kolmogorov-Smirnov检验显示源tracer DOL分布与受害者tracer DOL分布差异显著（KD≈0.8-0.9），而不同受害者之间差异极小（KD≈0.04-0.09）。
- **可扩展性**：即使增至10个副本，ResNet-CIFAR10仍保持>96%追溯准确率。
- **协同攻击应对**：面对双模型协同攻击，追溯率从98%降至50%，但引入梯度正交自适应防御后可恢复至97.5%。

## 相关工作脉络
1. **黑盒对抗攻击**：Boundary[3]、HSJA[5]、QEBA[17]、SurFree[18]等决策边界攻击方法，本文以此验证不同攻击机制下的溯源鲁棒性。
2. **神经网络水印**：Boenisch[2]综述中模型溯源技术，本文追溯对象为对抗样本生成源而非模型拷贝，比水印更强。
3. **买家-卖家分散模型**：Zhang et al.[28]（arXiv:2111.15160）提出分发不同副本以削弱攻击可迁移性，本文在此基础上增加源追溯能力，形成完整防御闭环。
4. **VAE与对抗鲁棒性**：VAE结构常用于生成与表征学习，本文创新性将其编码器作为易受攻击的tracer组件。
5. **Logits差异分析**：与前作依赖硬标签或概率输出不同，本文充分利用模型中间logits信息，提升溯源精度。

## 局限性与未来方向
1. **协同攻击削弱效果**：当攻击者可访问多个模型进行联合攻击时，追溯准确率显著下降（98%→50%）。
2. **排列置换副本数受限**：当类别数K较小时，满足重叠约束的排列数有限（最多K×(K−1)个），限制了可分发模型数量。
3. **参数θ敏感**：θ控制tracer对噪声敏感度，过大过小均影响追溯性能，需针对具体任务调优。
4. **自适应防御仍需验证**：梯度正交化使各$\mathcal{C}_i$边界分化以应对协同攻击，但计算开销和泛化性待进一步研究。
5. **仅考虑黑盒单副本攻击假设**：实际场景中攻击者可能通过更多查询或白盒知识绕过tracer陷阱。

## 研究启发与可借鉴点
1. **并行网络结构范式**：主任务模型+辅助tracer的解耦设计，可在其他安全场景（如数据泄露溯源、模型窃取检测）中复用。
2. **VAE捕捉损失设计思路**：通过余弦相似度约束使模型对微小扰动敏感，可迁移至设计脆弱但功能独立的子模块。
3. **logits差异溯源方法**：利用模型中间层输出分布差异而非仅依赖最终预测，为取证分析提供更丰富的信号源。
4. **权衡参数α的实用性**：实验系统分析了α对分类精度与追溯率的 trade-off，为实际部署提供参数选择依据。
5. **分布检验验证方法**：使用Kolmogorov-Smirnov检验量化源/非源分布差异，可作为溯源模型有效性的通用验证手段。

## 关键术语表
**Black-box adversarial attack**：攻击者仅能通过查询获取模型硬标签输出，无法获得内部梯度信息。
**Buyer-seller setting**：模型卖家向多个买家分发功能相同但略有差异的模型副本的应用场景。
**Parallel network structure**：主分类器与tracer并联的结构，最终输出为两者加权和。
**Variational Autoencoder (VAE)**：基于变分推断的生成模型，本文利用其编码器作为tracer。
**Trap loss (L_Tr、ap)**：约束tracer输出与加噪输入输出之间余弦相似度，使tracer易被攻击的特性损失。
**Permutation-based separation**：通过对tracer输出类别进行排列置换来生成多个差异化tracer。
**Difference of Logits (DOL)**：攻击标签与正确标签对应位置的tracer输出logits差值，用于溯源判断。
**Non-transferable vs transferable**：非可迁移样本仅对源模型有效，可迁移样本可跨模型生效，后者溯源更有意义。

## 可复现要素
- **数据集**：CIFAR10、GTSRB、Mini-ImageNet（均为公开数据集）。
- **代码开源情况**：论文提到Boundary和HSJA使用ART库，QEBA和SurFree从GitHub获取实现，但未声明本文方法代码是否开源。
- **关键超参**：$\alpha\in\{0.05, 0.1, 0.15\}$，$\theta=75°$，$\lambda=0.001$，u=1，训练epoch=400（tracer）/200（主分类器）。
- **优化器**：Adam，学习率1e-4。
- **硬件**：NVIDIA RTX 2080ti，PyTorch实现。
