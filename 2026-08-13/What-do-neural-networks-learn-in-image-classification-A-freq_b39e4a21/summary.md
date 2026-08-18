---
title: "What-do-neural-networks-learn-in-image-classification-A-freq"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_What_do_neural_networks_learn_in_image_classification_A_frequency_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:41:12"
field: "计算机视觉可解释性与泛化"
keywords: ["频率捷径", "简单性偏差", "图像分类", "OOD泛化", "频域分析", "纹理偏见", "数据增强"]
innovations: ["提出DFM频率捷径识别方法并扩展至纹理/形状/颜色三种类型", "从频域角度系统解释分类任务中网络学习动态的数据依赖性", "揭示频率捷径的跨架构可迁移性及对OOD泛化的误导性影响"]
benchmarks: ["ImageNet-10", "ImageNet-SCT"]
---

# 论文速读：What-do-neural-networks-learn-in-image-classification-A-freq

## 一句话总结
本文从频率视角实证研究了深度神经网络在图像分类任务中的学习动态，发现网络倾向于利用数据中独特的频率特征（即"频率捷径"）以简化学习过程，这些捷径可以是基于纹理或形状的，且无法通过增大模型容量或常见数据增强完全避免。

## 研究问题与动机
- **现有研究的空白**：此前关于神经网络频率偏好（spectral bias）的研究主要集中于回归任务（发现网络优先学习低频分量），但对分类任务的频率学习动态缺乏系统分析。
- **简单性偏差的驱动机制**：神经网络的简单性偏差（simplicity bias）使其倾向找到最简单的解决方案，这可能诱导网络依赖数据中的虚假相关性而非语义信息。
- **捷径学习对泛化的影响不明确**：虽然已知捷径学习会损害分布外（OOD）泛化，但频率捷径的具体形式（纹理/形状/颜色）及其可迁移性尚未被系统揭示。
- **现有频率捷径识别方法的局限**：前作（如[31]）的频率捷径识别算法受频率去除顺序影响较大，且仅观察到基于纹理的捷径。

## 核心贡献（创新点）
- **扩展了分类任务的频率偏差认知**：与前作发现回归任务偏向低频不同，本文证明分类任务中网络可优先学习低频或高频特征，取决于数据的频率特性，揭示了数据驱动的灵活性。
- **提出了系统性的频率捷径识别方法**：设计了基于频域削减（frequency culling）的 Dominant Frequency Map (DFM) 方法，通过记录去除特定频率后损失值的变化来评估频率重要性，避免了对频率去除顺序的依赖。
- **扩展了频率捷径的类型认知**：首次系统展示了频率捷径可基于纹理、形状或颜色，取决于何种特征最能简化目标函数，并提出了 ADCS 度量来量化类别间的频率差异。
- **揭示了频率捷径的跨数据集可迁移性**：通过 ViT-B 在 ImageNet-SCT 上的测试，证明基于 ResNet18+SIN 学到的频率捷径可迁移至不同架构，且大模型与数据增强（AutoAugment、AugMix、SIN）无法完全规避。
- **构建了 ImageNet-SCT 评估基准**：设计了包含7种图像风格的OOD测试集，系统验证了频率捷径如何导致"泛化提升的假象"（present shortcuts）或"泛化下降"（absent shortcuts）。

## 方法详解
**ADCS 度量（Accumulative Difference of Class-wise average Spectrum）**：
计算每个类别在频域 $(u,v)$ 的平均幅度谱与其他类别的差异累积符号和：$ADCS^{c_i}(u,v) = \sum_{c_j \neq c_i} sign(E_{c_i}(u,v) - E_{c_j}(u,v))$，其中 $E_{c_i}(u,v) = \frac{1}{|X^i|}\sum_{x \in X^i}|\mathcal{F}_x(u,v)|$。该值范围从 $1-|C|$ 到 $|C|-1$，正值表示该类在该频率具有更多能量。

**频率捷径识别方法（DFM构建）**：
对每个类别 $c_i$，计算其样本在频域 $(u,v)$ 的平均幅度谱，然后逐一去除每个频率分量并记录测试损失增量 $\Delta L_{(u,v)}^{c_i}$。按增量降序排列所有频率，选取 top-X% 构成该类的 DFM。通过比较原始测试集与 DFM-filtered 测试集的 TPR（真阳性率）和 FPR（假阳性率）来量化捷径的判别力与特异性。

**Band-stop 实验设计（合成数据）**：
将傅里叶频谱均分为四个频带 $B_1$（最低频）至 $B_4$（最高频），生成具有不同频率偏置的合成数据集 $\text{Syn}_b$。测试时移除两个频带，通过相对混淆矩阵 $\Delta^{C_i,C_j} = (Pred_{bs}^{C_i,C_j} - Pred_{org}^{C_i,C_j})/N_c \times 100$ 分析性能变化。

## 实验与结果
- **合成数据实验**：四个 $\text{Syn}_b$ 数据集上，ResNet18 始终优先学习类别 $C_3$（即使其仅含高频成分），且对所有数据集，$C_2$ 类在 band-stop 测试下仍保持良好性能，证实频率捷径的存在。
- **ImageNet-10 ID 测试**：ResNet18 对 'zebra'（TPR=0.96, FPR=0.0222）和 'container ship'（TPR=0.94, FPR=0.0133）存在显著频率捷径；使用 SIN 增强后，'siamese cat' 成为新的捷径目标（TPR=0.88, FPR=0.2444）。
- **模型容量影响**：ResNet50 对 'zebra' 的 TPR 降低至 0.78，但对 'airliner' 和 'container ship' 产生新的捷径；VGG16 对 'container ship'（TPR=0.7, FPR=0.42）存在强捷径。
- **跨架构可迁移性**：ViT-B 使用 ResNet18+SIN 的 DFM 过滤后，对 'siamese cat'（TPR=0.82, FPR=0.22）和 'container ship'（TPR=0.92, FPR=0.25）仍表现出频率捷径。
- **数据增强效果**：AugMix 恶化了 'container ship' 的捷径但缓解了 'zebra'；AutoAugment 部分缓解两者；SIN 引入 'siamese cat' 捷径。
- **OOD 测试（ImageNet-SCT）**：所有模型平均 TPR 显著下降；依赖 'zebra' 纹理捷径的模型在 'horse' 上性能差（TPR<0.15）；而 'fishing vessel'（纹理捷径存在）表现优于平均；'fire truck'（无捷径）展示了真实泛化能力。

## 相关工作脉络
- **Spectral bias in regression [23]**：证明 NN 优先学习低频分量，本文将其扩展至分类任务并发现数据偏置决定学习方向（低频或高频）。
- **Texture bias [12]**：Geirhos 等人发现 CNN 偏向纹理特征，本文从频域角度解释其成因——数据中纹理对应的频率特征更易被学习为捷径。
- **Frequency shortcut definition [31]**：前作首次定义频率捷径但算法依赖频率去除顺序且仅观察到纹理型捷径，本文提出顺序无关的 DFM 方法并发现形状/颜色型捷径。
- **High-frequency bias in classification [1]**：指出分类中中高频的重要性由数据决定，本文通过 ADCS 和 DFM 提供了系统量化工具。
- **Background dependency [33]**：发现背景信息影响图像分类，本文从频域角度补充解释了此类捷径的频谱机制。

## 局限性与未来方向
- **实验规模限制**：主要在 ImageNet-10（小样本子集）上验证，全量 ImageNet 或其他大规模数据集的推广性有待考察。
- **合成数据的简化假设**：合成数据的频率分布和类别设计较为理想化，与真实数据复杂性存在差距。
- **DFM 阈值敏感性**：top-5% 作为默认阈值，未系统探索不同比例对捷径识别稳定性的影响。
- **缓解策略的不足**：本文主要揭示问题而非提供有效解决方案，自述需"focus on effective training schemes mitigating frequency shortcut learning"。
- **未来方向**：作者建议研究应聚焦于利用 DFM 诱导网络利用更多频率分量（而非捷径频率）的训练策略，具体可参考其同期工作 DFM-X [30]。

## 研究启发与可借鉴点
- **DFM 识别框架可直接迁移**：基于损失增量排序的频域重要性评估方法适用于任意图像分类模型，可作为通用捷径检测工具。
- **ADCS 度量可用于数据诊断**：类别间频域差异分析可帮助理解数据本身的偏置特性，为数据集质量评估提供新视角。
- **图像风格增强（如SIN）的双刃剑效应**：改变数据偏置方向会转移而非消除捷径，提示在数据增强设计时需同时监测频率特征的变化。
- **OOD 基准设计的启示**：ImageNet-SCT 通过"对应类别+风格变换"的方式系统验证捷径影响，该范式可复用于其他泛化能力评估。
- **结合团队方向的创新机会**：可将 DFM 作为正则化约束（如 DFM-X [30]）引入模型训练，或在对比学习/自监督学习中显式惩罚捷径频率的过度依赖。

## 关键术语表
- **Frequency Shortcut（频率捷径）**：神经网络利用的特定频率子集，能简化分类目标但可能忽略语义信息，可导致基于纹理、形状或颜色的错误决策。
- **Simplicity Bias（简单性偏差）**：神经网络倾向于以"最简单"方式完成优化任务的固有偏好，是频率捷径学习的根本驱动力。
- **Dominant Frequency Map (DFM)**：按频率对分类贡献度排序后选取 top-X% 重要频率构成的频域掩码，用于识别和量化捷径。
- **ADCS（累积类别谱差异）**：衡量单个类别相对于其他类别在各频点能量优势的统计量，用于评估频率特征的区分度。
- **Band-stop 测试**：通过移除特定频带构造测试集，验证模型是否仅依赖部分频率即可分类，从而检测捷径。
- **ImageNet-SCT**：本文构建的 OOD 测试集，包含10个类别和7种图像风格（art、cartoon、sketch等），用于系统评估频率捷径对泛化的影响。

## 可复现要素
- **代码与数据**：已开源，地址 https://github.com/nis-research/nn-frequency-shortcuts
- **数据集**：ImageNet-10（公开子集）、ImageNet-SCT（本文构建，通过 GitHub 获取）
- **模型**：ResNet18、ResNet50、VGG16、ViT-B
- **数据增强**：AutoAugment、AugMix、SIN（Shape-INduced augmentation）
- **关键超参**：DFM 保留 top-5% 频率；合成数据中频带划分；训练迭代数（合成数据500步、自然图像1200步评估早期学习）
- **评估指标**：TPR（真阳性率）、FPR（假阳性率）、F₁-score
