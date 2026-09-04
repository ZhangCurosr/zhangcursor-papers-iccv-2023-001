---
title: "Beyond-Single-Path-Integrated-Gradients-for-Reliable-Input-A"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Jeon_Beyond_Single_Path_Integrated_Gradients_for_Reliable_Input_Attribution_via_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:20:22"
field: "可解释机器学习"
keywords: ["Input Attribution", "Integrated Gradients", "Explainable AI", "Stick-Breaking Process", "Path-based Explanation", "Neural Network Interpretability"]
innovations: ["提出基于Stick-Breaking Process的随机路径采样框架，将单路径归因扩展为路径分布期望", "提出SPI-P概率可视化方法，基于高斯假设计算特征贡献置信度", "系统性验证多路径期望归因在像素插入删除和ROAR指标上的优越性"]
benchmarks: ["ImageNet Classification", "CIFAR-10 Classification"]
---

# 论文速读：Beyond-Single-Path-Integrated-Gradients-for-Reliable-Input-Attribution

## 一句话总结
本文提出 **Stick-breaking Path Integration (SPI)**，通过 Stick-Breaking Process (SBP) 采样多条随机积分路径并对属性取期望，解决了单路径 Integrated Gradients (IG) 因路径选择不同而产生高方差、噪声大的问题，显著提升了输入归因的可靠性与物体对齐程度。

## 研究问题与动机
1. **单路径归因不稳定**：不同积分路径穿过不同的 piece-wise 线性区域，导致同一输入的单路径 IG 在不同路径选择下产生高方差结果。
2. **现有降噪方法局限**：SmoothGrad（加噪声到输入）、NoiseGrad（加噪声到权重）、XRAI（局部池化）等方法从其他角度降噪，但未系统性地建模路径分布。
3. **路径选择的主观性**：现有路径方法（如 Guided IG）依赖人工设计的路径，缺乏对"所有可能路径"的期望估计。
4. **解释可靠性需求**：在视觉任务中，归因 heatmap 需要与目标物体对齐，而非被无关背景像素干扰。

## 核心贡献（创新点）
1. **提出路径期望归因框架**：将单路径归因扩展为对所有可能积分路径的期望估计，本质区别在于将问题从"选一条好路径"转变为"建模路径分布并取期望"。
2. **基于 SBP 的随机路径采样机制**：借用 Stick-Breaking Process 生成多样化的概率质量函数（PMF），进而构建随机积分路径，本质区别在于这是一种无偏的、理论可控制的随机过程，相比均匀采样更具表达力。
3. **提出 SPI-P 概率可视化方法**：将每条路径的归因视为随机变量，基于高斯假设计算特征贡献超过阈值的概率，本质区别在于提供置信度信息而非仅有点估计。
4. **系统实验验证**：在 ImageNet 和 CIFAR-10 上通过像素插入/删除游戏和 ROAR 指标验证，证明 SPI 在三类架构（VGG-16、Inception-v3、ResNet-18）上均优于所有基线。

## 方法详解
1. **积分路径定义**：路径 γ: [0,1] → X，需满足（1）γ(0)=x̄（基线），γ(1)=x（输入）；（2）单调性 dγᵢ/dt = C(xᵢ - x̄ᵢ)，C ≥ 0。
2. **Piece-wise 线性分析**：ReLU 网络的 f(x) 为分片线性函数，IG 可表示为各线性区域权重的加权求和：φᵢ(γ) = Σⱼ [wᵢ⁽ʲ⁾ αᵢ⁽ʲ⁾(γ)] / Σⱼ' αᵢ⁽ʲ'⁾(γ)，其中 α 为路径在第 j 个区域经过的第 i 维投影长度。
3. **Stick-Breaking Process 采样**：
   - 定义 PMF：G(t) = Σₖ πₖ δₜₖ(t) ~ SBP(H, α)
   - 其中 πₖ = βₖ ∏ᵢ₌₁ᵏ⁻¹(1-βᵢ)，βₖ ~ Beta(1, α)，tₖ ~ H（基准分布，本文取 U(0,1)）
   - 超参数 α（浓度参数）控制路径方差：α 越大，路径越接近 IG 直线；α 越小，路径越分散。
4. **SPI 计算公式**：
   - 路径构造：γᵢ(t; αᵢ) = x̄ᵢ + F_Gᵢ(t)(xᵢ - x̄ᵢ)，其中 F_Gᵢ 为 Gᵢ 的 CDF
   - 最终归因：SPIᵢ(x; α) = E_G[(xᵢ - x̄ᵢ) ∫₀¹ (∂f/∂γᵢ) · (dF_Gᵢ/dt) dt]
5. **SPI-P 概率可视化**：假设 φᵢ(γ) ~ N(μ, σ)，阈值 Θ 取 SPI 值的 top 5% 分位数，SPI-Pᵢ = P(φᵢ > Θ)，用高斯 CDF 计算。

## 实验与结果
- **数据集与模型**：ImageNet 验证集（50k 图像）+ ResNet-18 在 CIFAR-10 训练集（50k）+ 测试集（10k）；预训练模型：VGG-16、Inception-v3、ResNet-18。
- **基线方法**：Grad*Input、GuidedBProp、IG、FullGrad、GuidedIG。
- **Pixel Insertion/Deletion（Table 1）**：
  - SPI 在所有三架构上均取得最优；以 Inception-v3 为例，Insertion=**0.704**（第二名为 FullGrad 的 0.558），Deletion=**0.051**（更低更好）。
- **ROAR（Table 2）**：
  - SPI 在移除 10% 重要像素时测试准确率降至 **19.70±0.40%**，显著低于 IG 的 26.23±1.11%，证明 SPI 更准确地定位训练相关特征。
- **定性结果（Figure 5）**：SPI 热力图更聚焦目标物体（如项链），背景噪声更少；SPI-P 进一步区分重要特征与无关背景。

## 相关工作脉络
1. **Integrated Gradients (IG) [28]**：本文基础，基于 Aumann-Shapley 值，沿单一路径积分梯度；本文扩展为多路径期望。
2. **Guided IG [9]**：自适应路径降噪，但路径仍为单条确定性路径；本文从分布角度建模路径不确定性。
3. **SmoothGrad [24]**：通过对输入加噪声再平均，属于输入扰动视角；本文从路径随机性切入，两者正交。
4. **ROAR 评估 [7]**：本文采用此因果评估指标验证归因与模型训练特征的相关性。
5. **Stick-Breaking Process [20]**：Dirichlet 过程的构造性定义，本文为首次将其引入路径采样领域。

## 局限性与未来方向
1. **超参数 α 的选择**：不同 α 值影响路径方差和结果，目前依赖实验调参，缺乏自动选择策略。
2. **计算开销**：需采样多条路径（N 次前向/反向传播），相比单路径 IG 开销增大 N 倍。
3. **仅验证了图像分类任务**：未扩展到目标检测、分割或 NLP 等其他视觉/语言任务。
4. **高斯分布假设**：SPI-P 假设属性服从高斯分布，实际分布可能非对称或多峰。
5. **未来方向**：使用非均匀基准分布（如高斯）、随机化基线 x̄、探索多模态路径分布。

## 研究启发与可借鉴点
1. **SBP 路径采样的通用性**：Stick-Breaking Process 可用于其他基于路径的方法（如 Path Integral 类解释方法），提供理论保证的随机采样框架。
2. **期望归因范式**：将"选最优路径"转为"对路径分布取期望"的思路可迁移到任何依赖路径选择的积分型算法。
3. **SPI-P 的概率解释**：用统计分布而非单点估计描述归因不确定性，为可解释性提供置信区间概念，值得在其他解释方法中推广。
4. **α 超参数的语义**： concentration parameter 控制探索-利用平衡，可在后续工作中探索自动学习 α 的策略。

## 关键术语表
- **Integrated Gradients (IG)**：沿基线到输入的直线路径积分梯度，满足完整性公理的路径归因方法。
- **Stick-Breaking Process (SBP)**：一种连续打破 stick 的随机过程，用于构造 Dirichlet 过程先验，本文将其转化为路径采样工具。
- **Integration Path**：从基线到输入输入的连续映射函数，决定梯度积分的轨迹。
- **Piece-wise Linear Region**：ReLU 网络中梯度为常数的线性区域，不同路径穿过不同区域导致归因差异。
- **Pixel Insertion/Deletion Game**：按归因重要性依次插入/删除像素，测量模型输出变化，评估归因因果性。
- **ROAR (RemOve-And-Retrain)**：移除归因重要像素后重新训练模型，测试准确率下降越大说明归因越准确。
- **Concentration Parameter (α)**：SBP 中超参数，控制路径方差，α→∞ 时退化为 IG 直线路径。
- **SPI-P**：基于归因高斯分布假设，计算特征贡献超过阈值的概率的可视化方法。

## 可复现要素
- **数据集**：ImageNet（公开）、CIFAR-10（公开）；实验使用 ImageNet 验证集 50k 图像，CIFAR-10 训练集 50k、测试集 10k。
- **代码/权重**：论文未提及开源；预训练模型 VGG-16、Inception-v3、ResNet-18 为常见公开模型。
- **关键超参**：浓度参数 α（低值降噪效果好，高值趋近 IG）、采样路径数 N、阈值分位数（top 5%）。
