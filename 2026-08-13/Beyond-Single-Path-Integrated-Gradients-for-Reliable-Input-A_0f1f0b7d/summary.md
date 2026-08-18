---
title: "Beyond-Single-Path-Integrated-Gradients-for-Reliable-Input-A"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Jeon_Beyond_Single_Path_Integrated_Gradients_for_Reliable_Input_Attribution_via_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:28:39"
field: "可解释人工智能/输入归因"
keywords: ["input attribution", "integrated gradients", "explainable AI", "Stick-Breaking Process", "model interpretability", "path-based attribution", "gradient integration"]
innovations: ["将归因从单路径积分扩展为路径分布的期望，通过 SBP 随机路径采样降低归因方差", "提出 SPI-P 不确定性感知可视化方法，输出归因显著性的概率热力图"]
benchmarks: ["ImageNet Pixel Insertion/Deletion Game", "CIFAR-10 ROAR"]
---

# 论文速读：Beyond-Single-Path-Integrated-Gradients-for-Reliable-Input-Attribution-via-Randomized-Path-Sampling

## 一句话总结
本文针对 Integrated Gradients（IG）等路径归因方法在单一路径积分下产生噪声、不稳定归因的问题，提出基于 Stick-Breaking Process（SBP）的随机路径采样方法 Stick-breaking Path Integration（SPI），通过对多路径归因分布求期望来获得更可靠、目标对齐的输入归因热力图。

## 研究问题与动机
- **单一路径归因的高方差问题**：DNN（含 ReLU 等分段线性激活）具有大量线性区域，沿不同路径积分时聚合的区域权重差异大，导致单次路径归因方差高、不可靠。
- **现有基线方法的局限**：Grad*Input、GuidedIG、SmoothGrad 等虽通过自适应路径或噪声平均降低噪声，但仍未从根本上考虑"路径分布期望"这一视角。
- **路径分布难以显式定义**：高维输入空间中路径分布 $P(\gamma)$ 无法直接建模，需设计一种能高效采样多样化路径的随机过程。
- **归因可靠性缺乏统一定量评估标准**：归因无 ground truth，评测依赖 Pixel Flip、ROAR 等间接指标，亟需更稳健的归因来对齐模型真实行为。

## 核心贡献（创新点）
1. **提出从单一路径积分到路径分布期望的范式转换**：将归因可靠性问题建模为对路径分布 $P(\gamma)$ 求期望 $\mathbb{E}_\gamma[\phi_i(\gamma)]$，与 SmoothGrad 对输入扰动平均、GuidedIG 使用自适应路径有本质区别。
2. **设计基于 Stick-Breaking Process（SBP）的随机路径采样器**：将单位 Stick 按 Beta 分布随机折断得到的分段 PMF 转换为 CDF，进而生成每条特征维度独立的集成路径；与 GuidedIG 的梯度引导路径相比，本方法不依赖梯度信号，通过超参 $\alpha$ 控制路径多样性。
3. **提出 SPI 与 SPI-P 两种归因变体**：SPI 直接取多路径归因均值作为最终得分；SPI-P 进一步假设各特征归因服从高斯分布，输出归因值超过阈值 $\Theta$ 的概率，提供不确定性感知的热力图可视化。
4. **系统性的定性与定量验证**：在 ImageNet（VGG-16/Inception-v3/RN-18）和 CIFAR-10（ROAR）上全面评测，SPI 在所有指标上达到最优或接近最优。

## 方法详解
- **基础框架**：给定可微模型 $f$、输入 $\mathbf{x}$ 与基线 $\bar{\mathbf{x}}$，路径方法沿从基线到输入的连续路径 $\gamma(t)$（$t \in [0,1]$）积分梯度，满足 $\gamma(0)=\bar{\mathbf{x}}$、$\gamma(1)=\mathbf{x}$ 且单调前进（$\frac{d\gamma_i}{dt} \ge 0$）。
- **路径多样性与归因方差**：DNN 在 ReLU 下为分段线性函数，每个线性区域内梯度为常数；不同路径穿过不同区域组合，导致归因结果方差大。
- **Stick-Breaking Process（SBP）采样**：
  - 每个特征维度独立采样 $G_i \sim SBP(U(0,1), \alpha)$，其中 $\beta_k \sim Beta(1,\alpha)$，$\pi_k = \beta_k \prod_{i=1}^{k-1}(1-\beta_i)$，折段位置 $t_k \sim U(0,1)$。
  - 浓度参数 $\alpha$：$\alpha$ 越大， realized PMF 越接近均匀分布（趋向 IG 直线型路径）；$\alpha$ 越小，路径越曲折、多样性越高。
  - 由 PMF $G_i$ 构造 CDF $F_{G_i}(t)$，得到第 $i$ 维路径：$\gamma_i(t) = \bar{x}_i + F_{G_i}(t)(x_i - \bar{x}_i)$。
- **SPI 计算**：采样 $N$ 条路径，对每条路径计算 IG 式梯度积分后取均值：
  $$SPI_i(\mathbf{x};\alpha) = \frac{1}{N}\sum_{k=1}^N \int_0^1 \frac{\partial f(\gamma^{(k)}(t))}{\partial \gamma_i^{(k)}(t)} \cdot \frac{d\gamma_i^{(k)}(t)}{dt} dt$$
- **SPI-P 可视化**：假设每条路径归因 $\phi_i(\gamma^{(k)}) \sim N(\mu,\sigma)$，取 SPI 得分 top 5% 分位为阈值 $\Theta$，则 $SPI\text{-}P_i = P(\phi_i > \Theta)$，用高斯 CDF 计算概率热力图。

## 实验与结果
- **数据集与模型**：ImageNet 验证集（50k 样本），使用 VGG-16、Inception-v3、ResNet-18 预训练模型；CIFAR-10 训练集（50k）+ 测试集（10k），使用 ResNet-18。
- **基线方法**：Grad*Input、Guided Backpropagation、IG、FullGrad、GuidedIG。
- **Pixel Insertion/Deletion Game（ImageNet）**：
  - **Insertion**（越高越好）：SPI 在 VGG-16 上 **0.443**（↑0.028 vs FullGrad）、Inception-v3 上 **0.704**（↑0.146 vs FullGrad）、RN-18 上 **0.515**（↑0.067 vs FullGrad），三项均为最优。
  - **Deletion**（越低越好）：SPI 在 VGG-16 上 **0.019**（↓0.091 vs GuidedIG）、Inception-v3 上 **0.051**（↓0.010 vs FullGrad）、RN-18 上 **0.018**（↓0.002 vs GuidedIG），三项均为最优。
- **ROAR（CIFAR-10, ResNet-18）**：移除 top-k% 重要性像素后重训练，测试准确率越低说明归因越准确捕捉关键特征。SPI 在移除 10% 时准确率仅 **31.00±0.48%**（IG 为 39.70±0.79%，降幅最大）；移除 70% 时 SPI 为 **13.09±0.72%**（次低，FullGrad 14.82%）。
- **定性分析**：SPI 热力图更集中于目标物体（如项链、动物主体），背景噪声显著减少；SPI-P 可更好区分重要特征与无关背景。
- **强结果摘要**：SPI 在所有 ImageNet 插入/删除指标上**全面最优**，ROAR 在关键低移除比例（10%）上表现最优。

## 相关工作脉络
- **Integrated Gradients（IG）** [28]：本文基线方法，沿单一直线型路径积分梯度；SPI 扩展为对路径分布求期望，解决单路径方差问题。
- **GuidedIG** [9]：通过梯度符号自适应选择路径以减少噪声；区别在于 GuidedIG 仍为单路径，SPI 通过多路径平均消除路径依赖。
- **SmoothGrad** [24]：对输入加随机噪声后多次积分取平均；区别在于 SmoothGrad 扰动输入空间，SPI 保持输入不变而扰动路径形状。
- **XRAI** [8]：通过多尺度 superpixel 平均降低梯度噪声；区别在于 XRAI 是空间平滑，SPI 是路径空间期望。
- **RISE** [16]：对随机 mask 输入收集输出变化来估计归因；属于采样类方法，与 SPI 思路不同但同为统计聚合思想。
- **ROAR 评测框架** [7]：本文 ROAR 实验的直接基线来源，通过重训练测归因有效性，成为可解释性领域标准 benchmark 之一。

## 局限性与未来方向
- **超参 $\alpha$ 的选择依赖经验**：$\alpha$ 控制路径多样性，过小导致背景棋盘格噪声，过大趋近 IG 失去意义；文中未给出自动调参策略。
- **基线选择固定为全零（或 blur）**：未讨论不同 baseline（如黑图、均值图、噪声图）与 SPI 的交互影响。
- **每维度独立 SBP 假设**：各特征维度独立采样忽略了特征间相关性，可能在高维空间产生不合理解释路径。
- **计算开销增加**：需要采样 $N$ 条路径并分别计算梯度积分，相比单次 IG 推理成本显著提高。
- **未来方向**：作者提出可探索非均匀基分布（如高斯分布）、多模态路径分布、随机化 baseline 等方向。

## 研究启发与可借鉴点
- **"期望替代单点"的思想迁移**：将单一确定性计算扩展为对分布求期望以降低方差，此思路可迁移至其他归因方法（如 LRP、LRP-ε 等）及任何依赖路径/采样的可解释性方法中。
- **SBP 路径采样的通用性**：Stick-Breaking 过程只需每维独立生成 CDF 即可构造有效路径，该采样框架可适配不同任务（如 NLP 序列归因、时序数据归因），无需修改核心梯度积分逻辑。
- **SPI-P 的不确定性量化思路**：通过多次采样估计归因分布的均值与方差，并转化为概率热力图，为归因的可信度评估提供了轻量级方案，可与其他不确定性估计方法结合。
- **与团队方向的结合机会**：本文针对视觉输入归因，其 SBP 路径采样和分布期望框架可直接迁移至医学影像归因、时序信号归因等团队相关方向；SPI-P 的概率输出也可作为下游可信度过滤模块。

## 关键术语表
- **Integrated Gradients（IG）**：基于 Aumann-Shapley 值的归因方法，沿基线到输入的直线路径积分梯度，满足完整性与敏感性公理。
- **Stick-Breaking Process（SBP）**：Dirichlet 过程的构造性定义，通过连续从剩余 Stick 上折断一段（Beta 分布）生成随机概率质量分布。
- **SPI（Stick-breaking Path Integration）**：本文提出的归因方法，对 SBP 采样的多条路径的梯度积分结果取期望作为最终归因。
- **SPI-P（SPI Probability）**：SPI 的可视化变体，假设各路径归因服从高斯分布，输出归因超过阈值的概率热力图。
- **Pixel Insertion/Deletion Game**：按归因重要性从高到低依次插入/删除像素，观察模型输出变化，用于定量评测归因质量。
- **ROAR（RemOve-And-Retrain）**：移除归因重要性 top-k% 像素后重训练模型，测试准确率下降幅度越大说明归因越准确。
- **Concentration Parameter（α）**：SBP 中控制路径多样性的超参，越大路径越趋向均匀（直线），越小路径越曲折多样。
- **Piece-wise Linear Region**：ReLU DNN 在输入空间划分的线性区域，不同区域内梯度为常数，是单路径归因方差高的根本原因。

## 可复现要素
- **数据集**：ImageNet（验证集 50k，公开）、CIFAR-10（训练集 50k + 测试集 10k，公开）；代码与权重**论文未提及**开源状态。
- **关键超参**：
  - $\alpha$（浓度参数）：控制 SBP 路径多样性，文中建议低值（如 $\alpha=0.01$）表现更优，高值（$\alpha=10$）趋近 IG。
  - 采样路径数 $N$：论文未明确给出具体数值，需从实验细节推断（通常 50~200）。
  - 阈值 $\Theta$：取 SPI 得分 top 5% 分位值。
- **模型**：VGG-16、Inception-v3、ResNet-18（均为公开预训练权重）；CIFAR-10 训练使用 Adam（lr=3e-4，100 epochs）。
- **基线实现**：Grad*Input、GuidedBProp、IG、FullGrad、GuidedIG（均来自公开代码库或官方实现）。
