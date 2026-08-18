---
title: "GameFormer-Game-theoretic-Modeling-and-Learning-of-Transform"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:08:29"
field: "自动驾驶交互预测与规划"
keywords: ["交互预测", "博弈论", "Transformer", "自动驾驶规划", "多智能体轨迹预测", "level-k推理"]
innovations: ["将level-k博弈论引入Transformer解码器实现层级交互建模", "提出基于斥力势场的交互损失约束多智能体安全交互"]
benchmarks: ["Waymo Open Motion Dataset交互预测", "nuPlan规划基准"]
---

# 论文速读：GameFormer: Game-theoretic-Modeling-and-Learning-of-Transformer-based-Interactive-Prediction-and-Planning-for-Autonomous-Driving

## 一句话总结
GameFormer将level-k博弈论引入Transformer解码器，设计了层级交互解码结构来迭代建模多智能体间的认知交互，实现了联合交互预测与自动驾驶规划；在Waymo交互预测任务上达到SOTA，并在nuPlan规划基准上展现领先性能。

## 研究问题与动机
1. **现有Transformer预测模型重编码、轻解码**：Scene-Transformer、WayFormer等主流方法集中于场景编码，解码阶段缺少对智能体未来交互的显式建模。
2. **条件预测模型仅实现单向交互**：M2i、Pip等条件预测模型以AV内部计划为条件预测其他智能体响应，忽略了AV与其他智能体间的动态双向互影响。
3. **预测结果对下游规划呈被动反应**：传统预测模块仅输出轨迹，规划模块被动接收，在汇入口、变道、无保护左转等关键场景中，AV需要主动协调交互行为。
4. **需要显式的多轮交互推理能力**：从博弈论视角看，现有模型相当于有限深度的Stackelberg博弈，缺乏对认知交互深度（level-k）的系统建模。

## 核心贡献（创新点）
1. **提出GameFormer框架**：首次将level-k博弈论系统性地应用于Transformer解码器设计，以层级解码实现多智能体认知交互的深度建模，区别于仅依赖编码端隐式建模的现有方法。
2. **设计了level-k交互解码器**：每层解码器以前一层预测轨迹为输入，通过自注意力+加权平均池化建模前一层次级未来交互，再与共享环境上下文做交叉注意力，实现交互过程的逐层细化。
3. **提出层级交互训练损失**：在模仿损失基础上引入基于斥力势场的交互损失（$\mathcal{L}_{Inter}$），约束当前层智能体轨迹避免与前一层其他智能体轨迹过近，本质区别在于将安全交互约束内嵌到训练目标而非后处理规则。
4. **端到端联合预测与规划验证**：不仅预测SOTA，且在WOMD开环/闭环规划和nuPlan基准上均超越DIPP等联合预测规划基线，体现了交互建模对规划能力的提升价值。

## 方法详解
**整体架构**：Transformer Encoder + 层级Decoder（Level-0到Level-K），共享编码的场景上下文供所有解码层使用。

**场景编码（Sec 3.2）**：
- 智能体历史：LSTM编码 $S_p \in \mathbb{R}^{N \times T_h \times d_s}$ → $A_p \in \mathbb{R}^{N \times D}$
- 矢量地图：MLP编码车道/人行横道线段 → max-pooling聚合 → $M_r \in \mathbb{R}^{N \times N_{mr} \times D}$
- 关系编码：拼接 $C^i = [A_p, M_p^i]$，经 $E$ 层Transformer Encoder → 场景上下文 $C_s \in \mathbb{R}^{N \times (N+N_{mr}) \times D}$

**Level-0解码（Sec 3.3）**：
- 可学习模态嵌入 $I \in \mathbb{R}^{N \times M \times D}$ 作为Query
- Cross-Attention：Query $=(C_{s,A_p}+I)$，Key/Value=$C_s$
- 两路MLP分别解码GMM参数 $(\mu_x, \mu_y, \log\sigma_x, \log\sigma_y)$ 和各模态分数

**Level-k交互解码（k≥1，Fig 3）**：
- 接收Level-(k-1)均值轨迹 $S_f^{L_{k-1}}$，MLP+max-pooling编码为 $A_{mf}^{L_{k-1}}$
- 按Score加权平均池化 → 智能体未来特征 $A_f^{L_{k-1}}$
- 自注意力建模多智能体未来间交互 → 与 $C_s$ 拼接得更新上下文 $C_{L_k}^i$
- Cross-Attention：Query $=Z_{L_{k-1}}^i + A_{mf}^{L_{k-1}}$，Key/Value=$C_{L_k}^i$，用masking阻止访问自身未来信息
- MLP解码GMM参数和分数

**学习过程（Sec 3.4）**：
- **模仿损失 $\mathcal{L}_{IL}$**：对最佳预测组件（离GT最近）计算负对数似然（Eq. 2-3）
- **交互损失 $\mathcal{L}_{Inter}$**（仅 k≥1）：基于斥力势场，鼓励避开前一层其他智能体可能的轨迹（Eq. 4），仅在距离小于阈值时激活
- **总损失**：$\mathcal{L}_i^k = w_1 \mathcal{L}_{IL} + w_2 \mathcal{L}_{Inter}$（Eq. 5）

## 实验与结果
**数据集**：Waymo Open Motion Dataset（WOMD）+ nuPlan

**预测任务（WOMD交互预测）**：
- 联合预测（M=6）：**minADE=0.9161，minFDE=1.9373，Miss=0.4531**，超越Scene-Transformer（0.9774/2.1892）、MTR（0.9181/2.0633），为SOTA
- 边际预测（M=64+EM聚合）：minADE=0.9721，mAP=0.1923，接近MTR

**开环规划（WOMD，K=4最优）**：
- 碰撞率**1.98%**（对比DIPP的2.33%、MTR-e2e的2.32%），规划ADE@5s=2.451m，预测ADE=0.853m/FDE=1.919m

**闭环规划（WOMD）**：
- 无精炼：**成功率73.16%**（vs DIPP 68.12%）；**加精炼后94.50%**（vs DIPP 92.16%），位置误差@8s=11.13m（vs DIPP 12.53m）

**nuPlan基准**：Overall 0.8288，超过Urban Driver（0.7467）和IDM Planner（0.5912），略低于Hoplan（0.8745）和Multi_path（0.8477）；论文说明因算力限制缩小了模型规模。

**消融**：移除未来信息/自注意力/交互损失均导致性能下降；交互预测最优K=6；规划最优K=4。

## 相关工作脉络
1. **Scene-Transformer / WayFormer**：聚焦统一Transformer编码架构，解码端相对简单；GameFormer弥补了解码端交互建模的不足。
2. **Motion Transformer（MTR）**：提出解码端的迭代局部运动细化，但未引入博弈论层次化交互深度建模；GameFormer的迭代是在认知层级层面而非空间局部层面。
3. **M2i / Pip（条件预测）**：将AV计划作为条件输入预测其他智能体，属于单向交互；GameFormer通过level-k框架实现双向互影响建模。
4. **DIPP**：可微分联合预测规划方法，但交互建模深度有限；GameFormer在交互损失和层次化推理上更具系统性。
5. **Social LSTM / Scept**：早期图神经网络和基于策略的预测方法；GameFormer以Transformer+博弈论提供更强表达力。
6. **MultiPath++**：多模态轨迹预测强基线，但仅预测ego轨迹，不建模智能体间交互；GameFormer通过joint/marginal预测和交互解码克服此局限。

## 局限性与未来方向
1. **推理层级深度有限**：实验中仅测试到K=8，更深层次的博弈推理可能带来进一步收益，但未充分探索。
2. **nuPlan性能受模型规模限制**：测试时因算力限制缩小了编码器/解码器层数，未达到理论上限性能。
3. **仍依赖模仿学习范式**：主损失为NLL模仿损失，离线训练模式难以像强化学习那样探索极端交互场景。
4. **交互损失为启发式斥力势场**：非严格最优控制意义上的碰撞避免，安全边界依赖超参数。

## 研究启发与可借鉴点
1. **level-k博弈论作为解码器结构先验**：可将此框架迁移到多智能体系统预测、机器人协同导航等其他需要显式交互建模的场景。
2. **层级交互损失的通用性**：基于斥力势场的$\mathcal{L}_{Inter}$可复用于其他多智能体轨迹预测任务中促进安全交互。
3. **前一层预测作为下一层上下文的自回归思想**：与MTR的迭代细化思路正交互补，可结合使用。
4. **可学习模态嵌入+交叉注意力的解耦设计**：将模态不确定性初始化和场景上下文编码分离，有利于理解和改进。

## 关键术语表
**Level-k博弈论**：认知分层博弈框架，level-0智能体独立行动，level-k智能体假设他人为level-(k-1)并据此优化自身策略。

**GMM（高斯混合模型）**：用多个高斯分布的加权和表示多模态未来轨迹分布，每个分量含均值、方差和概率权重。

**模仿损失（Imitation Loss）**：以专家驾驶数据为监督信号，对最佳预测分量计算负对数似然，模拟人类驾驶风格。

**交互损失（Interaction Loss）**：基于斥力势场的辅助损失，约束当前层智能体避免与前一层其他智能体的可能轨迹过于接近。

**模态嵌入（Modality Embedding）**：可学习的初始查询向量，编码未来可能行为的多种模态不确定性。

**开环规划 vs 闭环规划**：开环指在固定场景日志上评估规划轨迹质量；闭环指在仿真器中实际执行规划并观察与环境的实时交互结果。

**EM聚合**：期望最大化算法，用于从边际预测中组合出一致性强的联合预测轨迹集。

## 可复现要素
- **数据集**：Waymo Open Motion Dataset（公开）、nuPlan（公开，需注册）
- **代码**：项目主页 https://mczhi.github.io/GameFormer/，论文未明确声明GitHub仓库
- **关键超参**：Encoder层数E=6，隐藏维度D=256，邻居智能体数N=10/20，模态数M=6/64，最优解码层数K=4（规划）/K=6（预测），观察 horizon $T_h$ 和预测 horizon $T_f$（规划5s/预测8s）
