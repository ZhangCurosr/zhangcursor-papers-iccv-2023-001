---
title: "GameFormer-Game-theoretic-Modeling-and-Learning-of-Transform"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:05:24"
field: "自动驾驶交互式预测与规划"
keywords: ["interactive prediction", "level-k game theory", "autonomous driving", "motion planning", "transformer", "multi-agent trajectory prediction"]
innovations: ["基于level-k博弈论的层次Transformer解码器结构", "结合模仿损失与排斥势场交互损失的学习框架", "统一评估交互预测精度与开环/闭环规划性能"]
benchmarks: ["WOMD Interaction Prediction", "nuPlan Planning Benchmark"]
---

# 论文速读：GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving

## 一句话总结
本文提出GameFormer，一个基于Transformer的交互式预测与规划框架，利用level-k层次博弈论设计解码器结构，通过迭代推理建模智能体间的动态交互；在Waymo数据集上实现最先进的交互预测性能，并在nuPlan规划基准上取得领先表现。

## 研究问题与动机
1. **未来交互建模缺失**：现有Transformer预测模型擅长编码场景上下文和历史轨迹，但无法显式建模智能体的未来交互及其与AV的交互，导致规划模块处于被动反应状态。
2. **条件预测的单向性局限**：现有条件预测模型（如Pip、M2I）仅将AV的内部计划作为条件输入预测其他智能体响应，忽略了AV与其他道路使用者之间的双向动态影响。
3. **规划与预测割裂**：多数工作将预测模块与规划模块分离训练，缺乏联合推理机制，难以在并线、 unprotected left turn等需要主动协调的场景中实现人机共驾式决策。

## 核心贡献（创新点）
1. **提出GameFormer框架**：基于level-k博弈论设计层次Transformer解码器，每一层利用上一层预测轨迹作为上下文迭代细化交互推理；与已有工作本质区别在于将博弈论中的认知层级显式建模为解码器结构，而非仅依赖编码器处理历史交互。
2. **设计level-k交互解码器**：level-0 decoder独立预测模态轨迹，level-k（k≥1）decoder通过自注意力建模其他智能体上一级未来轨迹，并通过交叉注意力更新场景上下文；创新点在于引入了"上一级预测→当前级推理"的级联机制，使模型能够捕捉多轮交互动态。
3. **提出结合模仿损失与交互损失的学习过程**：交互损失利用排斥势场鼓励当前level智能体避免与上一级其他智能体轨迹碰撞；区别于仅优化轨迹拟合精度的方法，该设计将安全性约束显式嵌入训练目标。
4. **统一预测与规划评估**：在WOMD和nuPlan上同时验证预测精度与规划性能（开环+闭环），证明交互式预测对下游规划的增益；与仅关注预测指标的工作相比，本文强调端到端实用性。

## 方法详解
**场景编码（Scene Encoding）**：
- 使用LSTM编码agent历史状态序列 $S_p$，得到特征 $A_p \in \mathbb{R}^{N \times D}$
- MLP编码局部矢量地图折线 $M$，经max-pooling聚合后得 $M_r \in \mathbb{R}^{N \times N_{mr} \times D}$
- 拼接 $C^i = [A_p, M_p^i]$，通过E层Transformer编码器生成共享场景上下文 $C_s \in \mathbb{R}^{N \times (N+N_{mr}) \times D}$

**层次解码（Level-k Decoding）**：
- **Level-0**：可学习模态嵌入 $I \in \mathbb{R}^{N \times M \times D}$ 与历史编码拼接后作为query，$C_s$作为key/value，经交叉注意力后通过MLP解码GMM分量 $G_{L_0}$ 和分数 $P_{L_0}$
- **Level-k（k≥1）**：将level-(k-1)的均值轨迹 $S_f^{L_{k-1}}$ 经MLP+max-pooling编码为 $A_{mf}^{L_{k-1}}$，再按分数加权平均得 $A_f^{L_{k-1}}$；通过自注意力建模agent间未来交互特征，拼接到 $C_s$ 得到更新上下文 $C_{L_k}^i$；最后以level-(k-1)的query内容 $Z_{L_{k-1}}^i$ 和 $A_{mf}^{L_{k-1}}$ 拼接为query，与 $C_{L_k}^i$ 进行交叉注意力，输出当前level的GMM分量和分数；使用masking防止agent访问自身未来信息

**学习过程（Learning Process）**：
- **模仿损失** $\mathcal{L}_{IL}$：对最佳预测分量计算负对数似然 $\mathcal{L}_{NLL} = \log\sigma_x + \log\sigma_y + \frac{1}{2}\left[\left(\frac{dx}{\sigma_x}\right)^2 + \left(\frac{dy}{\sigma_y}\right)^2\right] - \log p_{m*}$
- **交互损失** $\mathcal{L}_{Inter} = \sum_{m=1}^M \sum_{t=1}^{T_f} \max_{j \neq i} \frac{1}{d(\hat{s}_{m,t}^{(i,k)}, \hat{s}_{n,t}^{(j,k-1)}) + 1}$，仅在距离小于阈值时激活排斥力
- **总损失** $\mathcal{L}_i^k = w_1 \mathcal{L}_{IL} + w_2 \mathcal{L}_{Inter}$

## 实验与结果
**数据集**：Waymo Open Motion Dataset (WOMD) 用于预测与规划评估；nuPlan用于规划基准测试

**评估基线**：预测任务对比LSTM baseline、Heat、AIR2、SceneTrans、DenseTNT、M2I、MTR；规划任务对比Vanilla IL、DIM、MultiPath++、MTR-e2e、DIPP

**主要结果**：
- **交互预测**（Table 1）：GameFormer (J, M=6) 取得minADE=0.9161、minFDE=1.9373，超越MTR（0.9181/2.0633）；GameFormer (M, M=64) 的mAP=0.1923接近MTR的0.2037
- **开环规划**（Table 3）：碰撞率1.98%、遗漏率7.53%，Planning ADE 0.129/0.836/2.451m@1/3/5s，优于DIPP（2.33%/8.44%）
- **闭环规划**（Table 4）：成功率73.16%，配合refinement后达94.50%，位置误差@8s为21.02m，优于DIPP的92.16%/26.10m
- **nuPlan基准**（Table 5）：总分0.8288，在learning-based方法中仅次于Hoplan（0.8745）和Multi_path（0.8477）

**消融实验**（Tables 6-8）：移除未来轨迹编码、自注意力模块或交互损失均导致性能下降；最优解码层数K=6（预测）/K=4（规划）

## 相关工作脉络
1. **Scene-Transformer/WayFormer**：采用统一Transformer架构编码场景，但解码端侧重单一阶段预测；本文在解码端引入层次推理，显式建模多轮交互
2. **Motion Transformer**：提出迭代局部运动细化机制；本文在此基础上引入博弈论框架，使细化过程具有交互语义
3. **THOMAS/Intersim**：多智能体联合预测框架，关注轨迹一致性但未考虑AV主动规划的影响；本文通过level-k机制将AV纳入交互循环
4. **Conditional prediction (Pip, M2I)**：单向条件预测，将AV计划作为先验；本文扩展为双向迭代交互，逼近真实博弈过程
5. **DIPP**：可微分集成预测规划方法；本文通过显式交互建模提升预测质量，进而改善规划性能（碰撞率降低0.35%）

## 局限性与未来方向
1. **解码层数依赖任务调参**：预测任务最优K=6，规划任务K=4，缺乏统一理论指导层数选择
2. **计算复杂度随层数和agent数增长**：多次注意力计算可能影响实时部署，论文未讨论模型压缩策略
3. **nuPlan性能受限于模型尺寸**：因服务器算力限制缩减了encoder/decoder层数，完整规模性能未报告
4. **交互损失仅考虑碰撞规避**：未显式建模社交规范（如让行礼仪、换道默契），可能限制在复杂社交场景的泛化

## 研究启发与可借鉴点
1. **层次推理架构的可迁移性**：level-k博弈论驱动的级联解码结构可推广至机器人编队控制、体育战术预测等多智能体交互场景
2. **势场交互损失设计**：排斥势场形式的碰撞避免损失易于实现且可微，可复用至其他多智能体强化学习或模仿学习任务
3. **编码-解码职责分离范式**：编码器专注场景理解、解码器专注交互推理的分工模式，为感知-预测-规划联合架构设计提供参考
4. **联合评估方法论**：同时报告预测精度与规划指标（开环+闭环），为模型实用性验证建立完整评估体系，值得后续工作借鉴

## 关键术语表
**Level-k Game Theory**：层次博弈论，假设agent具有不同推理深度，level-k agent假设其他agent为level-(k-1)并据此优化自身策略
**Gaussian Mixture Model (GMM)**：高斯混合模型，本文用于表示多模态未来轨迹分布，每模态由均值和协方差参数化
**Interaction Loss**：交互损失，通过排斥势场惩罚当前level智能体与上一级其他智能体轨迹的近距离接触
**Modality Embedding**：模态嵌入，初始化为可学习参数的query张量，表征未来可能出现的不同行为模式
**Open-loop Planning**：开环规划评估，在固定场景回放中测试规划器，其他智能体不响应AV决策
**Closed-loop Planning**：闭环规划评估，规划器与仿真环境实时交互，衡量实际执行安全性和任务完成度

## 可复现要素
- **数据集**：Waymo Open Motion Dataset（公开）、nuPlan（需申请）
- **代码/权重**：论文未提及开源；项目网站：https://mczhi.github.io/GameFormer/
- **关键超参**：Encoder层数E=6，隐藏维度D=256，解码层数K=4（规划）/K=6（预测），模态数M=6（joint预测）/M=64（marginal预测），观测窗口与未来 horizon依任务而定
