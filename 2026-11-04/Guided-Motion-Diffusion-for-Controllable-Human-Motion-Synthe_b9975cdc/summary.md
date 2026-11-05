---
title: "Guided-Motion-Diffusion-for-Controllable-Human-Motion-Synthe"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:42:51"
field: "3D 人类动作生成与空间条件控制"
keywords: ["human motion synthesis", "diffusion model", "spatial conditioning", "controllable generation", "trajectory generation"]
innovations: ["Emphasis Projection：通过随机投影矩阵放大轨迹维度权重以缓解姿态-轨迹不协调", "Dense Signal Propagation：利用预训练去噪器反向传播将稀疏关键帧引导信号稠密化", "两阶段轨迹-动作生成框架：先采样满足关键帧的轨迹，再条件生成局部姿态"]
benchmarks: ["HumanML3D"]
---

# 论文速读：Guided-Motion-Diffusion-for-Controllable-Human-Motion-Synthesis

## 一句话总结
本文提出 Guided Motion Diffusion (GMD)，通过引入 Emphasis Projection 和 Dense Signal Propagation 两项技术，将空间约束（如轨迹、关键帧、障碍物）有效集成到扩散模型的人类动作生成过程中，同时保持甚至提升了纯文本驱动动作生成的质量。

## 研究问题与动机
- **空间约束难以融入扩散模型**：现有基于扩散的动作生成模型（如 MDM）擅长文本条件生成，但无法处理全局轨迹、障碍物等空间约束条件，这些条件对真实环境中的人物动画至关重要。
- **稀疏的帧级引导信号易被忽略**：扩散模型在反降噪过程中会连续去噪，若引导信号仅存在于少量帧（如稀疏关键帧），这些信号容易被模型误判为噪声而丢弃，导致生成结果偏离约束。
- **运动表示中局部姿态与全局方向的失衡**：常见运动表示（如 HumanML3D）每帧使用 4 个值编码全局朝向，却使用 259 个值编码局部姿态；仅通过 Imputation 注入轨迹信息后，模型倾向于忽略全局方向的改变，产生脚部滑动（foot skating）等不协调现象。

## 核心贡献（创新点）
1. **Emphasis Projection**：通过可学习的随机投影矩阵放大轨迹相关维度（骨盆旋转与地面位置）在运动表示中的相对权重，使空间信息在去噪过程中获得充分关注，从根本上缓解局部姿态与全局轨迹之间的不协调问题；与简单增加轨迹损失权重的方法相比，不依赖重新训练且能一致性地提升 FID 与降低 foot skating。
2. **Dense Signal Propagation**：受强化学习中稀疏奖励分配问题的启发，通过反向传播经过已训练的去噪器（denoiser）将稀疏的空间引导信号扩散到相邻帧，将稀疏信号转化为稠密信号，使极少关键帧也能可靠地约束生成轨迹；无需额外模型。
3. **GMD 整体框架**：将上述两项技术与两阶段采样流程相结合——先基于目标函数和关键帧位置生成满足空间约束的高质量轨迹（使用 trajectory DPM），再在强调投影空间下生成与轨迹一致的局部姿态，同时采用 $\epsilon_\theta$ 建模而非 $\mathbf{x}_{0,\theta}$ 以缓解 DPM 对训练分布的固有偏见对引导信号的抵消。

## 方法详解

**整体架构**：GMD 采用 UNet + AdaGN  backbone，包含两个独立训练的扩散模型——motion DPM（$\mathbf{x}_{0,\theta}$）与 trajectory DPM（$\epsilon_\phi$），均基于 DDPM 框架，$T=1000$ 步去噪。

**Emphasis Projection（投影变换）**：
- 定义随机正交投影矩阵 $A = A' B$，其中 $A' \in \mathbb{R}^{N \times N}$ 元素来自 $\mathcal{N}(0,1)$，$B$ 是对角矩阵，轨迹相关索引（骨盆的 rot, x, z）置为 $c$，其余为 1。
- 投影后运动表示 $\mathbf{x}^{\mathrm{proj}} = \frac{1}{N-3+3c^2} A \mathbf{x}$，归一化因子保持单位方差。
- 投影空间的 imputation：先逆变换回原始空间做 imputation，再重新投影：
$$\tilde{\mathbf{x}}_0^{\mathrm{proj}} = A\left((1-M_z^x)\odot(A^{-1}\mathbf{x}_{0,\theta}^{\mathrm{proj}}) + M_z^x\odot P_z^x\mathbf{z}^*\right)$$

**Dense Signal Propagation（稠密引导）**：
- 核心思想：利用去噪器 $f(\mathbf{x}_t) = \mathbf{x}_{0,\theta}(\mathbf{x}_t)$ 的时空上下文学习能力，通过自动微分计算 $\nabla_{\mathbf{x}_t} G_z(P_x^z f(\mathbf{x}_t))$，将仅在稀疏关键帧上定义的 $G_z$ 梯度传播至所有帧。
- Masked classifier guidance：在 imputation 区域使用显式 imputation，在非 imputation 区域使用 classifier guidance：
$$\mu_t^{\mathrm{proj}} = \tilde{\mu}_t^{\mathrm{proj}} + A(1-M_z^x)\odot\Delta_\mu$$
其中 $\Delta_\mu = -s\Sigma_t A^{-1}\nabla_{\mathbf{x}_t^{\mathrm{proj}}} G_z(P_x^z A^{-1}f(\mathbf{x}_t^{\mathrm{proj}}))$。

**$\epsilon_\theta$ vs $\mathbf{x}_{0,\theta}$ 建模的选择**：
- $\mathbf{x}_{0,\theta}$ 在 $t=0$ 时对采样均值影响最大，与渐弱引导信号矛盾；$\epsilon_\theta$ 在 $t=T$ 时影响最大，更符合直觉——早期步骤强引导、后期少干扰。故 trajectory DPM 采用 $\epsilon_\phi$ 建模。

**两阶段生成流程（Algorithm 1）**：
- Stage 1：从先验 $\mathbf{z}_T \sim \mathcal{N}(0,I)$ 出发，经 trajectory DPM 迭代采样，期间用 imputation 注入关键帧位置 $\mathbf{y}$，并用 dense classifier guidance 优化目标函数 $G_z$。
- Stage 2：在投影空间中对运动进行迭代采样，使用 emphasis projection + masked classifier guidance，最终返回原始空间的 motion。

## 实验与结果

**数据集**：HumanML3D（14,646 个动作序列，44,970 条文本标注）。

**文本到动作生成（Table 1）**：
| 模型 | FID↓ | R-precision↑ | Diversity→ |
|------|-------|-------------|-----------|
| MDM | 0.556 | 0.608 | 9.446 |
| MLD | 0.473 | 0.772 | 9.724 |
| PhysDiff | 0.433 | 0.631 | — |
| **Ours** | **0.212** | 0.670 | 9.440 |
| Ours (proj) | 0.235 | 0.652 | 9.726 |

本文方法在 FID 上达到 **0.212**，显著优于所有基线（相对 MDM 提升约 62%），同时保持相近的 Diversity。

**轨迹条件生成（Table 2）**：
- 使用真实轨迹做 imputation，Ours + $c=10$ + $\mathbf{x}^{\mathrm{proj}}$ 达到 **FID=0.198，Foot skating ratio=0.128**，远优于 MDM baseline（FID=0.904，skating=0.284）和简单增加 loss weight 的方案。

**关键帧条件生成（Table 3，N=5 关键帧）**：
- 最佳配置（$\tau=100$）：**FID=0.523，Loc. err.=0.049，Avg. err.=0.139m**；MDM 在相同任务上 Loc. err. 为 0.127（且 traj. err. 为 0.326），说明 MDM 几乎完全忽略关键帧约束。
- 引入 trajectory DPM 两阶段方案后，Loc. err. 进一步降至 **0.020**（$\tau=100$），较单阶段降低超过 50%。

**障碍物回避（Figure 7）**：定性展示模型可在给定起点/终点及 2D SDF 障碍物条件下生成合理避障轨迹。

**计算效率**：单卡 RTX 3090，trajectory DPM 吞吐 2,048 samples/sec，motion DPM 吞吐 256 samples/sec，单次推理约 110 秒。

## 相关工作脉络
- **MDM**（Tev et al., 2022）：最强的文本条件动作生成基准，采用 UNet + classifier-free guidance，但不支持空间约束条件；本文在其基础上扩展空间可控能力。
- **MLD**（Chen et al., 2022）：在 latent space 执行运动扩散，文本到动作 FID=0.473；本文方法在相同指标上达到 0.212。
- **PhysDiff**（Yuan et al., 2023）：物理约束的运动生成，侧重物理合理性而非空间路径控制；本文关注轨迹/障碍物等几何约束。
- **MotionDiffuse**（Zhang et al., 2022）：早期基于 diffusion 的文本到动作生成工作，FID=0.630；本文优于该基线。
- **Classifier-free vs Classifier guidance**：MDM 等依赖 classifier-free；本文通过 classifier-based guidance（结合 dense propagation）实现不重新训练即可适应新空间约束目标。
- **Imputation/Inpainting in DPMs**（Chung et al., 2022；Choi et al., 2021）：已有图像编辑工具；本文首次将其系统应用于 3D 动作生成，并证明直接套用会导致严重不协调。

## 局限性与未来方向
- **SDF 障碍物的假设**：当前障碍物回避仅支持 2D 地面 SDF 表示，尚未扩展到复杂三维障碍物场景。
- **目标函数设计依赖人工**：$G_z$ 需要针对每个任务手动设计，对于隐式目标（如社交距离、自然性）的自动化设计仍具挑战。
- **推理速度较慢**：单次采样约 110 秒，难以满足实时应用需求。
- **轨迹 DPM 需额外训练**：虽然共用同一数据集但需要单独训练一个小型 trajectory DPM，增加了管线复杂度。
- 作者提及后续工作可扩展更多类型的引导信号，包括非显式目标。

## 研究启发与可借鉴点
- **Emphasis Projection 思路可迁移**：任何存在维度权重失衡的多模态/多模态条件生成任务（如视频+音频、3D shape+texture），均可考虑类似投影加权来平衡不同条件的重要性，且无需修改损失函数即可生效。
- **Denoiser 作为稀疏信号传播器**：将预训练 denoiser 用于梯度传播以实现稀疏→稠密的信号扩展，这一思路不限于动作生成，可推广至图像修复、视频插值等任何需要局部约束扩散的任务。
- **两阶段生成策略**：先解耦生成中间变量（轨迹），再条件生成最终输出（姿态），可简化复杂条件的联合优化，且提高可控性；类似思路可借鉴于文本→场景布局→渲染的两阶段 pipeline。
- **$\epsilon_\theta$ 建模对 classifier guidance 的稳定性**：在需要强外部引导的扩散模型任务中，使用 $\epsilon_\theta$ 而非 $\mathbf{x}_{0,\theta}$ 建模可显著改善引导信号的维持，这是一个简单但有效的实现细节。
- **Masked hybrid conditioning**（imputation + classifier guidance 的联合使用）：在部分条件已知、部分条件未知的混合场景下，按 mask 分区使用不同引导策略是一种通用范式。

## 关键术语表
- **Emphasis Projection**：一种通过随机投影矩阵放大特定维度（如轨迹）权重的表示变换方法，使扩散模型在去噪过程中更关注空间信息。
- **Dense Signal Propagation**：利用预训练去噪器的时空上下文能力，通过反向传播将稀疏关键帧引导信号扩散到相邻帧的技术。
- **Classifier Guidance**：通过梯度 $\nabla_{\mathbf{x}_t}\log p(d|\mathbf{x}_t)$ 偏移采样均值来实现条件引导的后处理方法，无需重新训练扩散模型。
- **Imputation / Inpainting**：在扩散模型采样过程中，将已知部分直接替换为目标值（imputation）或对未观测部分进行修复（inpainting）的技术。
- **Foot Skating Ratio**：衡量生成动作中脚部在地面滑动帧占比的指标，用于评估轨迹与局部姿态之间的空间一致性。
- **Goal Function $G_z(\cdot)$**：一个标量函数，用于度量当前动作轨迹与目标约束（如轨迹、关键帧、障碍物）的偏离程度。
- **Classifier-Free Guidance**：通过同时在条件/无条件denoiser之间插值（参数 $w$）来增强条件生成质量的方法，是 MDM 等工作的核心条件机制。
- **$\epsilon_\theta$ vs $\mathbf{x}_{0,\theta}$ 建模**：扩散模型预测目标的两种选择；前者预测噪声，后者预测干净输入，两者在不同时间步对采样均值的影响权重不同。

## 可复现要素
- **数据集**：HumanML3D（公开，来自 AMASS + HumanAct12）；代码与项目页面：https://korrawe.github.io/gmd-project/
- **代码/权重**：论文未提供公开代码下载链接，仅给出项目主页；权重未明确声明开源状态。
- **关键超参**：DDPM $T=1000$ 步；Emphasis Projection 系数 $c \in \{1,2,5,10\}$；两阶段切换步数 $\tau$；classifier guidance 强度 $s$；UNET + AdaGN backbone；CLIP 文本编码器。
