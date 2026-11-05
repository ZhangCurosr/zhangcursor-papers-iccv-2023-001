---
title: "Delving-into-Motion-Aware-Matching-for-Monocular-3D-Object-T"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Delving_into_Motion-Aware_Matching_for_Monocular_3D_Object_Tracking_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:58:21"
field: "单目3D视觉感知与跟踪"
keywords: ["单目3D多目标跟踪", "运动感知匹配", "Motion Transformer", "特征空间关联", "对比学习", "nuScenes", "KITTI"]
innovations: ["在特征空间而非输出空间进行运动感知的数据关联，缓解单目检测噪声影响", "设计 tracklet-conditioned 候选特征与双阶段时空Transformer，显式建模多帧运动与轨迹间空间交互", "引入对比运动特征学习，增强运动表征对遮挡和检测噪声的鲁棒性"]
benchmarks: ["nuScenes 3D MOT", "KITTI 3D MOT"]
---

# 论文速读：Delving-into-Motion-Aware-Matching-for-Monocular-3D-Object-Tracking

## 一句话总结
本文提出 MoMA-M3T，一种面向单目3D多目标跟踪的运动感知匹配框架，通过将历史轨迹的相对运动信息编码为特征空间表示，利用运动Transformer建模时空运动模式，在特征空间而非输出空间进行数据关联，有效缓解了单目3D检测器观测噪声问题，在 nuScenes 和 KITTI 上均达到SOTA性能。

## 研究问题与动机
1. **长程运动信息利用不足**：现有方法多仅匹配相邻帧特征（如Time3D），或预测当前状态后在输出空间匹配（如QD-3DT、DEFT），难以捕捉多帧运动的长程依赖关系。
2. **单目检测噪声敏感**：单目3D检测器提供的物体位置、姿态估计存在较大误差，直接在输出空间（如位置坐标）进行匹配容易受噪声干扰。
3. **缺乏运动感知的表征设计**：现有工作较少探索如何将物体运动信息显式编码为鲁棒的特征表示，以支持在特征空间中完成数据关联。
4. **泛化能力有限**：多数方法需针对特定检测器联合训练，难以迁移到不同3D检测器的输出。

## 核心贡献（创新点）
1. **提出 MoMA-M3T 框架，将运动感知匹配机制引入单目3D MOT**：与 Time3D/DEFT 等在输出空间匹配的不同在于，本文在特征空间计算匹配概率，从而规避单目检测器位置估计噪声的影响。
2. **设计 Motion Transformer 模块，从时空视角建模轨迹运动表示**：相比 QD-3DT 使用的 LSTM，本文的Transformer同时包含时间编码器和空间编码器，可更好捕获长程时空依赖与轨迹间空间交互。
3. **提出 Tracklet-conditioned 运动特征生成策略**：将每个当前检测与所有历史轨迹的最新位置配对，生成多组候选运动特征，使匹配过程考虑所有可能的关联假设，而非仅依赖单一预测。
4. **引入对比运动特征学习（Contrastive Motion Feature Learning）**：通过对同一条轨迹的不同采样子集施加对比损失，学习更鲁棒的运动特征表示，减少遮挡和检测噪声带来的影响；这与常规分类损失的本质区别在于显式拉近同类轨迹、推远异类轨迹的表征空间结构。
5. **实现即插即用的灵活架构**：无需重新训练即可与不同预训练单目3D检测器（FCOS3D、EPro-PnP 等）结合，验证了运动特征的通用性。

## 方法详解
MoMA-M3T 遵循 tracking-by-detection 范式，包含三个核心组件：

**1）运动特征生成（Motion Feature Generation）**
- 定义物体的运动状态为 $s_t = (\mathbf{r}, \theta, h, w, l)_t$，其中 $\mathbf{r} = \mathbf{p}_a - \mathbf{p}_b$ 是相邻帧间的相对位移向量，$\theta$ 为朝向角，$(h,w,l)$ 为尺寸。
- 通过 MLP 编码为运动特征：$\mathbf{f}_t = \mathrm{MLP}(s_t) \in \mathbb{R}^C$。
- **Tracklet-conditioned 策略**：对当前帧每个检测 $\mathbf{p}_{obs}$，计算其相对于每条轨迹最新位置 $\mathbf{p}_l$ 的相对运动 $\mathbf{r}_{obs|l}$，生成 $N \times M$ 组候选运动特征。
- 维护 **Motion Feature Bank**：存储历史运动特征 $\mathbf{F}_{bank} \in \mathbb{R}^{N_{max} \times T_{max} \times C}$ 和全局位置 $\mathbf{P}_{bank} \in \mathbb{R}^{N_{max} \times T_{max} \times 3}$。

**2）Motion Transformer**
- **时间编码**：对历史 $T$ 帧中缺失帧引入可学习的时间位置嵌入，建模非连续观测。
- **时间编码器**：将可学习的 motion token $\mathbf{F}_m$ 拼接至序列头部，经 Multi-Head Self-Attention + FFN 聚合时序信息，输出 $\hat{\mathbf{F}}_m$。
- **空间编码器**：将轨迹绝对位置 $\mathbf{X}_p = \mathrm{MLP}(\mathbf{p}_l)$ 作为 Key/Value，与运动特征 Query 交互，捕获轨迹间空间依赖，输出最终运动特征 $\tilde{\mathbf{F}} \in \mathbb{R}^{M \times C}$。

**3）运动感知匹配（Motion-Aware Matching）**
- 计算检测与轨迹运动特征的差异，经 MLP + Sigmoid 得到匹配概率：
  $$\mathbf{A}_{ij} = \mathrm{Sigmoid}(\mathrm{MLP}(\mathbf{f}_{i|p_j} - \tilde{\mathbf{F}}_j))$$
- 匹配损失采用 Binary Focal Loss：
  $$\mathcal{L}_{match} = \frac{1}{NM}\sum_i\sum_j \mathrm{FL}(\mathbf{A}_{ij}, \hat{\mathbf{A}}_{ij})$$
- **对比运动特征学习损失**：
  $$\mathcal{L}_{con} = -\frac{1}{|\mathbf{N}_p|}\sum_{(i,j)\in\mathbf{N}_p}\log\frac{\exp(\tilde{\mathbf{F}}_i\cdot\tilde{\mathbf{F}}_j/\tau)}{\sum_{(i,k)\in\mathbf{N}_a}\exp(\tilde{\mathbf{F}}_i\cdot\tilde{\mathbf{F}}_k/\tau)}$$
  其中 $\tau=0.1$，正样本来自同一条轨迹的不同采样子集，负样本来自不同轨迹。
- 总损失：$\mathcal{L} = \mathcal{L}_{match} + \mathcal{L}_{con}$。

**推理流程**：每帧利用匈牙利算法匹配检测与轨迹，匹配分 > 0.5 则关联成功；未匹配轨迹保留 10 帧后淘汰（track rebirth 策略）；匹配成功的检测特征和位置更新入 Feature Bank。

## 实验与结果
**数据集**：nuScenes（700 train / 150 val / 150 test）、KITTI（13 train / 8 val / 29 test），后者无官方3D MOT基准，采用 AB3DMOT 评估协议。

**评估基线**：Time3D、QD-3DT、DEFT、CenterTrack、MUTR3D、CC-3DT 等。

**主要结果**：
- **nuScenes 单摄像头设置（Table 1）**：MoMA-M3T 达到 AMOTA **24.2%**（标准），AMOTA **28.5%**（更强检测器 + 对比学习）；MOTA **24.6%**，MOTP **0.695m**；相比 Time3D（AMOTA 21.4%）**+2.8%**；MT 达 2236，ML 降至 2642。
- **KITTI 验证集 Car 类别（Table 2）**：sAMOTA **47.17**，AMOTA **16.12**；相比 MonoDLE + AB3DMOT（sAMOTA 46.16 / AMOTA 13.00）**+1.01 / +3.12**。
- **多摄像头 nuScenes（Table 6）**：相比 MUTR3D + DETR3D **(+9.0 AMOTA)**，相比 CC-3DT + BEVFormer **(+1.5 AMOTA)**。
- **推理速度**：33.3 FPS（NVIDIA 3090，batch=1）。

**消融结论**：
- 运动表征 + 特征空间匹配组合（Table 3）使 AMOTA 达 30.7，优于任一单独设计；加入对比学习后进一步提升至 31.1。
- Motion Transformer 各组件均有效（Table 4）：Temporal encoder（+1.6 AMOTA）贡献最大，Spatial encoder 和全局位置特征亦有明显增益；替换为 LSTM 后性能下降（29.7 vs 31.1）。

## 相关工作脉络
1. **Time3D [24]**：端到端联合检测与跟踪，利用 Transformer 建模相邻帧间关系；本文与其区别在于不使用相邻帧直接匹配，而是利用多帧历史运动特征在特征空间关联。
2. **QD-3DT [15]**：联合学习3D状态和 Re-ID 特征，使用 LSTM 建模轨迹运动；本文用 Transformer 替代 LSTM，且在特征空间而非输出空间做匹配，对检测噪声更鲁棒。
3. **DEFT [6]**：利用历史特征预测当前状态并在输出空间匹配；本文避免在含噪输出空间匹配，转而学习运动特征表示。
4. **CenterTrack [61]**：基于关键点的方法在相邻帧间匹配；本文通过多帧历史运动建模实现更长程的关联能力。
5. **MUTR3D [57] / CC-3DT [11]**：多摄像头3D MOT方法；本文方法可无缝适配多摄像头检测输出（Table 6），且在不依赖多摄像头信息的情况下提升显著。
6. **AB3DMOT [49]**：基于3D IoU + Kalman Filter 的传统方法；本文用学习到的运动特征匹配取代手工设计的度量，在单目设置下显著优于传统方案。

## 局限性与未来方向
1. **严重遮挡下的性能瓶颈**：tracklet-conditioned 特征依赖轨迹最新位置，若连续多帧丢失可能导致特征失效（track rebirth 策略缓解但非根治）。
2. **运动特征对极远距离物体敏感度有限**：单目深度估计在远距离误差大，相对位移的准确性下降，可能影响远距离目标的跟踪稳定性。
3. **实时性仍有优化空间**：33.3 FPS 虽满足部分需求，但对资源受限平台仍需进一步加速（如减少 Feature Bank 查询开销）。
4. **未探索3D运动特征与外观特征的融合**：当前仅使用几何运动信息，结合 Re-ID 外观特征可能进一步提升复杂场景下的关联能力。
5. **对比学习的正负样本构造依赖轨迹完整性**：若某轨迹被频繁截断，对比学习的有效样本数将减少。

## 研究启发与可借鉴点
1. **特征空间匹配替代输出空间匹配**：将多模态观测对齐到统一特征空间后再做关联，可有效缓解各模态（或检测器）的噪声，这一思路可迁移至 LiDAR-Camera 融合跟踪等任务。
2. **Tracklet-conditioned 候选特征设计**：将当前观测与每条历史轨迹的相对运动分别编码为候选特征，使匹配网络能显式考虑所有可能的关联假设，而非依赖单一预测值，设计简洁且有效。
3. **对比学习增强运动表征鲁棒性**：对同一条轨迹的随机子集施加对比损失，可有效学习对局部遮挡和检测噪声不敏感的运动特征，该策略可推广到其他时序关联任务。
4. **时空 Transformer 在轨迹建模中的分层设计**：先时间编码器聚合时序运动，再空间编码器建模轨迹间交互，这种"时序→空间"的两阶段设计兼具清晰性和有效性，可作为通用轨迹建模的参考范式。
5. **即插即用的解耦架构**：检测与跟踪模块完全解耦、冻结权重即可迁移，大幅降低部署成本；该思路对构建模块化3D感知系统有参考价值。

## 关键术语表
**Monocular 3D Multi-Object Tracking (3D MOT)**：仅依靠单个相机传感器，在视频序列中同时完成3D物体检测与跨帧身份关联的任务。
**Motion Feature**：由物体相对位移、朝向角和尺寸等运动状态经 MLP 编码得到的特征向量，用于表征物体的运动模式。
**Motion Transformer**：包含时间编码、时序编码器和空间编码器的 Transformer 模块，从时空双视角建模轨迹的运动表示。
**Tracklet-conditioned Motion Feature**：将当前帧每个检测分别与所有活跃轨迹的最新位置配对，生成多组候选运动特征，使匹配过程考虑所有可能的关联假设。
**Motion Feature Bank**：存储所有活跃轨迹历史运动特征和全局3D位置的矩阵，维度为 $N_{max} \times T_{max} \times C$。
**Motion-Aware Matching**：通过计算检测与轨迹运动特征之差，经 MLP + Sigmoid 输出匹配概率的关联模块。
**Contrastive Motion Feature Learning**：对同一条轨迹的不同采样子集施加 InfoNCE 对比损失，拉近同类轨迹表征、推远异类轨迹表征的学习策略。
**Tracking-by-Detection**：先独立执行3D检测生成候选框，再通过数据关联模块将检测与历史轨迹匹配的跟踪范式。

## 可复现要素
- **数据集**：nuScenes（公开）、KITTI（公开）
- **代码**：已开源，https://github.com/kuanchihhuang/MoMA-M3T
- **权重**：已提供（论文声明）
- **关键超参**：特征维度 $C=128$，$N_{max}=50$，$T_{max}=10$，对比学习温度 $\tau=0.1$，匹配阈值 0.5，轨迹保留 10 帧后淘汰；训练 Adam，lr=0.0001，batch=128，100 epoch；每 mini-batch 采样 16 条轨迹（$T=6$ 帧）和 16 个检测，对比学习每轨迹采样 $k=2$ 个子集。
- **检测器**：nuScenes 用 PGD3D [46]，KITTI 用 MonoDLE [33]；鲁棒性实验使用 FCOS3D [45] 和 EPro-PnP [7]。
