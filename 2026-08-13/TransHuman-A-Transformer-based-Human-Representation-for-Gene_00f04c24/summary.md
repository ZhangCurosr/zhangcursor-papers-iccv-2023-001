---
title: "TransHuman-A-Transformer-based-Human-Representation-for-Gene"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_TransHuman_A_Transformer-based_Human_Representation_for_Generalizable_Neural_Human_Rendering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:35:47"
field: "可泛化神经人渲染"
keywords: ["generalizable neural human rendering", "transformer", "NeRF", "SMPL", "canonical space", "human representation"]
innovations: ["提出TransHuman框架，首次在canonical space下用transformer处理paint SMPL捕获人体全局关系", "设计DPaRF将token从canonical空间变形回观测空间进行鲁棒编码", "通过FDI实现粗粒度人体先验与细粒度外观特征的coarse-to-fine融合"]
benchmarks: ["ZJU-MoCap", "H36M"]
---

# 论文速读：TransHuman-A-Transformer-based-Human-Representation-for-Gene

## 一句话总结
本文提出TransHuman框架，用于可泛化的神经人渲染任务，通过在canonical space下学习paint SMPL并利用transformer捕获人体各部分的全局关系，显著优于现有SPC-based方法，在ZJU-MoCap和H36M上实现新的SOTA性能。

## 研究问题与动机
1. **动态人体渲染的应用需求**：高保真自由视角视频合成对混合现实、游戏、远程在场等应用至关重要，但人体动态形变使该任务比静态场景更具挑战性。
2. **现有可泛化方法的局限**：之前方法主要依赖SparseConvNet (SPC)处理paint SMPL，在观测空间下优化导致训练/推理阶段姿态不对齐，且3D卷积局部感受野有限，对自遮挡导致的不完整SMPL敏感。
3. **通用性需求**：传统方法需要逐subject的繁琐优化且依赖密集训练视角，无法泛化到新主体，限制了实际应用。

## 核心贡献（创新点）
1. **提出TransHuman框架**：首个基于transformer的可泛化神经人渲染框架，在ZJU-MoCap和H36M上达到显著新SOTA，比次优方法提升+2.20 PSNR和45% LPIPS。
2. **Canonical space下的人体表征学习**：首次在静态canonical space下处理paint SMPL以消除训练/推理的姿态不对齐，并通过DPaRF将表征变形回观测空间进行鲁棒查询点编码。
3. **首次探索transformer用于paint SMPL全局关系建模**：利用transformer捕获人体各部分的全局语义关联，相比SPC的局部感受野，能更好处理严重自遮挡问题。

## 方法详解
**整体架构**：TransHuman由三个核心模块组成——TransHE（Transformer-based Human Encoding）、DPaRF（Deformable Partial Radiance Fields）、FDI（Fine-grained Detail Integration）。

**TransHE（Transformer-based Human Encoding）**：
- **Canonical Body Grouping**：在canonical space（T-pose）下对SMPL顶点进行k-means聚类，得到分组字典D^c，避免observation space下因姿态变化导致的语义歧义（时序语义变化和空间语义纠缠问题）。
- **特征聚合**：$\hat{F} = \mathcal{G}_{\mathcal{D}^c}(F)$，将同簇顶点特征通过平均池化聚合为N_t个token。
- **Canonical Learning**：$\hat{F}' = \mathcal{T}(\hat{F}, \gamma_1(\hat{V}^c))$，在静态canonical位置$\hat{V}^c$下使用transformer学习全局关系，避免observation位置随时间步变化导致的模式不固定问题。

**DPaRF（Deformable Partial Radiance Fields）**：
- **坐标系统变形**：每个token绑定一个局部辐射场，其坐标系统$W_i^c$在canonical space初始化，随姿态变化通过旋转矩阵$\hat{R}_i$变形：$W_i^o = \hat{R}_i W_i^c$。
- **坐标编码**：查询点p在i-th token的变形坐标系下的坐标：$\bar{\mathbf{p}}_i = W_i^o(\mathbf{p} - \widehat{V}_i^o)$，最终human representation：$\mathbf{h}_i = [\widehat{F}_i'; \gamma_2(\bar{\mathbf{p}}_i)]$。
- **K近邻聚合**：将查询点分配给N_k个最近DPaRF，基于距离加权聚合：$\mathbf{h} = \sum_{i=1}^{N_k} \text{softmax}(\cdot) \mathbf{h}_i$。

**FDI（Fine-grained Detail Integration）**：
- **Appearance Features**：除CNN深度特征外，额外拼接投影的RGB原始信息，通过FC层融合以弥补CNN下采样的细节丢失。
- **Coarse-to-fine Integration**：使用cross-attention，以human representation $\mathbf{h}$为query，appearance feature $\mathbf{a}$为key/value，实现从粗粒度人体先验到细粒度外观的融合。

**体积渲染与损失**：
- 密度和颜色由MLP预测：$\sigma(\mathbf{p}) = MLP_\sigma(\mathbf{f})$, $\mathbf{c}(\mathbf{p}) = MLP_\mathbf{c}(\mathbf{f}, \gamma_3(\mathbf{d}))$。
- 训练损失：$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \mathcal{L}_{PER}$（perceptual loss）。

## 实验与结果
**数据集**：ZJU-MoCap（10个subject，23同步相机）、H36M（8个subject，4相机）。

**评估设置**：
- **Pose Generalization**：训练ZJU-7，测试ZJU-7（未见过姿态）
- **Identity Generalization**：训练ZJU-7，测试ZJU-3
- **One-shot Generalization**：推理时仅1个参考视角
- **Cross-dataset Generalization**：ZJU-MoCap训练，H36M测试

**主要结果**：
| 设置 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|------|--------|--------|---------|
| Pose Generalization | GP-NeRF | 25.05 | 0.909 | 0.159 |
| Pose Generalization | **Ours** | **27.25** | **0.936** | **0.087** |
| Identity Generalization | GP-NeRF | 24.55 | 0.902 | 0.157 |
| Identity Generalization | **Ours** | **26.15** | **0.918** | **0.098** |
| Cross-dataset | NHP | 18.84 | 0.820 | 0.222 |
| Cross-dataset | **Ours** | **20.48** | **0.856** | **0.169** |

**效率对比**：参数量6.08M，推理时间17min，推理显存6.2GB，与GP-NeRF（9min/10.3GB）相比，在相同时间内PSNR提升0.84。

## 相关工作脉络
1. **PixelNeRF [51] / IBRNet [42]**：可泛化静态场景的NeRF方法，本文受其启发将其扩展到动态人体场景。
2. **NHP [19] / GP-NeRF [6]**：现有可泛化神经人渲染SOTA方法，采用SPC-based人体表征，本文通过transformer替代SPC解决其姿态不对齐和局部感受野问题。
3. **Neural Body [33] /HumanNeRF [44]**：per-subject优化方法，需要密集视角和逐subject训练，无法泛化到新主体。
4. **MVSNeRF [5] / KeyNeRF [28]**：基于多视图立体或关键点的方法，本文利用human prior（SMPL）提供更强的几何约束。
5. **Transformer + NeRF结合**：[22, 18, 36]将transformer作为特征聚合器，本文首次将transformer应用于paint SMPL表面的全局关系建模。

## 局限性与未来方向
1. **依赖预拟合SMPL**：当前方法假设SMPL已准确拟合，未考虑joint optimization of fitted SMPL under the generalizable setting。
2. **采集环境限制**：目前仅在多视图校准场景下验证，未来可扩展到unconstrained capture setups。
3. **计算效率**：虽然比NHP快，但相比GP-NeRF仍有推理时间差距，可通过减少采样点（如16pts）来平衡。

## 研究启发与可借鉴点
1. **Canonical space + Transformer组合**：将对象表征在canonical空间学习（静态、无歧义），再通过变形映射回observation空间，可迁移到其他形变对象的表征学习。
2. **Deformable Partial Radiance Fields设计**：为每个语义token绑定局部可变形辐射场的思路，可用于其他需要细粒度局部编码的场景。
3. **Coarse-to-fine Integration策略**：先用强prior（人体/物体结构）提供几何约束，再引入细粒度外观信息，这种分层融合策略值得借鉴。
4. **Semantic Disentanglement via Grouping**：通过canonical space的grouping避免semantic ambiguity，对解决复杂形变场景的表征歧义问题有启发。

## 关键术语表
**Generalizable Neural Human Rendering**：从多视角人类视频学习条件NeRF，能够零样本泛化到新主体的自由视角渲染任务。
**Painted SMPL**：将参考图像的深度特征通过相机投影到拟合的SMPL模型顶点上形成的特征化参数化人体模型。
**Canonical Space**：标准T-pose的人体空间，姿态固定且身体舒展，用于消除训练/推理的姿态不对齐。
**Observation Space**：实际观测姿态下的人体空间，随时间步和姿态变化。
**Transformer-based Human Encoding (TransHE)**：在canonical space下通过transformer处理paint SMPL，捕获人体各部分全局关系的编码管道。
**Deformable Partial Radiance Fields (DPaRF)**：绑定每个token与可变形局部辐射场的机制，坐标系统随姿态变化，用于编码观测空间中查询点的鲁棒人体表征。
**Fine-grained Detail Integration (FDI)**：通过cross-attention从像素对齐的外观特征中整合细粒度细节（光照、纹理）到粗粒度人体表征。
**Canonical Body Grouping**：在canonical space下对SMPL顶点进行聚类分组，避免observation space下因姿态变化导致的语义歧义。

## 可复现要素
- **数据集**：ZJU-MoCap和H36M，论文公开详细split信息（见附录）
- **代码/权重**：项目主页https://pansanity666.github.io/TransHuman/，论文未明确说明代码开源状态
- **关键超参**：$N_t=300$（token数量），$N_k=7$（k近邻数），$N_v=3$（参考视角数），每射线64采样点，ResNet-18（前3层）+ ViT-Tiny，$\lambda=0.1$（perceptual loss权重）
- **训练分辨率**：512×512
