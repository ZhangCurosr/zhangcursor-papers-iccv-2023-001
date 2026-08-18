---
title: "WaterMask-Instance-Segmentation-for-Underwater-Imagery"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:41:25"
field: "水下计算机视觉"
keywords: ["水下图像分割", "实例分割", "图注意力网络", "边界学习", "UIIS数据集"]
innovations: ["提出DSGAT通过图注意力重建退化特征细节", "设计BMS融合多尺度边界预测精化掩码", "构建首个通用水下实例分割数据集UIIS"]
benchmarks: ["UIIS", "COCO-style instance segmentation metrics"]
---

# 论文速读：WaterMask-Instance-Segmentation-for-Underwater-Imagery

## 一句话总结
本文提出了首个通用水下图像实例分割数据集UIIS（4628张图像、7个类别、像素级标注），并设计了首个专为水下图像定制的实例分割模型WaterMask，通过图注意力机制恢复退化细节、多尺度特征精炼和边界掩码策略实现高精度实例分割。

## 研究问题与动机
1. **数据集匮乏**：现有水下图像数据集多针对特定任务（如鱼检测、语义分割），缺乏支持多类别实例分割的通用数据集。
2. **水下图像质量退化**：波长和距离相关的衰减散射、浮游生物引起的海雪噪声导致图像质量严重下降，直接迁移自然图像分割算法效果不佳。
3. **密集聚类实例分割困难**：鱼群、珊瑚礁等实例高度聚集且相互遮挡，传统方法因多次下采样丢失边界细节。
4. **尺度变化大**：数据集包含极小实例（<14×14像素，占11.7%）和极大实例（>128×128像素，占22.8%），挑战传统分割框架。

## 核心贡献（创新点）
1. **构建首个通用水下实例分割数据集UIIS**：与现有特定目标数据集（如DeepFish仅针对鱼类）不同，覆盖7类通用水下对象，采用COCO格式便于广泛使用。
2. **提出差异相似图注意力模块DSGAT**：与BMask R-CNN等使用后验边界信息的不同，DSGAT通过图注意力在网络前端重建最高分辨率特征，从相似斑块聚合退化残留细节，本质区别在于"特征级细节恢复"而非"预测后边界优化"。
3. **设计多尺度特征精炼模块MFRM**：将DSGAT重建的全局细粒度特征与FPN的局部粗粒度特征迭代融合，不同阶段分别输出前景掩码和边界掩码，区别于传统单尺度预测。
4. **提出边界掩码策略BMS与边界学习损失BLL**：BMS通过拉普拉斯算子生成边界掩码并融合高低分辨率预测；BLL针对性增强边界像素的分类权重，解决水下图像模糊边界导致的过拟合问题。

## 方法详解
**整体框架**：WaterMask基于Mask R-CNN/Cascade R-CNN构建，由DSGAT、MFRM、BMS三模块组成。

**DSGAT（差异相似图注意力模块）**：
- 输入FPN最高分辨率层$P_2$（$h \times w \times c$），经$s \times s$步长卷积降采样为$P_2'$
- 将每行像素视为图节点$h_i \in \mathbb{R}^d$，每个节点对应原图$4s \times 4s$像素块
- 仅连接每个节点距离最远的$k$个邻居，降低计算复杂度
- 注意力系数计算：$a_{ij} = \frac{\exp(\sigma(l^T[W h_i \| W h_j]))}{\sum_{n \in \mathcal{N}_i} \exp(\sigma(l^T[W h_i \| W h_n]))}$
- 特征更新：$h_i' = \delta(\sum_{n \in \mathcal{N}_i} a_{ij} W h_j)$，$\delta$为ELU激活
- 反卷积上采样生成残差流$P_{res}$补充丢失细节

**MFRM（多尺度特征精炼模块）**：
- 初始特征：14×14 RoIAlign + 两个$3 \times 3$卷积生成$F_1$
- 迭代精炼：每次从$P_2^*$提取局部细粒度特征，与前阶段特征拼接后经$1 \times 1$卷积（通道减半）和$2 \times$上采样
- 执行两次迭代，输出$F_2$（前景预测）和$F_3$（边界预测）
- 特点：$1 \times 1$卷积减半通道后上采样，不增加参数开销

**BMS（边界掩码策略）**：
- 拉普拉斯边界生成：$\nabla^2 p$卷积核定义$p_{ij} = b^2-1$（中心）或$-1$（其他）
- 边界判定：$B(M) = 1$若$|\nabla^2 p(M)| \leq \mu b^2$，否则0（$\mu=0.15$）
- 最终输出：$M_{out} = f_{2\times}(M_2) \odot B_{2\times} + M_3 \odot (1 - B_{2\times})$

**边界学习损失BLL**：
- $\mathcal{L}_B = \frac{\sum_i R_{2\times}^i \cdot BCE(M_3^i, G_3^i)}{\sum_i R_{2\times}^i}$，其中$R_{2\times}^i = f_{2\times}(B(M_2) \vee B(G_2))$
- 总掩码损失：$\mathcal{L}_{mask} = \mathcal{L}_B + \sum_{k \in [1,2]} \lambda_k \mathcal{L}_{BCE}(M_k, G_k)$，$\lambda_1=0.25, \lambda_2=0.65$

## 实验与结果
**数据集划分**：UIIS共4628张图像，3937张训练，691张验证，7个类别（Fish、Reefs、Aquatic plants、Wrecks/ruins、Human divers、Robots、Sea-floor）。

**评估指标**：标准Mask AP（mAP、AP50、AP75、APS、APM、APL），以及按类别划分（f=鱼类、h=潜水员、r=沉船）。

**基线对比结果**（3× schedule + ResNet-101-FPN）：
| 方法 | mAP | AP50 | AP75 | APS | APM | APL | APf | APh | APr |
|------|-----|------|------|-----|-----|-----|-----|-----|-----|
| Mask R-CNN | 23.4 | 40.9 | 25.3 | 9.3 | 19.8 | 32.5 | 43.6 | 49.0 | 18.0 |
| **WaterMask R-CNN** | **27.2** | 43.7 | **29.3** | 9.0 | 21.8 | **38.9** | 46.3 | **54.8** | 20.9 |
| Cascade Mask R-CNN | 25.5 | 42.8 | 27.8 | 7.5 | 20.1 | 35.0 | 43.9 | 52.9 | 22.3 |
| **Cascade WaterMask R-CNN** | **27.1** | 42.9 | **30.4** | 8.3 | 21.0 | **38.9** | **47.0** | **55.8** | **22.5** |

**提升幅度**：
- WaterMask R-CNN较Mask R-CNN：ResNet-50提升2.9 mAP，ResNet-101提升3.8 mAP
- 严格IoU阈值下（AP80）提升达6.1 mAP
- 在QueryInst、Mask2Former等Transformer方法中保持竞争力（APf领先Mask2Former 5.9点）

**消融实验**（ResNet-101-FPN，1× schedule）：
- w/o DSGAT：mAP↓1.4（24.2），APS↓0.1，APL↓2.7
- w/o MFRM：mAP↓2.5（23.1），APL↓4.2
- w/o BMS：mAP↓3.1（22.5）
- w/o BLL：mAP↓1.7（23.9）

**超参敏感性**：
- Patch大小：12×12最优（mAP=25.6），8×8显存超限，16×16以上性能下降
- 邻居数k：k=11最优（mAP=25.6），k>11性能略降

## 相关工作脉络
1. **Mask R-CNN [12]**：两阶段实例分割基线，通过RoIAlign提取低分辨率特征预测掩码。WaterMask在其基础上加入DSGAT和BMS解决水下退化问题。
2. **Cascade Mask R-CNN [3]**：多阶段精化检测框提升定位精度。WaterMask同样适配此框架，展示模块通用性。
3. **BMask R-CNN [7]**：利用边界信息改进掩码对齐。与WaterMask区别在于BMask使用后验边界预测，而WaterMask在前端通过DSGAT主动恢复细节。
4. **PointRend [20]**：将分割视为渲染问题，自适应选择点预测。WaterMask采用多尺度特征融合策略，计算效率更高。
5. **QueryInst [9] / Mask2Former [6]**：基于Transformer的实例分割方法。WaterMask在经典CNN框架内通过图注意力和边界策略达到类似效果，参数量更低（67M vs 191M）。
6. **DeepFish [10]**：针对鱼类实例分割的专用数据集。UIIS更通用，覆盖7类对象，适合多场景应用。

## 局限性与未来方向
1. **小目标性能下降**：DSGAT重建对极小实例（<14×14）有一定负面影响（APS落后R³-CNN 0.7点），需进一步优化细节保留与小目标检测的平衡。
2. **数据集规模有限**：4628张图像对于深度学习训练相对较小，复杂水下场景覆盖不足。
3. **测试时边界核尺寸调整**：训练使用7×7卷积核，测试改用9×9，暗示超参需针对部署场景调优。
4. **未来方向**：作者提出扩展数据集至更大规模、更复杂水下探索场景，可考虑引入时序信息（视频分割）或3D点云数据。

## 研究启发与可借鉴点
1. **图注意力用于特征恢复**：DSGAT将图像块建模为图节点、仅连接最远k个邻居的策略，为低质量图像细节恢复提供了高效方案，可迁移至医学影像、遥感图像等退化场景。
2. **边界感知分割的通用设计**：BMS通过拉普拉斯算子显式生成边界掩码并融合多尺度预测，BLL针对性强化边界像素权重，这种"预测-边界生成-融合"范式适用于任何边界敏感任务（如细胞分割、遥感地物 delineation）。
3. **多阶段特征精炼不增参设计**：MFRM通过1×1卷积减半通道后再上采样，在增加感受野的同时不引入额外参数，可在资源受限场景中复用。
4. **数据集构建流程**：UCIQE/UIQM质量筛选→多人标注→第三方评审→共识过滤的流程，为领域专用数据集建设提供了规范化参考。

## 关键术语表
**UIIS（Underwater Image Instance Segmentation）**：首个通用水下图像实例分割数据集，包含4628张RGB图像、7个类别、像素级COHO格式标注。
**DSGAT（Difference Similarity Graph Attention Module）**：差异相似图注意力模块，通过图神经网络在最高分辨率特征层聚合相似图像块信息，恢复因退化和下采样丢失的细节。
**MFRM（Multi-level Feature Refinement Module）**：多尺度特征精炼模块，迭代融合DSGAT重建的细粒度特征与FPN粗粒度特征，分别生成前景掩码和边界掩码。
**BMS（Boundary Mask Strategy）**：边界掩码策略，利用拉普拉斯算子生成边界判定矩阵，融合高低分辨率掩码预测以精化边界。
**BLL（Boundary Learning Loss）**：边界学习损失，通过边界区域权重重分配强化网络对模糊边界的分类能力，缓解水下图像退化导致的边界过拟合。
**AP/S/M/L**：按实例面积划分的平均精度，S（小，<32²）、M（中，32²~96²）、L（大，>96²）像素。
**RoIAlign**：Region of Interest Alignment，从特征金字塔提取固定大小实例特征的操作，保持空间对齐避免量化误差。

## 可复现要素
- **数据集**：UIIS公开可用，链接见论文GitHub
- **代码**：开源于https://github.com/LiamLian0727/WaterMask
- **预训练权重**：论文未明确提及，基于MMDetection标准实现
- **关键超参**：
  - DSGAT：patch大小12×12（stride=3），邻居数k=11
  - BMS：训练时$\nabla^2 p$核尺寸7×7，测试时9×9，$\mu=0.15$
  - 损失权重：$\lambda_1=0.25, \lambda_2=0.65$
  - 优化器：SGD，初始学习率2.5e-3，batch size=2/GPU
  - 训练策略：1×（12 epoch）和3×（36 epoch，含多尺度训练）
