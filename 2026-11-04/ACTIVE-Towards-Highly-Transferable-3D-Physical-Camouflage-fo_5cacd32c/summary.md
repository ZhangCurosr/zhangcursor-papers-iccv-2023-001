---
title: "ACTIVE-Towards-Highly-Transferable-3D-Physical-Camouflage-fo"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Suryanto_ACTIVE_Towards_Highly_Transferable_3D_Physical_Camouflage_for_Universal_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:33:58"
field: "计算机视觉安全"
keywords: ["对抗性伪装", "3D物理攻击", "可迁移性", "三平面映射", "目标检测防御", "对抗鲁棒性"]
innovations: ["首次将三平面映射引入对抗性纹理生成，实现对象无关的通用纹理", "提出Stealth Loss同时最小化分类置信度和物体存在性得分使目标不可检测", "在五维可迁移性（instance/class/model/task/domain）上全面验证泛用性"]
benchmarks: ["CARLA+UE4物理仿真", "YOLOv3/YOLOv7/SSD/Faster R-CNN/Mask R-CNN", "MaX-DeepLab-L/Axial-DeepLab分割模型", "真实世界1:10缩比Tesla Model 3实验"]
---

# 论文速读：ACTIVE-Towards-Highly-Transferable-3D-Physical-Camouflage-fo

## 一句话总结
ACTIVE 提出了一种面向3D车辆的高可迁移物理对抗性伪装攻击框架，通过引入三平面映射（Triplanar Mapping）、新型隐蔽损失（Stealth Loss）等创新技术，实现了跨车型、跨模型、跨任务和跨物理环境的泛用性对抗纹理生成，显著提升了攻击的 universality 与 robustness。

## 研究问题与动机
- **现有方法缺乏可迁移性**：多数对抗性伪装方法仅针对单一目标车型优化，更换车型后性能大幅下降，无法实现 instance-agnostic 的通用性。
- **纹理映射方式受限**：传统 UV mapping 等方法依赖对象特定的纹理坐标，导致生成的对抗纹理无法直接迁移到其他3D模型。
- **仅最小化单类别置信度**：现有方法通常只最小化特定目标类别的置信度，车辆可能仍能被检测（misclassification而非undetectable）。
- **物理模拟与实际部署差距大**：部分方法在仿真中表现良好，但在 UE4 物理模拟或真实场景中性能严重退化。

## 核心贡献（创新点）
1. **首次将三平面映射引入对抗性伪装生成**：采用 TPM 替代对象依赖的 UV 映射，使同一张纹理可泛化到任意车辆，区别于 DTA 基于简单投影的方式。
2. **提出 Stealth Loss 实现完全不可检测**：同时最小化分类置信度与物体存在性得分（objectness），使车辆从检测器中"消失"，而非仅误分类，区别于 DAS/FCA 仅关注类别置信度的做法。
3. **集成平滑损失与伪装损失提升自然性**：通过 TV loss 和 K-means 主导背景色匹配，生成类涂鸦的自然纹理，优于 DTA 的彩色马赛克风格。
4. **系统验证五维可迁移性**：在 instance-agnostic、class-agnostic（卡车/巴士）、model-agnostic（含 Transformer 模型）、task-agnostic（分割模型）、domain-agnostic（真实世界）五个维度全面证明泛用能力。

## 方法详解
- **三平面映射（Triplanar Mapping）**：从三个正交方向（X/Y/Z轴）将纹理投影到物体表面，利用深度图提取表面坐标和法向量，无需对象特定的 UV 展开，实现 texture-object 解耦，同一纹理可应用于任意车型。
- **Neural Texture Renderer（NTR）**：基于 DenseNet 的编码器-解码器网络，替代传统渲染器进行可微纹理渲染，保留物理特性（光照、反射）的同时支持梯度反传。
- **Stealth Loss（隐蔽损失）**：定义检测得分 $h_d(x) = h_c(x) \times h_o(x)$，其中 $h_c$ 为类别置信度，$h_o$ 为物体存在性得分；最小化 $L_{atk} = f_{log}(\max(h_d(x)))$，使车辆同时丧失分类置信度和存在性。
- **Random Output Augmentation（ROA）**：对渲染输出施加随机缩放、平移、亮度、对比度等数字变换，增强对抗纹理的数字和物理鲁棒性。
- **Smooth Loss（平滑损失）**：采用改进的 TV loss 约束相邻像素差异，公式为 $L_{sm}(\eta) = \frac{1}{N_{sm}} \sum f_{log}(|\eta_{i,j}-\eta_{i+1,j}|) + f_{log}(|\eta_{i,j}-\eta_{i,j+1}|)$。
- **Camouflage Loss（伪装损失）**：通过 K-means 提取背景主导色集 B，结合 NPS loss 形式迫使纹理颜色与背景融合，公式为 $L_{cm}(\eta, B) = \frac{1}{N_{cm}} \sum f_{log}(\min_{b \in B} |b - \eta_{i,j}|)$。
- **总损失**：$L_{total} = \alpha L_{atk} + \beta L_{sm} + \gamma L_{cm}$，默认 $\alpha=1.0, \beta=0.25, \gamma=0.25$，纹理分辨率 64×64。

## 实验与结果
- **数据集与环境**：基于 CARLA + UE4 构建物理仿真环境，选取5款车型用于训练、另5款用于泛化测试；NTR 训练使用 50,625 张（测试 150,000 张）照片级图像。
- **评估基线**：DAS、FCA、DTA、UPC、CAMOU、ER 等。
- **关键结果**：
  - **数字→物理仿真**：在 YOLOv3 上，ACTIVE 将汽车 AP@0.5 从 99.57% 降至 **7.29%**，远超 DTA（41.95%）和 FCA（92.28%）。
  - **多模型评估**：在 YOLOv3、SSD、Faster R-CNN、Mask R-CNN 上均取得最佳攻击效果（AP@0.5 最低）。
  - **跨模型泛化**：在 YOLOv7（新模型）上 AP@0.5 从 92.54% 降至 **41.55%**，显著优于 DTA（59.39%）。
  - **Transformer 模型**：在 PVT 和 DDTR 上同样有效（PVT 降至 14.69%），证明对 Transformer 架构同样具有威胁。
  - **跨类别**：在卡车和巴士上持续优于基线。
  - **跨任务**：在 MaX-DeepLab-L 和 Axial-DeepLab 分割模型上，Cityscape 预训练下像素准确率降至 **17.45%**（正常 90.70%）。
  - **真实世界**：1:10 比例 Tesla Model 3 实验中，在 YOLOv3 上 AP@0.5 从 90.83% 降至 **8.75%**，YOLOv7 降至 48.75%。
- **最强结果**：在 YOLOv3 目标模型上的数字→物理仿真转移后 AP@0.5 达 **7.29%**（对比 DTA 41.95%），提升幅度约 5.7×。

## 相关工作脉络
1. **CAMOU / ER（黑盒方法）**：早期对抗性伪装采用黑盒策略（克隆网络/遗传算法），缺乏白盒优化的精确性，性能显著低于本文方法。
2. **UPC（通用物理伪装）**：基于 patch 方法，虽提出通用性概念但受限于 patch-based 的固有缺陷，在多视角和多模型上可迁移性不足。
3. **DAS / FCA（早期神经渲染器方法）**：使用传统 NMR 渲染器，无法准确模拟光照反射等物理现象，在 UE4 物理仿真中性能大幅退化。
4. **DTA（最新基线）**：提出可微变换网络和物理特性渲染，但依赖简单纹理投影导致映射不准确，且生成纹理呈马赛克状不自然。
5. **本文定位**：在 DTA 基础上引入更精确的 TPM 纹理映射和新型 Stealth Loss，在保持物理仿真的同时实现五维可迁移性和更自然的纹理。

## 局限性与未来方向
- **纹理抽象性**：生成的纹理虽类涂鸦但仍显抽象，与真实环境融合度有待提升。
- **计算开销**：NTR 训练和纹理优化过程需要较多计算资源。
- **真实世界尺度限制**：当前真实世界实验仅使用 1:10 缩比模型，全尺寸部署效果需进一步验证。
- **防御潜力**：对 Transformer 架构的脆弱性揭示暗示未来可探索针对性的防御机制。

## 研究启发与可借鉴点
1. **三平面映射用于3D对抗攻击**：TPM 的 object-independent 特性可迁移至其他3D物体（行人、自行车等）的对抗纹理生成研究。
2. **Stealth Loss 的设计思路**：同时优化分类置信度和存在性得分的思路可推广至其他检测任务（如行人检测、骑行者检测）。
3. **多维度可迁移性评估范式**：instance/class/model/task/domain 五维评估体系可作为对抗性攻击论文的标准化实验设计参考。
4. **与视觉Transformer的交互研究**：本文发现 ViT 模型在物理对抗攻击下同样脆弱，这一结论对现有"ViT更鲁棒"的认知构成挑战，值得深入探究。
5. **K-means主导色匹配策略**：Camouflage Loss 中通过聚类提取背景色再匹配的思路，可迁移至其他需要隐蔽性的对抗攻击场景。

## 关键术语表
**Triplanar Mapping（三平面映射）**：从三个正交方向投影纹理到物体表面的映射技术，无需对象特定的UV展开，实现纹理与3D模型的解耦。

**Stealth Loss（隐蔽损失）**：同时最小化目标类别置信度和物体存在性得分的损失函数，使车辆从检测器中完全"消失"。

**Neural Texture Renderer（NTR）**：基于DenseNet的编码器-解码器网络，提供可微的纹理渲染以支持白盒对抗优化。

**Random Output Augmentation（ROA）**：对渲染输出施加随机缩放、平移、亮度、对比度等数字变换，增强对抗纹理的鲁棒性。

**Total Variation Loss（TV Loss）**：约束相邻像素差异的平滑损失，用于生成更自然的对抗纹理。

**Non-Printability Score Loss（NPS Loss）**：原用于限制可打印颜色的损失函数，本文改造为匹配背景主导色的伪装损失。

## 可复现要素
- **数据集**：基于 CARLA + UE4 自建，**未公开**（论文未提及开源数据集计划）
- **代码**：官网链接 https://islab-ai.github.io/active-iccv2023，**论文未明确声明是否开源**
- **关键超参**：纹理分辨率 64×64，Adam优化器，30轮迭代，α=1.0，β=0.25，γ=0.25，IoU阈值 t=0.5；ROA参数：亮度0.25、对比度[0.75,1.5]、缩放[0.25,1.0]、投影偏移[−0.5,0.5]、缩放[−0.25,0.25]
- **NTR训练**：DenseNet架构，20轮迭代，50,625训练图像/150,000测试图像
