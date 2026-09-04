---
title: "ACTIVE-Towards-Highly-Transferable-3D-Physical-Camouflage-fo"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Suryanto_ACTIVE_Towards_Highly_Transferable_3D_Physical_Camouflage_for_Universal_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:18:42"
field: "对抗机器视觉的物理攻击"
keywords: ["物理对抗攻击", "3D伪装", "目标检测", "对抗纹理", "可微渲染", "通用性评估"]
innovations: ["首次将三平面映射引入对抗性伪装生成，实现实例无关的通用纹理", "提出Stealth Loss同时最小化类别置信度和目标存在性得分实现完全不可检测", "通过Camouflage Loss和ROA联合优化提升自然性和物理鲁棒性"]
benchmarks: ["YOLOv3/YOLOv7", "SSD", "Faster R-CNN", "Mask R-CNN", "Deformable DETR", "PVT", "MaX-DeepLab-L", "Axial-DeepLab", "MobileNetV2", "EfficientDet-D2", "YOLOX-L"]
---

# 论文速读：ACTIVE-Towards-Highly-Transferable-3D-Physical-Camouflage-fo

## 一句话总结
本文提出 ACTIVE 框架，用于生成通用且鲁棒的 3D 物理对抗性伪装纹理，使自动驾驶车辆检测器从任意视角都无法检测到目标车辆。核心创新包括三平面映射纹理技术、Stealth Loss 和 Camouflage Loss，在 15 种检测模型和真实世界中均显著优于现有方法。

## 研究问题与动机
- 现有对抗性伪装方法（如 DAS、FCA）主要优化单一目标类的置信度分数，车辆可能仍被检测为其他类别，无法实现真正"不可见"；DTA 虽考虑全类攻击但纹理映射不够精确。
- 现有方法多依赖 UV 映射等对象相关纹理映射，导致生成的纹理无法迁移到其他车辆实例或类别。
- 现有方法的物理仿真性能不佳（FCA 在 UE4 中检测分从 52.05% 骤升至 92.28%），缺乏对光照反射、物理形变等现实因素的充分建模。
- 生成的伪装图案视觉上不自然（彩色马赛克），难以在实际场景中应用。

## 核心贡献（创新点）
1. **三平面映射（Triplanar Mapping）首次引入对抗性伪装生成**：通过表面坐标和法向量将纹理从三个正交方向投影，无需依赖对象的 UV 映射，实现实例无关（instance-agnostic）的通用攻击纹理。
2. **Stealth Loss 使目标完全不可检测**：同时最小化所有合法类别的置信度得分和目标存在性得分（objectness score），使检测器不仅误分类，而是直接认为画面中无目标。
3. **引入 Camouflage Loss 增强视觉隐蔽性**：通过 K-means 聚类提取背景主导色，结合 NPS loss 约束纹理颜色接近背景，使伪装图案既符合人类视觉又辅助对抗机器视觉。
4. **提出 ROA（Random Output Augmentation）模块**：在渲染后对对抗样本进行随机缩放、平移、亮度、对比度等数字增强，进一步模拟真实世界变化，这在现有伪装方法中未被考虑。
5. **在 15 种检测模型、跨类别/跨任务/跨域设置下验证高度可迁移性**：包括 Transformer 架构（PVT、Deformable DETR）和分割模型（MaX-DeepLab-L、Axial-DeepLab），以及真实世界 1:10 比例车辆实验。

## 方法详解
- **Triplanar Mapping（TPM）**：从深度图提取表面坐标和法向量，将纹理从三个正交方向（X/Y/Z）投影到 3D 物体表面，实现与具体纹理贴图无关的对象无关纹理应用。
- **Neural Texture Renderer（NTR）**：改进自 DTA 的 DTN，基于 DenseNet 架构构建四个编码器-解码器层，能更准确地保留物理特性（如光照反射），且训练数据减少 82%（仅需 9 种颜色即可泛化到 50 种随机颜色，SSIM=0.985）。
- **Stealth Loss（Latk）**：定义为 $L_{atk}(x) = f_{log}(\max(h_d(x)))$，其中检测得分 $h_d(x) = h_c(x) \times h_o(x)$（置信度 × 目标存在性），仅对 IoU > t 的有效框计算损失，$f_{log}(n) = -log(1-n)$，迫使检测器同时降低类别置信度和目标存在判断。
- **Smooth Loss（Lsm）**：改进版 Total Variation loss，$L_{sm}(\eta) = \frac{1}{N_{sm}}\sum f_{log}(|\eta_{i,j}-\eta_{i+1,j}|) + f_{log}(|\eta_{i,j}-\eta_{i,j+1}|)$，使相邻像素值接近，提升纹理平滑度和自然感。
- **Camouflage Loss（Lcm）**：$L_{cm}(\eta,B) = \frac{1}{N_{cm}}\sum f_{log}(\min_{b\in B}|b-\eta_{i,j}|)$，B 为通过 K-means 聚类提取的背景主导色集合，约束纹理颜色与背景相似。
- **总损失**：$L_{total} = \alpha L_{atk} + \beta L_{sm} + \gamma L_{cm}$，默认超参 $\alpha=1.0, \beta=0.25, \gamma=0.25$，IoU 阈值 $t=0.5$，纹理分辨率 64×64。
- **ROA 模块**：对渲染后的对抗样本施加随机亮度 [0.75,1.5]、对比度 [0.25,1.0]、缩放 [0.25,1.0] 及三平面投影随机平移/缩放增强。

## 实验与结果
- **数据集与环境**：基于 CARLA（UE4）物理仿真器，使用 10 种不同车辆（5 种用于优化，5 种用于泛化验证），50,625 张训练图像 + 150,000 张测试图像用于 NTR 训练。
- **关键数字**：
  - 数字→物理仿真转移（YOLOv3，Audi E-Tron）：ACTIVE 从 1.28% 降至 7.29% AP@0.5，显著优于 DTA（16.91%→41.95%）和 FCA（52.05%→92.28%）。
  - 多模型物理仿真攻击（目标 YOLOv3，黑盒测试）：ACTIVE 在单阶段检测器（YOLOv3/AP=19.52%，SSD/AP=33.56%）和双阶段检测器（FrRCNN/AP=41.70%，MkRCNN/AP=45.08%）上均最优。
  - **跨模型泛化（黑盒，YOLOv7 等 5 种新模型）**：ACTIVE 在 YOLOv7 上降至 41.55%（其他方法几乎无效），在 Transformer 模型 PVT 上降至 51.54%。
  - **跨类别（卡车/巴士）**：在所有检测器上均显著优于基线。
  - **跨任务（语义分割）**：在 MaX-DeepLab-L (Cityscape) 上像素准确率降至 17.45%，在 Axial-DeepLab (COCO) 上降至 23.46%。
  - **真实世界（1:10 Tesla Model 3）**：在 YOLOv3 上 AP 从 90.83% 降至 8.75%，在 YOLOv7 上降至 48.75%，在 EfficientDet-D2 上降至 26.25%。
- **消融实验**：仅用 Stealth Loss 时 AP=22.48%，加 Smooth Loss 后降至 20.21%；TPM+ROA 组合平均提升 35%。

## 相关工作脉络
- **CAMOU [45] / ER [41]**：早期黑盒方法，利用克隆网络或遗传算法优化纹理，无法保证最优攻击性能，且无全类不可检测能力。
- **UPC [16]**：基于 patch 的通用物理伪装攻击，受限于 patch 方法无法覆盖非平面 3D 表面，跨视角/跨模型泛化有限。
- **DAS [39] / FCA [35]**：基于神经网络渲染器的白盒方法，但 FCA 依赖传统渲染器（NMR），无法准确表达光照反射等物理特性，物理仿真中性能大幅退化。
- **DTA [30]**：本文直接前身，提出可微变换网络（DTN）和重复纹理投影（RTP），但纹理映射精度不足（简单投影无法处理非平面形状），且图案不自然。
- **Vision Transformer 脆弱性发现**：本文首次揭示 PVT、Deformable DETR 等 Transformer 架构在 3D 物理对抗伪装下同样脆弱，挑战了"Transformer 对对抗攻击更鲁棒"的既有认知。

## 局限性与未来方向
- **纹理抽象性**：生成的图案仍为抽象样式（类似涂鸦），虽比彩色马赛克更自然，但在真实场景中可能引起人类怀疑。
- **泛化场景有限**：目前实验主要在 CARLA 模拟器中进行，真实道路复杂光照和天气条件（雨雪、夜间）下的鲁棒性有待验证。
- **未来可探索方向**：结合更多物理现象建模（如阴影、雨滴）、面向人类视觉感知的隐蔽性优化、拓展至行人/骑行者等其他目标类别。

## 研究启发与可借鉴点
1. **三平面映射用于对抗纹理生成**：可作为通用的对象无关纹理优化策略，适用于任何需要"一张纹理适用于多个 3D 实例"的场景（如游戏、数字孪生）。
2. **Stealth Loss 设计思路**：同时最小化类别置信度和目标存在性得分，对追求"完全隐藏"而非"误分类"的任务具有借鉴价值（如隐私保护中的目标消除）。
3. **NTR 的训练效率优化**：仅用 9 种颜色训练即可泛化到 50 种随机颜色，且数据量减少 82%，为神经渲染器的低资源训练提供了可行方案。
4. **跨 Transformer 架构的对抗迁移性验证**：本文方法可用于进一步测试 ViT/DETR 类模型在物理空间中的鲁棒性，填补该方向评测空白。
5. **Camouflage Loss 的可迁移性**：将背景主导色约束引入对抗纹理优化，对图像隐藏、数字水印等任务有借鉴意义。

## 关键术语表
**Triplanar Mapping（三平面映射）**：从三个正交方向将纹理投影到 3D 物体表面的映射技术，无需依赖对象的 UV 展开，实现对象无关的纹理应用。

**Stealth Loss（隐匿损失）**：同时最小化目标检测器所有类别置信度和目标存在性（objectness）得分的损失函数，使目标从检测结果中完全消失。

**Neural Texture Renderer（NTR）**：基于 DenseNet 的神经纹理渲染器，通过可微渲染保留物理特性（如光照反射），用于对抗纹理的白盒优化。

**Random Output Augmentation（ROA）**：在神经渲染输出后施加随机亮度、对比度、缩放等数字增强，模拟真实世界变化以提升纹理鲁棒性的模块。

**IoU Threshold（交并比阈值）**：用于限定损失函数仅在检测框与真实框重叠度超过阈值时才生效的机制，避免对无效检测框施加梯度。

**Average Precision@0.5（AP@0.5）**：目标检测评估指标，表示在 IoU 阈值 0.5 下的平均精度，值越低表示检测器性能下降越多（攻击效果越好）。

## 可复现要素
- **数据集**：CARLA 模拟器合成数据（未公开原始数据集，实验环境依赖 CARLA+UE4）
- **代码/权重**：论文未明确声明开源，项目主页 https://islab-ai.github.io/active-iccv2023
- **关键超参**：Adam 优化器，30 轮迭代，α=1.0，β=0.25，γ=0.25，IoU 阈值 t=0.5，纹理分辨率 64×64，NTR 训练 20 轮
- **硬件**：论文未提及
