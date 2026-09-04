---
title: "HyperReenact-One-Shot-Reenactment-via-Jointly-Learning-to-Re"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Bounareli_HyperReenact_One-Shot_Reenactment_via_Jointly_Learning_to_Refine_and_Retarget_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:16:39"
---

# 论文速读：HyperReenact-One-Shot-Reenactment-via-Jointly-Learning-to-Re

## 一句话总结
提出 HyperReenact，一种基于预训练 StyleGAN2 与超网络（Hypernetwork）的单帧人脸重演方法，通过联合学习身份特征精炼与目标姿态重定向，在极端头部姿态变化及跨主体场景下生成无伪影的真实 talking head 图像。

## 研究问题与动机
1. **极端姿态下的严重伪影**：现有 SOTA 方法在单帧（one-shot）设置下容易产生明显视觉伪影，尤其在源/目标头部姿态差异较大时失效。
2. **身份保持依赖昂贵微调**：多数方法需依赖配对数据或 few-shot fine-tuning（多视角源图像）才能忠实保留源身份特征，限制了实际泛化能力。
3. **外部反演模块的性能瓶颈**：利用预训练 GAN 的方法虽能解耦身份与姿态，但严重依赖外部 GAN inversion 模块，受限于“重建质量-可编辑性”权衡（reconstruction-editability trade-off），全局姿态编辑时伪影突出。
4. **既有超网络方法不适配重演**：HyperStyle / HyperInverter 等
