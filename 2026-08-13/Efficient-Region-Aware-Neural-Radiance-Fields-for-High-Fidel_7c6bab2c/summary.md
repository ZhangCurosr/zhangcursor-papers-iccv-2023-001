---
title: "Efficient-Region-Aware-Neural-Radiance-Fields-for-High-Fidel"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Efficient_Region-Aware_Neural_Radiance_Fields_for_High-Fidelity_Talking_Portrait_Synthesis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:04:39"
---

# 论文速读：Efficient-Region-Aware-Neural-Radiance-Fields-for-High-Fidel

## 一句话总结
本文提出ER-NeRF，一种面向听觉驱动说话肖像合成的条件NeRF框架，通过Tri-Plane Hash Representation降低3D动态头部建模的哈希碰撞，并结合Region Attention Module显式建立语音特征与空间区域的跨模态关联，在保持紧凑模型规模的同时实现实时渲染、快速收敛与SOTA级高保真合成。

## 研究问题与动机
- **核心问题**：现有NeRF-based听觉驱动方法（如RAD-NeRF）在实时性优化过程中，仍受限于3D哈希网格在动态人脸场景下的严重碰撞问题，以及音频-区域关联仅靠大型MLP隐式学习的低效性。
- **动机1**：动态头部体渲染仅表面区域贡献有效信息，大量空间区域为空，可通过正交二维平面分解显式剪枝，降低冗余采样与解码器负担。
- **动机2**：不同面部区域对语音的响应差异显著（如唇部敏感、背景/躯干不敏感），现有方法缺乏对这种非均匀区域贡献的显式建模。
- **动机3**：头颈分离时复杂位姿变换直接作为条件易导致躯干渲染错位，需将姿态映射为更直观的空间坐标以辅助torso-NeRF隐式学习。

## 核心贡献（创新点）
- **Tri-Plane Hash Representation**：将3D特征体素正交分解为XY、YZ、XZ三个2D多分辨率哈希网格，将碰撞复杂度从$O(R^2N)$降至$O(R^2+2RN)$，使MLP解码器更专注于音频条件与动态表面重建；与Instant-NGP/RAD-NeRF依赖MLP容忍碰撞的本质区别在于主动降维剪枝。
- **Region Attention Module**：提出基于External Attention的跨模态通道级加权机制，利用几何特征生成区域感知向量$\mathbf{v}$对音频特征$\mathbf{a}$进行逐通道调制（$\mathbf{a}_r = \mathbf{v} \odot \mathbf{a}$），显式建立音频与空间位置的映射；与纯concatenation+大MLP隐式学习的区别在于“按区域动态增强/抑制条件信息”，静态区域自动趋零降噪。
- **Adaptive Pose Encoding**：将头部位姿逆变换作用于$N=3$个可训练3D关键点，投影为2D图像坐标后作为torso-NeRF的条件输入，以极简方式解决头身分离时的姿态对齐问题。

## 方法详解
- **Tri-Plane Hash Representation**：对
