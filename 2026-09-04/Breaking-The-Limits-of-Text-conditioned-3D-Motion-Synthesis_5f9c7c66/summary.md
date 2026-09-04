---
title: "Breaking-The-Limits-of-Text-conditioned-3D-Motion-Synthesis"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Qian_Breaking_The_Limits_of_Text-conditioned_3D_Motion_Synthesis_with_Elaborative_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:19:33"
---

# 论文速读：Breaking-The-Limits-of-Text-conditioned-3D-Motion-Synthesis

## 一句话总结
提出 EMS（Elaborative Motion Synthesis）两阶段模型，将复杂长文本描述分解为原子动作序列分别生成，再通过上下文连接网络与 Natural Loss 实现平滑融合，在 KIT-ML 与 BABEL 基准上显著超越现有 SOTA。

## 研究问题与动机
1. **单一隐向量瓶颈**：现有方法将整段文本编码为单个潜在向量后直接解码，面对长段落或详细修饰时语义过载，且受 GPU 显存限制难以处理完整描述。
2. **忽略上下文依赖**：简单的自回归“逐句拼接”无法捕捉动作间的时序与语义关联（例如相同 “stand up” 在前置动作为 “sit down” 或 “squat down” 时姿态截然不同）。
3. **细粒度控制缺失**：多数工作仅支持动作标签（word-level）输入，无法显式建模速度、方向、身体部位等原子动作级属性。
4. **长序列生成困难**：公开数据集原子动作组合数量随描述长度呈指数增长，现有模型在长时、复杂动作上易出现抖动、悬浮或语义漂移。

## 核心贡献（创新点）
1. **两阶段 elaborative 生成框架**：首次将复杂动作分解为原子
