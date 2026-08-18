---
title: "Compatibility-of-Fundamental-Matrices-for-Complete-Viewing-G"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Bratelund_Compatibility_of_Fundamental_Matrices_for_Complete_Viewing_Graphs_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:31:17"
---

# 论文速读：Compatibility-of-Fundamental-Matrices-for-Complete-Viewing-G

## 一句话总结
本文从代数几何视角系统刻画了多视图基本矩阵集合的兼容性条件，针对完全图给出了仅含基本矩阵与其极线的显式齐次多项式充要条件，证明四元组兼容即可保证任意规模相机图的全局兼容，并指出先前基于 $n$ 视图基本矩阵秩与特征值的条件在一般与共线情形下是冗余的。

## 研究问题与动机
- **核心问题**：给定一组基本矩阵 $F^{ij}$（本文以完全图为主），是否存在一组相机矩阵 $P_1,\dots,P_n$ 使得每对相机恰好生成对应的 $F^{ij}$？即“兼容性（compatibility）”的判定与显式刻画。
- **现有方法不足**：
  1. $n=3$ 的经典三元组条件无法直接推广至 $n\geq4$，领域内长期误信“三元组兼容即全局兼容”，本文通过反例纠正。
  2. Kasten 等 [15] 与 Geifman 等 [9] 的充要条件依赖 $n$-view fundamental matrix 的秩与特征值，且必须预先固定每个矩阵的缩放比例，**不具备射影不变性**（not projectively well-defined）。
  3. 实际多视图 SFM 管线需处理完整或部分基本矩阵集合，缺乏显式代数约束使得全局优化难以严格保证几何一致性。

## 核心贡献（创新点）
- **引入基本矩阵作用（Fundamental Action）**：定义 $\{F^{ij}\}\mapsto\{H_i^T F^{ij} H_j\}$ 的群作用，证明其保持兼容性，从而可在保射影等价的前提下将矩阵标准化，大幅简化后续推导。
- **给出完全图的显式多项式充要条件**：将 $K_3$、$K_4$ 及一般 $K_n$ 的兼容性完全用基本矩阵 $F^{ij}$ 与极线 $e_j^i$ 表达的齐次多项式刻画，彻底摆脱对人为缩放的依赖。
- **证明四元组兼容性蕴含全局兼容性**：对任意 $n\geq4$，若所有相机四元组的六对基本矩阵均兼容，则整体兼容，将指数级约束降维至常数局部检查。
- **修正并简化先前理论**：借助符号计算验证并严格证明，在一般位置与全部极线重合的共线情形下，[15,9] 中的特征值条件可被删除，仅需秩与循环条件。
- **提出适用于任意视角图的循环定理**：将兼容性等价转化为图中所有有向环的加权基本矩阵之和为零，建立与平行刚性（parallel rigidity）理论的代数同构。

## 方法详解
- **基本矩阵作用（Fundamental Action）**：对变换 $(H_1,\dots,H_n)\in\mathrm{GL}_3^n$，定义 $\tilde{F}^{ij}=H_i^T F^{ij} H_j$。由 Proposition 2.1 知兼容性与否不变，结合 $\mathrm{PGL}_4$ 作用相机空间，可将各像内的极线标准正交化，从而将复杂几何问题转化为纯代数消元。
- **极线数（Epipolar Numbers）**：定义 $\mathbf{e}_{sijt}=(e_i^s)^T F^{ij} e_j^t$，在基本矩阵作用下不变（Lemma 2.4）。几何意义为：$\mathbf{e}_{sijt}=0$ 当且仅当四个相机中心 $c_s,c_i,c_j,c_t$ 共面（Lemma 2.5）。
- **$K_3$ 情形**：
  - 非共线：极线互异 + 三元组条件 $(e_1^3)^T F^{12} e_2^3 = (e_1^2)^T F^{13} e_3^2 = (e_2^1)^T F^{23} e_3^1 = 0$（Theorem 3.2）。
  - 共线：给出新充要条件 $e_1^2=e_1^3$ 等 + $(F^{12})^T[e_1^2]_\times F^{13}=F^{23}$（Proposition 3.4），纠正了 [11] 的断言。
- **$K_4$ 情形（按极线构型分类）**：
  - **Case 1（一般位置）**：各像内三极线不共线，充要条件为三元组条件 + 一个六次极线数等式（Theorem 3.6, Eq. 15）。
  - **Case 2（四中心共面）**：各像内极线共线但互异，需附加内积与范数组合条件（Theorem 3.8, Eq. 21, 22）。
  - **Case 3（恰三中心共线）**：部分极线重合，给出含交叉比结构的有理方程（Theorem 3.10, Eq. 24, 25）。
  - **Case 4（四中心共线）**：所有极线重合，直接退化为 $K_3$ 共线条件。
- **$K_n$ 推广**：Theorem 3.12 证明任意 $n\geq4$ 时，所有四元组兼容 $\Rightarrow$ 全局兼容；若所有极线重合则三元组兼容已足够，重建出的相机中心共线。
- **循环定理（Theorem 4.1）**：兼容的充要条件是存在 $H_i\in\mathrm{GL}_3$ 与对称非零标量 $\lambda_{ij}$，使得 $G^{ij}=\lambda_{ij}H_i^T F^{ij} H_j$ 满足对图 $\mathcal{G}$ 每条有向环 $C$ 均有 $\sum_{(ij)\in E(C)} G^{ij}=0$。由此可直接推出 $\mathrm{rank}(\mathbf{F})\leq6$，并重新导出三元组与 $K_4$ 的必要条件。

## 实验与结果
- 本文属纯代数几何理论工作，**无实证数据集与定量 Benchmark**。
- 提供两个显式反例：Example 3.3 驳斥“共线相机仅需三元组条件”的错误断言；Example 3.5 驳斥“三元组兼容即保证 $K_4$ 全局兼容”的长期直觉，验证六个矩阵满足 Theorem 3.2 但仍无解。
- 借助计算机代数系统 **Macaulay2** [10] 完成符号消元与理想基计算，验证 $K_4$ 多项式条件的完备性，并证明 [15,9] 的特征值条件在一般与共线情形下冗余（Theorem 3.16）。
- 所有结论以严格定理形式呈现，未报告数值误差、运行时间或消融实验。

## 相关工作脉络
- **Hartley & Zisserman [12]**：奠定 $n=3$ 三元组兼容的经典代数条件；本文将其系统化推广至 $n\geq4$，并明确区分布尔共线/非共线构型。
- **Kasten 等 [15] (GP-SfM)**：提出基于 $n$-view fundamental matrix 秩与
