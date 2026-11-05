---
title: "MARS-Model-agnostic-Biased-Object-Removal-without-Additional"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Jo_MARS_Model-agnostic_Biased_Object_Removal_without_Additional_Supervision_for_Weakly-Supervised_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:00:18"
---

# 论文速读：MARS-Model-agnostic-Biased-Object-Removal-without-Additional

## 一句话总结
本文提出MARS，一种无需额外数据与人工标注的模型无关（model-agnostic）去偏框架，首次将无监督语义分割（USS）引入弱监督语义分割（WSSS），通过聚类特征与背景距离度量自动分离并剔除伪标签中的偏置对象，显著提升分割精度。

## 研究问题与动机
- WSSS依赖图像级标签生成初始CAMs，假阳性（FP）是制约其与全监督性能差距的核心瓶颈，其中“偏置对象”（如`train`类对应的铁轨、`boat`类对应的海面）占比极高。
- 现有去偏方法（如CLIMS、W-OoD）需借助CLIP文本先验逐类手动识别偏置对象，或依赖人工从Open Images收集纯偏置图像，难以扩展至包含近两万个类别的真实复杂场景。
- 偏置问题本质上源于图像级监督的模糊性，而USS因无类别先验，其特征中偏置干扰更少，具备将目标对象与背景偏置分离的潜力。
- 传统WSSS与USS研究长期独立发展，缺乏利用USS语义一致性特征自动化修正WSSS伪标签的系统性方案。

## 核心贡献（创新点）
1. 首次将USS特征引入WSSS用于自动去偏：通过K-means聚类USS嵌入向量分离同类别内的偏置与目标区域；与以往WSSS内部优化策略不同，本文跨范式借用USS的无偏特征空间实现自动化对象解耦。
2. 提出基于背景距离的前景中心度量准则：计算各前景聚类中心与全量背景中心的余弦距离，距离越短越判定为偏置对象；区别于CLIMS依赖文本匹配、W-OoD依赖人工收集的监督需求，该准则完全数据驱动且零额外标注。
3. 设计模型无关的即插即用去偏框架：将去偏流水线作为后处理与训练策略模块直接插入现有WSSS管线；与绑定特定骨干网络或预训练权重的方法不同，本文可无缝适配IRNet、SEAM、AdvCAM、RS+EPM等多种主流基线。
4. 提出在线互补训练与确定性加权损失：用EMA教师网络动态预测填补被剔除的偏置像素，配合`certain mask`控制训练置信度；区别于静态伪标签修正，该方法在训练过程中实时修复因激进去偏引入的假阴性（FN）。

## 方法详解
MARS整体分为四个阶段：
- **训练设置**：WSSS网络与USS模型均在目标数据集上从零训练。WSSS输出含偏置的伪标签$Y_i^b$，冻结的USS网络输出像素级嵌入$F_i \in \mathbb{R}^{D \times H \times W}$（不含类别特定信息）。
- **去偏中心选择**：对每类前景像素的$F_i$执行K-means聚类（设$K_{fg}=2$，$K_{bg}$可变量），得到图像级中心$v_i^c$。计算每个前景中心与全量背景中心（共$N^{bg}=N \cdot K_{bg}$个）的平均距离：$dist_k^c = \frac{1}{N^{bg}}\sum_{j=0}^{N^{bg}} D(v_k^c, v_j^0)$，其中$D$为余弦距离。按距离降序排列后，取Top $\alpha\%$中心的均值作为去偏中心$\mathcal{V}^c$。
- **生成去偏标签**：计算$\mathcal{V}^c$与$F_i$的像素级相似度，结合图像内真实类别集合$\mathcal{C}_{I_i}$，通过ReLU与逐像素最大值生成二值去偏掩码$\hat{M}_i^{db}$，经CRF细化得$M_i^{db}$。将$Y_i^b$中“前景但$M_i^{db}=0$”的像素标记为`-1`（偏置类），得到去偏标签$Y_i^{db}$。
- **在线互补训练**：引入EMA教师网络生成预测$Y_i^{te}$，将$Y_i^{db}$中的`-1`位置替换为$Y_i^{te}$对应类别，得到互补标签$Y_i^{co}$。定义确定性掩码$W_i$为教师网络在所有前景类上的最大概率像素图，采用加权交叉熵损失$\mathcal{L}_{WCE} = -\sum_{c,y,x} W_i(y,x) \cdot O[Y_i^{co}](c,y,x) \log P_i^\theta(c,y,x)$进行训练，避免早期不确定预测污染监督信号。

## 实验与结果
- **数据集**：PASCAL VOC 2012（21类）、MS COCO 2014（81类）。
- **评估指标**：mIoU、FP、FN；VOC结果取自官方在线评测服务器。
- **最强结果**：在PASCAL VOC 2012 val/test上分别达到**77.7% / 77.2%**（ResNet-101），在MS COCO 2014 val上达到**49.4%**，刷新WSSS SOTA，且与全监督FSSS（VOC test 81.0%）的差距缩小至约3.8个百分点。
- **基线提升**：对IRNet（+6.3%）、SEAM（+6.3%）、AdvCAM（+3.3%）、RS+EPM（+3.3%）均实现稳定提升；偏置敏感类`boat`提升9.0%，`train`提升29.0%。
- **消融结论**：加入互补策略后mIoU提升3.5%，但引入WCE损失后进一步稳定至最优（81.8% vs 80.9%），证明确定性掩码有效抑制了互补过程中的假阳性反弹；超参$K_{bg} \in [1,5]$表现稳定，$\alpha < 0.5$时效果最佳。

## 相关工作脉络
- **WSSS主流管线**（SEAM, CPN, PPC, RS+EPM等）：聚焦于扩展CAMs前景区域以缓解FN，未系统处理FP中的跨类别背景偏置；本文方法与其正交，可作为即插模块叠加。
- **去偏相关工作**（CLIMS, W-OoD）：CLIMS依赖CLIP文本先验需逐类人工确认偏置对象；W-OoD需从
