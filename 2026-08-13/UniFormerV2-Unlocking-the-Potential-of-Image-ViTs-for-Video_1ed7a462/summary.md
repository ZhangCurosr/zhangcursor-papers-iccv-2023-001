---
title: "UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_UniFormerV2_Unlocking_the_Potential_of_Image_ViTs_for_Video_Understanding_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:31:33"
field: "视频理解与动作识别"
keywords: ["Vision Transformer", "Video Understanding", "Action Recognition", "Pretrained ViT", "Spatiotemporal Modeling"]
innovations: ["Local UniBlock分离局部时序与全局空间建模，复用预训练ViT空间表征", "Query-based Cross MHRA以O(L)复杂度聚合全时空token", "Kinetics-710去重合并策略以33%训练成本实现更强泛化"]
benchmarks: ["Kinetics-400/600/700", "Something-Something V1/V2", "Moments in Time", "ActivityNet", "HACS"]
---

# 论文速读：UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video

## 一句话总结
论文将预训练的图像ViT与UniFormer的高效设计相结合，通过重新设计局部时序与全局跨注意力模块，使图像ViT无需从头训练即可直接在视频理解任务中达到SOTA性能，首次在Kinetics-400上突破90% top-1准确率。

## 研究问题与动机
- **图像-视频域差距大**：纯图像ViT缺乏时序建模能力，在时序相关任务（如Something-Something）上显著落后于CNN方法。
- **专用视频模型扩展受限**：UniFormer等专用架构需要大量图像预训练才能迁移到视频，难以利用开源大模型快速迭代。
- **插件式适配器效果有限**：TimeSformer、ST-Adapter等模块虽能提升时序建模，但在复杂动作识别上仍明显弱于专用模型。
- **开源图像ViT潜力未释放**：CLIP、MAE等预训练模型性能强大，但缺乏适配视频的轻量级设计。

## 核心贡献（创新点）
1. **Local UniBlock**：在预训练ViT块前插入局部时序MHRA，仅用低成本卷积式聚合消除时序冗余，同时继承ViT的空间表征能力。
2. **Global UniBlock（Query-based Cross MHRA）**：用单个可学习query与全部时空token做cross attention，将复杂O(L²)全局聚合降至O(L)，高效提取视频级表征。
3. **Multi-Stage Fusion Block**：提出顺序/并行/层级KV/Q四种融合策略，实证简单顺序融合即可获得最优性价比。
4. **Kinetics-710统一训练集**：合并K400/600/700训练集并去重，训练视频从1.14M降至0.66M，节省33%成本的同时提升泛化。
5. **首个K400 90%突破**：UniFormerV2-L在CLIP-ViT+K710 post-pretraining设置下达90.0% top-1，超越 MTV-H、CoCa等参数量大数倍的方法。

## 方法详解
- **整体流程**：输入视频经3D卷积（3×16×16）投影为时空token，依次通过若干Local UniBlock + Global UniBlock，最终多阶段融合输出分类。
- **Local UniBlock**：先经LT_MHRA（局部时序关系聚合，affinity矩阵大小为t×1×1管状区域）捕捉相邻帧运动，再经GS_MHRA（全局空间自注意力）保留ViT原有表征，最后过FFN。
- **Global UniBlock**：加入DPE（动态位置编码，3×3×3 depthwise时空卷积）后，用learnable query q与所有时空token做cross attention，输出单条视频token；计算复杂度从O(L²)降至O(L)。
- **Multi-Stage Fusion**：每个stage的Global UniBlock输出视频tokenX_i^G，本文验证顺序融合（上一stage输出作当前stage query）效果最佳（SSV2 69.5%），优于并行拼接或多query设计。
- **最终分类**：取最后一stage Local Block的class token F^C 与融合结果F加权求和：Z = α·F + (1-α)·F^C，α由Sigmoid学习。

## 实验与结果
- **数据集**：Kinetics-400/600/700（场景类）、Moments in Time（异构类）、Something-Something V1/V2（时序类）、ActivityNet与HACS（未裁剪长视频）。
- **关键结果**：
  - K400：UniFormerV2-L (CLIP+K710) 达**90.0%** top-1，首破90%门槛；B模型85.6%。
  - K600：L模型90.1%，K700：82.7%。
  - SSV2：L模型73.0%，V1：62.7%（新SOTA）。
  - MiT：L模型47.8%。
  - ActivityNet：94.7%；HACS：95.5%。
- **效率优势**：相较MViTv2-L（86.1% K400 / 42.4 TFLOPs），UniFormerV2-L以8.0 TFLOPs实现89.3%；相较MTV-H（89.1% / 1000+参数），仅用35%参数量。
- **消融结论**：Global块对场景类关键，Local块对时序类关键；temporal downsampling×2 +双倍输入帧可在同FLOPs下扩大时序感受野；DPE与中间层特征对时序区分均必要。

## 相关工作脉络
- **MViTv1/V2、VideoSwin**：专用分层视频Transformer，依赖长时图像预训练；UniFormerV2直接利用开源图像ViT，无需从头预训。
- **VideoMAE**：自监督MAE在视频上的扩展，需1600 epoch训练；UniFormerV2复用CLIP等预训练权重，仅少量finetune即可。
- **ST-Adapter、EVL**：基于Adapter的轻参数视频适配方法；本文以轻量MHRA替代普通depthwise卷积，在SSV2上获得明确增益（69.1% vs 68.0%）。
- **X-CLIP**：引入文本监督增强视频识别；UniFormerV2纯视觉路径即超越或匹敌，说明时空结构改进足以弥补跨模态缺失。
- **原始UniFormer**：同一团队前作，MHRA统一卷积与注意力但需从头预训练；V2将其拆解为可插拔模块，无缝对接预训练ViT。

## 局限性与未来方向
- 最大模型（L，64帧）需75.3 TFLOPs，离实际部署仍有距离；B模型虽高效但性能略低。
- 仅验证了单一视觉分支，未探索与语言/多模态模型的联合训练（如CLIP文本分支的利用）。
- 未在dense预测任务（检测、分割）上系统评估，泛化范围待扩展。
- 未来可探索更轻量的时序聚合方式、与MAE等自监督预训练的融合，以及向视频生成/多模态方向的迁移。

## 研究启发与可借鉴点
1. **时空分离设计范式**：将时序建模（局部管状MHRA）与空间建模（预训练ViT）解耦，为后续 ViT-to-Video 适配提供通用模板。
2. **Query-based 全局聚合**：用单个 learnable query 替代全token自注意力，显著降复杂度且保留全图信息，可复用于其他长序列视频任务（检测、分割）。
3. **数据集去重合并策略**：合并多个Kinetics变体并剔除重复/泄露视频，以更少数据获得更强泛化，值得在多基准联合训练中推广。
4. **可学习融合权重α**：class token与全局token的Sigmoid加权求和，实现细粒度与全局信息的自适应平衡，简单有效。
5. **与团队方向结合机会**：可将Local/Global UniBlock作为插件接入本团队的视觉-语言预训练管线，或在时序定位/动作分割任务上验证其dense预测能力。

## 关键术语表
- **MHRA（Multi-Head Relation Aggregator）**：UniFormer核心算子，统一卷积（局部）与自注意力（全局）的关系聚合方式。
- **LT_MHRA / GS_MHRA**：分别指局部时序MHRA与全局空间MHRA，前者建模t×1×1管状关系，后者为标准ViT空间注意力。
- **DPE（Dynamic Position Embedding）**：3×3×3 depthwise时空卷积，为token注入3D位置信息。
- **Cross MHRA**：Global UniBlock中的query-to-all跨注意力聚合，复杂度O(L)。
- **K710（Kinetics-710）**：合并K400/600/700并去重的统一训练集，含0.66M视频、710个动作类别。
- **SSV2**：Something-Something V2，侧重细粒度时序交互的动作识别数据集。
- **ST-Adapter**：将时序depthwise卷积作为Adapter引入ViT的参数高效迁移方法。
- **Sparse Sampling**：稀疏帧采样策略（如8×3×1），降低时序冗余同时保持计算效率。

## 可复现要素
- **代码**：https://github.com/OpenGVLab/UniFormerV2（已开源）。
- **预训练权重**：基于CLIP-ViT-B/L、MAE、BEiT等公开权重微调，权重随代码提供。
- **数据集**：Kinetics、Something-Something、ActivityNet、HACS、Moments in Time均为公开基准；K710为作者自行构建的去重合并集，可在原数据集基础上按论文描述复现。
- **关键超参**：输入分辨率224，稀疏采样"8×3×1"（K400）/ "16×1×3"（SSV）；BN用于Local MHRA前，LN用于Global MHRA与FFN前；全局块插入最后4层（SSV用最后8/16层）。
- **训练细节**：详见论文补充材料（Supplemental）。
