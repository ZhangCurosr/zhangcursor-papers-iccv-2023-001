---
title: "Learning-Image-Adaptive-Codebooks-for-Class-Agnostic-Image-R"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Learning_Image-Adaptive_Codebooks_for_Class-Agnostic_Image_Restoration_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:45:35"
field: "图像恢复与生成"
keywords: ["image restoration", "codebook", "super-resolution", "inpainting", "discrete generative prior", "class-agnostic"]
innovations: ["提出AdaCode，通过多basis codebook加权融合实现类无关离散生成先验", "将3D LUT自适应融合思想迁移到离散latent空间", "三阶段训练策略结合对比学习code-level损失提升恢复质量"]
benchmarks: ["OST", "Set5", "Set14", "BSD100", "Urban100", "Manga109", "DIV2K", "DIV8K"]
---

# 论文速读：Learning-Image-Adaptive-Codebooks-for-Class-Agnostic-Image-R

## 一句话总结
本文提出AdaCode，通过预训练一组类特定的basis codebooks并结合图像自适应权重图，构建类无关的离散生成先验，用于图像重建、超分辨率与inpainting等恢复任务，在多项基准上达到SOTA性能。

## 研究问题与动机
- 现有基于codebook的离散生成先验（如VQGAN、FeMaSR）需要针对每类图像单独训练，无法泛化到任意自然图像。
- 单通用codebook表达能力受限，难以刻画复杂图像中来自不同类别的纹理与结构内容。
- 图像恢复任务中退化类型多样，单一离散表示容易产生伪影或过度平滑。
- 需要一种灵活且表达力强的类无关离散先验，以统一处理多种图像恢复任务。

## 核心贡献（创新点）
- 提出AdaCode框架，学习一组类特定basis codebooks并通过图像自适应权重图进行加权组合，实现对任意图像内容的灵活离散表示。（与VQGAN/FeMaSR单一固定codebook的本质区别：从单分区到多视角自适应融合）
- 将3D LUT的思想迁移到离散latent空间，用预训练的VQGAN codebook作为基底，构建图像自适应codebook。（与前作3D-LUT的区别：前者用于颜色变换，本文用于生成式离散表征）
- 三阶段训练策略：类特定预训练、AdaCode表示学习、下游恢复任务微调，保证训练稳定且代码库规模与已有方法可比。（与前作端到端训练的区别：分阶段解耦，Stage I固定codebooks以提升效率）
- 在重建、超分辨率、inpainting三个任务上全面验证，多个数据集达到SOTA，且使用可比甚至更小的codebook尺寸。（与前作相比的优势：在更小参数量下获得更强表达力）

## 方法详解
- **Stage I 类特定Codebook预训练**：使用SegFormer对HR图像进行150类语义分割，并将类别聚类为5个super-class（Architectures、Indoor objects、Natural scenes、Street views、Portraits），在每个HR子集上独立训练VQGAN，获得K个class-specific basis codebook $Z_k$。损失函数包括L1、perceptual、adversarial、VQ code-level loss以及VGG19语义正则项$\mathcal{L}_{sem}$。
- **Stage II AdaCode表示学习**：给定输入HR patch，编码器E生成embedding $\hat{z}$，分别量化得到K个$z_{q_k}$；权重预测模块由4个RSTB和一层卷积构成，输出K-channel权重图$w \in \mathbb{R}^{h \times w \times K}$；AdaCode表示为$z = \sum_i w_i \times z_{q_i}$。此阶段固定所有basis codebook $Z_k$，仅训练E、权重预测模块、解码器G和判别器D，损失为$\mathcal{L}_{stage2} = \mathcal{L}_1 + \mathcal{L}_{per} + \lambda \cdot \mathcal{L}_{adv} + \mathcal{L}_{VQ}(E,G)$。
- **Stage III 图像恢复**：冻结解码器G和codebooks，将恢复问题转化为特征精炼。给定退化输入x，得到特征z，目标是最小化与GT特征$z_{gt}$的距离。引入基于SimCLR设计的code-level损失：$\mathcal{L}_{code} = \mathcal{L}_{InfoNCE}(z_{gt}, z) + \mathcal{L}_{style}(z_{gt}, z) + \beta \cdot ||\hat{z} - sg[z_{gt}]||_2^2$。整体损失$\mathcal{L}_{stage3} = \mathcal{L}_1 + \mathcal{L}_{per} + \lambda \cdot \mathcal{L}_{adv} + \mathcal{L}_{code}$。采用FeMaSR风格的encoder（特征提取模块+残差捷径）。
- **参数设置**：basis codebook尺寸分别为{512, 256, 512, 256, 256} × 256，对应5个语义子类；每图像表示为32×32的code序列；三阶段各训练350k次，batch size=32，Adam优化器，生成器学习率1e-4，判别器4e-4，在4×V100上训练。

## 实验与结果
- **数据集**：训练集包含DIV2K train、Flickr2K、DIV8K train及10,000张FFHQ人脸，共198,061个512×512 patch；退化模型采用BSRGAN。测试集：重建用OST；超分辨用Set5/Set14/BSD100/Urban100/Manga109（×2/×4）；inpainting用DIV2K/DIV8K验证集。
- **重建**：AdaCode（codebook size 1792×256）取得PSNR 25.7629、SSIM 0.7705，显著优于VQGAN（21.3557/0.5664）和VQGAN-aux（21.9219/0.6030）。
- **超分辨**：在多个基准上达到SOTA或接近SOTA，如×4 Set5 PSNR 25.868、SSIM 0.7731、LPIPS 0.1505；×2 Urban100 PSNR 23.663、SSIM 0.7894、LPIPS 0.1009。
- **Inpainting**：DIV8K上PSNR 32.701、LPIPS 0.0372、FID 1.1657，优于GPEN、MAT和FeMaSR。
- **结论**：AdaCode在可比或更小codebook规模下，重建与恢复质量全面领先，证明类自适应离散先验的有效性。

## 相关工作脉络
- VQGAN [7]：使用单一通用codebook进行图像生成，本文通过多basis codebook+权重融合扩展其表达力。
- FeMaSR [3]：引入单感知丰富codebook进行盲超分，本文指出单码本难以刻画多类别纹理，提出自适应组合策略。
- 3D-LUT [55]：学习多个LUT作为基底并用CNN预测融合权重进行颜色增强，本文将该思想迁移到离散latent表示。
- Real-ESRGAN [42]：基于纯合成数据训练盲超分，本文与之对比展示离散先验在纹理真实感上的优势。
- KX-Net [9]：从LR图像迭代学习退化核，本文从表征学习角度提供另一种恢复路径。
- MAT [19]、GPEN [52]：inpainting领域SOTA方法，本文在统一框架下与它们对比验证class-agnostic能力。

## 局限性与未来方向
- Basis codebooks的数量及每个codebook的条目数尚不明确，缺乏理论指导。
- 三阶段分离训练可能非全局最优，未来考虑端到端联合优化。
- 未引入高层显式语义信息（如语义分割图）参与自适应融合，潜在改进空间。
- 当前仅适用于2D图像，未来可扩展至视频与多光谱图像等高维视觉内容。

## 研究启发与可借鉴点
- 将连续权重融合思想从LUT迁移到离散codebook空间，为生成先验的设计提供了新范式。
- 三阶段解耦训练策略有效兼顾了codebook多样性学习与下游任务稳定性，可在其他先验学习方法中复用。
- InfoNCE + style loss的code-level对比学习设计增强了退化特征向HR特征的收敛，值得在特征对齐任务中借鉴。
- 语义子集划分虽不严格但实用，启发我们可通过弱语义监督提升多基底模型的表征多样性。
- 可探索将本方法与本团队方向结合，如在低资源机器翻译或跨模态生成中引入多基底离散先验。

## 关键术语表
- **AdaCode**：本文提出的图像自适应codebook学习方法，通过加权组合多个basis codebook实现类无关图像表示。
- **Basis Codebook**：在特定语义子集上预训练的离散codebook，作为AdaCode的可组合基础单元。
- **Weight Map**：由transformer预测的K通道权重图，决定每个basis codebook在空间各位置的贡献比例。
- **VQGAN**：基于矢量量化的生成对抗网络，本文作为basis codebook的预训练 backbone。
- **InfoNCE Loss**：对比学习损失，用于拉近退化特征与其HR对应特征的code表示。
- **Class-Agnostic**：指模型不依赖特定图像类别，可泛化到任意自然图像的通用性。
- **Semantic Subset**：基于SegFormer分割结果按super-class划分的HR图像子集，用于多样化预训练。
- **RSTB**：残差Swin Transformer Block，用作权重预测模块的核心组件。

## 关键要素
- 数据集：DIV2K train、Flickr2K、DIV8K train、FFHQ（10,000张人脸）；测试集OST、Set5、Set14、BSD100、Urban100、Manga109、DIV2K/DIV8K验证集。
- 代码开源：https://github.com/kechunl/AdaCode
- 关键超参：codebook尺寸{512,256,512,256,256}×256；patch尺寸512×512；code序列32×32；训练350k iter；batch size=32；生成器lr=1e-4，判别器lr=4e-4；λ=0.1（adv/sem权重），β=0.25（VQ损失）。
