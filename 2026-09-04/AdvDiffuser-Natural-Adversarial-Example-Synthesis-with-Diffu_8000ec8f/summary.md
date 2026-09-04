---
title: "AdvDiffuser-Natural-Adversarial-Example-Synthesis-with-Diffu"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:13:01"
field: "对抗机器学习与扩散生成模型"
keywords: ["adversarial examples", "diffusion models", "unrestricted adversarial examples", "adversarial training", "GradCAM", "inpainting"]
innovations: ["首次将扩散模型逆向去噪过程与对抗梯度扰动交替优化，用于自然无限制对抗样本合成", "提出基于GradCAM显著性mask的对抗性inpainting，在保留语义显著区域的同时仅扰动非关键区域", "揭示无威胁模型假设下扩散UAE对抗训练可泛化至多种未见攻击类型"]
benchmarks: ["CIFAR-10", "CelebA", "ImageNet", "RobustBench"]
---

# 论文速读：AdvDiffuser-Natural-Adversarial-Example-Synthesis-with-Diffu

## 一句话总结
AdvDiffuser首次利用扩散模型的逆向去噪过程，通过在去噪每一步注入有界的对抗梯度扰动，并结合基于GradCAM的对抗性inpainting保留显著区域，从而合成语义自然、感知隐蔽且攻击能力强的无限制对抗样本（UAE）。

## 研究问题与动机
- 传统 $\ell_p$ 有界对抗攻击无法反映人类对扰动的真实感知，且在高鲁棒性模型面前效力有限。
- 现有的无限制对抗样本（UAE）生成方法（如GAN-based的AC-GAN、VAE-based方法）通过扰动潜码来生成对抗样本，会导致高维语义信息丢失，生成的UAE视觉质量差、语义模糊、不自然。
- 梯度类UAE攻击（如GA-attack）依赖人工选取距离度量与代理模型，主观性强，且易在低信息区域（如背景天空）引入可见伪影，整体感知距离大。
- 扩散模型在数据分布建模和生成质量上已超越GAN，但尚未被探索用于UAE合成；其去噪过程天然具备"去除非自然噪声、保留自然结构"的能力，有望生成更自然且攻击性强的UAE。

## 核心贡献（创新点）
- **首次将扩散模型引入自然UAE合成**：与GAN/VAE方法在潜空间扰动不同，AdvDiffuser直接在像素级去噪轨迹中注入对抗扰动，避免了语义信息的不可逆丢失。
- **提出对抗性guidance机制**：在每一步逆向去噪过程中，使用有界PGD攻击扰动当前去噪结果，并将扰动幅度按 $\varepsilon_t = \sigma \beta_t$ 随噪声方差递减，使最终生成的扰动既有效攻击又保持与自然分布一致。
- **提出基于GradCAM的对抗性inpainting**：利用 defending classifier 的梯度类激活图生成显著区域mask，在去噪过程中通过插值保留显著物体区域、仅对背景/次要区域施加对抗扰动，大幅提升UAE的视觉自然度与语义一致性。
- **揭示无威胁模型假设下的对抗训练新范式**：基于AdvDiffuser合成的UAE进行对抗训练，可使模型在 $\ell_\infty$、$\ell_2$、JPEG压缩、ReColorAdv、LPA、StAdv等多种未见威胁模型下均获得显著提升的鲁棒性，突破了传统 norm-bounded 对抗训练的局限。

## 方法详解
- **总体框架**：算法以纯噪声 $\hat{\mathbf{x}}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 为起点，迭代 $T$ 步执行"去噪→对抗扰动→掩码插值"的循环，最终得到 $\hat{\mathbf{x}}_0$。若提供参考图像 $\mathbf{x}_0$，则先通过 $\mathbf{m} = \text{GradCAM}(f, \mathbf{x}_0, y)$ 计算显著区域mask。
- **对抗性guidance（Eq.5-6）**：在每一步 $t$，先由预训练扩散模型采样 $\tilde{\mathbf{x}}_{t-1} \sim p_\theta(\mathbf{x}_{t-1}|\hat{\mathbf{x}}_t)$，再对其施加有界PGD攻击：$\mathbf{z}_{i+1} = \mathcal{P}_{\mathbf{z}_0, \varepsilon_t}(\mathbf{z}_i + \text{sign}(\nabla_{\mathbf{z}_i} \mathsf{L}(f(\mathbf{z}_i), y)))$，其中 $\mathsf{L}$ 采用归一化softmax交叉熵（SCE），$\varepsilon_t = \sigma \beta_t$ 确保对抗扰动始终小于当前扩散噪声尺度。
- **对抗性inpainting（Eq.8）**：在每步得到攻击后的 $\hat{\mathbf{x}}_{t-1} = \mathbf{z}_I$ 后，与来自原始参考图的加噪版本 $\mathbf{x}_{t-1}^{\text{obj}} \sim \mathcal{N}(\sqrt{\overline{\alpha}_{t-1}}\mathbf{x}_0, (1-\overline{\alpha}_{t-1})\mathbf{I})$ 进行mask加权融合：$\mathbf{x}_{t-1} = \mathbf{m} \odot \mathbf{x}_{t-1}^{\text{obj}} + (\mathbf{1}-\mathbf{m}) \odot \hat{\mathbf{x}}_{t-1}$，从而保护显著物体的同时允许背景/非关键区域被充分扰动。
- **自生成模式（from scratch）**：当无参考图像时，mask初始化为全零，算法退化为直接从纯噪声生成对抗样本，但仍保留对抗性guidance机制。
- **对抗训练扩展**：将AdvDiffuser生成的UAE（含图像条件型与自生成型）作为对抗训练数据，模型无需预设威胁模型假设即可学习通用鲁棒特征。

## 实验与结果
- **数据集与基线**：CIFAR-10、CelebA、ImageNet；对比基线包括AC-GAN（自生成）、GA-PGD与GA-FSA（图像条件UAE）； defended models取自RobustBench榜单最强模型。
- **CelebA自生成对比**（Table 1）：AdvDiffuser攻击成功率 **99.1%** vs AC-GAN 91.1%，FID **8.4** vs 15.6，生成速度 **12.8s/image** vs 23.6s/image。
- **CIFAR-10攻击成功率**（Table 2）：对Rebuffi et al. A/B与Gowal et al.等鲁棒模型，AdvDiffuser成功率为9.81%~11.77%，而GA-PGD对Rebuffi et al. B仅80.15%（即鲁棒准确率约80%），说明AdvDiffuser在强鲁棒模型上仍具优势且生成样本自然度高。
- **ImageNet攻击效果**（Table 3）：对Salman et al. A/B、Engstrom et al.等当前最强鲁棒模型，AdvDiffuser将分类准确率降至0.2%~0.6%，同时LPIPS仅0.03~0.05（GA-PGD为0.24~0.34）、SSIM达0.97~0.99（GA-PGD为0.59~0.80）、FID为25.9~27.2（GA-PGD为48.9~49.5）。**相比GA-attack，LPIPS降低约6、FID降低约23、SSIM提升约0.28**。
- **未见威胁模型鲁棒性**（Table 4）：在CIFAR-10上，AdvDiffuser对抗训练后的模型在JPEG、ReColorAdv、LPA、StAdv等未见威胁下准确率普遍高于Engstrom et al.的标准 $\ell_2$ 对抗训练模型；Clean准确率虽从90.2%降至约67%，但多威胁泛化能力显著增强。
- **最强结果**：在ImageNet上对Salman et al. A模型攻击成功率达99.5%（准确率0.5%），同时感知指标全面优于SOTA UAE攻击。

## 相关工作脉络
- **AC-GAN / VAE-based UAE**（Song et al. 2018, Zhao et al. 2018）：在生成器潜空间扰动以产生对抗样本，语义保真度低；AdvDiffuser转向像素级去噪轨迹扰动，避免潜码扰动带来的语义损坏。
- **GA-attack**（Liu et al. 2023，CVPR 2021 UAE竞赛冠军）：基于几何感知的梯度UAE攻击，依赖代理模型与距离度量主观选择；AdvDiffuser不依赖此类先验，直接利用扩散分布天然约束扰动形态。
- **LPIPS/SSIM感知攻击**（Laidlaw et al. 2021, ReColorAdv等）：在感知距离空间优化扰动；AdvDiffuser通过扩散去噪隐式满足感知自然性，无需显式感知距离约束。
- **DiffPure**（Nie et al. 2022）：利用扩散模型去除对抗扰动以增强防御；本文反向利用扩散生成对抗样本，二者构成"攻击-防御"对称视角。
- **Classifier-free guided diffusion**（Ho & Salimans 2021）：为条件扩散模型的基础训练范式，AdvDiffuser在此基础上叠加对抗性guidance，形成"生成-攻击"联合优化。
- **对抗训练与数据增强**（Rebuffi et al. 2021）：证明DDPM生成数据可提升鲁棒性；本文进一步将扩散模型直接用于UAE合成并验证跨威胁泛化能力。

## 局限性与未来方向
- **白盒假设局限**：所有实验均在白盒设定下进行，未评估黑盒迁移攻击性能。
- **$\ell_\infty$ 指标较高**：作者承认 $\ell_\infty$ 距离并非优化目标，视觉伪影在个别区域仍存在，与感知指标（LPIPS/SSIM）存在一定脱节。
- **扩散采样计算开销**：尽管速度已优于AC-GAN（12.8s vs 23.6s），但相比传统梯度攻击仍慢一个数量级，限制了大规模实时应用。
- **鲁棒-准确率权衡**：对抗训练后clean准确率下降明显（90.2%→67%），如何在保持多威胁鲁棒性的同时恢复clean性能仍有待探索。
- **未来方向**：可扩展至黑盒/白盒混合设定、结合latent diffusion（如Stable Diffusion）以提升ImageNet高分辨率场景的效率、探索与DiffPure等防御方法的动态博弈。

## 研究启发与可借鉴点
- **扩散去噪+对抗扰动的交替优化范式**：将"去噪保自然"与"梯度攻击保有效"在每一步去噪中交替执行，是一种可迁移到视频/3D对抗样本生成的通用框架。
- **GradCAM mask的动态插值策略**：用显著性mask在去噪轨迹中动态融合"原始信息"与"对抗扰动"，可推广至其他需要语义保真的对抗生成任务（如医学图像、自动驾驶）。
- **无威胁模型假设的对抗训练思路**：放弃预设 $\ell_p$ 边界，转而用生成模型覆盖多样化扰动分布，对构建通用鲁棒模型具有方法论启发。
- **扰动幅度随噪声方差递减的设计**：$\varepsilon_t = \sigma \beta_t$ 使早期大扰动充分探索对抗空间、后期小扰动精细雕刻，这一调度策略可复用于其他基于生成的对抗优化问题。
- **与防御方法的对称研究**：本文与DiffPure形成天然对照，可作为后续"生成式攻击-防御博弈"研究的基准起点。

## 关键术语表
- **Unrestricted Adversarial Examples (UAE)**：不受 $\ell_p$ 范数约束、仅在感知上自然的对抗样本，可对人类可见语义保持忠实但使模型错误分类。
- **Adversarial Guidance**：在扩散逆向去噪的每一步中注入有界对抗梯度扰动，使去噪轨迹逐步偏离决策边界以攻击目标分类器。
- **Adversarial Inpainting**：基于GradCAM显著性mask，在去噪过程中将对抗扰动仅施加于非显著区域，保留主体对象语义一致性的插值技术。
- **Normalized Softmax Cross-Entropy (SCE) Loss**：论文采用的对抗最大化目标函数，相比标准SCE在生成有效对抗样本方面表现更优。
- **Classifier-free Guidance**：扩散模型的条件生成技术，无需额外分类器即可实现条件采样，本文预训练DDPM的基础。
- **DiffPure**：利用扩散模型去除图像中对抗扰动的防御方法，与AdvDiffuser形成攻击-防御对称关系。
- **RobustBench**：标准化的对抗鲁棒性评测基准平台，收录各模型在多种攻击下的鲁棒准确率。
- **GradCAM**：基于梯度加权类激活映射的可视化技术，用于定位图像中对分类决策贡献最大的显著区域。

## 可复现要素
- **数据集**：CIFAR-10（公开）、CelebA（公开）、ImageNet（公开）；论文声明使用预训练条件DDPM（OpenAI ImageNet模型）及自行复现的CIFAR-10/CelebA模型。
- **代码开源情况**：论文未明确声明代码开源（ICCV 2023版本未附GitHub链接），仅说明详细实验配置见Appendix A。
- **关键超参**：扩散步数 $T=100$（CIFAR-10）、$T=400$（ImageNet）；对抗指导尺度 $\sigma=0.1$（CIFAR-10）、$\sigma=0.4$（ImageNet）；PGD迭代次数 $I=1$（CIFAR-10）、$I=25$（ImageNet）；扰动边界 $\varepsilon_t = \sigma \beta_t$。
