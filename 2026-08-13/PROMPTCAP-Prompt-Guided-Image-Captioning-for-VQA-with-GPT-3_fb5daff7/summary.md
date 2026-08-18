---
title: "PROMPTCAP-Prompt-Guided-Image-Captioning-for-VQA-with-GPT-3"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_PromptCap_Prompt-Guided_Image_Captioning_for_VQA_with_GPT-3_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:25:53"
field: "视觉语言理解"
keywords: ["视觉问答", "图像描述", "大语言模型", "提示学习", "知识型VQA", "GPT-3", "多模态", "in-context learning"]
innovations: ["提出问题感知图像描述模型PROMPTCAP，通过自然语言提示控制视觉信息生成", "首次利用GPT-3合成Vision-Language任务训练数据并设计soft accuracy过滤机制", "PROMPTCAP+GPT-3在OK-VQA(60.4%)和A-OKVQA(59.6%)达到SOTA"]
benchmarks: ["OK-VQA", "A-OKVQA", "WebQA", "VQAv2"]
---

# 论文速读：PROMPTCAP-Prompt-Guided-Image-Captioning-for-VQA-with-GPT-3

## 一句话总结
论文提出了 PROMPTCAP，一种问题感知的图像描述模型，能够根据输入的问题提示生成针对性的图像描述，从而帮助黑盒大语言模型（如 GPT-3）更好地理解图像内容并完成知识型 VQA 任务；该模型通过 GPT-3 合成的数据训练，在 OK-VQA（60.4%）和 A-OKVQA（59.6%）上达到 SOTA。

## 研究问题与动机
1. **通用图像描述遗漏关键视觉细节**：现有方法将图像转换为通用文字描述后再输入 GPT-3 进行 VQA，但通用描述往往无法包含回答问题所需的关键视觉信息（如图中"McDonald's"招牌）。
2. **黑盒 LM 无法直接接收图像输入**：GPT-3 等先进大语言模型只能通过 API 调用，无法访问内部参数进行端到端微调，需要将图像信息转化为文本。
3. **缺乏问题感知型图像描述的标注数据**：没有现成的"问题-图像描述"配对数据可用于训练 PROMPTCAP。
4. **跨模态信息传递存在瓶颈**：从图像到文本的转换过程难以保留所有对下游任务有用的视觉细节。

## 核心贡献（创新点）
1. **提出 PROMPTCAP——问题感知的图像描述模型**：首次将自然语言提示（包含问题）作为条件输入控制图像描述生成，使生成的描述有针对性地包含回答问题所需的视觉信息，区别于传统通用描述模型。
2. **提出基于 GPT-3 的训练数据合成与过滤流水线**：利用 GPT-3 的 in-context learning 能力，将 VQA 数据集中的问题-答案对转换为问题感知的图像描述训练样本，并通过 GPT-3 QA 验证和 soft accuracy 指标筛选高质量样本，这是首次利用 GPT-3 合成 Vision-Language 任务训练数据的工作。
3. **在知识型 VQA 上实现 SOTA**：PROMPTCAP + GPT-3 组合在 OK-VQA 达到 60.4%、A-OKVQA 达到 59.6%，均超越此前所有基线方法；消融实验证实 PROMPTCAP 比同架构通用描述模型分别在 OK-VQA、A-OKVQA、VQAv2 上提升 3.8%、5.3%、9.2%。
4. **验证零样本跨域泛化能力**：在 WebQA 数据集上，PROMPTCAP 未进行任何目标领域训练即可优于通用描述方法和所有监督基线。

## 方法详解
**整体架构**：PROMPTCAP 基于 OFA（Open Multimodal Foundation）captioning 模型（470M 参数），采用 encoder-decoder 结构，输入为图像 patch + 自然语言提示的拼接 `[P_i : I_i]`，输出为问题感知的描述文本 $C_i$。

**训练数据合成（§3.1）**：
- 将 VQAv2 数据集中的问题-答案对视为图像和视觉细节的自然来源
- 使用 GPT-3（code-davinci-002，175B）进行 few-shot in-context learning 生成问题感知描述：给定 COCO 通用描述（5 条 human-written captions 拼接为 "Original Context"）和问题-答案，GPT-3 重写为有助于回答问题的描述
- 每个问题-答案对采样 5 个候选描述，通过过滤pipeline选优：
  - **Soft VQA Accuracy**：使用字符编辑距离（CER）计算软准确率，解决 open-ended generation 中表面形式差异导致的评分偏差问题，公式为：$\text{Acc}_{soft}(a) = \max_{x,y,z \in [n]} \sum_{i \in \{x,y,z\}} \frac{\max[0, 1 - \text{CER}(a, g_i)]}{3}$
  - **CIDEr 去重**：对于软准确率相同的候选，选择与 COCO ground-truth 描述 CIDEr 分数最高的

**训练损失（§3.2）**：使用负对数似然损失，端到端训练：
$$\mathcal{L} = -\sum_{\mathcal{D}} \sum_{t=1}^{|C_i|} \log p(c_t \mid [P_i : I_i], c_{\le t-1})$$

**VQA 推理流水线（§4）**：
- Step 1：用 PROMPTCAP 将图像和问题转换为问题感知的文本描述
- Step 2：将转换后的样本作为 in-context examples 输入 GPT-3，使用 CLIP 相似度检索选取最相似的 n=32 个训练样本作为 demonstrations，进行 zero-shot/few-shot VQA 推理

## 实验与结果
**数据集**：OK-VQA（14K 图像-问题对）、A-OKVQA（25K 图像-问题对）、WebQA（多跳多模态推理）

**基线对比**：
- OK-VQA：超越 KAT（56.6%）、REVIVE（58.0%）、PICa-Full（57.8%）、Flamingo（50.6%）等所有已有方法
- **最强结果**：PROMPTCAP + GPT-3 在 OK-VQA 验证集达到 **60.4%**，在 A-OKVQA 多项选择达到 **73.1%**、直接回答达到 **59.6%**
- **提升幅度**：相比同架构通用描述模型 OFA-Cap，分别在 OK-VQA（+3.8%）、A-OKVQA（+5.3%）、VQAv2（+9.2%）取得绝对提升

**消融实验**：
- GPT-3（175B）vs Flan-T5-XXL（11B）：知识型 VQA 上 GPT-3 优势显著（OK-VQA +18.4%，A-OKVQA +14.8%），VQAv2 上优势较小（+3.2%）
- CLIP 相似度检索 vs 随机检索：n=32 时在 OK-VQA 带来 +5.2% 绝对提升
- in-context 示例数量增加持续提升性能

**跨域泛化**：WebQA 上 PROMPTCAP + GPT-3（8-shot）FL*Acc=34.5，超越所有监督基线和通用描述方法。

## 相关工作脉络
1. **PICa（Yang et al., 2022）**：首次将 GPT-3 用于知识型 VQA 的 in-context learning，使用通用图像描述作为图像输入，本文识别其通用描述遗漏关键细节的问题并提出改进方案。
2. **KAT（Gui et al., 2022）**：引入 Wikidata 作为额外知识源并端到端微调多组件，达到 OK-VQA 56.6%，本文不依赖额外知识源但以 prompt-guided 描述超越之。
3. **REVIVE（Lin et al., 2022）**：当前 SOTA，引入 object-centric 视觉特征并与 caption 集成，本文表明 PROMPTCAP 可替换其 captioning 模块进一步提升性能。
4. **Flamingo（Alayrac et al., 2022）和 BLIP-2（Li et al., 2023）**：保持 LM 冻结但需要微调视觉编码器，依赖 LM 内部参数访问，不适用于 GPT-3 等黑盒 API 场景，本文方法无此限制。
5. **VLP + VinVL / VLP + x101fpn（Chang et al., 2022）**：WebQA 上的监督基线方法，本文以 8-shot in-context 方式超越。

## 局限性与未来方向
1. **任务局限性**：当前 PROMPTCAP 仅针对知识型 VQA 设计，可推广至其他 Vision-Language 任务（如 NLVR2）。
2. **图像信息的文本化损失**：部分视觉信息无法被完整抽象为文本描述，需与其他多模态方法结合使用。
3. **依赖 GPT-3 的数据合成**：数据合成质量受限于 GPT-3 的理解和生成能力，可能存在事实性错误。
4. **合成数据的多样性有限**：GPT-3 生成的问题感知描述通常较短且覆盖的视觉实体较少，可能限制模型上限。
5. **未来方向**：扩展训练数据规模和指令多样性、探索 VQA 之外的更多视觉语言任务、与视觉特征融合。

## 研究启发与可借鉴点
1. **GPT-3 in-context learning 数据合成范式**：利用 GPT-3 的 few-shot 能力将已有数据集自动转换为新任务格式的训练数据，无需人工标注，可迁移到其他 Vision-Language 任务的训练数据构建。
2. **Soft VQA Accuracy 评估设计**：通过字符编辑距离容忍表面形式差异的软准确率计算方式，可有效解决 open-ended generation 任务的评估噪声问题，适用于其他自由文本生成任务。
3. **CLIP 相似度检索优化 in-context examples**：使用 CLIP 检索与测试样本最相似的训练样本作为 demonstrations，显著提升 GPT-3 in-context learning 效果，是低成本提升 LLM 性能的有效策略。
4. **Prompt-guided 描述作为 LM 与图像的桥梁**：将问题/任务指令作为 prompt 引导视觉信息提取，这一思路可推广至视频描述、图表理解等场景。
5. **黑白盒协同范式**：在无法微调黑盒 LM 的前提下，通过优化输入侧的表示（问题感知描述）来提升 LM 性能，是一种实用的工程策略。

## 关键术语表
**Prompt-Guided Captioning**：根据自然语言提示（如问题）动态生成针对性图像描述，而非输出通用描述的技术。
**In-Context Learning (ICL)**：通过提供少量示例而非微调参数，使预训练模型适应新任务的学习范式。
**Soft VQA Accuracy**：基于字符编辑距离（CER）计算的多候选匹配准确率，容忍答案表面形式的小差异。
**OFA（Open Multimodal Foundation）**：由上海人工智能实验室提出的统一多模态基础模型，本文使用其 470M 参数的 captioning 版本。
**Code-Davinci-002**：GPT-3 系列中 175B 参数的代码生成专用引擎，本文用于 in-context learning 和训练数据合成。
**Knowledge-based VQA**：需要外部世界知识（不仅依赖图像内容）才能正确回答的视觉问答任务。
**CIDEr Score**：基于集合理论的图像描述自动评估指标，衡量生成描述与多条 ground-truth 描述的一致性。
**Character Error Rate (CER)**：基于字符级编辑距离的相似度度量，用于软准确率计算中的匹配判断。

## 可复现要素
- **数据集**：OK-VQA、A-OKVQA、WebQA、VQAv2、COCO caption——均为公开数据集
- **代码/权重**：OFA 官方开源权重已公开；论文提供了演示网站 https://yushi-hu.github.io/promptcap_demo/，但代码开源状态论文未明确声明
- **关键超参**：OFA checkpoint "caption-large-best-clean"（470M）；AdamW optimizer，learning rate {2e-5, 3e-5, 5e-5}，batch size 32/64/128，$\beta_1=0.9$，$\beta_2=0.999$；GPT-3 使用 code-davinci-002（175B），in-context examples n=32（CLIP 检索），VQAv2 合成数据
- **数据合成**：20 个人工手写示例 + VQAv2 数据，GPT-3 in-context learning 生成问题感知描述
