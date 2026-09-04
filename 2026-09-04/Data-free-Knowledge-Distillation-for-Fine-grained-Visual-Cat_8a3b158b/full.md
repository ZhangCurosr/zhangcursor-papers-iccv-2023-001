# Data-free Knowledge Distillation for Fine-grained Visual Categorization

Renrong Shao<sup>1</sup>, Wei Zhang<sup>1</sup>\*, Jianhua Yin<sup>2</sup>, Jun Wang<sup>1</sup>\* <sup>1</sup>School of Computer Science and Technology, East China Normal University <sup>2</sup>School of Computer Science and Technology, Shandong University roryshaw6613<sub>,</sub>zhangwei thu<sub>,</sub>jhyinmail<sub>,</sub> wongjun @gmail com

## Abstract

Data-free knowledge distillation (DFKD) is a promising approach for addressing issues related to model compression, security privacy, and transmission restrictions. Although the existing methods exploiting DFKD have achieved inspiring achievements in coarse-grained classification, in practical applications involving fine-grained classification tasks that require more detailed distinctions between similar categories, sub-optimal results are obtained. To address this issue, we propose an approach called DFKD-FGVC that extends DFKD tofine-grained visual categorization (FGVC) tasks. Our approach utilizes an adversarial distillation framework with attention generator, mixed high-order attention distillation, and semantic feature contrast learning. Specifically, we introduce a spatial-wise attention mechanism to the generator to synthesize fine-grained images with more details of discriminative parts. We also utilize the mixed high-order attention mechanism to capture complex interactions among parts and the subtle differences among discriminative features of the fine-grained categories, paying attention to both local features and semantic context relationships. Moreover, we leverage the teacher and student models of the distillationframework to contrast high-level semanticfeature maps in the hyperspace, comparing variances of different categories. We evaluate our approach on three widely-used FGVC benchmarks (Aircraft, Cars196, and CUB200) and demonstrate its superior performance. Code is available at https://github.com/RoryShao/DFKD-FGVC.git

## 1. Introduction

Fine-grained visual categorization (FGVC) aims at distinguishing subcategories from father categories, e.g., subcategories of birds [43], aircraft [29], and cars [23]. It has long been considered a more challenging issue than traditional image classification due to the subtle inter-class and large intra-class variations [42]. To distinguish subtle diversities, the current approaches commonly exploit deeper neural networks with elaborate designs [50, 26, 53] to excavate the discriminative features effectively. Inevitably, the network becomes more and more complex, which leads to another problem, i.e., complicated networks are not easily deployed on embedded or mobile devices. Besides, the training data of released pre-trained models are often unavailable due to transmission, privacy, or legal issues. For example, pre-trained models commonly need a large amount of data such as ImageNet [24]. If the data is transmitted directly, a large amount of bandwidth is consumed. Moreover, some sensitive data such as e-commerce items or medical data are usually not directly accessible to the public due to intellectual property rights or privacy protection considerations. To obtain a lightweight model, recent research has made significant progress, including pruning [25], quantization [49, 27], and knowledge distillation [16]. Among them, knowledge distillation (KD) is a popular and effective paradigm for model compression and knowledge transfer [16]. It works by transferring knowledge from a cumbersome teacher network to a lightweight student network. Thanks to this separable architecture, it can also be used to solve privacy protection in data-free scenarios, which is called data-free knowledge distillation (DFKD) [5] or zeroshot knowledge distillation (ZSKD) [33].

Fortunately, a series of DFKD methods have been proposed [5, 33, 30, 11, 47, 12]. The existing approaches can be divided into two paradigms. The first paradigm is based on the category distribution, which exploits the out distribution of teacher and student to optimize the student and generator, e.g., DFAL [5], ZSKT [30], DFAD [11], ZSKD [33]. Such a paradigm commonly fails to generate realistic samples due to the lack of semantic-related information, especially when it comes to complex samples. The second paradigm is based on prior distribution, which exploits the prior information (i.e., BatchNorm) to optimize synthetic images for distillation, e.g., MAD [10], CMI [12], DFQ [8],

ADI [47]. This paradigm can produce realistic features and, therefore, gives the student a noticeable improvement.

Although the existing methods have achieved inspiring achievements in coarse-grained classification, in practical applications, sub-optimal results are achieved due to the subtle variations widely found in different scenarios. The main reasons for this situation are as follows: Firstly, for FGVC tasks, the variances of the same category are more prominent than that of coarse-grained classification due to different factors, such as viewing angles, lighting, backgrounds, occlusion, etc. Secondly, compared to coarsegrained categories, the feature discrepancies of different categories in FGVC are not obvious. Besides, in the datafree scenario, the model can not access the raw data directly. For synthesized images, it is difficult for the teacher model to capture the subtle variances of discriminative features. To our best knowledge, there are still no specialized data-free studies on fine-grained DFKD. Therefore, this inspires us to explore this issue and tackle this task in a data-free scenario.

In this paper, we tackle this issue by extending DFKD to fine-grained visual classification (FGVC) tasks and propose an approach named DFKD-FGVC, which is achieved by exploiting the adversarial distillation framework with attention generator, mixed high-order attention distillation (MHAD) and semantic feature contrast learning (SFCL). Concretely, as shown in Fig. 1, to promote the generator to synthesize more fine-grained images, we exploit the generator with spatial-wise attention, which can help the generator synthesize the images with more details of discriminative parts. Then, to fully mine the knowledge of discriminative features for student, we exploit the mixed high-order attention mechanism to capture complex interactions among parts and the subtle differences among discriminative features of the fine-grained categories, paying attention to both local features and semantic context relationships. Besides, to compare variances of different categories, we skillfully exploit the teacher and student model of distillation framework to contrast semantic feature maps in the hyperspace. To verify our approach, massive experiments are conducted on three fine-grained benchmarks, such as Aircraft, Cars196, and CUB200 to evaluate the effectiveness of our approach.

In a nutshell, our contributions are four-fold: 1) We are the first to propose an approach for FGVC in the datafree distillation scenario, which aims to optimize the entire generation and distillation stages to focus on discriminative features. 2) To synthesize more fine-grained images for adversarial distillation, we employ the generator with spatial-wise attention, which motivates the generator to synthesize the images with more details of discriminative features. 3) Particularly, to effectively mine the potential semantic features and contextual relationships of the fine-grained categories, we provide two strategies, namely,

MHAD and SFCL, both of which can promote the performance of DFKD from different dimensions. 4) Extensive experiments demonstrate the effectiveness of our approach in the data-free setting, which achieves state-of-the-art performance on the mainstream FGVC benchmark datasets.

## 2. Related Works

## 2.1. Fine-Grained Visual Categorization.

Fine-grained visual classification (FGVC) [43, 29, 23] is much more challenging than traditional classification tasks due to the inherently subtle intra-class object variations [42, 18]. Benefiting from the recent development of neural networks, recent studies have moved from strongly supervised information with extra annotations such as bounding boxes [2, 48, 18] to weakly-supervised conditions with only category labels [51, 13, 41]. Current methods on FGVC can be roughly divided into localization-based methods [13, 41] and attention-based methods [3, 19, 31]. The core for solving FGVC is to learn the discriminative features of objects in images. However, current approaches tackle this problem in the data-driven setting, few approaches consider this problem in the data-free setting. Therefore, different from the above studies, we explore the FGVC tasks in the novel aspect of the data-free distillation scenario.

## 2.2. Attention Mechanism

The attention mechanism stems from human vision, which exploits a sequence of partial glimpses and selectively focuses on salient parts to capture visual structure better. In the field of computer vision, attention mechanism [40, 17, 44, 34] are mainly exploited to capture essential information in various tasks such as pedestrian reidentification [46, 20, 45], FGVC [3, 31], etc. For example, [40] proposes the residual attention network for large-scale classification tasks. Then Hu et al. [17] exploit a squeezeand-excitation (SE) block to compute channel-wise attention. CBAM [44] infers attention maps along two separate dimensions, i.e., channel and spatial. Similar to [44], BAM [34] also exploits the 3D attention map inference into channel and spatial. In terms of tasks, spatial attention is well-suited to dense prediction tasks such as semantic segmentation and object detection [14], while channel attention is a good choice for image classification. However, only exploiting spatial attention or channel attention is coarse, we can not capture the high-order and complex interactions among parts [4]. Therefore, in our data-free framework, we empirically exploit the spatial attention for our generator and the mixed high-order attention for distillation.

## 2.3. Data-free Distillation.

Data-free Distillation has become a hot topic in recent years, mainly due to privacy protection [28]. It exploits synthesized alternative samples to solve the dilemma that model can not directly access the original data and makes gratifying achievements in the task of classification [47, 5, 11, 12]. For example, ADI [47] utilizes batch normalization statistics (BNS) of the pre-trained teacher to optimize the noise to synthesize high-fidelity images for KD. CMI [12] exploits the local and global contrast of samples to optimize the generator diversity. This kind of method ordinarily can synthesize more realistic images and achieve relatively better performance. DFAL [5] adopts a generator to synthesize images, and then the student learns the knowledge from the teacher by distillation. ZSKT [30] exploits the adversarial distillation to transfer the knowledge from teacher to student by KL and spatial attention, while DFAD [11] only utilizes the MAE loss to fit the output distribution of the teacher. All kinds of the above methods can achieve relatively inspiring achievements in coarsegrained classification, and there is no specific research on FGVC. Motivated by this, we conduct the study for datafree fine-grained distillation.

![](images/b50eded7a70999dbec30ec14f82ce7fa69e15427fa7b8e0a9bf9e456583cc52d.jpg)  
Figure 1. The whole framework of our approach. The left: The spatial attention module is plugged into each block of generator , which aims to focus on fine-grained semantic information from the whole process of noise z to images xˆ. The intermediate: At each block of teacher and student, the feature maps are extracted by the mixed high-order attention module to achieve MHAD. The right: In the penultimate layer, exploiting the MLP to map the high-level semantic features of teacher and student to a common hyperspace and compare the variances by SFCL.

## 3. Preliminary

Our approach follows the basic thinking of DFKD, as depicted in Fig. 1. First, a generator  is employed to synthesize a batch of images from noise $z \sim \mathcal { N } ( 0 , 1 ) , \mathcal { G } ( z )  \hat { x }$ $\boldsymbol { B } = \left\{ \hat { x } _ { 1 } , \hat { x } _ { 2 } , . . . , \hat { x } _ { n } \right\} , n \in \left\{ 1 , . . . , \mathrm { N } \right\}$ , where N is the batch size. Then the synthesized image xˆ is input to the pretrained teacher  and student  to support their distillation.

Finally, the generator $\mathcal { G }$ is optimized by adversarial distillation.

Data-free Adversarial Distillation Essentially, Data-free adversarial distillation is a robust minimax optimization problem [1], which encourages the generator to minimize the possible loss for a worst-case scenario (maximum loss) through adversarial training under data uncertainty. In the data-free scenario, it can be denoted as

$$
\operatorname* { m i n } _ { \mathcal { S } } \operatorname* { m a x } _ { \mathcal { G } } \left\{ \mathbb { E } _ { p ( z ) } \left[ \mathcal { D } ( \mathcal { T } ( \mathcal { G } ( z ) ) , \mathcal { S } ( \mathcal { G } ( z ) ) ) \right] - \delta \mathcal { L } _ { \mathcal { G } } \right\} ,\tag{1}
$$

where  represents the discrepancy measure, which normally exploits the Kullback-Leibler (KL) divergence as an optimization term. $\delta \geq 0$ is the balance factor, and $\mathcal { L } _ { \mathcal { G } }$ is the optimization term of generator .

Knowledge Distillation. According to the principle of classic knowledge distillation [16], the soft output of the network (a.k.a. probability distribution) implies the similarity between the current sample and other categories. Therefore, traditional methods [8, 10, 33] usually adopt the KL Divergence to measure the difference between the two distributions of teacher and student. The probability distribution distillation can be formulated as

$$
\mathcal { L } _ { \mathrm { K D } } = \mathbb { E } _ { \hat { x } } \left[ D _ { \mathrm { K L } } \left( \sigma ( S ( \hat { x } ) ) \Vert \sigma ( \mathcal { T } ( \hat { x } ) ) \right) \right] ,\tag{2}
$$

where $D _ { \mathrm { K I } }$ represents the Kullback-Leibler (KL) divergence, and σ is the softmax operation.

Prior Information Regularization. Prior information regularization aims to regularize the feature distribution of synthesized images by prior distribution information, i.e., mean μ and variance $\sigma ^ { 2 }$ of BatchNorm [47], which motivates the synthetic samples to approach the distribution of the original samples.

![](images/186201c0ce13c1e0cad42ee187da2278e9342c3dd4a46b6ba3f8bdacc42c1995.jpg)  
Figure 2. The spatial attention module of the generator, in which denotes the element-wise multiplication and denotes the element-wise addition.

$$
\mathcal { L } _ { \mathrm { B N } } = \operatorname* { m i n } _ { \mathcal { G } } \sum _ { l } \left. \mu _ { l } - \mu _ { l } ( \mathcal { G } ( z ) ) \right. _ { 2 } + \left. \sigma _ { l } ^ { 2 } - \sigma _ { l } ^ { 2 } ( \mathcal { G } ( z ) ) \right. _ { 2 } ,\tag{3}
$$

where l donates the $l ^ { t h }$ BatchNorm layer of the teacher model, $\mu$ and $\sigma ^ { 2 }$ are the batch-wise mean and variance, respectively.

## 4. Proposed Approach

## 4.1. Discriminative Feature Synthesis

In the DFKD framework, it is common to exploit a generator to assist in generating alternative samples for coarsegrained classification. However, directly applying it to synthesize fine-grained samples often does not yield desirable discriminative features. This is because the conventional generator cannot focus on subtle discriminative features, which decreases the ability of teacher to extract representation from various semantic parts and thus hampers the effectiveness of the distillation. Differing from the traditional approaches, in our framework, we employ a DCGAN [35, 38] generator with the attention module to increase the representation ability of features and tell the generator where to focus. Inspired by preceding attention works such as CBAM [44] and CBM [34], which stacks channel attention and spatial attention in series, we exploit the attention mechanism in our approach. However, unlike the prior approaches, we implement the attention by the encoderdecoder manner, thinking that the non-linear convolution can pay attention to context knowledge of features, which is more suitable for dense generation tasks. Besides, in order to have stability training, the spectral normalization [32] is exploited to regularize the ConvTranspose2d layers of DC-GAN, which controls the weights of modules by the Lipschitz constant.

Concretely, as displayed in Fig. 2, the noise z is input to generator $\mathcal { G }$ to synthesize the alternative samples xˆ. We first divide the whole DCGAN module into four blocks. Then we plug the attention module at each block to compute the low-dimensional feature maps $\mathcal { A } _ { d } \in \mathbb { R } ^ { \mathrm { C } / r \times \mathrm { H } \times \mathrm { W } }$ from original feature maps $\mathcal { F } _ { g } \in \mathbb { R } ^ { \mathbf { \tilde { C } } \times \mathbf { H } \times \mathbf { W } }$ , where r is the scaled scalar, C denotes channel, H and W represent the size of the feature maps. This aims to achieve lightweight feature maps. Next, the encoder is employed to achieve the latent space as follows:

![](images/d0e9b3370181c8001007d59dc4e695afc200ec6f0e1a9d53bf5a26fb9b744df7.jpg)  
Figure 3. The MHA module of teacher and student in distillation stage.

$$
\left\{ \begin{array} { l l } { \mathcal { A } _ { d } = \mathrm { C o v } ^ { 1 \times 1 } ( \mathcal { F } _ { g } ) , } \\ { \Psi = \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { C o v } ^ { 3 \times 3 } ( \mathcal { A } _ { d } ) ) ) , } \\ { \Gamma = \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { C o v } ^ { 3 \times 3 } ( \mathrm { M P } ( \Psi ) ) ) ) , } \end{array} \right.\tag{4}
$$

where Ψ represents features of intermediate process, $\mathrm { C o v ^ { 1 \times 1 } }$ and $\mathrm { \dot { C } o v ^ { 3 \times 3 } }$ denote the convolution with kernel size of $1 \times 1$ and $3 \times 3$ , and MP represents maximum pooling.

By Eq. 4, we can get the representation of lowdimensional latent space Γ from $\mathcal { A } _ { d } ^ { \sf ^ { ^ { \bot } } } \in \mathbb { R } ^ { \mathbf { C } / r \times \mathrm { H } \times \mathbf { W } } ,$ , and then decode the space with maximum uppooling (MUP) to achieve spatial-wise attention $\mathcal { A } _ { s } \in \mathbb { R } ^ { \tilde { \mathrm { C } } / \tilde { r } \times \tilde { \mathrm { H } } \times \tilde { \mathrm { W } } }$ . This operation can preserve information of the key locations in the feature to achieve the 2D spatial attention map as follows:

$$
\left\{ \begin{array} { l l } { \Psi = \mathrm { M U P } ( \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { D C } ^ { 3 \times 3 } ( \Gamma ) ) ) ) , } \\ { A _ { s } = \mathrm { C o v } ^ { 1 \times 1 } ( \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { D C } ^ { 3 \times 3 } ( \Psi ) ) ) ) , } \end{array} \right.\tag{5}
$$

where $\mathrm { D C ^ { 3 \times 3 } }$ denotes the deconvolution with kernel size of $3 \times 3 .$ , MUP represents the maximize unpooling. Then, aggregating the attention maps to the original feature maps to achieve $\tilde { \mathcal { F } } _ { g }$ is formulated as:

$$
\tilde { \mathcal { F } } _ { g } = \lambda ( \mathrm { S o f t m a x } ( \mathcal { A } _ { s } ) \times \mathcal { F } _ { g } ) + \mathcal { F } _ { g } ,\tag{6}
$$

where $\lambda$ is the hyperparameter to balance the attention maps with features, which defaults to $5 e ^ { - 2 }$ in our experiments. More details about the contributions of the attention generator are presented in Tab. 6.

## 4.2. Mixed High-Order Attention Distillation

In the stage of distillation, traditional DFKD methods [5, 11, 30] to solve coarse-grained classification commonly exploit the distribution of output layers due to the significant inter-class variation (compared to intra-class variation), which enables deep networks to learn generalized discriminatory features of coarse-grained classification.

However, the distribution knowledge distillation only exploits category-related information with dark knowledge [16], which lacks semantically relevant information. We argue that this paradigm may not be ideal for FGVC, due to the data-free scenario.

To solve the above difficulties, recent methods commonly exploit attention mechanism [3, 31] to capture the discriminative features of the object. However, the existing FGVC methods of attention mechanism are mainly designed for data-available scenarios, and there is no related research in the data-free scenario. This motivates us to extend this strategy in a data-free setting. Besides, the related attention distillation works [52, 37] only consider the loworder attention information, which only focuses on the local information and cannot capture the complex interactions among parts, resulting in less discriminative attention proposals and failing in capturing the subtle differences among objects. In the data-free scenario, due to the semantic information being sparse [9], we believe that low-order attention distillation cannot fully express the knowledge of the features. Thus we propose to exploit mixed high-order attention (MHA) to distill the aggregated local features and semantic context relation of synthesized FGVC images.

Our mixed high-order attention module is shown in Fig. 3, in which mixed 3-order attention (i.e., $\mathrm { ~ R ~ } = \ 3 )$ is exploited. The feature $\mathcal { F } _ { m } \in \mathcal { R } ^ { \mathrm { H \times W \times C } } \mathrm { ~ i ~ }$ s first extracted by three route $1 \times 1$ convolutions to achieve 3-order intermediate representations. In each route, the convolution layer and produced relative representation are the same as the order R. Then we multiply the representations of each order to obtain aggregated representations. For each aggregated representation, we exploit RELU and 1  1 convolution to produce the new map which will be aggregated with global attention maps $A _ { m }$ . At last, the activated global attention map $A _ { m }$ will be multiplied with the original features $\mathcal { F } _ { m }$ to produce the final attention maps $\tilde { \mathcal { F } } _ { m } = \mathcal { A } _ { m } \times \mathcal { F } _ { m }$

For teacher and student, their channels may be different. Thus we first exploit the Adapter to upgrade the channel of the student to the same number as the teacher, which is also implemented by the 1  1 2D convolution. Therefore, at each block of the intermediate layer of and , we exploit mean square error (MSE) to measure the MHAD loss, which is formulated as:

$$
\mathcal { L } _ { \mathrm { M H A D } } = \frac { 1 } { \mathrm { N } \times \mathrm { C } } \sum _ { i = 1 } ^ { \mathrm { N } } \sum _ { j = 1 } ^ { \mathrm { C } } \mathrm { M S E } ( \mathcal { F } _ { m } ^ { t } , \mathcal { F } _ { m } ^ { s } ) ,\tag{7}
$$

where N and C represent the batch size and channel, while $\mathcal { F } _ { m } ^ { t }$ and $\mathcal { F } _ { m } ^ { s }$ denote the attention map of an intermediate block of teacher and student, respectively. It should be

Algorithm 1 The whole pipeline of DFKD-FGVC.   
Input: A pre-trained teacher model on real data, genera  
tor and student network ${ \mathcal { S } } .$   
Output: A well-trained student network ${ \mathcal { S } } .$   
1: // Ganerator Stage   
2: for number ofiterations do   
3: for t steps iterations do   
4: Generate random noise $z \sim \mathcal { N } ( 0 , 1 )$   
5: Synthesize supporting sample $\hat { x } = \mathcal G ( z )$   
6: Optimize the generator by $\mathcal { L } _ { \mathrm { B N } } ,$ and $- \mathcal { L } _ { \mathrm { K D } } ;$   
7: Freeze and , and update by Eq. 10 .   
8: end for   
9: end for   
10: // Distillation Stage   
11: for number ofiterations do   
12: for k steps iterations do   
13: Generate random noise $z \sim \mathcal { N } ( 0 , 1 )$   
14: Synthesize supporting sample $\hat { x } = \mathcal G ( z )$   
15: Calculate discrepancy by ${ \mathcal { L } } _ { \mathrm { K D } } .$ , <sub>MHAD</sub>, and   
LSFCL<sup>.</sup>   
16: Freeze  and $\tau ,$ and update  by Eq. 11 ;   
17: end for   
18: end for

noted that this strategy is only exploited during our training process, which does not participate in the inference. Therefore, this does not affect the efficiency of the model.

## 4.3. Semantic Feature Contrast Learning

Since the pre-trained teacher has a higher discriminative ability than the student, optimizing the student by comparing the features of the teacher is conducive to improving the ability of the student to distinguish right from wrong. Therefore, in our FGVC task, we not only focus on intermediate low-level feature variances but also high-level semantic variances of the penultimate layer. Unlike traditional paradigms [6, 7, 21, 39], which contrast the original [7, 21] and augmentation data or in data-driven scenarios [6, 39]. we exploit high-level semantic features to contrast feature representation of teacher and student and aim to learn the variances between different categories in the data-free scenarios, which are more difficult than data-driven scenarios.

Specifically, in the penultimate layer, we obtain their semantic feature representations and exploit the multi-layer perceptron (MLP) to map the representations to a common space to achieve 2N feature representations as $\mathcal { F } _ { s } ~ =$ $\mathcal { C } (  { \boldsymbol { S } } (  { \boldsymbol { G } } ( z ) ) )$ and $\mathcal { F } _ { t } = \mathcal { C } ( \mathcal { T } ( \mathcal { G } ( z ) ) )$ , where $\mathcal { C }$ is the MLP layer with two hidden linear layers. Then, we normalize the features to a unit hyperspace and measure their similarity as follows:

$$
s i m ( \mathcal { F } _ { t } , \mathcal { F } _ { s } ) = \frac { \mathcal { F } _ { t } \cdot \mathcal { F } _ { s } ^ { \top } } { \Vert \mathcal { F } _ { t } \Vert \cdot \Vert \mathcal { F } _ { s } \Vert } ,\tag{8}
$$

Table 1. Results of different data-free distillation methods on three fine-grained datasets.
<table><tr><td></td><td>Setting</td><td>Prior Info.</td><td colspan="2">Compression Info.</td><td colspan="3">Accuracy</td></tr><tr><td>Method</td><td>Data-free</td><td>BN</td><td>FLOPs</td><td>Params.</td><td>Aircraft</td><td>Cars196</td><td>CUB200</td></tr><tr><td>ResNet-34 (T.)</td><td>X</td><td>X</td><td>~3.67G</td><td>~22M</td><td>70.15</td><td>84.22</td><td>76.87</td></tr><tr><td>ResNet-18 (S.)</td><td>X</td><td>X</td><td>~1.82G</td><td>~11M</td><td>68.71</td><td>77.43</td><td>58.60</td></tr><tr><td>ZSKD [33]</td><td>√</td><td>X</td><td>~1.82G</td><td>~11M</td><td>37.32</td><td>26.21</td><td>30.53</td></tr><tr><td>ZSKT [30]</td><td>√</td><td>X</td><td>~1.82G</td><td>~11M</td><td>51.16</td><td>28.48</td><td>31.88</td></tr><tr><td>DAFL [5]</td><td>V</td><td>X</td><td>~1.82G</td><td>~11M</td><td>43.69</td><td>37.71</td><td>31.01</td></tr><tr><td>DFAD [11]</td><td>√</td><td>X</td><td>~1.82G</td><td>~11M</td><td>49.51</td><td>48.72</td><td>40.15</td></tr><tr><td>ADI [47]</td><td>√</td><td>√</td><td>~1.82G</td><td>~11M</td><td>58.14</td><td>65.24</td><td>47.63</td></tr><tr><td>DFQ [8]</td><td>√</td><td>V</td><td>~1.82G</td><td>~11M</td><td>60.22</td><td>66.14</td><td>48.43</td></tr><tr><td>MAD [10]</td><td>√</td><td>√</td><td>~1.82G</td><td>~11M</td><td>63.74</td><td>67.53</td><td>53.43</td></tr><tr><td>CMI [12]</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>~1.82G</td><td>~11M</td><td>63.57</td><td>68.74</td><td>53.53</td></tr><tr><td>Ours</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>~1.82G</td><td>~11M</td><td>65.76</td><td>71.89</td><td>56.93</td></tr></table>

where denotes the inner (dot) product. The cosine distance is used as the similarity metric to measure the relationship between two feature representations for contrastive loss, which is defined as

$$
\mathcal { L } _ { \mathrm { S F C L } } = \operatorname* { m i n } _ { S } \left\{ - \log \frac { \exp ( s i m ( \mathcal { F } _ { t } ^ { i } , \mathcal { F } _ { s } ^ { j } ) / \tau ) } { \sum _ { k } ^ { 2 \mathrm { N } } \mathbb { 1 } _ { [ k \neq i ] } \exp ( s i m ( \mathcal { F } _ { t } ^ { i } , \mathcal { F } _ { s } ^ { k } ) / \tau ) } \right\} ,\tag{9}
$$

where $\mathbb { 1 } _ { [ k \neq i ] }$ is an indicator function that returns 1 if $i = j ,$ i and $j \in 2 \mathrm { N }$ are the indexes of the samples in the representations, and τ denotes a temperature parameter. This loss maximizes the representations of the different categories, where the teacher can extract the effective features from noisy images and pull away from the other dissimilar features. Therefore, if one feature of the teacher is viewed as an anchor, and the student extracts another representation of this synthetic image as the positive. Due to the weak ability of sample representations of the student model, such operation of the student plays a role as augmented images. The other 2(N 1) features can be viewed as the negative. Therefore, the loss $\mathcal { L } _ { \mathrm { S F C L } }$ is used to optimize the student to close to the teacher model, i.e., improving the ability of students to distinguish different samples.

## 4.4. Total Objects

In the whole algorithm pipeline 1, we first optimize the generator to synthesize more realistic diverse samples. The total objective of the generator is

$$
\operatorname* { m i n } _ { \mathcal { G } } \alpha \mathcal { L } _ { \mathrm { B N } } - \mathcal { L } _ { \mathrm { K D } } .\tag{10}
$$

Then, with the above strategy for the generator, we can detail the total objective of the student:

$$
\operatorname* { m i n } _ { \mathcal { S } } \mathcal { L } _ { \mathrm { K D } } + \beta \mathcal { L } _ { \mathrm { M H A D } } + \gamma \mathcal { L } _ { \mathrm { S F C L } } ,\tag{11}
$$

where $\alpha , \beta ,$ and $\gamma$ are both hyper-parameters. The training plays an adversarial distillation to optimize both at each iteration.

## 5. Experiments

## 5.1. Datasets and Implementation Details

Datasets. To demonstrate the effectiveness of our approach, we conduct experiments on three fine-grained datasets.

Aircraft FGVC-Aircraft [29] contains 100 different aircraft variants formed by 10,000 annotated images, which is divided into two subsets, i.e., the training set with 6,667 images and the testing set with 3,333 images.

Cars196 The Stanford Cars dataset [23] contains 16,185 images from 196 categories of cars. The data is split into 8,144 training images and 8,041 testing images.

CUB200 The Caltech-UCSD birds dataset (CUB-200- 2011) [43] consists of 11,788 annotated images in 200 subordinate categories, including 5,994 images for training and 5,794 images for testing.

Implementation Details. Our method is implemented with the PyTorch library. All the models are trained on NVIDIA 3090 GPUs with 24G memory. ResNet-34 [15] is employed as the cumbersome teacher network for all experiments in this paper, and four architectures, i.e., ResNet-18 [15], WRN40-2 [15], MobileNetV2 [36], and ResNet-34 [15] are utilized as students. We first train the generator for 20 steps (i.e., t=20) where the generator follows the architecture of DCGAN [35]. Adam [22] is adopted to optimize the generator with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ and beta is set 0.5 to 0.99. Then, we train the student 15 steps (i.e., k=15) after the generator and optimize the parameters by the SGD optimizer with a momentum of 0.9, a batch size of 64 as default, and a weight decay of $5 \times 1 0 ^ { - 4 }$ The initial learning rate starts at $1 \times 1 0 ^ { - 2 }$ with cosine annealing for a total of 200 epochs. In the pre-trained stage, due to subtle discrepancies that are difficult to detect, the input images of fine-grained datasets are both resized and randomly cropped to 224 244. In the data-free distillation stage, all the synthetic images are the same as the size of the original input images in the pre-trained stage. As for the hyper-parameters, both α, β, and γ are set to 0.3, 10, and 8 by default, respectively. Floating point operations (FLOPs) and parameters (Params) are employed to measure the computation and storage cost of the networks.

![](images/1b21ea855e352bc15d9a2988f6fa6ea8106a42cd3d4ba51e20d03d34e3548939.jpg)  
Figure 4. Visualization synthetic images generated by some representative approaches on Aircraft, Cars196, and CUB200 datasets.

## 5.2. Results and Comparisons

As shown in Table 1, we focus on evaluating our approach and other compared methods on three public finegrained datasets, i.e., Aircraft, Cars196, and Cub200. To evaluate the effectiveness of our proposed method, we conduct fair comparison experiments with two kinds of DFKD methods which are primarily for general classification tasks: (1) Without ( ) prior information methods, including ZSKT, DAFL, and DFAD; (2) With (-) prior information methods, including ADI, DFQ, MAD, and CMI. The first two rows of the table show the results of the teacher and student with annotated data supervision in training, which is also our target to achieve by KD. Obviously, the performance of the methods exploiting prior information is better than those without. For example, DFAD only achieves 49.51%, 48.72%, and 40.15% on three datasets, while ADI can achieve 58.14%, 65.24%, and 47.63%. This is mainly because BN regularization has a good performance to inverse and generate relatively realistic images, which is particularly important for downstream distillation. Based on the BN regularization, our approach exploits the spatial attention generator to generate the images with semantic information, which can further improve the performance of the student.

Besides, almost all of the above approaches exploit the vanilla KD (e.g., KL divergence) to transfer the knowledge from the output layer, although they can perform well on coarse-grained classification, but do not perform well on fine-grained classification. Our method mainly adopts two strategies to further improve the performance of the student by about 3% on average, which indicates that vanilla KD alone cannot complete all knowledge transfer, and special design is necessary for FGVC tasks distillation in DFKD. Under identical conditions, thanks to two optimization strategies, i.e., MHAD and SFCL, our approach outperforms the other data-free methods to achieve state-of-the-art performance on three datasets.

Table 2. More comparisons of different architectures’ students with ResNet-34 on Aircraft dataset.
<table><tr><td>Student</td><td>ZSKT</td><td>DFAL</td><td>DFAD</td><td>ADI</td><td>DFQ</td><td>MAD</td><td>CMI</td><td>Ours</td></tr><tr><td>WRN40-2</td><td>49.13</td><td>36.83</td><td>50.44</td><td>57.83</td><td>58.26</td><td>59.85</td><td>62.43</td><td>64.54</td></tr><tr><td>MobileNetV2</td><td>24.39</td><td>18.51</td><td>23.01</td><td>53.66</td><td>53.93</td><td>54.61</td><td>55.04</td><td>57.37</td></tr><tr><td>ResNet-34</td><td>39.52</td><td>36.63</td><td>52.15</td><td>60.75</td><td>61.75</td><td>63.12</td><td>64.66</td><td>65.48</td></tr></table>

To verify the generality of our approach, we perform distillation on another three student models with different architectures, including heterogeneous distillation (i.e., WRN40-2 and MobileNetV2) and self-distillation (i.e., Resnet-34). For WRN40-2 and MobileNetV2, we leverage the MLP with two hidden layers to map the dimension to match the teacher and implement our two strategies both in the penultimate layer. As shown in Table 2, our approach can also achieve state-of-the-art performance in different architectures.

## 5.3. Visualization and Analysis

Synthetic images. To clearly evaluate the effect of synthesized images, we present a visualization analysis of some representative methods on Aircraft, Cars196, and CUB200 in this section. As we can see from Fig. 4, the first column is the real data for reference. However, for ZSKT, DAFL, and DFAD, there is a big gap between the generated images and the real data. Since ADI, CMI, and Ours both exploit the BN to regularize the features, the synthesized images are more realistic than the other data-free methods, which is beneficial for downstream distillation. With the assistance of two optimization strategies of MHAD, and SFCL, our approach can generate better and more discriminative foreground images compared to ADI, DFQ, and CMI. For example, we can clearly distinguish the outline of the car and the color of the different areas of the birds.

![](images/024e3ebcc728dee9dadeddaa373955b7381ae59902dcefafd20cc49582962fa4.jpg)  
Figure 5. Visualization of t-SNE distribution on Aircraft dataset.

t-SNE. To illustrate the advantages of our approach in synthesizing images having more similar distributions with real images, we sample 10 categories from the Aircraft dataset and visualize the representations of MobileNetV2 by t-SNE as Fig. 5. As shown in Fig. 5(f), our approach gains obviously better representations than the other methods, according to the comparison with each other. Compared the Tab. 1 with Fig. 4, we can conclude that the performance of the student primarily relies on the quality of the synthetic images and the effect of knowledge transfer in DFKD.

Attention map. To further verify the effect of our mixed high-order attention (MHA) modules feature selection, we visualized the generated samples through GradCAM <sup>1</sup>, as shown in Fig. 6. The first row is the synthesized alternative samples of CUB200 which are generated by our attention module. We can see the fine-grained semantic information of different synthesized birds. For example, we can distinguish different beaks or wings of birds, and different colors of features. When we employ the student embedded with MHA modules to visualize birds’ discriminative features by GradCAM, the attention maps are sparse and focus on the discriminative parts, as shown in the second row of the figure. For example, the wings of birds are activated, which indicates that the wings are being paid attention to. We can conclude that MHA modules can focus on contextual semantic information on features which is based on the attention of discriminative features.

## 5.4. Ablation Study

Contribution of loss. To verify the contribution of each component, we conduct ablation experiments on the three datasets with ResNet-18, as shown in Table 3. In the first row is the Baseline of each benchmark, which exploits the Eq. 10 to optimize the ${ \mathcal { G } } .$ , while only optimizing the by exploiting the ${ \mathcal { L } } _ { \mathrm { K L } }$ to distill the knowledge. Then, adding the $\mathcal { L } _ { \mathrm { S F C L } }$ component to the Baseline, the result of each benchmark is improved by 3.07%, 2.33%, 2.92%, respectively. Likewise, when we add $\mathcal { L } _ { \mathrm { M H A D } }$ to the baseline, it can also achieve significant improvement. Nevertheless, by comparing both, we can find that the contribution of <sub>SFCL</sub> is relatively weaker than $\mathcal { L } _ { \mathrm { M H A D } }$ , which proves the effectiveness of exploiting mixed high-order attention to model discriminative features, which has been ignored by other methods. Finally, we add both to the baseline and obtain the final state-of-the-art effect.

![](images/05ba8508281a3a35888571ccb28526106d2ae4ed189d252b6614466f4174e0ce.jpg)  
Figure 6. Visualization of synthetic images with attention map generated by GradCAM on CUB200 datasets.

Table 3. The ablation study of our approaches with different components. $\cdot _ { + } \cdot$ denotes the add operation.
<table><tr><td>Method</td><td>Aircraft</td><td>Cars196</td><td>Cub200</td></tr><tr><td>Baseline</td><td>60.30</td><td>64.80</td><td>51.34</td></tr><tr><td> $+ \mathcal { L } _ { \mathrm { S F C L } }$ </td><td>63.37</td><td>67.13</td><td>54.26</td></tr><tr><td> $+ \mathcal { L } _ { \mathrm { M H A D } }$ </td><td>64.86</td><td>69.92</td><td>55.71</td></tr><tr><td>+  $\mathcal { L } _ { \mathrm { M H A D } } + \mathcal { L } _ { \mathrm { S F C L } }$ </td><td>65.76</td><td>71.89</td><td>56.93</td></tr></table>

Effect of hyper-parameters. In our optimization, $\alpha , \beta ,$ and $\gamma$ are the major hyper-parameters for balancing the loss terms in our framework. By adjusting the BN hyperparameter in the interval between 0 to 5, we find that the optimal value of α is 0.3. Then, we investigate the effect of $\beta$ and $\gamma$ on the student ResNet-18 on the Cars196 dataset and show the results in Fig. 7. In Fig. 7(a), we set γ as 1.0 and vary $\beta$ from 0 to 100, in which 10 is a reasonable parameter verified by our experiments. Then, we set the optimal value of $\beta$ to 10 and vary $\gamma$ from 0 to 100, in which the student network achieves the best performance when $\gamma$ is set to $^ { 8 , }$ as shown in Fig. 7(b). It is clear that, when using different $\beta$ and $\gamma ,$ our model stably outperforms the baseline model. The experimental results show that our proposed framework is robust to the different parameters.

Control parameter of attention ganerator. Due to the parameter λ being exploited to control the aggregating of attention and feature maps, we perform a group analysis of this parameter. As shown in Tab. 4, we first fix the other parameters, and then the λ is set to 0, which indicates that the generator does not exploit the attention mechanism. And the results on the three datasets achieve 63.88, 69.24, and 54.81, respectively. From the interval 0 to 5e−<sup>2</sup>, the effect of the generator rises significantly while the effect of the model decays between 5e−<sup>2</sup> and $9 { \dot { e } } ^ { - 2 }$ , in which the reasonable parameter is $5 e ^ { - 2 } .$ . This indicates that the generator needs to be moderate when employing attention. When the generator pays too much attention to the attention image, it may destroy the original synthesized images resulting in the degradation of the model.

![](images/8c84f73f013589c3dab5bf114cc418c853a409efa190b2885bbe064efe87c062.jpg)

![](images/e27368130546f2647490f8f9d6f34a01e439316d3134ea3e8f71f590e4b675d0.jpg)  
(a) α = 0.3, γ = 1.0, adjust β  
(b) α = 0.3, β = 10, adjust γ  
Figure 7. Effect of hyper-parameter β and γ on Cars196 dataset.

Table 4. The effect of λ under different parameters.
<table><tr><td>λ</td><td>Aircraft</td><td>Cars196</td><td>CUB200</td></tr><tr><td>0</td><td>63.88</td><td>69.24</td><td>54.81</td></tr><tr><td> $1 e ^ { - 2 }$ </td><td>64.32</td><td>69.87</td><td>55.44</td></tr><tr><td> $5 e ^ { - 2 }$ </td><td>65.76</td><td>71.89</td><td>56.93</td></tr><tr><td> $7 e ^ { - 2 }$ </td><td>65.02</td><td>70.95</td><td>56.30</td></tr><tr><td> $9 e ^ { - 2 }$ </td><td>64.23</td><td>70.36</td><td>55.81</td></tr></table>

Order effectiveness of MHA. We adopt mixed 3-order attention distillation in our method, which is mainly due to the 3-order attention having the ability to pay attention to the context information. It has more information than the 1-order attention. In this section, we conduct experiments to verify the effect of different orders on different FGVC datasets. As can be seen from Tab 5, when exploiting the 1-order attention distillation, we can only achieve 64.31, 69.26, and 56.12 on three datasets. However, when we exploit the 3-order attention distillation, we can improve the scores of 1.5% on average. What exceeded our expectations is the lower effect when 2-order attention was used. We believe that the 2-order attention mainly focuses on the global information, including the background information, which confuses the foreground attention and reduces the effect of attention.

Table 5. The effect of different orders on different FGVC datasets.
<table><tr><td>Order</td><td>Aircraft</td><td>Cars196</td><td>CUB200</td><td>Avg.</td></tr><tr><td> $R = 1$ </td><td>64.31</td><td>69.26</td><td>56.12</td><td>63.23</td></tr><tr><td> $R = 2$ </td><td>63.12</td><td>70.35</td><td>55.06</td><td>62.84</td></tr><tr><td> $R = 3$ </td><td>65.76</td><td>71.89</td><td>56.93</td><td>64.86</td></tr></table>

## 5.5. Architecture of generator

As illustrated in Fig. 1, the generator with spatial-wise attention modules is adopted in our experiments. Therefore, we detail the architecture of the generator and attention module as indicated in Tab. 6. Concretely, our generator is isomorphic to DCGAN [35]. However, to facilitate the calculation of the spatial-wise attention module, we divide the generator into four blocks. At each block, we exploit spectral normalization to normalize the weights of deconvolution, which aims to stabilize the training of the generator. Then, the encoder-decoder spatial-wise attention module is plugged into each block of the generator, in which the indexes of Maxpool are also used in the MaxUnpool to focus on the key position of synthesized features.

Table 6. The Left. Attention Generator Architectures. The noise is mapped to the features which are upsampled to the required image size. The SN denotes the spectral normalization, while SAM represents spatial-wise attention modules corresponding to the Right.
<table><tr><td rowspan=1 colspan=1>Attention Generator</td><td rowspan=1 colspan=1>Spatial-wise Attention Modules</td></tr><tr><td rowspan=1 colspan=1>FC, Reshape, BN</td><td rowspan=1 colspan=1> $1 \times 1 C \to C / r { \mathrm { C o n v } }$ </td></tr><tr><td rowspan=1 colspan=1> $3 \times 3 , 5 1 2  2 5 6$ Deconv $\uparrow _ { 2 \times } ,$ SN, LReLU, SAM</td><td rowspan=1 colspan=1> $3 \times 3 , C / r \to 2 C / r , \mathrm { C o n v } ,$ BN, ReLU, Maxpool</td></tr><tr><td rowspan=1 colspan=1>3 × 3, 256 → 128, Deconv $\uparrow _ { 2 \times } ,$ SN, LReLU, SAM</td><td rowspan=1 colspan=1> $3 \times 3 , 2 C / r \to 4 C / r , \mathrm { C o n v } ,$ BN, ReLU</td></tr><tr><td rowspan=1 colspan=1> $3 \times 3 , 1 2 8$ → 64, Deconv $\uparrow _ { 2 \times } ,$ SN, LReLU, SAM</td><td rowspan=1 colspan=1> $3 \times 3 , 4 C / r \to 2 C / r .$ Decov,BN, ReLU, MaxUnpool</td></tr><tr><td rowspan=1 colspan=1> $3 \times 3 , 6 4$ → 64, Deconv ↑2×,SN, LReLU, SAM</td><td rowspan=1 colspan=1> $3 \times 3 , 2 C / r  C / r , \mathrm { D e c o v } ,$ BN, ReLU</td></tr><tr><td rowspan=1 colspan=1>3 × 3, 64 → 3, Conv, Tanh</td><td rowspan=1 colspan=1> $1 \times 1 , C / r \to C , \mathrm { C o n v , S o f t M a x }$ </td></tr></table>

\* C is the input channel of each block, while r is scale scalar.

## 6. Conclusion

In this paper, we address the data-free distillation for FGVC. We propose to exploit the generator with spatial attention to synthesize the images with discriminative features. Then, two effective strategies are exploited to optimize the student by MHAD and SFCL, where MHAD captures the discriminative features with context information and SFCL exploits the high-level semantic features to contrast the variances between the different categories. Experimental evidence demonstrates that both approaches can improve the performance of the student on FGVC tasks and outperform other data-free distillation approaches to achieve state-of-the-art performance.

## References

[1] Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust optimization, volume 28. Princeton university press, 2009.

[2] Thomas Berg and Peter N Belhumeur. Poof: Part-based onevs.-one features for fine-grained categorization, face verification, and attribute estimation. In CVPR, pages 955–962, 2013.

[3] Sijia Cai, Wangmeng Zuo, and Lei Zhang. Higher-order integration of hierarchical convolutional activations for finegrained visual categorization. In ICCV, pages 511–520, 2017.

[4] Binghui Chen, Weihong Deng, and Jiani Hu. Mixed highorder attention network for person re-identification. In CVPR, pages 371–381, 2019.

[5] Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, and Qi Tian. Data-free learning of student networks. In ICCV, pages 3514–3522, 2019.

[6] Liqun Chen, Dong Wang, Zhe Gan, Jingjing Liu, Ricardo Henao, and Lawrence Carin. Wasserstein contrastive representation distillation. In CVPR, pages 16296–16305, 2021.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, pages 1597–1607. PMLR, 2020.

[8] Yoojin Choi, Jihwan Choi, Mostafa El-Khamy, and Jungwon Lee. Data-free network quantization with adversarial knowledge distillation. In CVPR, pages 710–711, 2020.

[9] Yao Ding, Yanzhao Zhou, Yi Zhu, Qixiang Ye, and Jianbin Jiao. Selective sparse sampling for fine-grained image recognition. In ICCV, pages 6599–6608, 2019.

[10] Kien Do, Hung Le, Dung Nguyen, Dang Nguyen, HARIPRIYA HARIKUMAR, Truyen Tran, Santu Rana, and Svetha Venkatesh. Momentum adversarial distillation: Handling large distribution shifts in data-free knowledge distillation. In Advances in NeurIPS, 2022.

[11] Gongfan Fang, Jie Song, Chengchao Shen, Xinchao Wang, Da Chen, and Mingli Song. Data-free adversarial distillation. arXiv preprint arXiv:1912.11006, 2019.

[12] Gongfan Fang, Jie Song, Xinchao Wang, Chengchao Shen, Xingen Wang, and Mingli Song. Contrastive model inversion for data-free knowledge distillation. In IJCAI, 2021.

[13] Weifeng Ge, Xiangru Lin, and Yizhou Yu. Weakly supervised complementary parts models for fine-grained image classification from the bottom up. In CVPR, pages 3034– 3043, 2019.

[14] Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-Tao Jiang, Tai-Jiang Mu, Song-Hai Zhang, Ralph R Martin, Ming-Ming Cheng, and Shi-Min Hu. Attention mechanisms in computer vision: A survey. Computational Visual Media, 8(3):331–368, 2022.

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.

[16] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[17] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. pages 7132–7141, 2018.

[18] Shaoli Huang, Zhe Xu, Dacheng Tao, and Ya Zhang. Partstacked cnn for fine-grained visual categorization. In CVPR, pages 1173–1182, 2016.

[19] Ruyi Ji, Longyin Wen, Libo Zhang, Dawei Du, Yanjun Wu, Chen Zhao, Xianglong Liu, and Feiyue Huang. Attention convolutional binary neural tree for fine-grained visual categorization. In CVPR, pages 10468–10477, 2020.

[20] Zilong Ji, Xiaolong Zou, Xiaohan Lin, Xiao Liu, Tiejun Huang, and Si Wu. An attention-driven two-stage clustering method for unsupervised person re-identification. In ECCV, pages 20–36. Springer, 2020.

[21] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in NeurIPS, 33:18661–18673, 2020.

[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2014.

[23] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In ICCV workshops, pages 554–561, 2013.

[24] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in NeurIPS, 25, 2012.

[25] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In ICLR, 2016.

[26] Chuanbin Liu, Hongtao Xie, Zheng-Jun Zha, Lingfeng Ma, Lingyun Yu, and Yongdong Zhang. Filtration and distillation: Enhancing region attention for fine-grained visual categorization. In AAAI, volume 34, pages 11555–11562, 2020.

[27] Yuang Liu, Wei Zhang, and Jun Wang. Zero-shot adversarial quantization. In CVPR, pages 1512–1521, 2021.

[28] Yuang Liu, Wei Zhang, Jun Wang, and Jianyong Wang. Data-free knowledge transfer: A survey. arXiv preprint arXiv:2112.15278, 2021.

[29] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[30] Paul Micaelli and Amos J Storkey. Zero-shot knowledge transfer via adversarial belief matching. Advances in NeurIPS, 32, 2019.

[31] Shaobo Min, Hantao Yao, Hongtao Xie, Zheng-Jun Zha, and Yongdong Zhang. Multi-objective matrix normalization for fine-grained visual recognition. IEEE Transactions on Image Processing, 29:4996–5009, 2020.

[32] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In ICLR, 2018.

[33] Gaurav Kumar Nayak, Konda Reddy Mopuri, Vaisakh Shaj, Venkatesh Babu Radhakrishnan, and Anirban Chakraborty. Zero-shot knowledge distillation in deep networks. In ICML, pages 4743–4751. PMLR, 2019.

[34] Jongchan Park, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Bam: Bottleneck attention module. BMVC, 2018.

[35] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2015.

[36] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR, pages 4510–4520, 2018.

[37] Changyong Shu, Yifan Liu, Jianfei Gao, Zheng Yan, and Chunhua Shen. Channel-wise knowledge distillation for dense prediction. In ICCV, pages 5311–5320, 2021.

[38] Teik Toe Teoh and Zheng Rong. Deep convolutional generative adversarial network. In Artificial Intelligence with Python, pages 289–301. Springer, 2022.

[39] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. arXiv preprint arXiv:1910.10699, 2019.

[40] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang. Residual attention network for image classification. In CVPR, pages 3156–3164, 2017.

[41] Zhuhui Wang, Shijie Wang, Haojie Li, Zhi Dou, and Jianjun Li. Graph-propagation based correlation learning for weakly supervised fine-grained image classification. In AAAI, volume 34, pages 12289–12296, 2020.

[42] Xiu-Shen Wei, Yi-Zhe Song, Oisin Mac Aodha, Jianxin Wu, Yuxin Peng, Jinhui Tang, Jian Yang, and Serge Belongie. Fine-grained image analysis with deep learning: A survey. TPAMI, 2021.

[43] Peter Welinder, Steve Branson, Takeshi Mita, Catherine Wah, Florian Schroff, Serge Belongie, and Pietro Perona. Caltech-ucsd birds 200. 2010.

[44] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In ECCV, pages 3–19, 2018.

[45] Di Wu, Chao Wang, Yong Wu, Qi-Cong Wang, and De-Shuang Huang. Attention deep model with multi-scale deep supervision for person re-identification. IEEE Trans. ETCI, 5(1):70–78, 2021.

[46] Jing Xu, Rui Zhao, Feng Zhu, Huaming Wang, and Wanli Ouyang. Attention-aware compositional network for person re-identification. In CVPR, pages 2119–2128, 2018.

[47] Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong Li, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. In CVPR, pages 8715–8724, 2020.

[48] Ning Zhang, Jeff Donahue, Ross Girshick, and Trevor Darrell. Part-based r-cnns for fine-grained category detection. In ECCV, pages 834–849. Springer, 2014.

[49] Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Chris De Sa, and Zhiru Zhang. Improving neural network quantization without retraining using outlier channel splitting. In ICML, pages 7543–7552. PMLR, 2019.

[50] Yifan Zhao, Ke Yan, Feiyue Huang, and Jia Li. Graph-based high-order relation discovery for fine-grained recognition. In CVPR, pages 15079–15088, 2021.

[51] Heliang Zheng, Jianlong Fu, Tao Mei, and Jiebo Luo. Learning multi-attention convolutional neural network for finegrained image recognition. In ICCV, pages 5209–5217, 2017.

[52] Zaida Zhou, Chaoran Zhuge, Xinwei Guan, and Wen Liu. Channel distillation: Channel-wise attention for knowledge distillation. arXiv preprint arXiv:2006.01683, 2020.

[53] Peiqin Zhuang, Yali Wang, and Yu Qiao. Learning attentive pairwise interaction for fine-grained classification. In AAAI, volume 34, pages 13130–13137, 2020.