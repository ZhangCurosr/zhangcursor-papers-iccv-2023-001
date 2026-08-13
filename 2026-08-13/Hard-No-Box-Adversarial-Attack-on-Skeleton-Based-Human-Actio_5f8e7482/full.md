# Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient

Zhengzhi Lu<sup>1,2</sup> <sup>†</sup> He Wang<sup>3</sup> Ziyi Chang<sup>1</sup> Guoan Yang<sup>2</sup> Hubert P. H. Shum<sup>1</sup> <sup>‡</sup> <sup>1</sup>Durham University, UK <sup>2</sup>Xi’an Jiaotong University, China <sup>3</sup>University College London, UK

lu947867114@stu.xjtu.edu.cn he wang@ucl.ac.uk ziyi.chang@durham.ac.uk

gayang@mail.xjtu.edu.cn hubert.shum@durham.ac.uk

## Abstract

Recently, methods for skeleton-based human activity recognition have been shown to be vulnerable to adversarial attacks. However, these attack methods require either thefull knowledge ofthe victim (i.e. white-box attacks), access to training data (i.e. transfer-based attacks) or frequent model queries (i.e. black-box attacks). All their requirements are highly restrictive, raising the question of how detrimental the vulnerability is. In this paper, we show that the vulnerability indeed exists. To this end, we consider a new attack task: the attacker has no access to the victim model or the training data or labels, where we coin the term hard no-box attack. Specifically, we first learn a motion manifold where we define an adversarial loss to compute a new gradientfor the attack, named skeleton-motioninformed (SMI) gradient. Our gradient contains information ofthe motion dynamics, which is differentfrom existing gradient-based attack methods that compute the loss gradient assuming each dimension in the data is independent. The SMI gradient can augment many gradient-based attack methods, leading to a new family of no-box attack methods. Extensive evaluation and comparison show that our method imposes a real threat to existing classifiers. They also show that the SMI gradient improves the transferability and imperceptibility of adversarial samples in both no-box and transfer-based black-box settings.

## 1. Introduction

Deep learning models are vulnerable to adversarial attacks, which compute data perturbations strategically to fool trained networks. Since its discovery [31], a wide variety of models in different tasks have been attacked [1], raising severe concerns as these perturbations are imperceptible to humans. Recently, the adversarial attack in skeletonbased human activity recognition (S-HAR) has attracted attention as skeletal data have been widely used in securitycritical applications such as sports analysis, bio-mechanics, surveillance, and human-computer interactions [24].

<table><tr><td>Information Accessible</td><td>White- Box</td><td>Queried Black-Box</td><td>Transferred Black-Box</td><td>No- Box</td><td>Hard No-Box</td></tr><tr><td>Model Parameters</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Queries of Victims</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Training Samples</td><td>×</td><td>×</td><td>√</td><td>X</td><td>×</td></tr><tr><td>Labels</td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td></tr></table>

Table 1. Comparisons on different settings of adversarial attacks. ✓ and × indicate if a method needs to access the corresponding information.

Existing attacks in S-HAR are categorized into whitebox and black-box approaches. White-box approaches require prior knowledge of the full details of a victim model [18, 36] while black-box approaches require a large number of queries to the victim model [7] or the access to training data and labels [36]. On the one hand, the victim model details and the training data and labels are unlikely to be available to the attacker in real-world scenarios. On the other hand, making frequent and numerous queries (e.g. tens of thousands) to the victim model is time-consuming and raises suspicion. In other words, the settings of existing S-HAR attacks are overly restrictive. A key to a successful attack is to reduce the required information of the victim model, training data and labels.

In this paper, we introduce a new threat model that requires no access to the victim model, training data or labels. We name the new threat model the hard no-box attack, differentiating from the recent no-box attack on images [16] that does not require access to the victim model but still needs access to the labels (i.e. soft no-box attack). Table 1 demonstrates the comparison on different settings of adversarial attacks. Among all attack settings, our hard nobox attack requires the least amount of knowledge, as it can only access the testing data without labels. Designing such an attack is nontrivial and challenging. Without access to the victim model, the attack method cannot rely on the gradient of a classification loss [12], data manipulation during training [25], and the feedback of a classifier [2]. The challenge is further exacerbated by the requirement of no label and training sample access, where no surrogate model can be trained to attack or estimate the data distribution.

To tackle the challenges, we propose a contrastive learning (CL) [34] solution with a manifold-based no-box adversarial loss. First, we introduce a new application of CL to learn a latent data manifold where similar samples are naturally aggregated while dissimilar samples are dispersed without the need of class labels. It provides a good description of sample similarity that facilitates generating skeletal adversarial samples. CL is suitable for hard no-box attack settings due to its ability to capture the discriminative high-level features under our restricted attack conditions. Second, we compute the perturbation to drag a data sample away from its similar neighbors in the latent space, bounded by a pre-defined budget. In particular, we design a new nobox adversarial loss to maximize each adversary’s dissimilarity with positive samples while minimizing its similarity with negative samples. The loss serves as guidance for the adversary search in our gradient-based attack scheme.

While gradient-based attack methods like I-FGSM [14] are shown to be effective on S-HAR attacks [36, 18], the gradient is computed based on the victim model and the labels, making it unsuitable for hard no-box attacks. Since adversarial samples are likely to lie in or near the motion manifold [7], ideally, we want to explore along the manifold. That is, the computation of adversarial loss gradient should consider the local motion manifold.

To this end, we propose to explicitly model motion dynamics for describing the local manifold around a given motion. Specifically, we introduce the skeleton-motioninformed (SMI) gradient that employs dynamics models (e.g. Markovian and autoregressive) to represent motion dynamics for the loss gradient computation. As a result, while existing methods generally assume each dimension in a data sample to be independent when computing the loss gradient, SMI gradient explicitly considers the dependency between frames in time. Furthermore, the SMI gradient is compatible with existing gradient-based methods including I-FGSM and MI-FGSM [8], allowing us to effectively construct a new family of no-box attack methods.

Extensive experiments show that our method generates effective adversarial samples that successfully attack various victim models across datasets (HDM05, NTU60 and NTU120). Our SMI-gradient based attacks improve the attack transferability in both no-box and transferred black-box settings, with better imperceptibility. Codes are available in https://github.com/ luyg45/HardNoBoxAttack and our contributions are:

• We confirm the S-HAR threat by introducing a new hard no-box attack and proposing the first method to generate adversarial samples without access to the victim model or training data or labels, to the best of our knowledge.

• We propose a new skeleton-motion-informed gradient that guides the adversary search along the motion manifold, explicitly considering the spatial-temporal nature of skeletal motions.

• We present a family of novel gradient-based attack strategies facilitated by the new gradient, improving the transferability and imperceptibility of adversarial samples in no-box and transferred black-box attacks.

## 2. Related Works

Skeleton-Based Human Action Recognition S-HAR has attracted considerable attention in many applications [24] where deep learning-based approaches have achieved state-of-the-art performance [30, 13]. Recurrent neural networks are employed to model the temporal domain of human motions [9, 29]. Furthermore, unlike images and videos, the skeleton has a graph structure, so graph convolutional networks have shown to be effective in modelling the spatial or spatial-temporal features [28, 41]. The effectiveness is generally achieved by considering the skeleton as a topological graph where the joints and bones correspond to nodes and edges [21]. Improved graph designs and network architectures are subsequently proposed [20, 44, 45, 43].

Adversarial Attacks on Skeletons Adversarial attacks were initially introduced in [31], which showcases the vulnerability of deep neural networks and has been extended to other data types. Generally, adversarial attack is a special technique of data augmentation that aims to reveal the vulnerability of a system by finding new samples, while other data augmentation techniques may have diverse purposes, e.g. training efficiency and inference performance [27]. Recently, the attack on S-HAR has received increasing attention. Wang et al. [36] analyzed the perceptibility of adversarial skeletal samples and proposed a new perceptual loss. Liu et al. [18] focused on GCN-based models and utilized generative adversarial networks to synthesize adversarial examples. Tanaka et al. [32] proposed a new lower-dimensional attack, in which only the length of bones could be perturbed. These methods all require the complete knowledge of victim models, a setting known as the whitebox attack. In contrast, Diao et.al [7] introduced the first black-box S-HAR attack method, which searches motion manifolds for adversaries. Still, black-box attacks need to frequently query the victim models, which can be infeasible in real-world systems. In contrast, we consider a more practical threat setting named the hard no-box attack, where an attacker only has access to unlabeled skeletal data.

Gradient-Based Attack Strategies The core component of adversarial attacks is to generate adversarial samples [1]. Gradient-based attack methods have been widely used to introduce perturbations to a sample following the direction of the loss gradient. Goodfellow et al. [12] proposed the fast gradient sign method (FGSM) that perturbs a sample by a single step along the loss gradient. Kurakin et al. [14] proposed I-FGSM by extending FGSM to an iterative process. Dong et al. [8] presented MI-FGSM by adding momentum to the gradient, which boosted the transferability of adversarial samples. Xie et al. [40] applied diversified augmentations to the inputs before each iteration to craft more transferable samples. While these gradient-based strategies are successful in static data and have been adapted to skeletal motions, they neglect the dependency between frames for gradient computation, which is crucial in time series. This motivates us to propose our skeleton-motion-informed attack strategies, which explicitly model the motion dynamics in the temporal domain [39, 35].

![](images/846208364e899786b1ff759dacd624f5b68795acf14b0be7c6aa2de1058ded8e.jpg)  
Figure 1. The training (left) and attack (right) processes for the hard no-box attack. The trained query encoder in the training process is used for attacks in the attack process.

## 3. Hard No-Box Attack for Skeletal Data

Figure 1 shows the overview of our method. The left part is the training process where we adopt contrastive learning to obtain a latent data manifold to distinguish data samples. The attack process is shown on the right-hand side. We first design a new no-box adversarial loss in the trained latent space to guide the adversary search using samples that are dissimilar to the attacked sample. Then we propose a novel skeleton-motion-informed gradient and a new family of attack methods for generating adversarial samples.

## 3.1. Contrastive Learning for Motion Manifold

While the fundamental idea of adversarial attacks is to perturb a data sample to cross class boundaries, such boundaries cannot be estimated for hard no-box attacks due to the lack of labels. To estimate such boundaries without labels, we present a new application of contrastive learning (CL) [34] to aggregate similar data samples as soft class boundaries in latent space. Such boundaries enable us to adversarially perturb a sample to cross boundaries. We also train an encoder to extract discriminative high-level features for the motion manifold in latent space. Overall, our CL constructs boundaries in latent space without labels via aggregating similar samples and segregating dissimilar samples. Our attack is guided by the dissimilarity of high-level features between samples for generating adversarial samples, instead of using class boundaries to lead the attack [2].

To incorporate both spatial and temporal information, we train an encoder (Fig. 1 left) based on adaptive graph convolutional network (AGCN) [28]. To force encoders to focus on high-level features, we apply skeleton-specific data augmentations to an input sequence $S$ and obtain two different views $S _ { q }$ and $S _ { k }$ . Augmentations include spatial operations (e.g. pose transformations, joint jittering, etc.) and temporal operations (e.g. temporal crop and resize) [34] (detailed in supplementary material). Then, we feed $S _ { q }$ and $S _ { k }$ into the query encoder $f _ { q }$ and the key encoder $f _ { k }$ respectively for the info-noise-contrastive estimation (InfoNCE) [23]:

$$
\begin{array} { l c l } { { L _ { c o n t r a s t } = } } & { { ( 1 ) } } \\ { { \displaystyle - \log \frac { \exp { ( f _ { q } ( S _ { q } ) \cdot f _ { k } ( S _ { k } ) / \tau ) } } { \exp { ( f _ { q } ( S _ { q } ) \cdot f _ { k } ( S _ { k } ) / \tau ) } + \sum _ { F _ { n } \sim N } ^ { } \exp { ( f _ { q } ( S _ { q } ) \cdot F _ { n } / \tau ) } } , } } \end{array}
$$

where $\tau$ is the temperature parameter, N is the dynamic queue storing the features of negative samples $F _ { n }$ obtained in the training process. After training, we use the query encoder $f _ { q } ,$ , which encodes the motion manifold, for attack.

## 3.2. Adversarial Loss for Unlabeled Skeletal Data

The adversarial loss of hard no-box attack is significantly different from most existing methods that heavily depend on labels and class boundaries. Since class labels and class boundaries are unavailable in hard no-box attacks, we utilize data samples that are dissimilar to a given sample (i.e. negative samples) for defining the adversarial loss. Correspondingly, samples that are similar to the given sample are considered as positive samples. We argue when a given sample is perturbed towards its negative samples and away from its positive samples, it tends to become an adversary. This is because the negative samples generally indicate the high-density areas of other classes in the latent space.

The hard no-box adversarial loss is designed as:

$$
L _ { a d v } = - \log \frac { \exp \left[ S i m \left( f _ { q } \left( s \right) , f _ { q } \left( \tilde { s } \right) \right) \right] } { \sum _ { j } \exp \left[ S i m \left( f _ { q } \left( s \right) , f _ { q } \left( \tilde { s } _ { j } \right) \right) \right] } ,\tag{2}
$$

where Sim is the cosine similarity, s is the adversarial sample to be computed, se is the clean sample regarded as the positive sample, and $\widetilde { s _ { j } }$ are the negative samples. Maximizing $L _ { a d v }$ moves s away from $\widetilde { s }$ and towards $\widetilde { s _ { j } }$ in the latent manifold. With $L _ { a d v } .$ , gradient-based attacks are employed.

To maximize Eq. 2, the selection of negative samples $\widetilde { s _ { j } }$ is crucial and we design a method tailored for no-box attacks. Existing work [10] utilizes cluster-fit [42] to generate pseudo labels for selecting negative samples during adversarial training, which is less suitable for the no-box attack as it requires another pretrained off-line encoder to obtain pseudo labels. Instead, we adapt K-means, an unsupervised method, to select negatives, removing the need of any pretraining. We discard Q clusters whose cluster centers are the closest to the input sample, mitigating the risk of misleading attacks. The remaining cluster centers are considered as the negative samples s˜<sub>j</sub>.

## 4. Skeleton-Motion-Informed Gradient

Existing gradient-based attack methods treat each dimension of the data as an independent variable, i.e. raw gradient. Attacks based on raw gradients tend to drag a sample away from the data manifold [11]. With the guidance of class boundaries and a limit on the perturbation budget, the raw gradient can still find deceiving adversaries. However, this setting is infeasible in hard no-box attacks. Without class boundaries, raw gradients that point to the negative samples can drag the adversary far away from the manifold. This is because while the perturbations are towards negative samples, they are not necessarily in a direction orthogonal to the class boundary. Consequently, larger perturbations are needed to cross the boundary, leading to adversaries being far off the manifold. This creates the need to constrain the perturbation within or near the manifold, at least locally. Since the motion manifold is constrained by the motion dynamics [38], we argue that the gradient needs to explicitly capture the dynamics. Therefore, we propose a new gradient named skeleton-motion-informed (SMI) gradient, capturing the manifold information that has been largely ignored by existing methods in loss gradient computation.

## 4.1. Dynamics in the Gradient Structure

Given a skeletal sequence $S = [ S _ { 1 } , S _ { 2 } , \cdots , S _ { t } ]$ and the adversarial loss $J ( S )$ , a straightforward but effective strategy to craft adversarial perturbations is the gradient-based attack [1]. It utilizes backpropagation of the loss function $\nabla J ( S )$ to iteratively change input samples S:

$$
\hat { S } = S + \alpha \cdot \mathrm { s i g n } \left( \nabla J ( S ) \right) ,\tag{3}
$$

where α is the attack step size and $\hat { S }$ is the adversarial samples. In skeletal motions, this attack gradient $\nabla J ( S )$ consists of a set of partial derivatives over all frames:

$$
\nabla J ( S ) = \left[ \frac { \partial J ( S ) } { \partial S _ { 1 } } , \frac { \partial J ( S ) } { \partial S _ { 2 } } , \cdots , \frac { \partial J ( S ) } { \partial S _ { t } } \right] .\tag{4}
$$

The partial derivative $\frac { \partial J ( S ) } { \partial S _ { t } }$ assumes each frame is independent, and this is the raw gradient employed in existing methods [36]. However, human motions contain rich dynamics so that the system can be described as $S _ { t } ~ = ~ f ( S _ { < t } )$ . So far, various dynamics models have been attempted to model human motions, such as Markovian models [33, 37], autoregressive models [39], and many-to-many mapping [38], all of which can capture the dynamics at different scales in time. We explore these models to reveal the missing dynamics in the structure of the raw gradient and propose our SMI-gradients that consider motion dynamics.

Markovian Model We assume the motion dynamics can be captured by a Markovian model, i.e. $S _ { t } = f _ { d 1 } ( S _ { t - 1 } )$ This allows us to derive the 1st-order SMI-gradient:

$$
\left( { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } \right) _ { d 1 } = { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } + { \frac { \partial J ( S ) } { \partial S _ { t } } } \cdot { \frac { d S _ { t } } { d S _ { t - 1 } } } ,\tag{5}
$$

where $\frac { d S _ { t } } { d S _ { t - 1 } }$ is the temporal relationship between two consecutive frames that will be instantiated. Eq. 5 shows that the attack gradient along the motion manifold needs to consider the first-order information in the motion, e.g. velocity.

Autoregressive Model Besides the first-order dynamics, we also model the second-order dynamics by assuming $S _ { t } = f _ { d 2 } ( S _ { t - 1 } , S _ { t - 2 } )$ , as 2nd-order dynamics (i.e. joint acceleration) capture the smooth temporal dynamics of skeletal motion [36]. We extend Eq. 5 as:

$$
\begin{array} { r } { \left(  { \frac { \partial J ( S ) } { \partial S _ { t - 2 } } } \right) _ { d 2 } =  { \frac { \partial J ( S ) } { \partial S _ { t - 2 } } } +  { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } \cdot  { \frac { \partial S _ { t - 1 } } { \partial S _ { t - 2 } } } } \\ { +  { \frac { \partial J ( S ) } { \partial S _ { t } } } \cdot (  { \frac { \partial S _ { t } } { \partial S _ { t - 2 } } } +  { \frac { \partial S _ { t } } { \partial S _ { t - 1 } } } \cdot  { \frac { \partial S _ { t - 1 } } { \partial S _ { t - 2 } } } ) . } \end{array}\tag{6}
$$

While higher-order models can also be considered, there is an empirical evidence that the first three orders are the most important in skeletal motion adversarial attack [36]. Therefore, we express the SMI gradi ents of the whole skeletal sequence as $( \nabla J ( S ) ) _ { d 1 } =$ $\left\lceil { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { 1 } } } { \bigr ) } _ { d 1 } , { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { 2 } } } { \bigr ) } _ { d 1 } , \cdot \cdot \cdot , { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { t } } } { \bigr ) } _ { d 1 } \right\rceil$ and $( \nabla J ( S ) ) _ { d 2 } =$ $\left\lceil { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { 1 } } } { \bigr ) } _ { d 2 } , { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { 2 } } } { \bigr ) } _ { d 2 } , \cdot \cdot \cdot , { \bigl ( } { \frac { \partial J ( S ) } { \partial S _ { t } } } { \bigr ) } _ { d 2 } \right\rceil$

## 4.2. Time-Varying Autoregressive Models for Dynamics

We employ explicit models [39] to compute the dynamics-related derivatives in SMI gradients. While implicit models [33] may also be considered, they would require another network to be trained, making them less preferable in hard no-box attacks. To realize $f _ { d 1 }$ and $f _ { d 2 }$ we use time-varying autoregressive models (TV-AR) [3], which effectively estimates the dynamics of skeleton sequence [39] due to its ability of modelling the temporal non-stationary signals:

$$
f _ { d 1 } : S _ { t } = A _ { t } \cdot S _ { t - 1 } + B _ { t } + \gamma _ { t } ,\tag{7}
$$

$$
f _ { d 2 } : S _ { t } = C _ { t } \cdot S _ { t - 1 } + D _ { t } \cdot S _ { t - 2 } + E _ { t } + \gamma _ { t } ,\tag{8}
$$

where Eq. 7 and Eq. 8 are denoted as $\mathrm { T V - A R } ( 1 )$ and TV-AR(2) respectively. The model parameters $\beta _ { t } ^ { 1 } = [ A _ { t } , B _ { t } ]$ and $\beta _ { t } ^ { 2 } = [ C _ { t } , D _ { t } , E _ { t } ]$ are all time-varying parameters and determined by data-fitting. $\gamma _ { t }$ is a time-dependent white noise representing the dynamics of stochasticity.

Using Eq. 7 to compute $\frac { \partial S _ { t } } { \partial S _ { t - 1 } } , \mathrm { E q } .$ 5 becomes:

$$
\left( { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } \right) _ { d 1 } = { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } + { \frac { \partial J ( S ) } { \partial S _ { t } } } \cdot A _ { t } .\tag{9}
$$

Similarly, using Eq. 8, we can compute $\begin{array} { r } { C _ { t } ~ = ~ \frac { \partial S _ { t } } { \partial S _ { t - 1 } } } \end{array}$ and $\begin{array} { r } { D _ { t } = \frac { \partial S _ { t } } { \partial S _ { t - 2 } } } \end{array}$ . For $\frac { \partial S _ { t - 1 } } { \partial S _ { t - 2 } }$ , we use $S _ { t - 1 } = C _ { t - 1 } \cdot S _ { t - 2 } +$ $D _ { t - 1 } \cdot S _ { t - 3 } + E _ { t - 1 } + \gamma _ { t }$ to compute it: $\begin{array} { r } { C _ { t - 1 } = \frac { \partial S _ { t - 1 } } { \partial S _ { t - 2 } } } \end{array}$ Then, Eq. 6 becomes:

$$
\left( { \frac { \partial J ( S ) } { \partial S _ { t - 2 } } } \right) _ { d 2 } = { \frac { \partial J ( S ) } { \partial S _ { t - 2 } } } + { \frac { \partial J ( S ) } { \partial S _ { t - 1 } } } \cdot C _ { t - 1 } + { \frac { \partial J ( S ) } { \partial S _ { t } } } \cdot ( D _ { t } + C _ { t } \cdot C _ { t - 1 } ) .\tag{10}
$$

## 5. Skeleton-Motion-Informed Attack

We construct new gradient-based attack methods based on our novel SMI gradient. Due to its compatibility, our proposed gradient can be integrated with most existing gradient-based methods. We select I-FGSM and MI-FGSM, which have been proven for their efficiency on S-HAR attacks [36, 18]. We augment them with first and secondorder SMI-gradients, leading to four new attack methods.

Fast Gradient Sign Methods (FGSM) FGSM [12] is a single-step attack method that generates the adversarial samples $\hat { \boldsymbol S } ~ = ~ \boldsymbol S + \boldsymbol r$ by maximizing the adversarial loss function $J ( S )$ , where r denotes an adversarial perturbation that is constrained within a budget $\| r \| _ { p } < \epsilon .$ , where $\lVert \cdot \rVert _ { p }$ denotes the $l _ { p } { \mathrm { - n o r m } } .$ One variant of FGSM is the Iterative Fast Gradient Sign Method (I-FGSM) [14], which extends FGSM to an iterative process:

$$
\hat { S } ^ { i + 1 } = \hat { S } ^ { i } + \alpha \cdot \mathrm { s i g n } \left( \nabla _ { s } J ( S ) \right) ,\tag{11}
$$

where $\alpha$ is the attack step size and i means iteration. $\mathbf { A n - }$ other variant is MI-FGSM [8], which considers the momentum of the attack to avoid local maxima:

$$
g ^ { i + 1 } = \boldsymbol { \mu } \cdot \boldsymbol { g ^ { i } } + \frac { \nabla _ { s } J \left( S \right) } { \left\| \nabla _ { s } J \left( S \right) \right\| _ { 1 } }\tag{12}
$$

$$
\hat { S } ^ { i + 1 } = \hat { S } ^ { i } + \alpha \cdot \mathrm { s i g n } \left( g ^ { i + 1 } \right) ,\tag{13}
$$

where $\mu$ is the momentum decay factor and $g _ { i }$ is the gradient in iteration i.

SMI-gradient Based Attacks We replace the original gradient $\nabla J ( S )$ in I-FGSM and MI-FGSM with our SMI gradient $( \nabla J ( S ) ) _ { d 1 }$ or $( \nabla J ( S ) ) _ { d 2 }$ . This creates four new dynamic attack strategies: first-order SMI I-FGSM $( \mathrm { S } _ { 1 } \mathrm { I } -$ FGSM), second-order SMI I-FGSM $\mathrm { ( S _ { 2 } I { - } F G S M ) }$ , firstorder SMI MI-FGSM $( \mathrm { S _ { 1 } M I - F G S M } )$ , and second-order SMI MI-FGSM $\mathrm { ( S _ { 2 } M I - F G S M ) }$ . The processes of SI-FGSM are shown in Algorithm 1. The algorithm of SMI-FGSM can be found in the supplementary material.

Algorithm 1 S I-FGSM and $\mathrm { S _ { 2 } l }$ -FGSM   
Input: An encoder k with a loss function $J ;$ a skeletal sequence sample   
S; the size of attack step α; iterations I; the budget of perturbation ϵ.   
Output: An adversarial example $\hat { S }$ with $\| \hat { S } - S \| _ { p } < \epsilon .$   
1: Initialization: ${ \hat { S } } ^ { 0 } = S ;$   
2: Fitting $S$ with TV-AR model to obtain the time-varying parameters   
$\beta _ { t } ;$   
3: for i = 0 to I − 1 do   
4: Input ${ \hat { S } } ^ { i }$ to k;   
5: Obtain the raw gradient $\nabla J ( \hat { S } ^ { i } )$ on $J ;$   
6: Calculate the SMI gradient $( \nabla J ( \hat { S } ^ { i } ) ) _ { d 1 }$ with Eq. 9, or   
$( \nabla J ( \hat { S } ^ { i } ) ) _ { d 2 }$ with Eq. 10, using $\beta _ { t }$ and $\nabla J ( \hat { S } ^ { i } ) ;$   
7: Update $\hat { S } ^ { i + 1 }$ by applying the sign gradient as:   
$\hat { S } ^ { i + 1 } = \hat { S } ^ { i } + \alpha \cdot \mathrm { s i g n } \left( \nabla J ( \hat { S } ^ { i } ) ) _ { d 1 } \right) , o r$   
(14)   
$\hat { S } ^ { i + 1 } = \hat { S } ^ { i } + \alpha \cdot \mathrm { s i g n } \left( \nabla J ( \hat { S } ^ { i } ) ) _ { d 2 } \right)$   
8: end for   
9: return $\hat { S } = \hat { S } ^ { I }$

## 6. Experiments

We refer the readers to the supplementary material for extra experimental results.

Datasets We select three widely used skeletal datasets: HDM05 [22] (2,337 sequences of 130 classes performed by 5 actors), NTU60 [26] (56,880 sequences of 60 classes), and NTU120 [19] (114,480 sequences of 120 classes, an extended version of NTU60, one of the largest datasets in the field). We pre-process HDM05 following [9] and both NTU datasets following [28]. The different skeletons are mapped to a standard 25-joint structure as in [38]. For our hard no-box attack, we only use the testing data and do not use the training data, the training labels and the testing labels during attacks.

Target Models We choose multiple state-of-the-art models as victims: ST-GCN [41], 2s-AGCN [28], AS-GCN [15], SGN [44] and MS-G3D [20]. They are trained using the official implementations and following the training protocols. For 2s-AGCN, we attack both the single joint stream model (js-AGCN) and the two-stream model.

Implementation Details We pre-train the CL encoder $f _ { q }$ following [34]. The unsupervised network is trained with a temperature value $\tau = 0 . 0 7$ and SGD optimizer for 450 epochs. The learning rate is set to 0.01 with a weight decay of 0.0001. Due to the limitation of hard no-box settings, the attacker cannot access the training samples during the whole process. Therefore, the encoder is trained on the testing set. We adopt the $l _ { \infty }$ norm for the perturbation budget ϵ. For clusters of negative samples, the number of clusters is 120, and the number of deleted centers Q is 10.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Victimmodels</td><td rowspan=1 colspan=1>Self-sup  AGCNAttacker Attacker</td><td rowspan=1 colspan=1>No-box   No-box    No-box    No-box     No-box      No-boxI-FGSM S₁I-FGSM S2I-FGSM MI-FGSM S₁MI-FGSM S2MI-FGSM</td></tr><tr><td rowspan=6 colspan=1>00  31</td><td rowspan=3 colspan=1>js-AGCN2s-AGCNST-GCN</td><td rowspan=3 colspan=1>11.21%5.36%3.57%  12.93%</td><td rowspan=1 colspan=1>26.05%   28.09%    30.87%    30.68%     34.75%      36.58%</td></tr><tr><td rowspan=1 colspan=1>13.94%   15.04%    16.45%    16.47%     18.23%      19.31%</td></tr><tr><td rowspan=1 colspan=1>9.55%    9.86%     9.96%    11.11%     11.36%      11.56%</td></tr><tr><td rowspan=3 colspan=1>MS-G3DSGNASGCN</td><td rowspan=3 colspan=1>8.39%  35.23%21.25%  26.03%5.69%  20.87%</td><td rowspan=1 colspan=1>10.57%   11.14%    11.69%    11.76%     12.85%      14.11%</td></tr><tr><td rowspan=1 colspan=1>34.09%   34.46%    35.23%    38.49%    38.75%      38.81%</td></tr><tr><td rowspan=1 colspan=1>13.92%   14.85%    14.67%    15.95%    16.82%      17.75%</td></tr><tr><td rowspan=6 colspan=1>000 = 38</td><td rowspan=6 colspan=1>js-AGCN2s-AGCNST-GCNMS-G3DSGNASGCN</td><td rowspan=2 colspan=1>10.12%5.04%</td><td rowspan=1 colspan=1>22.84%   24.36%    26.46%    25.70%     29.64%      30.88%</td></tr><tr><td rowspan=1 colspan=1>11.36%   12.19%    13.03%    12.30%     13.85%      15.56%</td></tr><tr><td rowspan=1 colspan=1>3.26%   10.33%</td><td rowspan=1 colspan=1>7.87%    7.99%     8.04%    8.84%     9.07%      9.19%</td></tr><tr><td rowspan=1 colspan=1>5.19%  31.28%</td><td rowspan=1 colspan=1>9.18%    9.51%     9.99%    10.01%     10.26%      12.98%</td></tr><tr><td rowspan=2 colspan=1>20.95%  23.30%5.54%  18.77%</td><td rowspan=1 colspan=1>29.61%   30.20%    30.74%    33.32%    33.63%      34.54%</td></tr><tr><td rowspan=1 colspan=1>11.42%   12.05%    12.29%    12.77%    13.69%      14.37%</td></tr><tr><td rowspan=6 colspan=1>000  36</td><td rowspan=2 colspan=1>js-AGCN2s-AGCN</td><td rowspan=2 colspan=1>7.19%3.70%</td><td rowspan=1 colspan=1>19.70%   20.32%    21.23%    20.04%    22.76%      24.33%</td></tr><tr><td rowspan=1 colspan=1>7.93%    9.80%     10.56%    9.80%     11.26%      12.66%</td></tr><tr><td rowspan=1 colspan=1>ST-GCN</td><td rowspan=1 colspan=1>2.46%   7.88%</td><td rowspan=1 colspan=1>5.65%    5.85%     5.97%     6.25%     6.54%       6.66%</td></tr><tr><td rowspan=1 colspan=1>MS-G3D</td><td rowspan=1 colspan=1>4.46%  23.15%</td><td rowspan=1 colspan=1>7.30%    7.84%     7.61%    7.94%     8.05%       8.39%</td></tr><tr><td rowspan=2 colspan=1>SGNASGCN</td><td rowspan=1 colspan=1>16.76%  19.93%</td><td rowspan=1 colspan=1>23.92%   24.57%    25.75%    26.74%     27.04%      27.64%</td></tr><tr><td rowspan=1 colspan=1>3.94%  14.30%</td><td rowspan=1 colspan=1>8.79%    9.23%     9.07%    9.71%     10.29%      10.90%</td></tr></table>

Table 2. The fooling rate of different methods on the target models in NTU60, where ϵ is the perturbation budget.
<table><tr><td></td><td>Victim models</td><td>Self-sup Attacker</td><td>AGCN Attacker</td><td>No-box I-FGSM</td><td>No-box S₁I-FGSM</td><td>No-box S2I-FGSM</td><td>No-box MI-FGSM</td><td>No-box S1MI-FGSM</td><td>No-box S2MI-FGSM</td></tr><tr><td rowspan="4">00 31</td><td>js-AGCN</td><td>9.97%</td><td></td><td>23.92%</td><td>24.64%</td><td>25.79%</td><td>27.26%</td><td>27.93%</td><td>29.07%</td></tr><tr><td>2s-AGCN</td><td>6.38%</td><td></td><td>20.09%</td><td>21.11%</td><td>22.06%</td><td>23.51%</td><td>24.63%</td><td>24.96%</td></tr><tr><td>ST-GCN</td><td>12.18%</td><td>23.84%</td><td>23.53%</td><td>24.73%</td><td>25.77%</td><td>26.95%</td><td>27.31%</td><td>28.76%</td></tr><tr><td>MS-G3D SGN</td><td>10.63% 19.65%</td><td>24.63%</td><td>19.07%</td><td>20.11%</td><td>20.20%</td><td>21.57%</td><td>21.95%</td><td>22.73%</td></tr><tr><td rowspan="4">000 = 38</td><td>ASGCN</td><td>7.29%</td><td>37.90% 24.15%</td><td>31.43% 18.03%</td><td>32.64% 19.29%</td><td>33.75% 20.37%</td><td>38.47% 19.88%</td><td>37.96% 20.22%</td><td>38.85% 21.60%</td></tr><tr><td>js-AGCN</td><td>9.04%</td><td></td><td>21.58%</td><td>21.77%</td><td></td><td></td><td></td><td></td></tr><tr><td>2s-AGCN</td><td>5.79%</td><td></td><td>15.71%</td><td>15.23%</td><td>22.47% 15.94%</td><td>24.28% 18.68%</td><td>24.62%</td><td>25.74%</td></tr><tr><td rowspan="2">ST-GCN</td><td>11.23%</td><td>21.35%</td><td>20.74%</td><td>21.51%</td><td>22.22%</td><td>23.52%</td><td>18.76% 24.65%</td><td>19.57%</td></tr><tr><td>MS-G3D 9.98%</td><td>21.73%</td><td>17.06%</td><td>17.74%</td><td>17.92%</td><td>19.53%</td><td>19.29%</td><td></td><td>24.87% 20.33%</td></tr><tr><td rowspan="5"></td><td>SGN ASGCN</td><td>17.59%</td><td>35.88%</td><td>27.38%</td><td>28.17%</td><td>28.65%</td><td>32.43%</td><td>33.06%</td><td>32.55%</td></tr><tr><td></td><td>6.87%</td><td>21.87%</td><td>15.73%</td><td>16.93%</td><td>17.62%</td><td>17.41%</td><td>17.75%</td><td>18.70%</td></tr><tr><td>js-AGCN</td><td>8.23%</td><td></td><td>18.51%</td><td>18.88%</td><td>18.79%</td><td>20.74%</td><td>21.88%</td><td></td></tr><tr><td>2s-AGCN</td><td>5.09%</td><td></td><td>7.93%</td><td>9.80%</td><td>10.56%</td><td>9.80%</td><td>11.26%</td><td>22.06%</td></tr><tr><td>ST-GCN</td><td>9.54%</td><td>17.51%</td><td>17.60%</td><td>17.97%</td><td>18.67%</td><td>18.90%</td><td>19.91%</td><td>12.66% 20.34%</td></tr><tr><td rowspan="4">000= E</td><td>MS-G3D</td><td>9.42%</td><td>18.07%</td><td>15.06%</td><td>15.09%</td><td>15.20%</td><td>16.86%</td><td>17.25%</td><td></td></tr><tr><td>SGN</td><td>16.82%</td><td>34.10%</td><td>22.39%</td><td>22.65%</td><td>22.76%</td><td>25.43%</td><td>25.79%</td><td>17.56%</td></tr><tr><td>ASGCN</td><td>5.39%</td><td>19.02%</td><td>12.90%</td><td>13.45%</td><td>14.97%</td><td>14.08%</td><td>15.26%</td><td>25.92%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>16.33%</td></tr></table>

Table 3. The fooling rate of different methods on the target models in NTU120, where ϵ is the perturbation budget.

Evaluation Metrics We employ the fooling rate as a major metric. It is defined as the percentage of data samples whose predicted labels changed after adversarial attacks [18]. Besides, inspired by [36], we define a perceptual deviation indicator to evaluate the imperceptibility of adversarial skeletal samples:

$$
\Delta p = \frac { 1 } { M T } \sum _ { n = 0 } ^ { M } { \left\| { \boldsymbol { S } } - { \boldsymbol { \hat { S } } } \right\| } _ { 2 } + \frac { 1 } { M T } \sum _ { n = 0 } ^ { M } { \left\| { \boldsymbol { B } } - { \boldsymbol { \hat { B } } } \right\| } _ { 2 }\tag{15}
$$

where M is the number of adversarial samples, T is the total number of frames, and L is the number ofjoints in the skeleton. The three terms evaluate the deviations of joint position, bone-length, and acceleration, respectively. A smaller perceptual deviation indicates better imperceptibility.

## 6.1. Hard No-Box Attack

Baselines We establish two baselines using transfer-based attacks based on a self-supervised and a supervised classifier. This is because our method is the first hard no-box attack and there is no other similar method. The transferbased attack is the closest setting to ours as they do not require access to the victim model. The first baseline (Selfsup Attacker) is a self-supervised surrogate model with a linear layer appended to our CL encoder. We freeze our CL encoder after the manifold training and then train the linear layer supervisedly [17]. In this way, the need to access labels is only for training the linear layer, not the CL encoder, which is not strictly hard no-box but closer than existing methods. The second baseline is a standard transfer-based attack where we use js-AGCN as the surrogate model and SMART [36] as the white-box attacker (AGCN Attacker). Unlike our method, both baselines still require access to the training data and the training labels during attacks.

For hard no-box attack, different attack strategies are compared including I-FGSM, MI-FGSM, S<sub>1</sub>I-FGSM, S<sub>2</sub>I-FGSM, S MI-FGSM, and S MI-FGSM. We run all the attackers for 400 iterations and report the attack performance under different perturbation budgets ϵ in Table 2 for NTU60, Table 3 for NTU120, and Table 4 for HDM05. We omit the results on js-AGCN and 2s-AGCN under AGCN Attacker because AGCN is the surrogate model.

<table><tr><td rowspan="2"></td><td rowspan="2">Victim models</td><td rowspan="2">Self-sup Attacker</td><td rowspan="2">AGCN Attacker</td><td rowspan="2">No-box I-FGSM</td><td rowspan="2">No-box</td><td rowspan="2">No-box S2I-FGSM</td><td rowspan="2">No-box MI-FGSM</td><td rowspan="2">No-box S₁MI-FGSM</td><td rowspan="2">No-box S2MI-FGSM</td></tr><tr><td>S₁I-FGSM</td></tr><tr><td rowspan="5">0.001 ⅡI E</td><td>js-AGCN</td><td>10.12%</td><td></td><td>10.61%</td><td>10.98%</td><td>11.55%</td><td>13.45%</td><td>14.20%</td><td>14.20%</td></tr><tr><td>2s-AGCN</td><td>1.89%</td><td></td><td>5.30%</td><td>5.68%</td><td>5.68%</td><td>6.44%</td><td>6.44%</td><td>6.82%</td></tr><tr><td>ST-GCN</td><td>6.44%</td><td>24.26%</td><td>9.47%</td><td>9.85%</td><td>9.47%</td><td>8.90%</td><td>11.17%</td><td>11.74%</td></tr><tr><td>MS-G3D</td><td>4.17%</td><td>86.95%</td><td>41.67%</td><td>44.51%</td><td>43.94%</td><td>53.03%</td><td>56.06%</td><td>53.41%</td></tr><tr><td>SGN</td><td>1.89%</td><td>3.98%</td><td>2.46%</td><td>2.84%</td><td>3.03%</td><td>3.03%</td><td>3.40%</td><td>3.59%</td></tr><tr><td></td><td>ASGCN</td><td>1.89%</td><td>27.57%</td><td>3.22%</td><td>4.36%</td><td>3.22%</td><td>4.36%</td><td>5.68%</td><td>5.68%</td></tr><tr><td rowspan="5">0.0008 = 3</td><td>js-AGCN</td><td>9.46%</td><td></td><td>8.14%</td><td>9.09%</td><td>9.09%</td><td>10.61%</td><td>10.98%</td><td>11.55%</td></tr><tr><td>2s-AGCN</td><td>1.70%</td><td></td><td>4.36%</td><td>4.92%</td><td>4.73%</td><td>4.55%</td><td>5.30%</td><td>5.87%</td></tr><tr><td>ST-GCN</td><td>3.79%</td><td>21.69%</td><td>8.33%</td><td>8.52%</td><td>8.52%</td><td>8.33%</td><td>8.90%</td><td>8.33%</td></tr><tr><td>MS-G3D</td><td>3.60%</td><td>82.17%</td><td>25.76%</td><td>30.68%</td><td>29.17%</td><td>36.36%</td><td>39.39%</td><td>39.02%</td></tr><tr><td>SGN</td><td>1.51%</td><td>3.60%</td><td>0.57%</td><td>1.51%</td><td>2.08%</td><td>2.27%</td><td>2.27%</td><td>2.46%</td></tr><tr><td rowspan="5">0.0006</td><td>ASGCN</td><td>1.70%</td><td>22.43%</td><td>2.46%</td><td>2.84%</td><td>2.84%</td><td>2.27%</td><td>2.84%</td><td>3.03%</td></tr><tr><td>js-AGCN</td><td>7.19%</td><td></td><td>5.11%</td><td>7.01%</td><td>6.44%</td><td>6.06%</td><td>6.63%</td><td>7.77%</td></tr><tr><td>2s-AGCN</td><td>1.33%</td><td></td><td>2.08%</td><td>2.08%</td><td>2.27%</td><td>3.79%</td><td>3.98%</td><td>4.17%</td></tr><tr><td>ST-GCN</td><td>3.40%</td><td>19.85%</td><td>6.44%</td><td>7.01%</td><td>6.63%</td><td>7.58%</td><td>8.33%</td><td>7.77%</td></tr><tr><td>MS-G3D</td><td>2.84%</td><td>72.43%</td><td>11.55%</td><td>12.31%</td><td>11.92%</td><td>15.34%</td><td>18.94%</td><td>17.99%</td></tr><tr><td>1I e</td><td>SGN ASGCN</td><td>0.57% 1.52%</td><td>3.40% 17.82%</td><td>0.13% 0.38%</td><td>0.94% 0.76%</td><td>0.57% 1.89%</td><td>1.13% 0.57%</td><td>1.32% 2.27%</td><td>1.51% 2.27%</td></tr></table>

Table 4. The fooling rate of different methods on the target models in HDM05, where ϵ is the perturbation budget.

![](images/27c930e87ef6b2f7f4be58c2337439a93336ed8c0a6aeea3d791fb9f20bb0ca4.jpg)  
Figure 2. Visual comparisons between attack strategies in no-box attacks (ϵ=0.006) with key visual differences highlighted.

Table 2-4 show that the hard no-box attack poses real threats to a range of S-HAR classifiers. In general, the fooling rate of hard no-box attacks is higher than Self-sup Attackers. This is surprising as the Attacker utilizes training data and training labels while our method does not. Looking further, AGCN Attacker is much better than Self-sup Attacker and the major difference is their feature extraction (i.e. one is a CL encoder and one is a graph network). This shows transfer-based attack heavily relies on the feature extraction ability of the surrogate model that cannot bypass access to the training data and the training labels. Furthermore, when compared with AGCN Attacker, our method achieves similar fooling rates, varying across different victims and datasets. Given that AGCN Attacker requires access to training data, training labels and testing labels, we argue hard no-box attacker achieves superior results and provides a more realistic setting.

In addition, among various no-box attack strategies, S<sub>2</sub>MI-FGSM performs the best and often by big margins. All the SMI gradient-based methods generate stronger adversaries compared with baselines I-FGSM and MI-FGSM. The 2nd-order SMI attack method usually outperforms the corresponding 1st-order version. Last, we notice a variance in fooling rate across different victims and datasets. For instance, the multi-stream model (2s-AGCN) significantly enhances the robustness compared to the single-stream model (js-AGCN); the fooling rate of all methods drops by nearly half. This may be because the multi-stream model can ensemble features from different modalities, which improves the robustness. In general, it is still an open question why fooling rate can vary across victims and datasets and we leave the theoretical analysis for future work.

<table><tr><td>Surrogate Model</td><td>Victims</td><td>MI- I-FGSM FGSM</td><td>(Ours)</td><td>S2I-FGSM S2MI-FGSM (Ours)</td></tr><tr><td rowspan="2">2s-AGCN</td><td>STGCN</td><td>2.10% 2.10%</td><td>3.00%</td><td>3.01%</td></tr><tr><td>MS-G3D</td><td>2.58% 2.59%</td><td>2.90%</td><td>2.97%</td></tr><tr><td rowspan="2">ST-GCN</td><td>2sAGCN</td><td>2.20% 2.34%</td><td>2.44%</td><td>2.64%</td></tr><tr><td>MS-G3D</td><td>2.00% 2.10%</td><td>2.63%</td><td>2.92%</td></tr><tr><td rowspan="2">MS-G3D</td><td>STGCN</td><td>1.71% 1.69%</td><td>2.65%</td><td>2.67%</td></tr><tr><td>2sAGCN</td><td>1.76% 1.79%</td><td>2.03%</td><td>2.07%</td></tr></table>

Table 5. The fooling rate of different attack strategies in transferred SMART attacks, where attack budgets ϵ = 0.01.

## 6.2. Transfer-Based Black-Box Attack

SMI gradient not only improves the transferability in hard no-box attacks, but also enhances other gradient-based skeletal attacks. Here, we employ SMART [36], a whitebox attacker, as a baseline to compare different attack strategies. In the original SMART settings, I-FGSM is adopted to generate adversarial samples. We replace it with $\mathrm { S _ { 2 } I _ { - } }$ FGSM, MI-FGSM, and S<sub>2</sub>MI-FGSM to make a comparison. As all strategies achieve similar fooling rates in whitebox attacks, we mainly focus on their transferability using different surrogate models. We utilize SMART to attack 2s-AGCN, ST-GCN, and MS-G3D on NTU60 and transfer the obtained samples to other victim networks.

The performance of the transferred black-box attack is shown in Table 5. S MI-FGSM gives the best performance in transfer-based black-box attacks. $\mathrm { S _ { 2 } I { - } F G S M }$ also improves the transferability of adversarial samples compared with baselines. In contrast, MI-FGSM, which succeeds in the image transfer-based attack, struggles in skeletal data. Its performance declines to 1.69% when it attacks STGCN via MS-G3D. Overall, the success rate is low in Table 5 because SMART is sensitive to the chosen surrogate [36]. Transfer-based attacks are proven to suffer from lower fooling rates in S-HAR attack [36] and improving the transferability is still an open problem. Table 5 aims to show our SMI gradient can improve it by incorporating motion dynamics into the attack gradient, compared with other alternative gradients. We will include in future work how to further explore this dynamics for better attack transfer.

## 6.3. Perceptual Analysis

A key feature of SMI gradient-based attacks is the improvement in the perceptual quality of the adversarial samples due to the consideration of the motion manifold. To verify this, we employ quantitative comparison and qualitative visual analysis on the no-box adversarial samples under various strategies. We compare the perceptual quality $\Delta p$ on NTU60 in Table 6. We find that S I-FGSM achieves the best imperceptibility and obtains a massive improvement compared with I-FGSM. In contrast, MI-FGSM’s deviation is twice that of S I-FGSM. Although S MI-FGSM does not achieve the best visual performance, it is still slightly better than I-FGSM and achieves a better trade-off between fooling rate and perceptual quality. This is understandable because our method considers dynamics that help to generate more on-manifold adversarial samples. Moreover, the 1storder SMI attacks outperform baselines but cannot compete with 2nd-order SMI attacks. This demonstrates the importance of considering acceleration in the skeletal attack.

We show the visual comparison of poses under various attack strategies in no-box attacks in Figure 2. The spinal joints demonstrate the most obvious differences. S I-FGSM outperforms the other attack methods and gets the most natural poses, whereas I-FGSM has slight but noticeable joint displacements in the neck and head. S MI-FGSM performs better than its baseline, MI-FGSM, which shows zig-zag bending in the frame $t _ { 0 } + 2 .$ . The samples produced by MI-FGSM have the worst imperceptibility, where we can easily find the unnatural jittery movements.

We also evaluate perceptual quality $\Delta p$ on SMART adversarial samples obtained with different gradients. Results conducted on the NTU60 dataset are shown in Table 7. $\mathrm { S _ { 2 } I }$ FGSM reaches the best perceptual performance compared with all attack strategies. MI-FGSM slightly declines the imperceptibility. S<sub>2</sub>MI-FGSM performs slightly worse in MS-G3D and ST-GCN. The reason is mainly that S<sub>2</sub>MI-FGSM takes more iterations in the white-box attack, leading to late stopping and slightly worse visual performance.

<table><tr><td>Strategies</td><td> $\overline { { \epsilon = 0 . 0 1 } }$ </td><td> $\overline { { \epsilon = 0 . 0 0 8 } }$ </td><td>€ = 0.006</td></tr><tr><td>I-FGSM</td><td>95.87</td><td>68.56</td><td>43.13</td></tr><tr><td>MI-FGSM</td><td>131.77</td><td>90.44</td><td>54.61</td></tr><tr><td>S₁I-FGSM</td><td>84.63</td><td>60.77</td><td>38.15</td></tr><tr><td>S₁MI-FGSM</td><td>114.51</td><td>78.06</td><td>47.04</td></tr><tr><td>S2I-FGSM</td><td>65.60</td><td>45.67</td><td>27.67</td></tr><tr><td>S2MI-FGSM</td><td>90.93</td><td>62.04</td><td>37.46</td></tr></table>

Table 6. The perceptual deviation of different attack strategies in no-box attacks with different budgets ϵ.

<table><tr><td>Victims</td><td>I-FGSM MI-FGSM</td><td>(Ours)</td><td>S2I-FGSM S2MI-FGSM (Ours)</td></tr><tr><td>2s-AGCN</td><td>1.52 1.62</td><td>1.25</td><td>1.49</td></tr><tr><td>MS-G3D</td><td>2.39 2.46</td><td>1.69</td><td>3.02</td></tr><tr><td>ST-GCN</td><td>1.13 1.18</td><td>1.10</td><td>1.47</td></tr></table>

Table 7. The perceptual deviation of different attack strategies in SMART for different victim models.

## 7. Conclusions and Discussions

In this paper, we have verified potential threats to S-HAR solutions. A new setting is proposed: the hard no-box attack on skeletal motions without access to the victim model, the training samples or the labels. We validate our setting by proposing the first pipeline for hard no-box attacks. Moreover, as far as we know, we are the first to explore motion dynamics in the adversarial gradient computation, leading to a new SMI gradient compatible with existing gradientbased attacks. By extensive evaluation and comparison, our method has been proven to be threatening and imperceptible, relying on the least prior knowledge.

The SMI gradient also improves the transferability of transferred black-box attacks. Nonetheless, boosting the transferability is still an open problem [36] and we will explore this further with our SMI gradient in the future. We will explore other models (e.g., diffusion models [5]) and other time-series data (e.g. stock price, videos) for the proposed attack. Also, our SMI gradient describes dynamics and may be beneficial for motion synthesis [4].

We call for attention to intensify the S-HAR robustness by considering defences against our hard no-box attack. We validate randomized smoothing [6] as a potential defence method in supplementary materials. Due to the least prior knowledge requirement, security risks posed by our attack can be reduced with such defenses. Otherwise, our attacks become a significantly threat to S-HAR.

## Acknowledegments

This research is supported in part by National Natural Science Foundation of China (ref: 61673314, Yang), EP-SRC (ref: EP/X031012/1, NortHFutures, Shum) and EU Horizon 2020 (ref: 899739, CrowdDNA, Wang).

## References

[1] Naveed Akhtar, Ajmal Mian, Navid Kardan, and Mubarak Shah. Advances in adversarial attacks and defenses in computer vision: A survey. IEEE Access, 9:155161–155196, 2021.

[2] Wieland Brendel, Jonas Rauber, and Matthias Bethge. Decision-based adversarial attacks: Reliable attacks against black-box machine learning models. In International Conference on Learning Representations, 2018.

[3] Laura F Bringmann, Ellen L Hamaker, Daniel E Vigo, Andre Aubert, Denny Borsboom, and Francis Tuerlinckx.´ Changing dynamics: Time-varying autoregressive models using generalized additive modeling. Psychological methods, 22(3):409, 2017.

[4] Ziyi Chang, Edmund JC Findlay, Haozheng Zhang, and Hubert PH Shum. Unifying human motion synthesis and style transfer with denoising diffusion probabilistic models. arXiv preprint arXiv:2212.08526, 2022.

[5] Ziyi Chang, George A Koulieris, and Hubert PH Shum. On the design fundamentals of diffusion models: A survey. arXiv preprint arXiv:2306.04542, 2023.

[6] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pages 1310–1320. PMLR, 2019.

[7] Yunfeng Diao, Tianjia Shao, Yong-Liang Yang, Kun Zhou, and He Wang. Basar: Black-box attack on skeletal action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7597– 7607, 2021.

[8] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial attacks with momentum. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9185–9193, 2018.

[9] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1110–1118, 2015.

[10] Lijie Fan, Sijia Liu, Pin-Yu Chen, Gaoyuan Zhang, and Chuang Gan. When does contrastive learning preserve adversarial robustness from pretraining to finetuning? Advances in Neural Information Processing Systems, 34:21480–21492, 2021.

[11] Reuben Feinman, Ryan R Curtin, Saurabh Shintre, and Andrew B Gardner. Detecting adversarial samples from artifacts. arXiv preprint arXiv:1703.00410, 2017.

[12] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

[13] Qiuhong Ke, Mohammed Bennamoun, Senjian An, Ferdous Sohel, and Farid Boussaid. A new representation of skeleton sequences for 3d action recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3288–3297, 2017.

[14] Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Adversarial examples in the physical world, pages 99–112. Chapman and Hall/CRC, 2018.

[15] Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian. Actional-structural graph convolutional networks for skeleton-based action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3595–3603, 2019.

[16] Qizhang Li, Yiwen Guo, and Hao Chen. Practical no-box adversarial attacks against dnns. Advances in Neural Information Processing Systems, 33:12849–12860, 2020.

[17] Lilang Lin, Sijie Song, Wenhan Yang, and Jiaying Liu. Ms2l: Multi-task self-supervised learning for skeleton based action recognition. In Proceedings of the 28th ACM International Conference on Multimedia, pages 2490–2498, 2019.

[18] Jian Liu, Naveed Akhtar, and Ajmal Mian. Adversarial attack on skeleton-based human action recognition. IEEE Transactions on Neural Networks and Learning Systems, 2020.

[19] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+ d 120: A largescale benchmark for 3d human activity understanding. IEEE transactions on pattern analysis and machine intelligence, 42(10):2684–2701, 2019.

[20] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang, and Wanli Ouyang. Disentangling and unifying graph convolutions for skeleton-based action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 143–152, 2020.

[21] Qianhui Men, Edmond S. L. Ho, Hubert P. H. Shum, and Howard Leung. A quadruple diffusion convolutional recurrent network for human motion prediction. IEEE Transactions on Circuits and Systems for Video Technology, 31(9):3417–3432, 2021.

[22] Meinard Muller, Tido R¨ oder, Michael Clausen, Bernhard¨ Eberhardt, Bjorn Kr¨ uger, and Andreas Weber. Mocap¨ database hdm05. Institutfur Informatik II, Universit¨ at Bonn¨ , 2(7), 2007.

[23] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[24] Bin Ren, Mengyuan Liu, Runwei Ding, and Hong Liu. A survey on 3d skeleton-based action recognition using learning method. arXiv preprint arXiv:2002.05907, 2020.

[25] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pirsiavash. Hidden trigger backdoor attacks. In Proceedings of the AAAI conference on artificial intelligence, pages 11957– 11965, 2020.

[26] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1010–1019, 2016.

[27] Divya Shanmugam, Davis Blalock, Guha Balakrishnan, and John Guttag. Better aggregation in test-time augmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 1214–1223, October 2021.

[28] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. Twostream adaptive graph convolutional networks for skeletonbased action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12026–12035, 2019.

[29] Sijie Song, Cuiling Lan, Junliang Xing, Wenjun Zeng, and Jiaying Liu. An end-to-end spatio-temporal attention model for human action recognition from skeleton data. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.

[30] Tae Soo Kim and Austin Reiter. Interpretable 3d human action analysis with temporal convolutional networks. In Proceedings ofthe IEEE conference on computer vision andpattern recognition workshops, pages 20–28, 2017.

[31] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

[32] Nariki Tanaka, Hiroshi Kera, and Kazuhiko Kawamoto. Adversarial bone length attack on action recognition. arXiv preprint arXiv:2109.05830, 2021.

[33] Xiangjun Tang, He Wang, Bo Hu, Xu Gong, Ruifan Yi, Qilong Kou, and Xiaogang Jin. Real-time controllable motion transition for characters. ACM Trans. Graph., 41(4), jul 2022.

[34] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. Skeleton-contrastive 3d action representation learning. In Proceedings of the 29th ACM International Conference on Multimedia, pages 1655–1663, 2021.

[35] He Wang, Yunfeng Diao, Zichang Tan, and Guodong Guo. Defending black-box skeleton-based human activity classifiers. arXiv preprint arxiv.2203.04713, 2022.

[36] He Wang, Feixiang He, Zhexi Peng, Tianjia Shao, Yong-Liang Yang, Kun Zhou, and David Hogg. Understanding the robustness of skeleton-based action recognition under adversarial attack. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14656– 14665, 2021.

[37] He Wang, Edmond SL Ho, and Taku Komura. An energydriven motion planning method for two distant postures. IEEE transactions on visualization and computer graphics, 21(1):18–30, 2015.

[38] He Wang, Edmond SL Ho, Hubert PH Shum, and Zhanxing Zhu. Spatio-temporal manifold learning for human motions via long-horizon modeling. IEEE transactions on visualization and computer graphics, 27(1):216–227, 2019.

[39] Shihong Xia, Congyi Wang, Jinxiang Chai, and Jessica Hodgins. Realtime style transfer for unlabeled heterogeneous human motion. ACM Transactions on Graphics (TOG), 34(4):1–10, 2015.

[40] Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, and Alan L Yuille. Improving transferability of adversarial examples with input diversity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2730–2739, 2019.

[41] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeleton-based action

recognition. In Thirty-second AAAI conference on artificial intelligence, 2018.

[42] Xueting Yan, Ishan Misra, Abhinav Gupta, Deepti Ghadiyaram, and Dhruv Mahajan. Clusterfit: Improving generalization of visual representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6509–6518, 2020.

[43] Jiahang Zhang, Lilang Lin, and Jiaying Liu. Hierarchical consistent contrastive learning for skeleton-based action recognition with growing augmentations. arXiv preprint arXiv:2211.13466, 2022.

[44] Pengfei Zhang, Cuiling Lan, Wenjun Zeng, Junliang Xing, Jianru Xue, and Nanning Zheng. Semantics-guided neural networks for efficient skeleton-based human action recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1112–1121, 2020.

[45] Yujie Zhou, Haodong Duan, Anyi Rao, Bing Su, and Jiaqi Wang. Self-supervised action representation learning from partial spatio-temporal skeleton sequences. arXiv preprint arXiv:2302.09018, 2023.