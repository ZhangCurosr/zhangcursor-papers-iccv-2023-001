# Learning Hierarchical Features with Joint Latent Space Energy-Based Prior

Jiali Cui<sup>1</sup>, Ying Nian Wu<sup>2</sup>, Tian Han<sup>1</sup> <sup>1</sup>Department of Computer Science, Stevens Institute of Technology <sup>2</sup>Department of Statistics, University of California, Los Angeles {jcui7,than6}@stevens.edu, ywu@stat.ucla.edu

## Abstract

This paper studies the fundamental problem of multilayer generator models in learning hierarchical representations. The multi-layer generator model that consists of multiple layers of latent variables organized in a top-down architecture tends to learn multiple levels of data abstraction. However, such multi-layer latent variables are typically parameterized to be Gaussian, which can be less informative in capturing complex abstractions, resulting in limited success in hierarchical representation learning. On the other hand, the energy-based (EBM) prior is known to be expressive in capturing the data regularities, but it often lacks the hierarchical structure to capture different levels of hierarchical representations. In this paper, we propose a joint latent space EBM prior model with multi-layer latent variables for effective hierarchical representation learning. We develop a variational joint learning scheme that seamlessly integrates an inference model for efficient inference. Our experiments demonstrate that the proposed joint EBM prior is effective and expressive in capturing hierarchical representations and modelling data distribution.

## 1. Introduction

In recent years, deep generative models have achieved remarkable success in generating high-quality images [36, 31], texts [40, 13], and videos [28, 35]. However, learning generative models that incorporate hierarchical structures still remains a challenge. Such hierarchical generative models can play a critical role in enabling explainable artificial intelligence, thus representing an important area of ongoing research.

To tackle this challenge, various methods of learning hierarchical representations have been explored [30, 42, 21, 3, 30]. These methods learn generative models with multiple layers of latent variables organized in a top-down architecture, which have shown the capability of learning increasingly abstract representations (e.g., general structures, classes) at the top layers while capturing low-level data features (e.g., colors, background) at the bottom layers. In general, one could categorize these methods into two classes: (i) conditional hierarchy and (ii) architectural hierarchy.

The conditional hierarchies [30, 12, 25, 21, 3] employ stacked generative models layered on top of one another and assume conditional Gaussian distributions at different layers, while the architectural hierarchies [42, 20] instead leverage network architecture to place high-level representations at the top layers of latent variables and low-level representations at the bottom layers. However, they typically assume conditional Gaussian distribution or isotropic Gaussian as the prior, which could have limited expressivity [26]. Learning a more expressive and informative prior model for multiple layers of latent variables is thus needed.

The energy-based models (EBMs) [19, 39, 7, 24, 26, 4], on the other hand, are shown to be expressive and proved to be powerful in capturing data regularities. Notably, [26] studies the EBM in the latent space, where the energy function is considered as a correction of the non-informative prior and tilts the non-informative prior to a more expressive prior distribution. The low dimensionality of the latent space makes EBM more effective in capturing regularities in the data. However, prior methods often rely on expensive MCMC methods for posterior sampling of the latent variables, and more importantly, the latent variables are not hierarchically organized, making it difficult to capture data variations at different levels.

In this paper, we propose to combine the strengths of latent space EBM and the multi-layer generator model for effective hierarchical representation learning. In particular, we build the EBM on the latent variables across different layers of the generator model, where the latent variables at different layers are concatenated and modelled jointly. The latent variables at different layers capture different levels of information from the data via the hierarchy of the generative model, and their inter-relations are further tightened up and better captured through EBM in joint latent space.

Learning the EBM in the latent space can be challenging since it usually requires MCMC sampling for both the prior and posterior distributions of the latent variables. The prior sampling is efficient with the low dimensionality of the latent space and the lightweight network of the energy function, while the MCMC sampling of posterior distribution requires the backward propagation of generation network, which is usually heavy and thus renders inefficient inference. To ensure efficient inference, we introduce an inference model and develop a joint training scheme to efficiently and effectively learn the latent space EBM and multi-layer generator model. In particular, the inference model aims to approximate the exact posterior of the multilayer latent variables while also serving to extract levels of data variations through the deep feed-forward inference network to facilitate hierarchical learning.

Contributions: 1) We propose the joint EBM prior model for hierarchical generator models with multiple layers of latent variables; 2) We develop a variational training scheme for efficient learning and inference; 3) We provide strong empirical results on hierarchical learning and image modeling, as well as various ablation studies to illustrate the effectiveness of the proposed model.

## 2. Background

In this section, we present the background of multi-layer generator models and the latent space EBM, which will serve as the foundation of the proposed model.

## 2.1. Multi-layer generator models

Let $\textbf { x } \in \ \mathbb { R } ^ { D }$ represent the high-dimensional observed examples, and $\textbf { z } \in \mathbb { R } ^ { d }$ denote the low-dimensional latent variables. The multi-layer generator model [30, 36, 21] consists of multiple layers of latent variables that are organized in a top-down hierarchical structure and modelled to be conditionally dependent on its upper layer, i.e., $\begin{array} { r } { p _ { \theta } ( \mathbf { z } ) = \prod _ { i = 1 } ^ { L - 1 } p _ { \theta _ { i } } ( \mathbf { z } _ { i } \vert \mathbf { z } _ { i + 1 } ) p _ { 0 } ( \mathbf { z } _ { L } ) } \end{array}$ , where p<sub>θ</sub> ${ \bf \Pi } _ { i } ( { \bf z } _ { i } | { \bf z } _ { i + 1 } ) \sim$ $\mathcal { N } ( \mu _ { \theta _ { i } } ( \mathbf { z } _ { i + 1 } ) , \sigma _ { \theta _ { i } } ( \mathbf { z } _ { i + 1 } ) )$ and $p _ { 0 } ( \mathbf { z } _ { L } ) \sim \mathcal { N } ( 0 , I )$ . Although such a modelling strategy has seen significant improvement in data density estimation, it can be less effective in learning hierarchical representations.

To demonstrate, we train BIVA<sup>1</sup> [21] on MNIST with 6 layers of latent variables and conduct hierarchical sampling by generating latent variables via the repamaramization trick, i.e., $\mathbf { z } _ { i } = \mu _ { \theta _ { i } } ( \mathbf { z } _ { i + 1 } ) + \sigma _ { \theta _ { i } } ( \mathbf { z } _ { i + 1 } ) \cdot \boldsymbol { \epsilon } _ { i } ,$ where $\epsilon _ { i }$ is the added Gaussian noise. Specifically, the latent variables are generated by randomly sampling $\epsilon _ { i }$ for the i-th layer, while fixing $\epsilon _ { j \neq i }$ for other layers, and the corresponding image synthesis should deliver the variation of representation learned by $\mathbf { z } _ { i }$ . We show the results in Fig. 1, where the major semantic variation is only captured by the top layer of the latent variables, suggesting an ineffective hierarchical representation learned (compare with our model shown in Fig. 5).

![](images/b4ef67c7b7f4968098dcf8251a7baecb9953c1f3c816204bc1189ac8ab55b3dd.jpg)  
Figure 1. Visualization of hierarchical representations on BIVA. Left: sampling $\epsilon _ { 1 } , \epsilon _ { 2 }$ for bottom layers. Middle: sampling ϵ<sub>3</sub>, ϵ<sub>4</sub> for middle layers. Right: sampling $\epsilon _ { 5 } , \epsilon _ { 6 }$ for top layers.

## 2.2. Latent space EBM

The energy-based models (EBMs) [7, 6, 24, 8, 38] present a flexible and appealing way for data modelling and are known for being expressive and powerful in capturing data regularities. Albeit their expressivity, learning the EBM on high-dimensional x-space can be challenging since the energy function needs to support the entire data space. As such, another line of works [26, 1, 37] seek to build EBM on low-dimensional z-space with the probability density defined as

$$
p _ { \theta } ( \mathbf { z } ) = \frac { 1 } { \mathbb { Z } ( \theta ) } \exp { [ f _ { \theta } ( \mathbf { z } ) ] } p _ { 0 } ( \mathbf { z } )\tag{1}
$$

However, with a single-layer non-hierarchical z-space, it is typically infeasible to capture the data abstraction of different levels. To see this, we pick up the digit classes $\cdot _ { 0 } , \cdot _ { 1 } ,$ and ‘2’ of the MNIST dataset, on which we train LEBM [26] with the latent dimension set to be 2. We show in Fig. 2 that changing the value of each latent unit could lead to a mixed variations in digit shape and class (compare to our proposed method in Fig. 6).

![](images/b623609c14037b5224a255a6bfd714d85f5d1295ff0a636e22d4c60b4140b942.jpg)  
Figure 2. Visualization for latent space EBM [26] by changing each unit of 2-dimensional z. Left: changing the first unit. Right: changing the second unit. Top: the value of each unit, where the orange color indicates the first unit and the blue color indicates the second unit.

## 3. Methodology

In this paper, we study a joint latent space EBM prior where latent variables are partitioned into multiple groups but modelled jointly for hierarchical representation learning. To ensure efficient inference, we introduce an inference model to approximate the generator posterior and develop a variational joint training scheme in which each model can be trained efficiently and effectively.

## 3.1. Latent variable generative model

A latent variable generative model (i.e., generator model) can be specified using joint distribution:

$$
p _ { \beta , \alpha } ( \mathbf { x } , \mathbf { z } ) = p _ { \beta } ( \mathbf { x } | \mathbf { z } ) p _ { \alpha } ( \mathbf { z } )\tag{2}
$$

which consists of the generation model $p _ { \beta } ( \mathbf { x } | \mathbf { z } )$ and the prior model $p _ { \alpha } ( \mathbf { z } )$

Joint latent space EBM: The prior model $p _ { \alpha } ( \mathbf { z } )$ is usually assumed to be single-layer, which can be infeasible for learning hierarchical representations (see Sec. 2.2). In this work, we propose a joint latent space EBM prior, where latent variables are separated into multiple groups, i.e., $\mathbf { z } = [ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } ]$ , and our EBM prior model is defined as

$$
p _ { \alpha } ( \mathbf { z } ) = { \frac { 1 } { \mathbb { Z } ( { \boldsymbol { \alpha } } ) } } \exp [ f _ { \alpha } ( [ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { L } ] ) ] p _ { 0 } ( [ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { L } ] )\tag{3}
$$

where [.] refers to the concatenation. $f _ { \alpha } ( . )$ is the negative energy function that can be parameterized by a small multilayer perceptron with the parameters $\alpha , p _ { 0 } ( [ \dots ] )$ is a reference distribution assumed to be a unit Gaussian, and $\mathbb { Z } ( \alpha )$ is the normalizing constant or partition function. Such a prior model can be considered as an energy-based refinement of the non-informative reference distribution, thus the relationship between different groups of latent codes can be well captured via the energy function $f _ { \alpha } ( [ { \bf z } _ { 1 } , \ldots , { \bf z } _ { L } ] ) ] )$

For notation simplicity, we denote $\mathbf { z } = [ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { L } ]$ in subsequent discussions.

Generation Model. The generation model is defined as

$$
p _ { \beta } ( \mathbf { x } | \mathbf { z } ) \sim \mathcal { N } ( g _ { \beta } ( \mathbf { z } ) , \sigma ^ { 2 } I _ { D } )\tag{4}
$$

which models the observed data using top-down generator network $g _ { \beta } ( \mathbf { z } )$ with additive Gaussian noise, i.e., $\mathbf { x } =$ $g _ { \boldsymbol { \beta } } ( \mathbf { z } ) + \epsilon , \epsilon \mathrm { ~ \sim ~ } \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { D } )$ To facilitate the hierarchical representation learning with multi-layer latent variables, we consider multi-layer hierarchical generator network $g _ { \beta }$ $( = \{ g _ { 1 } , g _ { 2 } , \ldots , g _ { L } \} )$ that is designed to explain the observation x by integrating data representation from the above layers, i.e.,

$$
\begin{array} { r l } & { h _ { L } = g _ { L } ( \mathbf { z } _ { L } ) } \\ & { h _ { i } = g _ { i } ( [ \mathbf { z } _ { i } , h _ { i + 1 } ] ) , \ i = 1 , 2 , \dots , L - 1 } \\ & { \mathbf { x } \sim \mathcal { N } ( h _ { 1 } , \sigma ^ { 2 } I _ { D } ) } \end{array}\tag{5}
$$

in which $\mathbf { z } _ { L }$ is at the top layer, and $g _ { i }$ is a shallow network that decodes latent code $\mathbf { z } _ { i }$ while integrating features from the upper layer.

Joint EBM prior vs. independent Gaussian prior: With such a multi-layer hierarchical generation model, it is also tempting to consider learning with the unit Gaussian prior, i.e., $p ( \mathbf { z } ) \sim \mathcal { N } ( 0 , I _ { d } )$ . Such a prior choice is adopted in [42, 20] and has seen some success in learning hierarchical features. However, the independent Gaussian prior does not account for the relations between levels of representations and can be less expressive in capturing complex data features. The proposed EBM prior is expressive and can couple the latent variables across different layers within a joint modelling framework.

![](images/81fff0a2dce589b0e947b1e1f86fe1cfa4153dd9de9a79ef7edfb0183f09fdff.jpg)  
Figure 3. The illustration of the proposed joint EBM prior model (Left). Red lines indicates the modelling of intra-layer relation, and blue lines indicate inter-layer relation. Compared to the Gaussian prior model and single-layer LEBM, our prior model is expressive and effective in learning hierarchical representations.

Our model is illustrated in the left panel of Fig. 3.

## 3.2. Variational learning scheme

Suppose we have observed examples $( \mathbf { x } _ { i } , i = 1 , \dots , n )$ that come from (true) data distribution $p _ { \mathrm { d a t a } } ( \mathbf { x } )$ and denote $\theta = ( \beta , \alpha )$ that collects all parameters from the generator model. Learning the generator model can be done by maximizing the log-likelihood of all observed examples as

$$
\sum _ { i = 1 } ^ { n } \log p _ { \boldsymbol \theta } ( \mathbf { x } _ { i } ) = \sum _ { i = 1 } ^ { n } \log \int p _ { \boldsymbol \alpha } ( \mathbf { z } ) p _ { \boldsymbol \beta } ( \mathbf { x } _ { i } | \mathbf { z } ) d \mathbf { z }\tag{6}
$$

When n becomes sufficiently large, maximizing the above log-likelihood is equivalent to minimizing the Kullback-Leibler (KL) divergence between the model distribution and data distribution, i.e., min $D _ { \mathrm { K L } } ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) | | p _ { \theta } ( \mathbf { x } ) )$ .

However, directly maximizing Eq. 6 can be challenging, as the inference of generator posterior (by chain rule, $p _ { \theta } ( { \bf z } | { \bf x } ) ~ = ~ p _ { \theta } ( { \bf x } , { \bf z } ) / p _ { \theta } ( { \bf x } ) )$ is typically intractable. To tackle the challenge, we employ an inference model $q _ { \phi } ( { \bf z } | { \bf x } )$ with a separate set of parameters $\phi$ to approximate the generator posterior.

Inference model: The inference model maps from data space to latent space and is typically assumed to be Gaussian distributed,

$$
q _ { \phi } ( \mathbf { z } | \mathbf { x } ) \sim \mathcal { N } ( \mu _ { \phi } ( \mathbf { x } ) , V _ { \phi } ( \mathbf { x } ) )\tag{7}
$$

where $\mu _ { \phi } ( \mathbf { x } )$ and $V _ { \phi } ( \mathbf { x } )$ are the d-dimensional mean vector and diagonal covariance matrix. To match with the hierar-

chical generator model and facilitate the hierarchical representation learning, we consider the multi-layer inference model defined as

$$
\begin{array} { l } { r _ { 1 } = h _ { 1 } ( \mathbf { x } ) } \\ { r _ { i } = h _ { i } ( r _ { i - 1 } ) \ i = 2 , 3 , \dots , L } \\ { \mathbf { z } _ { i } \sim \mathcal { N } ( \mu _ { i } ( r _ { i } ) , V _ { i } ( r _ { i } ) ) } \end{array}\tag{8}
$$

where $\mathbf { z } _ { L }$ is the inferred latent code at the top layer and $\mathbf { z } _ { 1 }$ is the inferred bottom layer latent code. Each latent code $\mathbf { z } _ { i }$ can be inferred and sampled through reparametrization trick [17] using $\mathbf { z } _ { i } = \mu _ { i } ( r _ { i } ) \mathbf { \bar { + } } V _ { i } ( r _ { i } ) ^ { 1 / 2 } { \epsilon }$ ϵ where $\epsilon \sim \mathcal { N } ( 0 , I _ { d _ { i } } )$ with $d _ { i }$ being the dimensionality of $z _ { i }$

Variational learning: We show that the proposed generator model and the inference model can be naturally unified and jointly trained within a variational learning scheme.

Specifically, we can define two joint densities, one for generator model density, i.e., $p _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { z } ) = p _ { \alpha } ( \mathbf { z } ) p _ { \beta } ( \mathbf { x } | \mathbf { z } )$ , and one for data density, i.e., $q _ { \phi } ( \mathbf { x } , \mathbf { z } ) = p _ { \mathrm { d a t a } } ( \mathbf { x } ) q _ { \phi } ( \mathbf { z } | \mathbf { x } )$ . We propose joint learning through KL minimization, and we denote the objective to be $L ( \theta , \phi )$ , i.e.,

$$
\operatorname* { m i n } _ { \theta } \operatorname* { m i n } _ { \phi } L ( \theta , \phi ) = \operatorname* { m i n } _ { \theta } \operatorname* { m i n } _ { \phi } D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { x } , \mathbf { z } ) | | p _ { \theta } ( \mathbf { x } , \mathbf { z } ) )\tag{9}
$$

where the generator model density is learned to match the true joint data density and capture extracted hierarchical features.

Learning $p _ { \alpha } ( \mathbf { z } ) { : }$ : With this joint KL minimization, the prior model can be learned with the gradient $- \nabla _ { \alpha } L ( \theta , \phi )$

$$
\begin{array} { r l } & { \mathbb { E } _ { q _ { \phi } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } | \mathbf { x } ) } \bigl [ \nabla _ { \alpha } f _ { \alpha } \bigl ( \bigl [ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } \bigr ] \bigr ) \bigr ] - } \\ & { \mathbb { E } _ { p _ { \alpha } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } ) } \bigl [ \nabla _ { \alpha } f _ { \alpha } \bigl ( \bigl [ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } \bigr ] \bigr ) \bigr ] } \end{array}\tag{10}
$$

Learning $p _ { \beta } ( \mathbf { x } | \mathbf { z } ) \colon$ The generation model can be learned with the gradient $- \nabla _ { \beta } L ( \theta , \phi )$ computed as

$$
\mathbb { E } _ { q _ { \phi } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } | \mathbf { x } ) } [ \nabla _ { \beta } \log p _ { \beta } ( \mathbf { x } | \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } ) ]\tag{11}
$$

Learning $q _ { \phi } ( \mathbf { z } | \mathbf { x } )$ : The inference model can be learned by computing the gradient $- \nabla _ { \phi } L ( \theta , \phi )$ as

$$
\begin{array} { r l } & { \nabla _ { \phi } \mathbb { E } _ { q _ { \phi } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } | \mathbf { x } ) } [ \log p _ { \beta } ( \mathbf { x } | \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } ) ] - } \\ & { \nabla _ { \phi } D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } | \mathbf { x } ) | | p _ { \alpha } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L } ) ) } \end{array}\tag{12}
$$

We provide further theoretical understanding in Sec. 3.4.

## 3.3. Prior sampling

Training the proposed joint EBM prior model requires sampling $\mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { L }$ from the proposed EBM prior (see Eq. 10). To do so, we can use Langevin dynamic [22].

For arbitrary target distribution $p ( \mathbf { z } )$ , the Langevin $\mathrm { d y } .$ namic iterates:

$$
\mathbf { z } _ { t + 1 } = \mathbf { z } _ { t } + \frac { s ^ { 2 } } { 2 } \nabla _ { \mathbf { z } } \log p ( \mathbf { z } _ { t } ) + s \epsilon _ { t }\tag{13}
$$

where t indexes the time step of Langevin dynamics. s is the step size and $\epsilon _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ . We run the noise-initialized Langevin dynamics for K steps. Noted that as $s  0 ,$ , and $K  \infty$ , the distribution of $\mathbf { z } _ { t }$ will converge to the target $p ( \mathbf { z } )$ regardless of the initial distribution of $\mathbf { z } _ { 0 }$ [22].

Specifically, for prior sampling, we use the prior distribution $p _ { \alpha } ( \mathbf { z } )$ as our target distribution, and the gradient $\nabla _ { \mathbf { z } } \log p _ { \alpha } ( \mathbf { z } )$ is computed as:

$$
\nabla _ { \mathbf { z } } f _ { \alpha } \big ( [ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { L } ] \big ) + \nabla _ { \mathbf { z } } \log p _ { 0 } \big ( [ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { L } ] \big )\tag{14}
$$

The Langevin dynamics for prior sampling can be efficient due to the low-dimensional latent space and the lightweight network structure for the energy function.

The overall procedure is summarized in Algorithm 1.

Algorithm 1 Learning latent EBM prior for hierarchical   
generator model   
Input: Training iterations T, observed training examples   
$\{ x _ { i } \} _ { i = 1 } ^ { n } ,$ , batch size $m ,$ network parameters $\alpha , \beta , \phi ,$ learn  
ing rate $\eta _ { \alpha } , \eta _ { \beta } , \eta _ { \phi } ,$ , Langevin steps $K ,$ Langevin step size   
$s ,$   
Let $t = 0$ and $\theta = ( \alpha , \beta )$   
repeat   
Prior sampling for $\{ z _ { i } ^ { - } \} _ { i = 1 } ^ { m }$ using Eq. 13 and Eq. 14   
with K and s.   
Inference sampling for $\{ z _ { i } ^ { + } \} _ { i = 1 } ^ { m }$ using Eq. 8 with   
$\{ x _ { i } \} _ { i = 1 } ^ { m }$ and reparametrization.   
Learn joint EBM prior: Given $\{ z _ { i } ^ { - } , z _ { i } ^ { + } \} _ { i = 1 } ^ { m }$ , update   
$\alpha = \alpha - \eta _ { \alpha } \nabla _ { \alpha } L ( \theta , \phi )$ using Eq. 10 with $\eta _ { \alpha }$   
Learn generation model:Given $\{ x _ { i } , z _ { i } ^ { + } \} _ { i = 1 } ^ { m }$ update   
$\beta = \beta - \eta _ { \beta } \nabla _ { \beta } L ( \theta , \phi )$ using Eq. 11 with $\eta _ { \beta }$   
Learn inference model: Given $\{ x _ { i } , z _ { i } ^ { + } \} _ { i = 1 } ^ { m }$ update   
$\phi = \phi - \eta _ { \phi } \nabla _ { \phi } L ( \theta , \phi )$ using Eq. 12 with $\eta _ { \phi }$   
Let $t = t + 1 ;$   
until $t = T$

## 3.4. Theorectical understanding

Divergence perturbation and ELBO. The KL joint minimization (Eq. 9) can be viewed as a surrogate of the MLE objective with the KL perturbation term,

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { x } , \mathbf { z } ) | | p _ { \theta } ( \mathbf { x } , \mathbf { z } ) ) } \\ { = } & { D _ { \mathrm { K L } } ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) | | p _ { \theta } ( \mathbf { x } ) ) + D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) | | p _ { \theta } ( \mathbf { z } | \mathbf { x } ) ) } \end{array}
$$

where $D _ { \mathrm { K L } } ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) | | p _ { \boldsymbol { \theta } } ( \mathbf { x } ) )$ is the MLE loss function, and the perturbation term $D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) | | p _ { \theta } ( \mathbf { z } | \mathbf { x } ) )$ ) measures the KL-divergence between inference distribution and generator posterior. The inference $q _ { \phi } ( { \bf z } | { \bf x } )$ is learned to match the posterior distribution of the generator without expensive posterior sampling. In fact, such KL minimization in the joint space is equivalent to evidence lower bound (ELBO). To see this, noted that $D _ { \mathrm { K L } } ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) | | p _ { \boldsymbol \theta } ( \mathbf { x } ) ) =$ $- H ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) ) - \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \log p _ { \boldsymbol { \theta } } ( \mathbf { x } ) ]$ where $H ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) )$ is the entropy of the empirical data distribution and can be treated as constant $C \equiv - H ( p _ { \mathrm { d a t a } } ( \mathbf { x } ) )$ w.r.t model parameters θ. Then $L ( \theta , \phi )$ is computed as

$$
\begin{array} { r l } & { - \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \log p _ { \theta } ( \mathbf { x } ) ] + D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } | \mathbf { x } ) ) + C } \\ { = } & { \mathbb { E } _ { p _ { \mathrm { d a t a } } } \left[ \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } \left( \log \frac { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } { p _ { \theta } ( \mathbf { z } | \mathbf { x } ) } \right) - \log p _ { \theta } ( \mathbf { x } ) \right] + C } \\ { = } & { \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ - \mathrm { E L B O } ( \mathbf { x } ; \theta , \phi ) ] + C } \end{array}
$$

where

$$
\begin{array} { l l l } { { \displaystyle \mathrm { E L B O } ( { \bf x } ; \theta , \phi ) } } & { { = } } & { { \displaystyle \log p _ { \theta } ( { \bf x } ) - \mathbb { E } _ { q _ { \phi } ( { \bf z } | { \bf x } ) } \left[ \log \frac { q _ { \phi } ( { \bf z } | { \bf x } ) } { p _ { \theta } ( { \bf z } | { \bf x } ) } \right] } } \\ { { } } & { { = } } & { { \displaystyle \mathbb { E } _ { q _ { \phi } ( { \bf z } | { \bf x } ) } \left[ \log \frac { p _ { \theta } ( { \bf x } , { \bf z } ) } { q _ { \phi } ( { \bf z } | { \bf x } ) } \right] } } \end{array}
$$

and is referred to as evidence lower bound (ELBO) in the literature [17]. Noted that the prior model $( { \mathrm { i . e . , ~ } } p _ { \theta } ( \mathbf { x , z } ) =$ $p _ { \beta } ( { \bf x } | { \bf z } ) p _ { \alpha } ( { \bf z } ) )$ is now parameterized to be our joint EBM prior. Minimizing Eq. 9 is equivalent to maximizing the evidence lower bound of the log-likelihood.

Short run sampling and divergence perturbation. The learning of the energy-based model in Eq. 10 requires MCMC sampling from $p _ { \alpha } ( \mathbf { z } )$ We adopt the short-run Langevin dynamics (Eq. 13) with fixed K steps to sample from prior $p _ { \alpha } ( \mathbf { z } )$ for efficient computation. Such prior sampling amounts to approximates the objective $L ( \theta , \phi )$ in Eq. 9 with yet another KL perturbation term, i.e.,

$$
\tilde { L } ( \boldsymbol { \theta } , \boldsymbol { \phi } ) = L ( \boldsymbol { \theta } , \boldsymbol { \phi } ) - D _ { \mathrm { K L } } \big ( \tilde { p } _ { \alpha ^ { ( t ) } } ( \mathbf { z } ) \big | p _ { \alpha } ( \mathbf { z } ) \big )
$$

where $\tilde { p } _ { \alpha ^ { ( t ) } } ( \mathbf { z } )$ refers to the distribution of latent codes after K steps of Langevin dynamics (Eq. 13) in the t-th iteration of the learning. The corresponding learning gradient for the prior model is thus

$$
- \nabla _ { \alpha } \tilde { L } ( \theta , \phi ) = \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \nabla _ { \alpha } f _ { \alpha } ( \mathbf { z } ) ] - \mathbb { E } _ { \tilde { p } _ { \alpha ^ { ( t ) } } ( \mathbf { z } ) } [ \nabla _ { \alpha } f _ { \alpha } ( \mathbf { z } ) ]
$$

When K is sufficiently large, the perturbation term $D _ { \mathrm { K L } } ( \tilde { p } _ { \alpha ^ { ( t ) } } ( { \bf z } ) | p _ { \alpha } ( { \bf z } ) )  0$ . The update rule for the energybased prior model (also Eq. 10) can be interpreted as selfadversarial learning. The prior model $p _ { \alpha } ( \mathbf { z } )$ serves as both generator and discriminator if we compare it to GAN (generative adversarial networks) [11]. In contrast to GAN, our learning follows MLE with divergence perturbations, which in general does not suffer from issues such as mode collapsing and instability, as it does not involve competition between two separate models. The energy-based model can thus be seen as its own adversary or its own critic.

## 4. Related Work

Generator model. The generator model has a top-down network that maps the low-dimensional latent code to the observed data space. VAEs [17, 27, 33, 32] proposes variational learning by introducing an approximation of the true intractable posterior, which allows a tractable bound on loglikelihood to be maximized. Another line of work [14, 25] trains the generator model without inference model by using Langevin dynamics for generator posterior sampling. We follow the former approach and use the inference model for efficient training.

Generator model with informative prior. The majority of the existing generator models assume that the latent code follows a simple and known prior distribution, such as isotropic Gaussian distribution. Such assumption may render an ineffective generator as observed in [34, 5]. Our work utilizes the informative and expressive energy-based prior in the joint space of latent variables and is related to the line of previous research on introducing flexible prior distributions. For example, [34] parameterizes the prior based on the posterior inference model. [5, 9] adopt a two-stage approach which first trains a latent variable model with simple prior and then trains a separate prior model (e.g., VAE or Gaussian mixture model) to match the aggregated posterior distribution.

Generator model with hierarchical structures. The generator models that consist of multi-layers of latent variables are recently attracting attention for hierarchical learning. [29, 2, 21, 3, 36] stack generative models on top of each other and assume conditional Gaussian distributions at different layers. However, as shown in Sec. 2.1, the bottom layer could act as a single generative model and absorb most features, leading to ineffective representation learning. [42, 20] propose to learn hierarchical features with unit Gaussian prior by adapting the generator into multiple layers. However, the latent variables at different layers are distributed independently a priori and are only loosely connected. Instead, we propose a joint EBM prior that could tightly couple and jointly model latent variables at different layers and show superior results through joint training.

Hierarchical generator model with informative prior. Recent advances [4, 1] have started exploring learning informative prior for multi-layer generator models. JEBM [4] and NCP-VAE [1] build latent space EBM on hierarchical generator models and showcase the capability of improving the generation quality. However, these models consider the conditional hierarchical models, which can be limited in learning effective hierarchical representations (see Sec. 2.1). This work studies hierarchical representation learning by introducing a novel framework where latent space EBM can be jointly trained with the architectural hierarchical models to capture complex representations of different levels.

![](images/8095009db4639b3ca82e371eef266631f53760a2e106c344e15587fc6d2aab1c.jpg)  
Figure 4. Hierarchical sampling on SVHN. Left: The latent code at bottom layer (z<sub>1</sub>) represents the background light and shading. Centerleft: the latent code at second bottom layer (z<sub>2</sub>) represents the color schemes. Center-right: the latent code at second top layer (z<sub>3</sub>) encodes the shape variations of the same digit. Right: the latent code at top layer (z ) captures the digit identity and the general structure.

![](images/0e3d73df3eb8a1531bd2c6d795ab32ca38f02d28a9e80b47b7276e438253a44a.jpg)  
Figure 5. Hierarchical sampling on MNIST. Left: The latent code at bottom layer (z<sub>1</sub>) indicates the stroke width. Center: the latent code at second layer (z<sub>2</sub>) encodes geometric changes among similar digits. Right: the latent code at top layer (z<sub>3</sub>) learns the digit identity and general structure.

## 5. Experiments

To demonstrate the effectiveness of our method, we conduct the following experiments: 1) hierarchical representation learning, 2) analysis of latent space, 3) image modelling, and 4) robustness. For a better understanding of the proposed method, we perform the disentanglement learning experiment, various ablation studies, and an analysis of parameter complexity.

## 5.1. Hierarchical Representation Learning

We first demonstrate the capability of our model in hierarchical representation learning by experiments of hierarchical sampling and latent classifier.

Hierarchical sampling. With partitioned latent variables and the hierarchical structure of generator, the latent variables at higher layers should capture semantic features, while the lower layers should capture low-level features. We train our model on MNIST using the same setting as in Sec. 2.2 with only $^ { \cdot } 0 , ^ { \cdot } 1 ^ { \cdot }$ , and $\cdot _ { 2 } \cdot _ $ digit classes available and the latent dimension set to 2. In Fig. 6, by changing the $\mathbf { z } _ { 1 }$ (at the bottom layer), our model can successfully deliver the variations in shape (low-level features), and changing the $\mathbf { z } _ { 2 }$ (at the top layer) leads to the variations in digit class (high-level features). This indicates a successfully learned hierarchical representation.

Further, we train our model on MNIST with all digit classes and the more challenging SVHN dataset, and we generate the images by sampling (using Eq. 13) the latent code of one layer while keeping other layers fixed. The results shown in Fig. 4 and Fig. 5 also demonstrate the hierarchical representation learned by our model.

![](images/f81a873c0f0f9b7ece9c23d6674b51b618bf2309ea914654f5ac63a91e512fb3.jpg)  
Figure 6. Visualization for our model by changing each unit of 2- dimensional z. Left: changing the first unit. Right: changing the second unit. Top: the value of each unit, where the orange color indicates the first unit and the blue color indicates the second unit.

Latent Classifier. We next show that our model can achieve better performance in hierarchical representation learning. Note that the inference model $q _ { \phi } ( { \bf z } | { \bf x } )$ should be learned to approximate the true posterior $p _ { \theta } ( \mathbf { z } | \mathbf { x } )$ of the generator, which integrates both the hierarchical inductive bias and the energy-based refinement. Therefore, to measure the learned hierarchical features, we learn classifiers on the inferred latent codes at different layers to predict the label of the data and measure the testing classification accuracy. The latent codes that carry rich high-level semantic features should achieve higher classification accuracy.

Table 1. Testing accuracy of inferred latent codes at each layer. We denote (↓) for the layer that has unexpectedly lower accuracy.
<table><tr><td></td><td></td><td>L = 1</td><td>L = 2</td><td>L = 3</td><td>L = 4</td></tr><tr><td>MNIST</td><td>Ours VLAE</td><td>32.60% 35.69%</td><td>58.32% 57.86%</td><td>67.64% 52.08% (↓)</td><td>= =</td></tr><tr><td>Fashion-MNIST</td><td>Ours VLAE</td><td>39.27% 52.43%</td><td>44.81% 35.65% (↓)</td><td>84.27% 83.24%</td><td>= 1</td></tr><tr><td>SVHN</td><td>Ours VLAE</td><td>21.00% 22.59%</td><td>25.29% 26.97%</td><td>30.58% 56.70%</td><td>86.63% 52.14% (↓)</td></tr></table>

In practice, we first train our model on standard benchmarks, such as MNIST, Fashion-MNIST, and SVHN. Then, we train classifiers on the inferred latent codes at each layer. We consider VLAE [42] as our baseline model, which learns a structural latent space with standard Gaussian prior. We show comparison results in Tab. 1, where our learned inference model $q _ { \phi } ( \mathbf { z } | \mathbf { x } )$ extracts varying levels of semantic features for different layers of latent codes. The top layer carries the most significant features, and the bottom layer carries the least. The baseline model that assumes Gaussian prior, however, extracted mixed semantic features across different layers. For fair comparisons, we use the same generation and inference structure as the baseline model<sup>2</sup>. The structure of the classifier contains two linear layers with a hidden dimension set to 256 and uses ReLU activation function. The classifier has softmax as its output layer. We do not employ drop-out and normalization strategies.

## 5.2. Image Modelling

In this section, we evaluate our model in image modelling by measuring the quality of generated and reconstructed images. We consider the baseline models that assume Gaussian prior, such as Alternating Back-propagation (ABP) [14], Ladder-VAE (LVAE) [30], BIVA [21], Shortrun MCMC (SRI) [25], and VLAE [42], as well as generator models with informative prior, such as RAE [9], Twostages VAE (2s-VAE) [5], NCP-VAE [1], Multi-NCP [37] and LEBM [26]. To make fair comparisons, we follow the protocol in [26].

Table 2. Testing reconstruction by MSE, and generation evaluation by FID on SVHN and CelebA-64.
<table><tr><td colspan="2">SVHN</td><td colspan="2">CelebA-64</td></tr><tr><td>Model</td><td>MSE E (↓) FID (↓)</td><td>MSE (↓)</td><td>FID (↓)</td></tr><tr><td>ABP</td><td></td><td>49.71</td><td>51.50</td></tr><tr><td>LVAE</td><td>0.014</td><td>39.26</td><td>0.028 53.40</td></tr><tr><td>BIVA</td><td>0.010</td><td>31.65</td><td>0.010 33.58</td></tr><tr><td>SRI</td><td>0.011</td><td>35.23</td><td>0.011 36.84</td></tr><tr><td>VLAE</td><td>0.016</td><td>43.95</td><td>0.010 44.05</td></tr><tr><td>2s-VAE</td><td>0.019</td><td>42.81</td><td>0.021 44.40</td></tr><tr><td>RAE</td><td>0.014</td><td>40.02</td><td>0.018 40.95</td></tr><tr><td>NCP-VAE</td><td>0.020</td><td>33.23</td><td>0.021 42.07</td></tr><tr><td>Multi-NCP</td><td>0.004</td><td>26.19</td><td>0.009 35.38</td></tr><tr><td>LEBM</td><td>0.008</td><td>29.44</td><td>0.013 37.87</td></tr><tr><td>Ours</td><td>0.008</td><td>24.16</td><td>0.004 32.15</td></tr></table>

Synthesis: We use Fre´chet Inception Distance (FID) [16] to quantitatively evaluate the sample quality. In Tab. 2, it can be seen that our model obtains superior generation quality compared to the baseline models.

Reconstructions: To evaluate the accuracy of our inference model, we compute mean square error (MSE) [26, 24] for testing reconstruction. As shown in Tab. 2, the proposed model leads to more accurate and faithful reconstructions.

## 5.3. Analysis of Latent Space

Visualization of toy data. To illustrate the expressivity of the proposed EBM prior, we train our model on MNIST (using the same setting as shown in Fig. 2 and Fig. 6) and set the latent dimension for both $\mathbf { z } _ { 1 }$ and $\mathbf { z } _ { 2 }$ to 2 for better visualization. We visualized the Langevin transition of the prior sampling in Fig. 7, which shows that the latent codes at the top layer $\mathbf { \Gamma } ( \mathbf { z } _ { 2 } )$ can effectively capture the multi-modal posterior, where using a Gaussian prior might be infeasible.

![](images/c5ea21e347a8a728d84784f4ea9f807cce0f889aa36c222955fd49f7c2690c9b.jpg)  
Figure 7. Visualization of the latent codes sampled from our EBM prior (Top row: z<sub>2</sub>). Blue, Orange color indicate prior and posterior, respectively.

Langevin trajectory. We explore the energy landscape of the learned prior $p _ { \alpha } ( \mathbf { z } )$ via transition of Langevin dynamics initialized from $p _ { 0 } ( \mathbf { z } )$ towards $p _ { \alpha } ( \mathbf { z } )$ on CelebA-64. If the EBM is well-learned, the energy prior should render local modes of the energy function, and traversing these lo cal modes should present diverse, realistic image synthesis and steady-state energy scores. Existing EBMs often suffer from oversaturated synthesis via the challenging longrun Langevin dynamic (see oversaturated example in Fig.3 in [23]). We thereby test our model with 2500 Langevin steps, which is much longer than the 40 steps used in training. Fig. 8 shows the image synthesis and energy profile, where our EBM prior model does not exhibit the oversaturated phenomenon and successfully delivers diverse and realistic synthesis with steady-state energy.

![](images/d58ffff0b19d99031a1037ab2fdd2bd55c6705b0599778601282d9b39c88e53d.jpg)  
Figure 8. Transition of Markov chains initialized from p<sub>0</sub>(z) towards $p _ { \alpha } ( \mathbf { z } )$ for 2500 steps. Top: Trajectory in the CelebA-64 data space for every 100 steps. Bottom: Energy profile over time.

## 5.4. Robustness

In this section, we examine the robustness of our model in adversarial attack and outlier detection.

Adversarial robustness. For adversarial robustness, we consider the latent attack [10], which is shown to be a strong attack for the generative model with latent variables, and it allows us to attack the latent codes at different layers. Specifically, let $\mathbf { x } _ { t }$ be the target and $\mathbf { x } _ { o }$ be the original image. The attack aims to create the distorted image $\tilde { \textbf { x } } = \textbf { x } _ { o } + \epsilon$ such that its learned inference distribution $q _ { \phi } ( { \bf z } | \tilde { \bf x } )$ is close to the inference $q _ { \phi } ( \mathbf { z } | \mathbf { x } _ { t } )$ on target images under KL-divergence (Eq. 5 in [10]).

## 3633363336333656

Figure 9. Adversarial latent attack on different layers. The leftmost figure indicates the bottom layer, and the rightmost figure indicates the top layer. 1st col: original image. 2nd col: target image. 3rd col: adversarial example. 4th col: reconstruction of adversarial example.

We illustrate the adversarial attack on different layers in Fig. 9. The learned inference model $q _ { \phi } ( { \bf z } | { \bf x } )$ carries different levels of semantics for different layers. The adversarial example created by attacking latent codes at the bottom layer only perturbs the data along low-level features direction from the target image; hence, the model can be robust and have faithful reconstruction from adversarial examples. The adversarial example coming from the top-layer attack can distort the original data along a high-level semantic direction from the target image, thus, the reconstruction from the adversarial example can be substantially different from the original data, indicating a successful attack.

Anomaly detection. We also evaluate the model on anomaly detection. If the model is well-learned, the inference model $q _ { \phi } ( { \bf z } | { \bf x } )$ should form an informative latent space that separates anomalies from the normal data. Following the protocols in [18, 26], we take latent samples from the learned inference model and use the unnormalized log-posterior log $p _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { z } )$ as our decision function to detect anomaly on MNIST. Only one digit class is held-out as an anomaly in training, while both normal and anomalous data are used for testing. We compare baseline models such as BiGAN-σ [41], MEG [18], LEBM [26], HVAE and VAE using area under the precision-recall curve (AUPRC). The results of baseline models are provided by [26], and we use $L ^ { > 0 }$ [15] as the decision function for HVAE model. Tab. 3 shows the result.

Table 3. AUPRC scores for unsupervised anomaly detection on MNIST. Following the protocol of [26], we average the results over last 10 epochs to account for variance.
<table><tr><td>hold-out digit</td><td>1</td><td>4</td><td>5</td><td>7</td><td>9</td></tr><tr><td>VAE</td><td>0.063</td><td>0.337</td><td>0.325</td><td>0.148</td><td>0.104</td></tr><tr><td>HVAE</td><td>0.494 ± 0.004</td><td>0.920 ± 0.004</td><td>0.913 ± 0.003</td><td>0.680 ± 0.006</td><td> $0 . 7 9 1 \pm 0 . 0 0 8$ </td></tr><tr><td>MEG</td><td>0.281 ± 0.035</td><td>0.401 ± 0.061</td><td>0.402 ± 0.062</td><td>0.290 ± 0.040</td><td>0.342 ± 0.034</td></tr><tr><td>BiGAN-σ</td><td>0.287 ± 0.023</td><td>0.443 ± 0.029</td><td>0.514 ± 0.029</td><td>0.347 ± 0.017</td><td> $0 . 3 0 7 \pm 0 . 0 2 8$ </td></tr><tr><td>Latent EBM</td><td>0.336 ± 0.008</td><td>0.630 ± 0.017</td><td>0.619 ± 0.013</td><td>0.463 ± 0.009</td><td> $0 . 4 1 3 \pm 0 . 0 1 0$ </td></tr><tr><td>Ours</td><td>0.722 ± 0.010</td><td>0.949 ± 0.002</td><td>0.980 ± 0.001</td><td>0.941 ± 0.003</td><td> $\mathbf { 0 . 9 3 5 \pm 0 . 0 0 3 }$ </td></tr></table>

## 5.5. Ablation Studies

The proposed expressive prior model enables complex data representations to be effectively captured, which in turn improves the expressivity of the whole model in generating high-quality synthesis. To better understand the influence of the proposed method, we conduct ablation studies, including 1) MCMC steps, 2) complexity of EBM, and 3) other image datasets.

MCMC steps: We examine the influence of steps of shortrun MCMC for prior sampling. Tab. 4 shows that increasing steps could result in a better quality of generation. Increasing the step number from 15 to 60 could result in a significant improvement in synthesis quality, while exhibiting only minor influence when increased beyond 60.

Table 4. Varying k (steps number) of our model on CelebA-64.
<table><tr><td>k =15</td><td>k=30</td><td>k=60</td><td>k =150</td><td>k=300</td></tr><tr><td>FID 56.42</td><td>39.89</td><td>32.15</td><td>31.20</td><td>30.78</td></tr></table>

Complexity of EBM: We further examine the influence of model parameters of EBM. The energy function is parameterized by a 2-layer perceptron. We fix the number of layers and increase the hidden units of EBM. As shown in Tab. 5, increasing the hidden units of EBM results in better performance.

Table 5. Increasing hidden units (denoted as nef) of EBM.
<table><tr><td>nef</td><td>nef= 0</td><td>nef= 10</td><td>nef= 20</td><td>nef= 50</td><td>nef= 100</td></tr><tr><td>FID</td><td>43.95</td><td>41.72</td><td>33.10</td><td>30.42</td><td>24.16</td></tr></table>

Other image datasets: We test the scalability of our model on high-resolution image dataset, such as CelebA-128. We show the image synthesis in Fig. 10.

![](images/2e83215ab150d3d3094107ceb30c4ac85a3172b79e8d93c2369c68cf458a5259.jpg)  
Figure 10. Generated image synthesis on CelebA-128

We also train our model on CIFAR-10 (32x32) and report the FID score for generation quality in Tab. 6, in which our model performs well compared to baseline models.

Table 6. FID on CIFAR-10
<table><tr><td>Ours</td><td>2s-VAE</td><td>RAE</td><td>NCP-VAE</td><td>Multi-NCP</td><td>LEBM</td></tr><tr><td>63.42</td><td>72.90</td><td>74.16</td><td>78.06</td><td>65.01</td><td>70.15</td></tr></table>

## 6. Conclusion

In this paper, we propose to build a joint latent space EBM with hierarchical structures for better hierarchical representation learning. Our joint EBM prior is expressive in modelling the inter-layer relation among multiple layers of latent variables, which in turn enables different levels of data representations to be effectively captured. We develop an efficient variational learning scheme and conduct various experiments to demonstrate the capability of our model in hierarchical representation learning and image modelling. For future work, we shall explore learning the hierarchical representations from other domains, such as videos and texts.

## References

[1] Jyoti Aneja, Alex Schwing, Jan Kautz, and Arash Vahdat. A contrastive learning approach for training variational autoencoder priors. Advances in neural information processing systems, 34:480–493, 2021. 2, 5, 7

[2] Philip Bachman. An architecture for deep, hierarchical generative models. Advances in Neural Information Processing Systems, 29, 2016. 5

[3] Rewon Child. Very deep vaes generalize autoregressive models and can outperform them on images. arXiv preprint arXiv:2011.10650, 2020. 1, 5

[4] Jiali Cui, Ying Nian Wu, and Tian Han. Learning joint latent space ebm prior model for multi-layer generator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3603–3612, 2023. 1, 5

[5] Bin Dai and David Wipf. Diagnosing and enhancing vae models. arXiv preprint arXiv:1903.05789, 2019. 5, 7

[6] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Improved contrastive divergence training of energy based models. arXiv preprint arXiv:2012.01316, 2020. 2

[7] Yilun Du and Igor Mordatch. Implicit generation and generalization in energy-based models. arXiv preprint arXiv:1903.08689, 2019. 1, 2

[8] Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, and Diederik P Kingma. Learning energy-based models by diffusion recovery likelihood. arXiv preprint arXiv:2012.08125, 2020. 2

[9] Partha Ghosh, Mehdi SM Sajjadi, Antonio Vergari, Michael Black, and Bernhard Scholkopf. From variational to de-¨ terministic autoencoders. arXiv preprint arXiv:1903.12436, 2019. 5, 7

[10] George Gondim-Ribeiro, Pedro Tabacof, and Eduardo Valle. Adversarial attacks on variational autoencoders. arXiv preprint arXiv:1806.04646, 2018. 7, 8

[11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 5

[12] Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin, David Vazquez, and Aaron Courville. Pixelvae: A latent variable model for natural images. arXiv preprint arXiv:1611.05013, 2016. 1

[13] Jiaxian Guo, Sidi Lu, Han Cai, Weinan Zhang, Yong Yu, and Jun Wang. Long text generation via adversarial training with leaked information. In Proceedings of the AAAI Conference on Artificial Intelligence, number 1, 2018. 1

[14] Tian Han, Yang Lu, Song-Chun Zhu, and Ying Nian Wu. Alternating back-propagation for generator network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017. 5, 7

[15] Jakob D Drachmann Havtorn, Jes Frellsen, Soren Hauberg, and Lars Maaløe. Hierarchical vaes know what they don’t know. In International Conference on Machine Learning, pages 4117–4128. PMLR, 2021. 8

[16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a

two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. 7

[17] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4, 5

[18] Rithesh Kumar, Sherjil Ozair, Anirudh Goyal, Aaron Courville, and Yoshua Bengio. Maximum entropy generators for energy-based models. arXiv preprint arXiv:1901.08508, 2019. 8

[19] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, and F Huang. A tutorial on energy-based learning. Predicting structured data, 1(0), 2006. 1

[20] Zhiyuan Li, Jaideep Vitthal Murkute, Prashnna Kumar Gyawali, and Linwei Wang. Progressive learning and disentanglement of hierarchical representations. arXiv preprint arXiv:2002.10549, 2020. 1, 3, 5

[21] Lars Maaløe, Marco Fraccaro, Valentin Lievin, and Ole´ Winther. Biva: A very deep hierarchy of latent variables for generative modeling. Advances in neural information processing systems, 32, 2019. 1, 2, 5, 7

[22] Radford M Neal et al. Mcmc using hamiltonian dynamics. Handbook ofmarkov chain monte carlo, 2(11):2, 2011. 4

[23] Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of mcmc-based maximum likelihood learning of energy-based models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 5272–5280, 2020. 7

[24] Erik Nijkamp, Mitch Hill, Song-Chun Zhu, and Ying Nian Wu. Learning non-convergent non-persistent short-run mcmc toward energy-based model. Advances in Neural Information Processing Systems, 32, 2019. 1, 2, 7

[25] Erik Nijkamp, Bo Pang, Tian Han, Linqi Zhou, Song-Chun Zhu, and Ying Nian Wu. Learning multi-layer latent variable model via variational optimization of short run mcmc for approximate inference. In European Conference on Computer Vision, pages 361–378. Springer, 2020. 1, 5, 7

[26] Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space energy-based prior model. Advances in Neural Information Processing Systems, 33:21994–22008, 2020. 1, 2, 7, 8

[27] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, pages 1278–1286. PMLR, 2014. 5

[28] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. 1

[29] Casper Kaae Sø nderby, Tapani Raiko, Lars Maalø e, Søren Kaae Sø nderby, and Ole Winther. Ladder variational autoencoders. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. 5

[30] Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational

autoencoders. Advances in neural information processing systems, 29, 2016. 1, 2, 7

[31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1

[32] Yuhta Takida, Takashi Shibuya, WeiHsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. arXiv preprint arXiv:2205.07547, 2022. 5

[33] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein auto-encoders. arXiv preprint arXiv:1711.01558, 2017. 5

[34] Jakub Tomczak and Max Welling. Vae with a vampprior. In International Conference on Artificial Intelligence and Statistics, pages 1214–1223. PMLR, 2018. 5

[35] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 1

[36] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. Advances in Neural Information Processing Systems, 33:19667–19679, 2020. 1, 2, 5

[37] Zhisheng Xiao and Tian Han. Adaptive multi-stage density ratio estimation for learning latent space energy-based model. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. 2, 7

[38] Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. Vaebm: A symbiosis between variational autoencoders and energy-based models. arXiv preprint arXiv:2010.00654, 2020. 2

[39] Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu. A theory of generative convnet. In International Conference on Machine Learning, pages 2635–2644. PMLR, 2016. 1

[40] Lantao Yu, Weinan Zhang, Jun Wang, and Yong Yu. Seqgan: Sequence generative adversarial nets with policy gradient. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017. 1

[41] Houssam Zenati, Chuan Sheng Foo, Bruno Lecouat, Gaurav Manek, and Vijay Ramaseshan Chandrasekhar. Efficient gan-based anomaly detection. arXiv preprint arXiv:1802.06222, 2018. 8

[42] Shengjia Zhao, Jiaming Song, and Stefano Ermon. Learning hierarchical features from generative models. arXiv preprint arXiv:1702.08396, 2017. 1, 3, 5, 6, 7