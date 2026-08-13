# Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation

Changqi Wang<sup>1\*</sup>, Haoyu Xie<sup>1∗</sup>, Yuhui Yuan<sup>2</sup>, Chong Fu<sup>1,4†</sup>, Xiangyu Yue<sup>3</sup>

<sup>1</sup> School of Computer Science and Engineering, Northeastern University, Shenyang, China <sup>2</sup> Microsoft Research Asia <sup>3</sup> The Chinese University of Hong Kong <sup>4</sup> Key Laboratory of Intelligent Computing in Medical Image, Ministry of Education, NEU, China

2101668@stu.neu.edu.cn, xiehaoyu@stumail.neu.edu.cn, yuyua@microsoft.com fuchong@mail.neu.edu.cn, xyyue@ie.cuhk.edu.hk

## Abstract

Semi-Supervised Semantic Segmentation (S4) aims to train a segmentation model with limited labeled images and a substantial volume of unlabeled images. To improve the robustness of representations, powerful methods introduce a pixel-wise contrastive learning approach in latent space (i.e., representation space) that aggregates the representations to their prototypes in afully supervised manner. However, previous contrastive-based S4 methods merely rely on the supervision from the model’s output (logits) in logit space during unlabeled training. In contrast, we utilize the outputs in both logit space and representation space to obtain supervision in a collaborative way. The supervision from two spaces plays two roles: 1) reduces the risk of over-fitting to incorrect semantic information in logits with the help of representations; 2) enhances the knowledge exchange between the two spaces. Furthermore, unlike previous approaches, we use the similarity between representations and prototypes as a new indicator to tilt training those under-performing representations and achieve a more efficient contrastive learning process. Results on two public benchmarks demonstrate the competitive performance of our method compared with state-of-the-art methods.

## 1. Introduction

Semantic segmentation is a fundamental task in computer vision, aiming to classify each pixel in an image. Significant progress [22, 4] has been made in training on highquality labeled images using segmentation models composed of an encoder and a segmentation head. However, annotating images is expensive and time-consuming. Semisupervised Semantic Segmentation (S4) alleviates the thirst for annotation by leveraging unlabeled images to train segmentation models.

![](images/0a8f9e8de1c949b1c0cbd96bc2ccc23e54e32cf4f8e474acb4e711cd4af0fe21.jpg)  
Figure 1. We enhance the knowledge exchange between the logit and representation spaces. Orange and blue represent different classes. Top: Existing contrastive-based S4 methods overlook the semantic information in representation space. Bottom: Our method uses dual-space collaborative supervision.

Most existing works learn from unlabeled images via self-training [44, 37, 15] or consistency regularization [40, 38, 13] strategies, both of which retrain the model with its predictions on unlabeled images. Recently, great success has been achieved by introducing pixel-wise contrastive learning to semantic segmentation, which endows the model with a stronger features-extracting ability by accessing a more discriminative representation space. Specially, these methods [45, 23] project each pixel to representation space as a representation and regularize it in a fully supervised manner, i.e., aggregating the representations with the same class and separating them with different classes. In semi-supervised settings, due to limited labels, most methods [1, 32, 46] obtain supervision from the model’s output logits in logit space during the unlabeled training process. However, recent contrastive-based semantic segmentation methods [1, 32, 46, 45] mainly focus on the learning process in logit space while only taking that in representation space as an auxiliary task. The unidirectional supervision makes training dominated by the predicted logits, leading to the neglect of information in the representation space. We argue that this kind of single-space supervision may incorrectly provide semantic guidance to representation learning and fails to facilitate knowledge exchange between the two spaces (see Sec. 5.1).

In this work, we extend the single-space supervision to a dual-space supervision for contrastive-based S4 and propose Collaborative Space Supervision (CSS). Our key insight is to: i) utilize the semantic information in representations to obtain more reliable guidance during unlabeled training, and to enhance knowledge exchange between two spaces; ii) provide a more accurate reference for the model’s performance on predicting each representation to tilt training those under-performing representations. To achieve objective i), we obtain dense semantic predictions by retrieving the nearest class prototype for each representation in the representation space and then engage with predictions from the logits for collaborative supervision of the model. For objective ii), we measure the similarity between the representations and prototypes and use the similarity after a normalization operation as the indicator for guiding the learning process in the representation space. Unlike previous works that utilize confidence as the indicator to involve representation learning, the similarity directly reflects the confusion level between representations and prototypes, resulting in more efficient representation learning.

To summarize, our main contributions are three-fold: 1) The dual-space collaboration for contrastive-based S4, enhances the knowledge exchange between the logit and representation spaces. 2) Utilizing similarity to provide a more accurate reference for the model’s performance in representation learning. 3) Extensive experiments on two S4 benchmarks demonstrate the effectiveness of our method.

## 2. Related Works

## 2.1. Semi-supervised Semantic Segmentation

The aim of S4 is to train a segmentation model with the semi-supervised setting (i.e., a few labeled images and a large number of unlabeled images) to classify each pixel in an entire image. The critical issue of S4 is how to leverage unlabeled images to train the model. Some methods [25,

27, 30, 35] based on GANs [16], adversarial training [36], and consistency regularization paradigm [38, 13, 40, 7, 56]. Meanwhile, self-training [28, 44, 49, 61, 53] is also a striking paradigm, which always generates pseudo-labels from model and retrains the model with the combined supervision of human annotations and pseudo-labels. One essential issue of self-training is the accuracy of pseudo-labels. Some methods [29, 33, 14, 42, 52] try to polish pseudo-labels and provide reliable guidance. Some methods [21, 47, 24, 17] focus on the class-imbalance problems in the dataset and try to alleviate the negative effect from class-biased pseudolabels generated by the model pre-trained on imbalanced labeled images. We build our framework based on the self-training and additionally explore semantic information among different images.

## 2.2. Pixel-wise Contrastive Learning

Pixel-wise contrastive learning explores semantic relations not only in the individual image but also among different images. Different from instance-wise contrastive learning [19, 6, 3], pixel-wise contrastive learning [50, 55, 5, 58] project each pixel to the representation in representation space with the cooperation of encoder and representation head. Representations are then aggregated in their prototypes and are separated from each other in different classes. In semi-supervised settings, most methods [1, 32, 46, 48] use pseudo-labels based on logits to provide semantic information contrastive learning process during training on unlabeled images. Meanwhile, the confidence of logit is used as an indicator to involve the contrastive learning process, e.g., [32] uses the hard representations whose corresponding logit confidence is lower than a threshold to contrast for effective training. As opposed to the above methods, we use collaborative space supervision for contrastive learning on unlabeled images and use a new indicator to involve the contrastive learning progress.

## 2.3. Prototype-based Learning

Prototype-based learning has been widely studied in few-shot learning [41, 11, 31, 34] and unsupervised domain adaption [54, 26, 43, 39, 57]. Recently, it is restudied in semantic segmentation as known as a non-parametric prototype-based classifier [60]. Concretely, the classes in the dataset are presented by a set of non-learnable prototypes, and the dense semantic predictions are thus achieved by assigning the output features to its most similar prototype. Under semi-supervised settings, some methods [51] maintain the consistency between predictions from a linear predictor and a prototype-based predictor. The two predictors are followed by the encoder and project the features to logit space and representation space, respectively. In this work, we combine the semantic information in the logit and representation spaces to provide supervision in a collaborative way during semi-supervised learning.

## 3. Methodology

In the S4 task, we have a small labeled set $\begin{array} { r l } { \mathcal { D } _ { l } } & { { } = } \end{array}$ $\{ ( \pmb { x } _ { i } ^ { l } , \pmb { y } _ { i } ^ { l } ) \} _ { i = 1 } ^ { N _ { l } }$ and a large unlabeled set $\mathcal { D } _ { u } ~ = ~ \{ \pmb { x } _ { i } ^ { u } \} _ { i = 1 } ^ { N _ { u } }$ where $\pmb { x } _ { i } ^ { l } , \pmb { x } _ { i } ^ { u } \in \mathbb { R } ^ { H \times W \times 3 }$ , H, W denote the height, and the width, respectively. And ground truth $\pmb { y } _ { i } ^ { l } \in \{ 0 , \overset { \smile } { 1 } \} ^ { H \times W \times | C | }$ with the set of class C. The goal is to boost model performance with $\mathcal { D } _ { u }$ . The base model consists of an encoder $f ( \cdot )$ and a segmentation head $g ( \cdot )$ , which projects features to the logit space $\mathbb { R } ^ { H \times W \times | C | }$ . We adopt Self-Training (ST) and pixel-wise contrastive learning to our framework, as described in Sec. 3.1. The supervision for $\mathcal { D } _ { u }$ is produced by the collaboration between the logit and representation spaces, as described in Sec. 3.3.

## 3.1. ST and Pixel-wise Contrastive Learning

The main idea of self-training is to pre-train a model on labeled images and use it to produce pseudo-labels as supervision for unlabeled images. A typical framework is the teacher-student framework [44], which consists of a student model and a teacher model. Both the student model and the teacher model are constructed by an encoder and a segmentation head. Parameters of the student model are optimized via Stochastic Gradient Decent (SGD) while parameters of the teacher model are updated by the Exponential Moving Average (EMA) of student model parameters. We denote the encoder and the segmentation head in the student model by $f ( \cdot )$ and $g ( \cdot )$ while denoting those in the teacher model by $f ^ { \prime } ( \cdot )$ and $g ^ { \prime } ( \cdot )$ . The pseudo-labels $\hat { \pmb { y } } _ { i } ^ { u , l g t }$ are produced based on the teacher model’s output logits $\hat { p } _ { i } ^ { u } = g ^ { \prime } ( f ^ { \prime } ( \pmb { x } _ { i } ^ { u } ) )$ in logit space, formulated as:

$$
\hat { \pmb { y } } _ { i } ^ { u , l g t } = \mathbf { 1 } _ { c } ( \arg \operatorname* { m a x } \{ \hat { \pmb { p } } _ { i , c } ^ { u } \} _ { c \in C } ) ,\tag{1}
$$

where $\mathbf { 1 } _ { c } ( \cdot )$ denotes the one-hot encoding of class c.

In order to enhance the ability of the model itself to extract features, recent works [32, 46, 1] additionally employ pixel-wise contrastive learning and introduce a representation head to both teacher and student models. We denote the representation head in the student model as $h ( \cdot )$ and that in the teacher model as $h ^ { \prime } ( \cdot )$ . The pixel $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ of the class c are projected as representations $z _ { c i }$ in representation space by the cooperating of $f ( \cdot )$ and $h ( \cdot ) , i . e . , z _ { c i } = h ( f ( \pmb { x } _ { i } ) )$ . And the representation $z _ { c i }$ is then aggregated to its class centroid (prototype) while separated from representations in different classes $z _ { \tilde { c } i }$ (negatives). The semantic guidance for contrastive learning is from the combination of ground truth $\mathbf { \Delta } y _ { i } ^ { l }$ and pseudo-labels $\hat { \pmb { y } } _ { i } ^ { u , l g t }$ in logit space. Moreover, in order to emphasize the reliable and crucial aspects during unlabeled and contrastive learning, a sampling strategy is adopted to select valid pixels $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ according to their corresponding confidence, i.e. the student model’s output logits $\pmb { p } _ { i }$ after a Softmax operation.

Discussion. In recent works [46, 1, 32], the supervision of unlabeled images is derived solely from the logit space. This overlooks the potential benefits of the supervision from the representation space, leading to two potential limitations: 1) the pseudo-labels $\hat { \pmb { y } } _ { i } ^ { u , l \breve { g } t }$ obtained from the logit space may contain noise and miss the opportunity to be corrected by semantic information from the representation space; 2) since the confidence from logit space is used as the indicator $\hat { j } _ { i }$ for the sampling strategy, learning in the representation space may not be critical enough due to the different confusing parts between the two spaces.

To mitigate these limitations, we produce pseudo-labels from the representation space and combine them with pseudo-labels from the logit space to provide higher-quality supervision during unlabeled training. Meanwhile, we obtain a new indicator from the representation space for the more effective sampling strategy.

## 3.2. Supervision from Representation Space

In this section, we detail the approach to obtain the pseudo-labels from the representation space. Meanwhile, we simultaneously access a new indicator for the sampling strategy in representation spaces, which provides a critical reference in the contrastive learning process.

Specifically, we first build a set of prototypes for each class and obtain the pseudo-labels by retrieving the nearest prototype for each representation. We calculate the centroid of all representations in the current class c as the prototype $\rho _ { c } ,$ , which is formulated as:

$$
\rho _ { c } = \frac { 1 } { N _ { c } } \sum _ { i } ^ { N _ { c } } z _ { i } ^ { \prime } ,\tag{2}
$$

where $N _ { c }$ is the total number of representations of current class c and $\boldsymbol { z } _ { i } ^ { \prime }$ is the representation projected by the cooperation of the $f ^ { \prime } ( \cdot )$ and $h ^ { \prime } ( \cdot )$ . Meanwhile, to include more representation information, we update the prototype along the sequential iterations with EMA as follows:

$$
\hat { \pmb { \rho } } _ { c } ( t ) = \alpha \hat { \pmb { \rho } } _ { c } ( t - 1 ) + ( 1 - \alpha ) \pmb { \rho } _ { c } ( t ) ,\tag{3}
$$

where $\hat { \rho } _ { c } ( t ) , \hat { \rho } _ { c } ( t - 1 )$ mean the current $t ^ { t h }$ prototype and last $( t - 1 ) ^ { t h }$ prototype in iterations , $\rho _ { c } ( t )$ means the prototype calculated by Eq. 2 in current iteration, and α is a hyper-parameter that controls the updating speed. Thus, the pseudo-label from the representation space is achieved by:

$$
\hat { y } _ { i } ^ { u , r e p } = \mathbf { 1 } _ { \hat { c } } ( \hat { c } ) , w i t h \hat { c } = \underset { c } { \arg \operatorname* { m a x } } \{ s i m ( z _ { i } ^ { \prime } , \hat { \rho } _ { c } ( t ) ) \} _ { c \in C } ,\tag{4}
$$

where $s i m ( \cdot )$ is defined as the cosine similarity.

As for the indicator for the sampling strategy in the representation space, we use the Softmax function on the similarity among the representation and all prototypes, which is

![](images/0bdc091a423754b5ecef53408dc2c1e5e3a7109c4c57ddd5ca96bb559bb740fc.jpg)  
Figure 2. Overview of our framework. Our training pipeline consists of learning in two spaces: logit space and representation space. The pseudo-labels $\hat { \mathbf { y } } _ { i } ^ { u }$ during unlabeled training are produced by the collaboration of two spaces with mix pseudo-labeling strategy (1) or cros pseudo-labeling strategy (2). The indicator for representation learning is produced by similarity $( s _ { 1 } , s _ { 2 } ,$ , and $s _ { 3 } )$ .

followed as:

$$
\hat { j } _ { i } ^ { u , r e p } = \frac { e ^ { s i m \left( z _ { c i } , \hat { \rho } _ { c } \left( t \right) \right) / \tau } } { e ^ { s i m \left( z _ { c i } , \hat { \rho } _ { c } \left( t \right) \right) / \tau } + \sum _ { \tilde { c } \in \tilde { C } } e ^ { s i m \left( z _ { c i } , \hat { \rho } _ { \tilde { c } } \left( t \right) \right) / \tau } } ,\tag{5}
$$

where $\hat { \rho } _ { \tilde { c } } ( t )$ means the prototype with different classes from $z _ { c i }$ and τ is a hyper-parameter. Different from using confidence from logit space as the indicator to involve representation learning [32, 46], the Softmax similarity directly helps the model to discover the confusion between representations and their prototypes and focus on them during the subsequent training.

## 3.3. Collaboration Between Two Spaces

With the pseudo-labels in two spaces, we propose two pseudo-labeling strategies to strengthen the collaboration between two spaces and obtain more reliable pseudo-labels.

• Mix pseudo-labeling. To mitigate the negative effects of inherent noise from both two spaces during pseudolabeling, we adopt the mix pseudo-labeling strategy that only considers the mutually agreeable pseudolabels between the two spaces. Specifically, we define the set of final pseudo-labels as $\hat { Y ^ { u } } = \hat { Y } ^ { u , \bar { l } g t } \cap \hat { Y } ^ { u , r e p }$ where $\hat { \pmb { y } } _ { i } ^ { u , l g t } \in \hat { Y } ^ { u , l g t }$ and $\hat { \pmb { y } } _ { i } ^ { u , r e p } \in \hat { Y } ^ { u , r e p }$

• Cross pseudo-labeling. Inspired by recent researches [7, 38] that maintain consistency among the predictions of the same image across different models or decoders in different views, we propose a cross pseudolabeling strategy that leverages pseudo-labels from one space to supervise the other. Specifically, we use pseudo-labels $\hat { y } _ { i } ^ { u , r e p }$ to supervise the logit space, and vice versa.

The strengths of using pseudo-labels from two spaces in the collaborative way are twofold: 1) obtaining more reliable supervision during unlabeled training, and 2) enabling the strengths of learning in different spaces to complement each other. Since the learning in different feature spaces concentrates on different parts of features, i.e., the logit space mainly focuses on the most discriminative part of features while the representation space treats all parts of features equally, the performance of pseudo-labels from two spaces varies across different classes or regions of images. Therefore, our collaborative pseudo-labeling strategies exchange knowledge between two spaces and provide higher-quality supervision during unlabeled training. The experimental proof is in Sec. 5.1.

As for indicators, we use confidence as the indicator $\hat { j } _ { i } ^ { u , l g t }$ for learning logit space and Softmax similarity as the indicator $\hat { j } _ { i } ^ { u , r e p }$ for learning representation space. We argue that the confusing parts of learning in both two spaces are varied due to the different parts of features being concentrated in each space. Therefore, this strategy allows the learning in different spaces to focus on their own confusing parts, which can be more effective than using a single indicator when mining confusing parts for both spaces in the training process. The experimental proof is in Sec. 5.2.

## 3.4. Training Objective

With the indicators $\hat { j } _ { i } ^ { u , l g t }$ and $\hat { j } _ { i } ^ { u , r e p }$ , we adopt some threshold sampling strategies. In logit space, we set a threshold $\delta _ { u }$ during unlabeled learning and logits $\hat { \pmb { p } } _ { i } ^ { u }$ whose indicator $\hat { j } _ { i } ^ { u , l g t }$ is higher than $\delta _ { u }$ will be regarded as the valid logits in logit space. In representation space, our sample strategy can be divided into three parts: 1) Valid Sampling Strategy. Similar to the sampling strategy in logit space, a threshold $\delta _ { w }$ is used to sample representations whose indicator $\hat { j } _ { i } ^ { u , r e p }$ is higher than $\delta _ { w }$ . 2) Hard Sampling Strategy. We adopt the hard sampling strategy for tilting to train those confusing representations. Specifically, we set a threshold $\delta _ { s }$ to sample representations whose indicator $\hat { j } _ { i } ^ { u , r e p }$ is lower than $\delta _ { s }$ . 3) Negative Sampling Strategy. We sample negatives according to the similarity between the prototype of current class c and other prototypes. Concretely, the negatives are more likely to be sampled if its prototype is more similar to the prototype of the current class.

Cooperated with the ground truth $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { i } ^ { l } } .$ , pseudo-labels $\mathbf { \Delta } _ { \mathbf { \mathcal { Y } } _ { i } ^ { u } }$ produced from two spaces in a collaborative way, and different sampling strategies in two spaces, the total learning object is composed with a supervised loss $\mathcal { L } _ { s }$ , an unsupervised loss $\mathcal { L } _ { u }$ , and a contrastive loss $\mathcal { L } _ { c }$ as follows:

$$
\mathcal { L } = \mathcal { L } _ { s } + \mathcal { L } _ { u } + \lambda _ { c } \mathcal { L } _ { c } ,\tag{6}
$$

where $\lambda _ { c }$ is used to tune the contribution between logit space and representation space. Specifically, $\mathcal { L } _ { s }$ and $\mathcal { L } _ { u }$ are constructed by the Cross-Entropy (CE) $\ell _ { c e }$ and can be formulated as:

$$
\mathcal { L } _ { s } = \frac { 1 } { \vert \mathcal { B } _ { l } \vert } \sum _ { \pmb { x } _ { i } ^ { l } \in B _ { l } } \ell _ { c e } ( \pmb { p } _ { i } ^ { l } , \pmb { y } _ { i } ^ { l } ) ,\tag{7}
$$

$$
\mathcal { L } _ { u } = \frac { 1 } { \vert \hat { B } _ { u } \vert } \sum _ { { \pmb x } _ { i } ^ { u } \in \hat { B } _ { u } } \ell _ { c e } ( { \pmb p } _ { i } ^ { u } , \hat { \pmb y } _ { i } ^ { u } ) ,\tag{8}
$$

where $\boldsymbol { B } _ { l }$ denotes the labeled images in a mini-batch and $\hat { B } _ { u }$ is the subsets that sampled from unlabeled images in a mini-batch according to the sampling strategy. Meanwhile, the contrastive loss $\mathcal { L } _ { c }$ is formulated as:

$$
\begin{array} { c } { \mathcal { L } _ { c } = - \displaystyle \frac { 1 } { | C | \times | \hat { \mathcal { Z } } _ { c } | } \sum _ { c \in C } \sum _ { z _ { c i } \in \hat { \mathcal { Z } } _ { c } } } \\ { l o g [ \frac { e ^ { s ( z _ { c i } , \hat { \rho } _ { c } ) / \tau } } { e ^ { s ( z _ { c i } , \hat { \rho } _ { c } ( t ) ) / \tau } + \sum _ { \tilde { c } \in \tilde { C } } \sum _ { z _ { \tilde { c } i } \in \hat { \mathcal { Z } } _ { \tilde { c } } } e ^ { s ( z _ { c i } , z _ { \tilde { c } i } ) / \tau } } ] , } \end{array}\tag{9}
$$

where $\hat { \mathcal { Z } } _ { c }$ is the subset sampled from the set of the representations which belong to class c according to the sampling strategy, $\hat { \mathcal { Z } } _ { \tilde { c } }$ is the subset sampled from the set of the representations which bot belong to class $c , { \tilde { C } }$ denotes the subset of C with class c removed, and the supervision information comes from the final pseudo-labels $\hat { \mathbf { \mathscr { y } } } _ { i } ^ { u }$ after the pseudo-labeling strategies.

The whole framework is shown in Fig. 2. All the pseudocode of producing pseudo-labels and their indicators from the logit and representation spaces is shown in Algorithm 1, and the pseudo-code of the mix pseudo-labeling strategy is shown in Algorithm 2. The pseudo-code is PyTorch-like.

Algorithm 1 Pseudo-code of producing pseudo-labels and   
indicators   
Network: Teacher’s encoder $f ^ { \prime } ,$ segmentation head $g ^ { \prime } { \mathrm { . } }$ , rep  
resentation head $h ^ { \prime } .$   
Input: Unlabeled mini-batch $B _ { u }$ consists of $X ^ { u }$ , the set of   
prototypes $\{ \hat { \rho } _ { c } ( t ) \} _ { c \in C } .$   
1: for $X _ { u } \in B _ { u }$ do   
2: # Predict logits from unlabeled images.   
3: $\hat { P } ^ { u , l g t } \gets g ^ { \prime } ( f ^ { \prime } ( X ^ { u } ) )$   
4: # Produce labels and indicators based on logits with   
Eq. 1.   
5: $\hat { Y } ^ { \hat { u } , l g t } , \hat { J } ^ { u , l g t } \gets l g t \_ p s e u d o ( \hat { P } ^ { u , l g t } )$   
6: # Predict representations from unlabeled images.   
7: $\hat { R } ^ { u , r e p } \gets \hat { h ^ { \prime } } ( f ^ { \prime } ( X ^ { u } ) )$   
8: # Produce labels and indicators based on representa  
tions with Eq. 4 and Eq. 5.   
9: $\hat { Y } ^ { u , r e p } , \hat { J } ^ { u , r e p } \gets r e p . \textmu s e u d o ( \hat { R } ^ { u , r e p } , \{ \hat { \pmb { \rho } } _ { c } ( t ) \} _ { c \in C } )$   
10: end for

Algorithm 2 Pseudo-code of mix pseudo-labeling strategy   
Input: Pseudo-labels from logit space $\hat { Y } ^ { u , l g t }$ and from rep  
resentation space $\hat { Y } ^ { u , r e p }$ . Indicators $\hat { J } ^ { u , l g t }$ and $\scriptstyle { \hat { J } } ^ { u , r e p }$   
Notation: Threshold $\delta _ { u }$ for sampling strategy in $\mathcal { L } _ { u } .$ weak   
threshold $\delta _ { w }$   
1: # Sampling valid pseudo-labels according to the sam  
pling strategy in logit space.   
2: $\hat { Y } _ { v a l } ^ { u , l g t } \gets s a m p l e { . } l g t ( \hat { Y } ^ { u , l g t } , \hat { J } ^ { u , l g t } , \delta _ { u } )$   
3: # Sampling valid pseudo-labels according to the valid   
sampling strategy in logit space.   
4: $\hat { Y } _ { v a l } ^ { u , \hat { r } e p } \gets s a m p l e . r e p ( \hat { Y } ^ { u , r e p } , \hat { J } ^ { u , r e p } , \delta _ { w } )$   
5: # Produce the mask for mix pseudo-labels.   
6: ma $\mathrm { ~ } _ { 3 } k _ { y } \gets \hat { Y } ^ { u , r e p } . e q ( \hat { Y } ^ { u , l g t } )$   
7: # Produce mix pseudo-labels according to the mask.   
8: $\hat { Y } ^ { u } \gets m a s k _ { - } p s e u d o ( \hat { Y } ^ { u , l g t } , m a s k _ { y } )$

## 3.5. Discussions

We discuss the relations between our framework and other most-related S4 frameworks.

Compare with PCR [51]. The key insight of PCR and our method lies in the promotion of consistency between the output in the logit and representation spaces. However, PCR exclusively relies on pseudo-labels derived from logit space as the guidance in the logit and representation spaces during the unsupervised training process, thereby neglecting the exploitation of the extensive semantic information inherent in representations and prototypes. In contrast, our method adopts a collaborative way by combining pseudo-labels obtained from both the logit and representation spaces, which improves the quality of pseudo-labels and enhances the knowledge exchange between the two spaces. In addition, since these two approaches are orthogonal, one can employ our pseudo-labeling and indicator strategies within the PCR framework, or alternatively incorporate multiple prototypes and consistency loss into our approach. Such combinations may yield further improvements in performance, however, also increase the computational complexity.

Compare with CPS [7]. CPS contains two segmentation models initialized differently. It leverages pseudo-labels generated by one model to supervise the other, thereby maintaining the consistency between the output of the two models in the logit space. In a similar vein to CPS, our cross pseudo-labeling strategy in Sec. 3.3 is also motivated by the objective of maintaining consistency during unsupervised training. However, our method is distinct from CPS in that we focus on preserving consistency between the logit space and the representation space. This distinction confers a distinct advantage in terms of memory efficiency, as we solely introduce a MLP as the representation head, rather than introducing an additional segmentation model as in CPS.

Compare with other contrastive based S4 works [32, 1, 46, 23]. We adhere to prior works to build our prototype, describe in Eq. 2. Different from them, we update our prototypes iteratively, similar to [60]. Our sampling strategies are based on the threshold, similar to [32, 23, 45]. However, it is worth noting that our indicators which serve as the basis for comparison with the threshold, are derived from both the logit and representation spaces. This stands in contrast to prior approaches where indicators solely originate from the logit space. The efficacy of this modification is substantiated and extensively discussed in Sec. 5.2.

## 4. Experiments

## 4.1. Setup

Datasets. We conduct experiments on PASCAL VOC 2012 dataset [12], Cityscapes dataset [9], ADE20K dataset [59], and COCO-Stuff 10K dataset [2] to validate the effectiveness of our proposed method. The original PASCAL VOC 2012 dataset contains 1,464 labeled images in train set and 1,449 images for validation in val set. Following [7, 37], we additionally introduce 9,118 images from SBD [18] as training images. Since the labels in SBD are coarsely annotated, following [46], we use both classic VOC train set (1,464 candidate labeled images) and blender VOC train set (10,582 candidate labeled images). Cityscapes dataset is a dataset for urban scene understanding, which contains 2,975 images in train set and 500 images in val set. ADE20K has 20,210 and 2,000 images in train and val set, with 150 classes in total. COCO-Stuff 10K has 9,000 images in train set and 1,000 images in val set, with 181 classes. The approach of preprocessing labels is followed by MMSegmentation [8].

Network structure. We use Deeplabv3+ [4] with ResNet-101 [20] pre-trained on ImageNet [10] as our network structure. The segmentation and representation head are composed of Conv-BN-ReLU-Conv.

Implementation details. For training on PASCAL VOC 2012 dataset and COCO-Stuff 10K dataset, we set the learning rate as 0.0064, weight decay as 0.0005, crop size as 512 $\times \ 5 1 2 .$ , batch size as 16, and a total of 40,000 iterations. For training on the Cityscapes dataset, we set the learning rate as 0.0038, weight decay as 0.0005, crop size as $7 6 8 \ \times$ 768, batch size as 8, and a total of 80,000 iterations. For training on ADE20K dataset, we set the learning rate as 0.0064, weight decay as 0.0005, crop size as $5 1 2 \times 5 1 2 ,$ batch size as 16, and a total of 80,000 iterations. We use the poly scheduling to decay the learning rate during the training process: $\begin{array} { r } { l r = l r _ { b a s e } \times ( 1 - \frac { e \overline { { p } } o c h } { t o t a l . e p o c h } ) ^ { 0 . 9 } } \end{array}$ . We use the mean of Intersection over Union (mIoU) as the metric in evaluation. We use the sliding window strategy to evaluate the performance of our method on the Cityscapes dataset, following [7]. In addition, when adopting our cross-labeling strategy, due to the requirement of a set of high-quality prototypes when classifying each representation, we first solely use the supervision from logit space for 20 epochs to initialize the prototypes.

## 4.2. Comparison with Existing Methods

In this subsection, we first reproduce three baselines: MT [44], CutMix [15], and a same contrastive-based framework with us but with only logit space pseudo-labels and indicators (Baseline) on classic VOC train set. Meanwhile, we make the comparison of our method with mix (CSS (mix)) and cross (CSS (crs.)) pseudo-labeling strategy on blender VOC train set and Cityscapes train set with following recent SOTA S4 methods: CCT [38], CPS [7], U<sup>2</sup>PL [46], ST++ [52], PRCL [48], PCR [51], and PSMT [33]. Since the data split will dramatically affect the performance in S4, i.e., choosing labeled images plays an important role in the final results, we conduct experiments with three different data splits and report the mean value and standard deviation (blue color numbers). Since the mix pseudo-labeling strategy has better performance, we only use CSS (mix) when compared with SOTAs. Meanwhile, we use ResNet-101 with deep stem blocks as our network structure when compared with SOTAs. Since there is no uniform data split, we use the data splits in U<sup>2</sup>PL [46]. We use the OHEM loss when training Cityscapes.

Results on PASCAL VOC 2012. Tab. 1 shows the comparison with our baselines on Classic PASCAL VOC 2012 set. Our method consistently outperforms baselines with an acceptable standard deviation on all label rates. Tab. 2 shows the comparison with the SOTAs on PASCAL VOC 2012.

Results on Cityscapes. Tab. 3 shows the performance of our method on Cityscapes.

Results on ADE20K and COCO-Stuff 10K. Tab. 4 shows the results of our method on ADE20K and COCO-Stuff.

(b) GT  
(c) lgt. mask  
(d) lgt. label  
Table 1. Results on classic VOC train set with four different label rate. Labeled data splits are from the original VOC train set. All approaches are reproduced with three runs and three different data splits.
<table><tr><td colspan="5">Pascal VOC 2012 (Classic)</td></tr><tr><td>Method</td><td>92</td><td>183</td><td>366</td><td>732</td></tr><tr><td>Sup. MT</td><td>51.57±3.58</td><td> $\overline { { 5 4 . 6 9 _ { \pm 2 . 4 4 } } }$ </td><td> $6 4 . 8 6 _ { \pm 1 . 0 4 }$ </td><td> $\overline { { 7 0 . 7 7 _ { \pm 0 . 7 6 } } }$ </td></tr><tr><td rowspan="3">CutMix Baseline</td><td> $5 8 . 9 2 _ { \pm 2 . 9 9 }$ </td><td> $6 1 . 6 3 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $6 6 . 7 9 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $7 1 . 5 8 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td>65.82±3.60</td><td>67.91±1.47</td><td>72.53±0.50 74.08±0.49</td><td></td></tr><tr><td>66.91±4.21</td><td> $7 0 . 3 2 { \scriptstyle \pm 1 . 8 6 }$ </td><td> $7 3 . 9 7 _ { \pm 1 . 0 4 }$ </td><td> $7 6 . 4 9 { \scriptstyle \pm 0 . 5 8 }$ </td></tr><tr><td rowspan="3">CSS(crs.) CSS(mix)</td><td> $\underline { { 6 7 . 0 3 } } _ { \pm 4 . 5 8 }$ </td><td> $\underline { { 7 1 . 4 1 } } _ { \pm 1 . 6 7 }$ </td><td> $\underline { { 7 4 . 4 7 } } _ { \pm 1 . 0 8 }$ </td><td> $\underline { { 7 7 . 0 8 } } _ { \pm 0 . 3 7 }$ </td></tr><tr><td> ${ \bf 6 8 . 0 9 } _ { \pm 4 . 8 9 }$ </td><td> $\mathbf { 7 1 . 9 3 _ { \pm 1 . 8 8 } }$ </td><td> $\mathbf { 7 4 . 9 1 _ { \pm 1 . 1 2 } }$ </td><td> $7 7 . 5 7 _ { \pm 0 . 7 3 }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 2. Results on blender VOC train set. All the results from the recent papers [46, 52, 13, 29, 33, 51]. Labeled data is from the augmented VOC train set and the data splits are from [46, 33].
<table><tr><td colspan="5">Pascal VOC 2012 (Blender)</td></tr><tr><td>Method</td><td>662</td><td>1323</td><td>2646</td><td>5291</td></tr><tr><td>CCT [38]</td><td>71.86</td><td>73.68</td><td>76.51</td><td>77.40</td></tr><tr><td>CPS [7]</td><td>74.48</td><td>76.44</td><td>77.68</td><td>78.64</td></tr><tr><td>U2PL [46]</td><td>77.21</td><td>79.01</td><td>79.30</td><td>80.50</td></tr><tr><td>ST++ [52]</td><td>74.70</td><td>77.90</td><td>77.90</td><td></td></tr><tr><td>PRCL [48]</td><td>76.96</td><td>78.16</td><td>79.02</td><td>79.59</td></tr><tr><td>PCR [51]</td><td>78.60</td><td>80.71</td><td>80.78</td><td>80.91</td></tr><tr><td>PSMT [33]</td><td>75.50</td><td>78.20</td><td>78.72</td><td>79.76</td></tr><tr><td>CSS (mix)</td><td>78.73</td><td>79.54</td><td>80.82</td><td>81.06</td></tr></table>

Table 3. Results on Cityscapes. The model is trained on the Cityscapes train set, which consists of 2,975 samples in total, and tested on Cityscapes val set. And all the results from the recent papers [46, 51, 33].
<table><tr><td colspan="5">Cityscapes</td></tr><tr><td>Method</td><td>186</td><td>372</td><td>744</td><td>1488</td></tr><tr><td>CCT [38]</td><td>69.32</td><td>74.12</td><td>75.99</td><td>78.10</td></tr><tr><td>CPS [7]</td><td>69.78</td><td>74.31</td><td>74.58</td><td>76.82</td></tr><tr><td>U2PL [46]</td><td>70.30</td><td>74.37</td><td>76.47</td><td>79.05</td></tr><tr><td>PCR [51]</td><td>73.41</td><td>76.31</td><td>78.40</td><td>79.11</td></tr><tr><td>PSMT [33]</td><td></td><td>76.89</td><td>77.60</td><td>79.09</td></tr><tr><td>CSS (mix)</td><td>74.02</td><td>76.93</td><td>77.94</td><td>79.62</td></tr></table>

## 5. Ablative Study

The main contribution of our work lies in 1) collaborative pseudo-labeling strategies and 2) a new indicator for representation learning. To further prove the effectiveness of our proposed method, we conduct ablative studies on these two points. We choose Deeplabv3+ with ResNet-101 pre-trained on ImageNet as our backbone and leverage 92 labeled images and 183 labeled images in PASCAL VOC 2012. The other settings are the same as those in Sec. 4.

## 5.1. Effectiveness of Collaborative pseudo-labeling

Quality of pseudo-labels. To illustrate the superiority of using pseudo-labels from two spaces in a collaborative way as supervision, we conduct experiments to show the quality of pseudo-labels obtained 1) from logit space (lgt.), 2) from representation space (rep.), and 3) from the mix pseudolabeling strategy (mix). The pseudo-labels are sampled with corresponding sampling strategies in Sec. 3.4. Tab. 5 illustrates the IoU of pseudo-labels for each class on PASCAL VOC 2012 with 92 labeled images. The results clearly indicate that employing pseudo-labels from the representation space enhances the accuracy of the final pseudo-labels in most classes. This improvement is particularly evident in classes that are originally under-performing, such as the IoU improvement of 11.42% for the chair class and 11.58% for the sofa class.

![](images/bf96d921a6fc5d7f2e807d61568c4108da9da4e2171578d6d755433eacc0a35a.jpg)  
(a) Image  
Figure 3. Differences between pseudo-labels in different spaces.

Meanwhile, we also visualize the pseudo-labels obtained from logit space (lgt.) and representation space (rep.) in Fig. 3. Fig. 3 (c) and (e) show the masks for pseudolabels produced by the sampling strategies. In particular, the white color represents the valid pixels used during unlabeled learning, while the black color indicates the discarded pixels. Fig. 3 (d) and (f) are the pseudo-labels we obtained from two spaces. The figure clearly illustrates the differences between pseudo-labels produced by different spaces. For example, the parts of the instance edge in pseudo-labels are usually discarded since they are challenging for learning in logit space. However, pseudo-labels from representation space will easily tackle this problem (first row). The pseudo-labels from representation space are more inaccurate in some complex scenes, which be resolved by combining pseudo-labels from logit space (second row).

We mainly attribute the differences in pseudo-label to the differing concentrations of learning in the two spaces. Specifically, learning in the logit space primarily emphasizes the most discriminative part of features, whereas that in the representation space treats each part of features equally. As a result, learning in the logit space may overlook minor feature differences, leading to sub-optimal performance in predicting instance edges and distinguishing between similar classes (e.g., chair and sofa). Conversely, learning in the representation space produces balanced performance across all image parts and classes. However, this can lead to erroneous predictions for classes with high intra-class variance (e.g., background). By leveraging pseudo-labels from the two spaces, we capitalize on the strengths of the learning in each space and enhance the knowledge exchange between the two spaces.

Table 4. Results on ADE20K and COCO-Stuff 10K with four different label rates and single data splits.
<table><tr><td></td><td colspan="4">ADE20K</td><td colspan="4">COCO-Stuff 10K</td></tr><tr><td>Method</td><td>1/16</td><td>1/8</td><td>1/4</td><td>1/2</td><td>1/16</td><td>1/8</td><td> $\overline { { 1 / 4 } }$ </td><td>1/2</td></tr><tr><td>Cutmix</td><td>28.91</td><td>31.89</td><td>34.62</td><td>39.15</td><td>23.09</td><td>26.14</td><td>29.88</td><td>30.91</td></tr><tr><td>Baseline</td><td>31.11</td><td>33.24</td><td>36.46</td><td>39.91</td><td>26.57</td><td>27.91</td><td>28.97</td><td>31.08</td></tr><tr><td>CSS (crs.)</td><td>32.01</td><td>33.92</td><td>37.84</td><td>39.97</td><td>27.06</td><td>28.85</td><td>30.59</td><td>31.73</td></tr><tr><td>CSS (mix)</td><td>32.55</td><td>34.21</td><td>37.01</td><td>40.85</td><td>27.46</td><td>29.51</td><td>29.98</td><td>31.91</td></tr></table>

Table 5. The quality of pseudo-labels from different pseudo-labeling strategies. The pseudo-labels are sampled by sampling strategies.
<table><tr><td>source</td><td>back</td><td>aero.</td><td>bicy.</td><td>bird</td><td>boat</td><td>bott.</td><td>bus</td><td>car</td><td>cat</td><td>chair</td><td>COw</td></tr><tr><td>lgt.</td><td>96.78</td><td>95.14</td><td>77.47</td><td>93.38</td><td>81.37</td><td>87.54</td><td>96.76</td><td>95.48</td><td>94.47</td><td>4.09</td><td>92.15</td></tr><tr><td>rep.</td><td>90.71</td><td>96.50</td><td>61.42</td><td>75.75</td><td>53.30</td><td>54.65</td><td>84.46</td><td>80.55</td><td>91.11</td><td>28.03</td><td>88.16</td></tr><tr><td>mix</td><td> $9 6 . 6 6 _ { \downarrow 0 . 1 2 }$ </td><td> $9 8 . 1 2 _ { \uparrow 2 . 9 8 }$ </td><td> $8 2 . 7 6 _ { \uparrow 5 . 2 9 }$ </td><td> $9 4 . 4 7 _ { \uparrow 1 . 0 9 }$ </td><td> $8 5 . 1 2 _ { \uparrow 3 . 7 5 }$ </td><td> $8 9 . 9 4 _ { \uparrow 2 . 4 0 }$ </td><td> $9 7 . 0 3 _ { \uparrow 0 . 2 7 }$ </td><td> $9 5 . 9 8 _ { \uparrow 0 . 5 0 }$ </td><td> $9 4 . 6 5 _ { \uparrow 0 . 1 8 }$ </td><td> $1 9 . 0 1 _ { \uparrow 1 4 . 9 2 }$ </td><td> $9 4 . 8 1 _ { \uparrow 2 . 6 6 }$ </td></tr><tr><td>source</td><td>tabel</td><td>dog</td><td>horse</td><td>motor</td><td>pers.</td><td>plant</td><td>sheep</td><td>sofa</td><td>train</td><td>tv</td><td>mIoU</td></tr><tr><td>lgt.</td><td>58.34</td><td>94.12</td><td>93.01</td><td>91.37</td><td>93.68</td><td>50.33</td><td>91.10</td><td>18.16</td><td>86.65</td><td>64.89</td><td>78.87</td></tr><tr><td>rep.</td><td>52.23</td><td>86.97</td><td>73.22</td><td>88.70</td><td>91.75</td><td>56.13</td><td>66.78</td><td>35.25</td><td>88.51</td><td>62.61</td><td>71.57</td></tr><tr><td>mix</td><td> $6 4 . 3 5 _ { \uparrow 6 . 0 1 }$ </td><td> $9 4 . 3 3 _ { \uparrow 0 . 2 1 }$ </td><td> $9 3 . 6 5 _ { \uparrow 0 . 6 4 }$ </td><td> $9 1 . 5 0 _ { \uparrow 0 . 1 3 }$ </td><td> $9 3 . 0 1 _ { \downarrow 0 . 6 7 }$ </td><td> $5 5 . 6 4 _ { \uparrow 5 . 3 1 }$ </td><td> $9 1 . 5 6 _ { \uparrow 0 . 4 6 }$ </td><td> $2 9 . 7 4 _ { \uparrow 1 1 . 5 8 }$ </td><td> $9 3 . 5 4 _ { \uparrow 6 . 8 9 }$ </td><td> $6 9 . 4 8 _ { \uparrow 4 . 5 9 }$ </td><td> $8 2 . 1 7 _ { \uparrow 3 . 3 3 }$ </td></tr></table>

Table 6. Results on pseudo-labels from different sources on two different label rates.
<table><tr><td>source</td><td>92 labels</td><td>183 labels</td></tr><tr><td>logit space</td><td>67.11</td><td>70.32</td></tr><tr><td>representation space</td><td>64.20</td><td>67.52</td></tr><tr><td>mix pseudo-labeling</td><td> $6 8 . 4 1 _ { \uparrow 1 . 3 0 }$ </td><td> $\overline { { 7 2 . 7 4 _ { \uparrow 2 . 4 2 } } }$ </td></tr><tr><td>cross pseudo-labeling</td><td> $6 7 . 8 5 _ { \uparrow 0 . 7 4 }$ </td><td> $7 1 . 9 8 _ { \uparrow 1 . 6 6 }$ </td></tr></table>

![](images/966dba0b8034fbe4fd7b36071bb97ffb6fe999c92d9c979f192759794dbb3d59.jpg)  
(b)

Results of different strategies To investigate the involvement of different pseudo-labeling strategies, we conduct the experiments as follows: 1) Using pseudo-labels from logit space. 2) Using pseudo-labels from representation space. 3) Using mix pseudo-labeling strategy. 4) Using cross pseudolabeling strategy. Tab. 6 shows the effectiveness of our proposed strategy in two different label rates. The results show that the performances of experiments with collaborative pseudo-labeling strategies are better than the ones whose pseudo-labels come from a single space with two different label rates, which proves the effectiveness of our proposed collaboration between the two spaces. It is worth noting that even though the quality of pseudo-labels from representation space is lower than that from logit space, the performance of the model is also boosted by using the cross pseudo-labeling strategy to maintain consistency between the predictions in two spaces. In addition, our method with the mix pseudo-labeling strategy outperforms that with the cross pseudo-labeling strategy.

## 5.2. Effectiveness of the Indicator

![](images/fb1a3a607bb3a9817a3612ce7f54b043b8a618dd0c6f064ca8133c4ff1616d0d.jpg)  
Figure 4. Relations between similarity and confidence.

Limitation of merely using confidence. To explain the limitation of merely using confidence for learning in the logit and representation spaces, we conduct experiments to show the relations between confidence and similarity. The similarity is the cosine similarity between representations and prototypes, which directly shows the confusion level between the representation and the prototype of its class.

Fig. 4 (a) shows the comparison between the confidence of each prediction and the corresponding similarity. We use the class person for demonstrating in our experiments. It shows clearly that even though fixing the confidence into a small range (from 0.8 to 0.81 in our settings), the similarity varies. Meanwhile, in Fig. 4 (b), different color bars stand for different intervals of confidence, and the lines denote the mean similarity between the prototype and each representation whose corresponding confidence is in the current interval. Fig. 4 (b) illustrates that the mean similarity of the class fluctuates when its interval of confidence rises. Both two figures imply that confidence is not able to represent the confusion level between representations and prototypes since there are no direct and close relations between confidence and similarity.

Fig. 5 visualizes the similarity and confidence of an image in both logit (lgt.) and representation (rep.) spaces, indicating the varying levels of confusion in the same region when learning in different spaces.

We also attribute it to the different concentrations of learning two spaces, $i . e .$ , the confusing region in one space can be more readily addressed in the other space.

Thus, it is inappropriate to choose confidence as the indicator to involve representation learning, e.g., sampling more hard samples with threshold and indicator. In contrast, our indicator directly employs similarity between the representation and prototype of its class, which directly reflects the confusion level in representation learning. It is more accurate to use similarity as the indicator to sample hard and critical samples in representation learning.

high  
![](images/2bac95bb9b57ec94d9818c7fcdec45b4d787c70891ea581fc5aff929a60296b4.jpg)

![](images/37a60684edf0d13b1af8b31c687cce9d605a3c55e80939e7b0bdd698ae3dbe11.jpg)

![](images/0f9d068fba5cc0f4ad7a4278f44b5e9c8c3705d1e34836433400ec20ef4e27e7.jpg)  
(a) Image  
(b) GT

low  
![](images/3a28bb2c667cb818b5a5cc599ee34c26f4a8f33c948e5c23ab4d545d23b0d543.jpg)  
(c) lgt. space  
(c) rep. space  
Figure 5. Visualization of the confusing part in different spaces.

Results of different indicators. Tab. 7 shows the impact of using different indicators. We conduct experiments on two different label rates (92 and 183) and three different indicators for pseudo-labels: only confidence (conf.), only similarity (smlr.), and confidence for learning logit space while similarity for learning representation space (mix). We use two different approaches to obtain pseudo-labels: from logit space only (seg label) and from mix pseudo-labeling strategy (mix label). It is clear that using both confidence and similarity to involve the learning in their own spaces obtains the best performance.

Table 7. Results on indicators from different spaces on two different label rates.
<table><tr><td colspan="2">92 labels</td><td colspan="2">183 labels</td></tr><tr><td>source</td><td>seg label mix label</td><td>seg label</td><td>mix label</td></tr><tr><td>conf.</td><td>67.11</td><td>68.33 70.32 66.89</td><td>71.92</td></tr><tr><td>smlr.</td><td>66.16</td><td>68.90</td><td>69.25</td></tr><tr><td>mix</td><td> $6 7 . 8 0 _ { \uparrow 0 . 6 9 }$ </td><td> $6 8 . 4 1 _ { \uparrow 1 . 3 0 }$   $7 1 . 5 0 _ { \uparrow 1 . 1 8 }$ </td><td> $7 2 . 7 4 _ { \uparrow 2 . 4 2 }$ </td></tr></table>

## 5.3. Ablation study of Components

In this section, we conduct experiments to introduce our components in CSS step by step, with results shown in Tab. 8. Our baseline is the conventional contrastive-based S4, achieving mIoU of 67.11% on 92 labels and 70.32% on 183 labels. Mix and cross means the pseudo-labels are from the mix pseudo-labeling strategy and cross pseudo-labeling strategy while the indicator is still the confidence in two spaces. Ind means we use the different indicators in different spaces while the pseudo-labels are from logit space. The last two rows represent our proposed two pseudo-labeling strategies with indicators from two spaces.

## 5.4. Qualitative Results

Fig. 6 shows the qualitative results of different methods on PASCAL VOC 2012 with 92 labeled images. Baseline means the conventional contrastive-based method. Compared with the original self-training methods (CutMix), thanks to introducing pixel-wise contrastive learning, the baseline, and our method perform better in some ambiguous regions. Furthermore, benefiting from the supervision of two spaces and different indicators in different spaces, our method outperforms the baseline.

Table 8. Ablation study on different components of our CSS.
<table><tr><td>component</td><td>92 labels</td><td>183 labels</td></tr><tr><td>baseline</td><td>67.11</td><td>70.32</td></tr><tr><td>mix</td><td> $6 8 . 3 3 _ { \uparrow 1 . 2 2 }$ </td><td> $\bar { 7 } 1 . 9 \bar { 2 } _ { \uparrow 1 . 6 0 } \bar $ </td></tr><tr><td>cross</td><td> $6 7 . 2 1 _ { \uparrow 0 . 1 0 }$ </td><td> $7 0 . 8 7 _ { \uparrow 0 . 5 5 }$ </td></tr><tr><td>ind</td><td> $6 7 . 8 0 _ { \uparrow 0 . 6 9 }$ </td><td> $\underline { { 7 1 . 5 0 } } _ { - } . \underline { { 1 . 1 8 } }$ </td></tr><tr><td> $\bar { \operatorname { m i x } } \bar { + } \bar { \operatorname { i n d } }$ </td><td> $6 8 . 4 1 _ { \uparrow 1 . 3 0 }$ </td><td> $7 2 . 7 4 _ { \uparrow 2 . 4 2 }$ </td></tr><tr><td> $\mathrm { c r o s s } + \mathrm { i n d }$ </td><td> $6 7 . 8 5 _ { \uparrow 0 . 7 4 }$ </td><td> $7 1 . 9 8 \ L _ { \uparrow 1 . 6 6 }$ </td></tr></table>

![](images/e4477793dd2b5bab94071455a24e50001a6c7c33c580043bb4a0b7a9ee4717d8.jpg)  
Figure 6. Visualization on PASCAL VOC 2012 with 92 labeled images. Yellow boxes highlight the main differences.

## 6. Conclusion

In this paper, we propose two collaborative pseudolabeling strategies to take full use of the semantic information in the representation space and enhance the knowledge exchange between the logit and representation spaces. Moreover, we employ a new indicator for the learning process in the representation space. Extensive experiments demonstrate that our pseudo-labeling strategies obtain more reliable supervision during unlabeled training and our indicator helps the model to concentrate on more critical parts during representation learning.

Future work: In this paper, we employ pseudo-labeling strategies to utilize the semantic information in both logit and representation spaces. In the future, we will investigate more powerful strategies to enhance the knowledge exchange between two spaces.

## References

[1] Inigo Alonso, Alberto Sabater, David Ferstl, Luis Monte-˜ sano, and Ana C. Murillo. Semi-supervised semantic segmentation with pixel-level contrastive learning from a classwise memory bank. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8219–8228, October 2021.

[2] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018.

[3] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural Information Processing Systems, 33:9912– 9924, 2020.

[4] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018.

[5] Mu Chen, Zhedong Zheng, Yi Yang, and Tat-Seng Chua. Pipa: Pixel-and patch-wise self-supervised learning for domain adaptative semantic segmentation. arXiv preprint arXiv:2211.07609, 2022.

[6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020.

[7] Xiaokang Chen, Yuhui Yuan, Gang Zeng, and Jingdong Wang. Semi-supervised semantic segmentation with cross pseudo supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2613–2622, 2021.

[8] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/ mmsegmentation, 2020.

[9] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[11] Nanqing Dong and Eric P Xing. Few-shot semantic segmentation with prototype learning. In BMVC, volume 3, 2018.

[12] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010.

[13] Jiashuo Fan, Bin Gao, Huan Jin, and Lihui Jiang. Ucc: Uncertainty guided cross-head co-training for semi-supervised semantic segmentation. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), pages 9947–9956, June 2022.

[14] Zhengyang Feng, Qianyu Zhou, Qiqi Gu, Xin Tan, Guangliang Cheng, Xuequan Lu, Jianping Shi, and Lizhuang Ma. Dmt: Dynamic mutual training for semi-supervised learning. Pattern Recognition, page 108777, 2022.

[15] Geoff French, Timo Aila, Samuli Laine, Michal Mackiewicz, and Graham Finlayson. Semi-supervised semantic segmentation needs strong, high-dimensional perturbations. In In 31th British Machine Vision Conference, 2020.

[16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications ofthe ACM, 63(11):139–144, 2020.

[17] Dayan Guan, Jiaxing Huang, Aoran Xiao, and Shijian Lu. Unbiased subclass regularization for semi-supervised semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9968–9978, 2022.

[18] Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev,´ Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In 2011 international conference on computer vision, pages 991–998. IEEE, 2011.

[19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[21] Ruifei He, Jihan Yang, and Xiaojuan Qi. Re-distributing biased pseudo labels for semi-supervised semantic segmentation: A baseline investigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6930–6940, October 2021.

[22] Judy Hoffman, Dequan Wang, Fisher Yu, and Trevor Darrell. Fcns in the wild: Pixel-level adversarial and constraint-based adaptation. arXiv preprint arXiv:1612.02649, 2016.

[23] Hanzhe Hu, Jinshi Cui, and Liwei Wang. Region-aware contrastive learning for semantic segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 16291–16301, 2021.

[24] Hanzhe Hu, Fangyun Wei, Han Hu, Qiwei Ye, Jinshi Cui, and Liwei Wang. Semi-supervised semantic segmentation via adaptive equalization learning. Advances in Neural Information Processing Systems, 34:22106–22118, 2021.

[25] Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, and Ming-Hsuan Yang. Adversarial learning for semi-supervised semantic segmentation. arXiv preprint arXiv:1802.07934, 2018.

[26] Zhengkai Jiang, Yuxi Li, Ceyuan Yang, Peng Gao, Yabiao Wang, Ying Tai, and Chengjie Wang. Prototypical contrast adaptation for domain adaptive semantic segmentation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXIV, pages 36–54. Springer, 2022.

[27] Tarun Kalluri, Girish Varma, Manmohan Chandraker, and CV Jawahar. Universal semi-supervised semantic segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5259–5270, 2019.

[28] Rihuan Ke, Angelica I Aviles-Rivero, Saurabh Pandey, Saikumar Reddy, and Carola-Bibiane Schonlieb. A three-¨ stage self-training framework for semi-supervised semantic segmentation. IEEE Transactions on Image Processing, 31:1805–1815, 2022.

[29] Donghyeon Kwon and Suha Kwak. Semi-supervised semantic segmentation with error localization network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9957–9967, 2022.

[30] Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, and Sanja Fidler. Semantic segmentation with generative models: Semi-supervised learning and strong out-of-domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8300– 8311, 2021.

[31] Jinlu Liu, Liang Song, and Yongqiang Qin. Prototype rectification for few-shot learning. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part I 16, pages 741–756. Springer, 2020.

[32] Shikun Liu, Shuaifeng Zhi, Edward Johns, and Andrew Davison. Bootstrapping semantic segmentation with regional contrast. In International Conference on Learning Representations, 2022.

[33] Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis, and Gustavo Carneiro. Perturbed and strict mean teachers for semi-supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4258–4267, 2022.

[34] Binjie Mao, Xinbang Zhang, Lingfeng Wang, Qian Zhang, Shiming Xiang, and Chunhong Pan. Learning from the target: Dual prototype network for few shot semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1953–1961, 2022.

[35] Sudhanshu Mittal, Maxim Tatarchenko, and Thomas Brox. Semi-supervised semantic segmentation with high-and lowlevel consistency. IEEE transactions on pattern analysis and machine intelligence, 43(4):1369–1379, 2019.

[36] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. IEEE transactions on pattern analysis and machine intelligence, 41(8):1979–1993, 2018.

[37] Viktor Olsson, Wilhelm Tranheden, Juliano Pinto, and Lennart Svensson. Classmix: Segmentation-based data augmentation for semi-supervised learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1369–1378, 2021.

[38] Yassine Ouali, Celine Hudelot, and Myriam Tami. Semisupervised semantic segmentation with cross-consistency training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[39] Yingwei Pan, Ting Yao, Yehao Li, Yu Wang, Chong-Wah Ngo, and Tao Mei. Transferrable prototypical networks for unsupervised domain adaptation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2239–2247, 2019.

[40] Jizong Peng, Guillermo Estrada, Marco Pedersoli, and Christian Desrosiers. Deep co-training for semi-supervised image segmentation. Pattern Recognition, 107:107269, 2020.

[41] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017.

[42] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33:596– 608, 2020.

[43] Korawat Tanwisuth, Xinjie Fan, Huangjie Zheng, Shujian Zhang, Hao Zhang, Bo Chen, and Mingyuan Zhou. A prototype-oriented framework for unsupervised domain adaptation. Advances in Neural Information Processing Systems, 34:17194–17208, 2021.

[44] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017.

[45] Wenguan Wang, Tianfei Zhou, Fisher Yu, Jifeng Dai, Ender Konukoglu, and Luc Van Gool. Exploring cross-image pixel contrast for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7303–7313, 2021.

[46] Yuchao Wang, Haochen Wang, Yujun Shen, Jingjing Fei, Wei Li, Guoqiang Jin, Liwei Wu, Rui Zhao, and Xinyi Le. Semi-supervised semantic segmentation using unreliable pseudo labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[47] Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan Yuille, and Fan Yang. Crest: A class-rebalancing self-training framework for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10857–10866, June 2021.

[48] Haoyu Xie, Changqi Wang, Mingkai Zheng, Minjing Dong, Shan You, and Chang Xu. Boosting semi-supervised semantic segmentation with probabilistic representations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2938–2946, 2023.

[49] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687– 10698, 2020.

[50] Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16684–16693, 2021.

[51] Hai-Ming Xu, Lingqiao Liu, Qiuchen Bian, and Zhen Yang. Semi-supervised semantic segmentation with prototypebased consistency regularization. Advances in Neural Information Processing Systems, 2022.

[52] Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, and Yang Gao. St++: Make self-training work better for semi-supervised semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4268–4277, 2022.

[53] Jianlong Yuan, Yifan Liu, Chunhua Shen, Zhibin Wang, and Hao Li. A simple baseline for semi-supervised semantic segmentation with strong data augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8229–8238, 2021.

[54] Xiangyu Yue, Zangwei Zheng, Shanghang Zhang, Yang Gao, Trevor Darrell, Kurt Keutzer, and Alberto Sangiovanni Vincentelli. Prototypical cross-domain self-supervised learning for few-shot unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13834–13844, 2021.

[55] Xiangyun Zhao, Raviteja Vemulapalli, Philip Andrew Mansfield, Boqing Gong, Bradley Green, Lior Shapira, and Ying Wu. Contrastive learning for label efficient semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10623–10633, October 2021.

[56] Xu Zheng, Yunhao Luo, Hao Wang, Chong Fu, and Lin Wang. Transformer-cnn cohort: Semi-supervised semantic segmentation by the best of both students, 2022.

[57] Xu Zheng, Jinjing Zhu, Yexin Liu, Zidong Cao, Chong Fu, and Lin Wang. Both style and distortion matter: Dualpath unsupervised domain adaptation for panoramic semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1285–1295, June 2023.

[58] Yuanyi Zhong, Bodi Yuan, Hong Wu, Zhiqiang Yuan, Jian Peng, and Yu-Xiong Wang. Pixel contrastive-consistent semi-supervised semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7273–7282, 2021.

[59] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.

[60] Tianfei Zhou, Wenguan Wang, Ender Konukoglu, and Luc Van Gool. Rethinking semantic segmentation: A prototype view. In CVPR, 2022.

[61] Yanning Zhou, Hang Xu, Wei Zhang, Bin Gao, and Pheng-Ann Heng. C3-semiseg: Contrastive semi-supervised segmentation via cross-set learning and dynamic classbalancing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7036–7045, 2021.