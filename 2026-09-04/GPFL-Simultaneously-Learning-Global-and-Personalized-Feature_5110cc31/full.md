# GPFL: Simultaneously Learning Global and Personalized Feature Information for Personalized Federated Learning

Jianqing Zhang<sup>1</sup>, Yang Hua<sup>2</sup>, Hao Wang<sup>3</sup>, Tao Song<sup>1</sup> Zhengui Xue<sup>1</sup>, Ruhui Ma<sup>1\*</sup>, Jian Cao<sup>1</sup>, Haibing Guan<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Queen’s University Belfast <sup>3</sup>Louisiana State University

{tsingz, songt333, zhenguixue, ruhuima, cao-jian, hbguan}@sjtu.edu.cn Y.Hua@qub.ac.uk, haowang@lsu.edu

## Abstract

Federated Learning (FL) is popular for its privacypreserving and collaborative learning capabilities. Recently, personalized FL (pFL) has received attentionfor its ability to address statistical heterogeneity and achieve personalization in FL. However, from the perspective of feature extraction, most existing pFL methods only focus on extracting global or personalized feature information during local training, whichfails to meet the collaborative learning and personal ization goals of pFL. To address this, we propose a new pFL method, named GPFL, to simultaneously learn global and personalizedfeature information on each client. We conduct extensive experiments on six datasets in three statistically heterogeneous settings and show the superiority of GPFL over ten state-of-the-art methods regarding effectiveness, scalability,fairness, stability, and privacy. Besides, GPFL mitigates overfitting and outperforms the baselines by up to 8.99% in accuracy.

## 1. Introduction

To maximize the value of data generated on massive clients while protecting privacy, Federated Learning (FL), an iterative machine learning scheme, comes along with various applications [25, 21, 38, 35, 34]. Traditional FL methods focus on collaborative learning and obtaining a reasonable global model. However, in practice, one single global model cannot meet the requirements of every client and performs poorly due to statistical heterogeneity [24, 1, 50].

Recently, personalized FL (pFL) has attracted increasing attention in addressing statistical heterogeneity and achieving personalization in FL [50, 49, 63, 62]. From the view of each client, it joins FL for additional server information (e.g., global model parameters) to enhance its model and address the data shortage problem. To obtain high-quality server information, each client also has to provide locally learned information for server aggregation. Thus, an ideal pFL is one kind of FL with two goals: (1) aggregating information for collaborative learning and (2) training reasonable personalized models. On the other hand, since every client is connected to the external environment and shares certain common information, the data present on each client comprises both global and personalized feature information.

However, from a feature extraction perspective, existing pFL methods only focus on one of these two goals on clients. For collaborative learning, FedRoD [8] trains the feature extractor to extract global feature information for its global objective, but it does not extract personalized feature information for personalized tasks. For personalization, FedPer [3] and FedRep [12] only use local data to train the model for the personalized objective, losing some global information during local training [10], which is not beneficial for collaborative learning. Although FedPHP [32]/ FedProto [51] utilizes global features/prototypes to guide personalized feature extraction, the quality of global fea tures/prototypes depends on the quality of feature extractors, which is paradoxical. Poor global features/prototypes mislead feature extraction in turn.

To simultaneously learn global and personalized feature information on each client, we propose a novel pFL framework, named GPFL. Inspired by the category anchors that introduce extra common information in domain adaptation [65], we learn the global feature information with the guidance of global category embeddings using the Global Category Embedding layer (GCE). Besides, we learn personalized feature information through personalized tasks. However, learning two contrary (global vs. personalized) objectives is confusing, so we devise and insert the Conditional Valve (CoV) after the feature extractor to create a global guidance route and a personalized task route in the client model. With CoV, we learn global and personalized feature information separately at the same time, unlike FedRoD, FedPer, and FedRep, which only learn one kind of feature information. Besides, GPFL leverages trainable category embeddings to guide feature extraction at both the magnitude and angle levels, unlike FedPHP and FedProto, which rely on the well-trained feature extractor. Furthermore, the global category embeddings in GPFL introduce extra global information besides local data, which can mitigate the overfitting of personalized models and enhance fairness and privacy-preserving ability.

To evaluate GPFL regarding effectiveness, scalability, fairness, stability, and privacy, we compare GPFL with ten state-of-the-art (SOTA) methods on six datasets in Computer Vision (CV), Natural Language Processing (NLP), and Internet ofThings (IoT) domains. Besides, we consider the label skew [40, 36, 27],feature shift [31], and real world [66, 15] settings to simulate different kinds of statistical heterogeneity in FL. Experimental results show that GPFL outperforms these baselines by up to 8.99% in accuracy. We provide the code in the supplementary materials. Overall, our key contributions are

• We emphasize the importance of achieving both collaborative learning and individualized goals in pFL and propose a pFL method GPFL that simultaneously learns the global and personalized feature information.

• We learn the global feature information through trainable category embeddings, and the additional global information in GCE mitigates the overfitting of the personalized model to local data.

• We conduct extensive experiments in the CV, NLP, and IoT domains under label skew, feature shift, and real world settings. The results show that our GPFL outperforms the SOTA method in terms of effectiveness, scalability, fairness, stability, and privacy.

## 2. Related Work & Background

## 2.1. Personalized Federated Learning

Meta-learning & fine-tuning. Per-FedAvg [13] and Fed-Meta [7] are similar methods that learn a global model with the aggregated model update trend to achieve good performance on each client with a few steps of local fine-tuning. However, the aggregated trend cannot meet the model update trends of every client.

Personalized heads. FedPer [3], FedRep [12], and FedRoD [8] split the given backbone into a feature extractor and a head. FedPer and FedRep are similar methods that only share the feature extractor between the server and clients. Different from them, each client in FedRoD owns a feature extractor and two heads. FedRoD trains the feature extractor as well as the shared head for its global objective (with the balanced softmax (BSM) loss [46]) and trains the personalized head for its personalized objective. However, it does not derive gradients, w.r.t. the feature extractor from the personalized objective, ignoring personalized feature extraction. Besides, the BSM loss is ineffective infeature shift settings. Regularization. Different from FedProx [30], which regularizes the difference between local model parameters and frozen global parameters, pFedMe [49]/Ditto [29] uses a proximal term for the additional personalized models. The personalized model in Ditto benefits from the global parameter guidance. However, they ignore the global and personalized feature information extraction.

Feature extraction guidance. FedPHP aligns the features outputted by the personalized feature extractor and the global feature extractor for each sample, and FedProto aligns the feature vectors to their corresponding category prototypes. However, FedProto only guides a feature vector to be close to its corresponding prototype rather than guiding it to stay away from other prototypes, which results in the intersection of classification boundaries (see Sec. 4.2). Besides, FedProto generates prototypes based on the learned feature vectors, so the prototypes in FedProto can be uninformative without the well-trained feature extractor. This problem exacerbates when large backbones are trained from scratch in FL because they have difficulty learning good feature extractors in early iterations.

## 2.2. Conditional Computation

Usually, the structures of most DNNs are static during training and inference. With conditional computing tech niques [6, 16, 18, 43], such as dynamic routing [18, 37, 57], a DNN can have a dynamic structure when given different conditional inputs. For example, using an auxiliary policy network, SpotTune [18] dynamically chooses which and how many blocks in a pre-trained residual network should be fine-tuned according to the input images. During inference, BlockDrop [57] executes specific layers of a residual network according to the decisions made by reinforcement learning. D<sup>2</sup>NN [37] proposes a kind of DNN that allows selective execution through controller modules.

These methods are designed in central learning scenarios for specific tasks. Inspired by them, we propose a CoV to create global and personalized routes in the client model for global and personalized feature information extraction.

## 3. GPFL

## 3.1. Problem Statement

We have a total of N clients and client $i ( i \in [ N ] )$ generates its private data $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ (labelled by $y _ { i } )$ via a distinct distribution $\mathcal { D } _ { i } .$ . For the personalized task, we denote the personalized objective on client i as

$$
\mathscr { F } _ { i } : = \mathbb { E } _ { ( \pmb { x } _ { i } , \pmb { y } _ { i } ) \sim \mathscr { D } _ { i } } \mathscr { L } _ { i } ( \pmb { x } _ { i } , \pmb { y } _ { i } ; W _ { i } ) ,\tag{1}
$$

where $\mathcal { L } _ { i }$ is the personalized loss function, and $W _ { i }$ is the parameters of all modules on client i. For all clients, our objective is

$$
\{ W _ { 1 } , \ldots , W _ { N } \} = \arg \operatorname* { m i n } \mathcal { G } ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N } ) ,\tag{2}
$$

where typically $\begin{array} { r } { \mathcal G ( \mathcal F _ { 1 } , \dots , \mathcal F _ { N } ) \ = \ \sum _ { i \in [ N ] } n _ { i } \mathcal F _ { i } , \ n _ { i } \ = \ } \end{array}$ $\frac { \left| \mathcal { D } _ { i } \right| } { \sum _ { j \in \left[ N \right] } \left| \mathcal { D } _ { j } \right| }$ measures the importance of client i and $| \mathcal { D } _ { i } |$ is the number of training samples on client i. With Eq. (1), Eq. (2) considers collaborative learning and personalization.

## 3.2. Method

![](images/6315412bfd349c77318f6361038582b975a2b62546659da91c14b4c635a3d4fc.jpg)  
Figure 1. Illustration of client modules and data flow between them. Client i shares $W ^ { f e }$ , V , C, and $\hat { C }$ while keeping $\boldsymbol { W } _ { i } ^ { h }$ locally. Global category embeddings in GCE<sup>[</sup> are frozen before local training. We simultaneously train $\phi ,$ CoV, GCE, and ψ in an end-to-end manner on the client. For training, we activate both the global guidance route (gray arrows) and the personalized task route (black arrows). For inference and evaluation, only the personalized task route is activated.

Overview. Focusing on the extracted features, we follow FedRep and split the backbone into a feature extractor (ϕ) and a head (ψ), where ψ is the last fully connected (FC) layers in the backbone, and ϕ represents the remaining layers. In Figure 1, to achieve the collaborative learning and personalization goals in pFL, we use CoV to transform the original feature vector $f _ { i }$ to two feature vectors $\pmb { f } _ { i } ^ { G }$ and $\mathbf { \Delta } f _ { i } ^ { P }$ for global and personalized feature extraction, respectively. Then we learn the global feature information with the guidance of global category embeddings in GCE and learn personalized feature information through personalized tasks. Using the embedding technique [53, 64], we can get any trainable category embedding u through the look up operation: $\pmb { u } = \mathrm { G C E } ( u ; C )$ , where u is a category ID. In each iteration, we share global parameters $W ^ { f e } , \bar { V }$ , and C among clients and obtain C<sup>ˆ</sup> by copying C after receiving C. We denote the trainable parameters as $W _ { i } : = \{ W ^ { f e } , \bar { V } , C , W _ { i } ^ { h } \}$ Feature extraction. We firstly obtain $f _ { i }$ through ϕ that maps the data samples to a lower feature space of dimension $K \colon \boldsymbol { \phi } : \mathbb { R } ^ { D }  \bar { \mathbb { R } ^ { K } }$ , where typically $D \gg K$ . Formally, $\forall ( \pmb { x } _ { i } , y _ { i } ) \sim \mathcal { D } _ { i } , \pmb { f } _ { i } = \phi ( \pmb { x } _ { i } ; W ^ { f e } )$

Feature transformation. Inspired by the conditional computation techniques [18, 54, 43], we transform $f _ { i }$ to $\pmb { f } _ { i } ^ { G }$ and $\mathbf { \Delta } f _ { i } ^ { P }$ through the affine mapping [5, 54, 43]:

$$
\begin{array} { r l } & { \pmb { f } _ { i } ^ { G } = \sigma [ ( \gamma + \pmb { 1 } ) \odot \pmb { f } _ { i } + \pmb { \beta } ] , } \\ & { \pmb { f } _ { i } ^ { P } = \sigma [ ( \gamma _ { i } + \pmb { 1 } ) \odot \pmb { f } _ { i } + \pmb { \beta } _ { i } ] , } \end{array}\tag{3}
$$

where 1 has the same shape as $f _ { i }$ with all the values equal to 1 and σ is the ReLU activation function [33]. ⊙ is the Hadamard product. $\gamma , \beta , \gamma _ { i }$ , and $\beta _ { i }$ are generated by CoV:

$$
\begin{array} { r } { \{ \gamma , \beta \} = \mathrm { C o V } ( \pmb { f } _ { i } , \pmb { g } ; V ) , } \\ { \{ \gamma _ { i } , \beta _ { i } \} = \mathrm { C o V } ( \pmb { f } _ { i } , \pmb { p } _ { i } ; V ) , } \end{array}\tag{4}
$$

where $\pmb { g } \in \mathbb { R } ^ { K }$ and $\pmb { p } _ { i } \in \mathbb { R } ^ { K }$ are the personalized and global conditional input (described later), respectively. $\mathrm { C o V }$ consists of two sub-modules $\mathrm { C o V } _ { \gamma }$ and $\mathrm { C o V } _ { \beta }$ with identical structures but different parameters. Concretely, $\mathrm { C o V } _ { \gamma }$ generates $\gamma / \gamma _ { i }$ by inputting ${ \pmb g } / p _ { i }$ sequentially to an FC layer, a ReLU activation, and a layer-normalization layer [4]. $\mathrm { C o V } _ { \beta }$ generates $\beta / \beta _ { i }$ in a similar way.

Generating g and $\mathbf { \nabla } p _ { i }$ . g is identical among clients, while $\pmb { p } _ { i }$ contains local data distribution information. Since all the clients share the same C<sup>ˆ</sup> during local training, we can generate $\mathbf { \pmb { g } }$ by averaging all the frozen category embeddings (with dimension $K )$

$$
\mathbf { \sigma } \mathbf { g } = \frac { \sum _ { u \in [ U ] } \widehat { \mathbf { G C E } } ( u ; \hat { C } ) } { U } .\tag{5}
$$

For $\pmb { p } _ { i } .$ , we first obtain data distribution information through the statistics on client i. Specifically, we generate the proportion coefficient of category u by

$$
\begin{array} { r } { \alpha _ { i } ^ { u } = \mathbb { E } _ { ( \mathbf { x } _ { i } , y _ { i } ) \sim \mathcal { D } _ { i } } \mathbb { I } \{ y _ { i } = u \} , u \in [ U ] , } \end{array}\tag{6}
$$

where $\mathbb { I } \{ \cdot \}$ is the indicator function, and U is the number of all the categories among clients. Then, we have

$$
p _ { i } = \frac { \sum _ { \boldsymbol { u } \in [ U ] } \widehat { \mathrm { G C E } } ( \boldsymbol { u } ; \hat { C } ) \cdot \alpha _ { i } ^ { u } } { U } .\tag{7}
$$

Angle-level global guidance. In addition to sharing model parameters between server and clients, sharing other information, such as prototype [51] and logits [61], has shown effectiveness in the literature. Thus, we propose to share the global category embeddings across clients. Inspired by the contrastive loss [28, 9, 48], we guide each feature vector to be close to its corresponding category embedding while staying away from other category embeddings, which spreads out the category embeddings during training. Formally, we have the angle-level guidance loss

$$
\mathcal { L } _ { i } ^ { a l g } = - \log \frac { \exp { ( \sin ( f _ { i } ^ { G } , \mathrm { G C E } ( y _ { i } ; C ) ) ) } } { \sum _ { u \in [ U ] } \exp { ( \sin ( f _ { i } ^ { G } , \mathrm { G C E } ( u ; C ) ) ) } } ,\tag{8}
$$

where sim $\begin{array} { r } { ( \pmb { f } , \pmb { v } ) = \frac { \pmb { f } ^ { T } \pmb { v } } { | | \pmb { f } | | _ { 2 } | | \pmb { v } | | _ { 2 } } } \end{array}$ is a cosine similarity function. Due to the L2-norm $| | \cdot | | _ { 2 } ,$ , sim(·) only measures the similarity between $\pmb { f } _ { i } ^ { G }$ and $\mathrm { G C E } ( u ; C ) , u \in [ U ]$ at the angle level ignoring their magnitude. Note that, all the embeddings in GCE are updated through Eq. (8).

Magnitude-level global guidance. Inspired by the proximal term that keeps the local model parameters close to the frozen global parameters [30], we keep $\pmb { f } _ { i } ^ { G }$ close to its corresponding frozen global embedding. Formally, we have the magnitude-level guidance loss

$$
\mathcal { L } _ { i } ^ { m l g } = | | \pmb { f } _ { i } ^ { G } , \widehat { \mathrm { G C E } } ( y _ { i } ; \hat { C } ) | | _ { 2 } .\tag{9}
$$

Personalized tasks. With $\mathbf { \Delta } f _ { i } ^ { P }$ , client i learns a head $\psi$ that maps from the transformed feature space to the category space: $\overline { { \psi } } : \mathbb { R } ^ { K }  \mathbb { R } ^ { U }$ . Formally, we have

$$
\mathcal { L } _ { i } ^ { P } = \ell ( \psi ( \pmb { f } _ { i } ^ { P } ; W _ { i } ^ { h } ) , y _ { i } ) ,\tag{10}
$$

where ℓ is the Cross Entropy (CE) loss function [68].

Local loss function $\mathcal { L } _ { i }$ . Combining the loss mentioned above together, we have

$$
\mathcal { L } _ { i } = \mathcal { L } _ { i } ^ { P } + \mathcal { L } _ { i } ^ { a l g } + \lambda \mathcal { L } _ { i } ^ { m l g } + \mu \vert \vert V \vert \vert _ { 2 } + \mu \vert \vert C \vert \vert _ { 2 } ,\tag{11}
$$

where $\lambda$ and $\mu$ are hyperparameters. The entire learning process is shown in Algorithm 1.

## 3.3. Privacy Analysis

Following FedCG [56], we consider a semi-honest scenario where the server follows the FL protocol but may recover original data from a victim client i with its model updates through Deep Leakage from Gradients (DLG) [69]. Given model (or feature extractor) Φ and model updates $\Delta ,$ DLG learns dummy input x˜ and dummy logits (or feature vectors) output z˜ by minimizing

$$
\mathcal { L } ^ { d l g } = | | \nabla \tilde { \ell } ( \Phi ( \tilde { \pmb { x } } ) , \tilde { \pmb { z } } ) - \Delta | | ^ { 2 } ,\tag{12}
$$

where we use Mean Squared Error (MSE) loss [59] for ${ \tilde { \ell } } .$ Then the server can obtain the recovered input $\tilde { \pmb { x } } ^ { * }$

In GPFL, $\boldsymbol { W } _ { i } ^ { h }$ and $\alpha _ { i } ^ { u } , u \in [ U ]$ are not shared outside the client, protecting most of the private information. Without $\alpha _ { i } ^ { u }$ , the CoV module can only perform transformation for the global route, like a regular layer in a backbone. The server can treat the combination of $\phi$ and CoV as a pseudo feature extractor $\tilde { \phi } : = \phi \circ \mathrm { C o V }$ and obtain $\tilde { f } ^ { G } : = \tilde { \phi } ( \tilde { \pmb { x } } ; W _ { i } ^ { f e , t } , V _ { i } ^ { t } )$ Compared to FedPer and FedRep which only share the feature extractor, either $\phi$ or $\tilde { \phi }$ learns more global information. Besides, the server can utilize GCE to calculate the logit output for category u by

$$
\begin{array} { r } { \mathrm { l o g i t } _ { u } = \sin ( \tilde { { \mathbf { f } } } ^ { G } , \mathrm { G C E } ( u ; C _ { i } ^ { t } ) ) , u \in [ U ] . } \end{array}\tag{13}
$$

In other words, the server can use GCE by integrating it with ϕ<sup>˜</sup> as a pseudo model like the uploaded client model in FedAvg and Ditto, but with more global information. According to previous work [41], the model with more global information has a better privacy-preserving ability. We show the experimental results in Sec. 4.7.

Algorithm 1 The Learning Process in GPFL   
Input: N clients with their local data; initial parameters   
$W ^ { f e , 0 } , \ : V ^ { 0 } , \ : C ^ { 0 } ; \eta \colon$ local learning rate; λ and $\mu { \vdots }$ hy  
perparameters; $\rho \colon$ client joining ratio; $T \colon$ total training   
iterations.   
Output: Personalized model parameters $\{ W _ { 1 } , \ldots , W _ { N } \}$   
1: Client $i , \forall i \in [ N ]$ initializes its $\psi$ and obtains $\boldsymbol { W } _ { i } ^ { h , 0 }$   
2: for iteration $t = 0 , \ldots , T$ do   
3: Server samples a client subset ${ \mathcal { T } } ^ { t }$ based on $\rho .$   
4: Server sends $\{ W ^ { f e , t } , V ^ { t } , C ^ { t } \}$ to $\mathcal { T } ^ { t } .$   
5: for Client $i \in \mathcal { T } ^ { t }$ in parallel do   
$\triangleright$ local initialization   
6: Initialize $\phi , \mathrm { C o V } ,$ , GCE with $\{ W ^ { f e , t } , V ^ { t } , C ^ { t } \}$   
7: Initialize GCE<sup>[</sup> with $C ^ { t } .$   
8: Generate $\mathbf { \pmb { g } }$ and $\pmb { p } _ { i }$ by Eq. (5) and Eq. (7).   
\triangleright local training   
9: Update $W ^ { f e , t } , V ^ { t } , C ^ { t } , W _ { i } ^ { h , t }$ simultaneously:   
10: $\boldsymbol { W } _ { i } ^ { f e , t } \gets \boldsymbol { W } ^ { f e , t } - \eta \nabla _ { \boldsymbol { W } ^ { f e , t } } \mathcal { F } _ { i } ;$   
11: $V _ { i } ^ { i } \gets V ^ { t } - \eta \nabla _ { V ^ { t } } \mathcal { F } _ { i } ;$   
12: $\dot { C _ { i } ^ { t } } \gets C ^ { t } - \eta \nabla _ { C ^ { t } } \mathcal { F } _ { i } ;$   
13: $\dot { W _ { i } ^ { h , t + 1 } }  W _ { i } ^ { h , t } - \eta \nabla _ { W _ { i } ^ { h , t } } \mathcal { F } _ { i } .$   
14: Upload $\{ W _ { i } ^ { f e , t } , V _ { i } ^ { t } , C _ { i } ^ { t } \}$ to the server.   
\triangleright Server aggregation   
15: Server calculates $n ^ { t } = \textstyle \sum _ { i \in \pmb { \mathscr { L } } ^ { t } }$ n<sub>i</sub> and obtains   
16: $\begin{array} { r } { W ^ { f e , t + 1 } = \sum _ { i \in \mathcal { T } ^ { t } } \frac { n _ { i } } { n ^ { t } } \bar { W } _ { i } ^ { f e , t } ; } \end{array}$   
17: $\begin{array} { r } { V ^ { t + 1 } = \sum _ { i \in \mathcal { T } ^ { t } } \frac { - \tilde { n } _ { i } } { n ^ { t } } \bar { V } _ { i } ^ { t } ; } \end{array}$   
18: $\begin{array} { r } { C ^ { t + 1 } = \overline { { \sum } } _ { i \in \mathbb { Z } ^ { t } } ^ { \cdots } \frac { \tilde { n } _ { i } } { n ^ { t } } \dot { C } _ { i } ^ { t } . } \end{array}$   
19: return $\{ W _ { 1 } , \ldots , W _ { N } \}$

## 4. Performance Comparison

We evaluate the performance of GPFL in terms of the learned features, effectiveness, scalability, fairness, stability, and privacy. Specifically, we compare GPFL with ten SOTA methods, including FedAvg [40], FedProx [30], Per-FedAvg [13], pFedMe [49], Ditto [29], FedPer[3], FedRep [12], FedRoD [8], FedPHP [32], and FedProto [51], on CV, NLP, and IoT tasks.

## 4.1. Setup

Datasets. For CV tasks, we use three public datasets: Fashion-MNIST (FMNIST) [58], Cifar100 [26], and Tiny-ImageNet [11]. For NLP tasks, we use two public datasets: AG News [67] and Amazon Review [14]. For the IoT task, we use a Human Activity Recognition (HAR) dataset [2]. Backbones. Following work [40, 39, 17], we use a 4- layer CNN on FMNIST, Cifar100, and Tiny-ImageNet. To evaluate GPFL on a backbone larger than the 4-layer CNN, we also run ResNet-18 [19] on Tiny-ImageNet. On AG News and Amazon Review, we use the fastText [23] and the 3-layer MLP [44] as the backbones, respectively. Following previous work [60], we use a HAR-CNN on HAR to process the sensor signal. As for the local learning rate η, we set $\eta = 0 . 0 0 5$ for 4-layer CNN and 3-layer MLP, set $\eta = 0 .$ 1 for ResNet-18 and fastText, and set $\eta = 0 . 0 1$ for HAR-CNN.

Statistically heterogeneous settings. With the above six datasets, we create three popular statistically heterogeneous settings to simulate the FL environment: label skew [40, 36, 27], feature shift [31], and real world [66, 15] settings. Specifically, we have two label skew settings: the pathological setting [40, 47] and practical setting [36, 28]. For the pathological label skew setting, we sample data with label amount 2/10/20 for each client on FMNIST/Cifar100/Tiny-ImageNet from a total of 10/100/200 categories, with disjoint data and different numbers of data samples. For the practical label skew setting, we sample data from FMNIST, Cifar100, Tiny-ImageNet, and AG News through the Dirichlet distribution [36], denoted by $D i r ( \beta )$ Concretely, we sample $q _ { c , i } \sim D i r ( \beta ) \ : ( \beta = 0 . 1 / \beta = 1 $ by default for CV/NLP tasks [55]) and allocate a ${ q _ { c , i } }$ proportion of the samples with label c to client i. For the feature shift setting, following existing methods [14, 31], we create four clients, each containing data from one domain on Amazon Review. For the real world setting, the sensor signal data on HAR is naturally collected and stored on 30 clients with six activities.

Implementation Details. Following pFedMe and FedRoD, unless explicitly specified, we have 20 clients with a client joining ratio $\rho = 1$ . On each client, we consider 75% data as the training dataset and use the remaining 25% data for evaluation. Following pFedMe, we report the best performance of the global model for traditional FL and the best average performance across personalized models for pFL. By default, we set the batch size to 10 and the number of local epochs to 1. We run 2000 iterations with three trials for all the methods on each task and report the mean and standard deviation. For more details and experimental results $( e . g .$ , the results in the feature shift setting with different statistical heterogeneity, i.e., different β), please refer to supplementary materials.

## 4.2. Learned Features

Here, for easy visualization, we experiment on FMNIST in the pathological label skew setting using ten clients and keep other settings constant. As shown in Figure 2, we use t-SNE [52] to visualize the feature vectors extracted by FedPer, FedProto, and our GPFL.

The classification boundary is not discriminative in Fed-Per and the boundaries in FedProto intersect, e.g., the boundaries of category 0 and 1 in Figure 2c, while Figure 2d has no intersection. By simultaneously considering global guidance and personalized tasks, GPFL can learn discriminative features and distinguish the data distribution of each client, revealing excellent personalization performance of GPFL in feature extraction.

![](images/0fd5e22ae848b308e04de6f828369188d3a1d027abdd13eed46873a7df4ef24c.jpg)

![](images/79213874092b767764eb27f0fb32ecf395180c82b73326c64f75dea452590f78.jpg)  
(b) FedPer

(a) Data distribution  
![](images/73efcb0e6dd34820219c280627c9acc9bb66d704a47198c5bb281ca799db1eca.jpg)  
(c) FedProto

![](images/5088c284f20bc931c385960b1c3c9c68318c7c3931c2bcab15492c9df0854a32.jpg)  
(d) GPFL  
Figure 2. (a): Data distribution on each client; the size of red circles means the number of samples. (b), (c), and (d): t-SNE visualizations of feature vectors on FMNIST with 10 clients; “cid” means client ID. Best viewed in color.

## 4.3. Effectiveness

Then, we compare our GPFL with baselines under the label skew, feature shift, and real world settings. Due to the limited space, we use “TINY” and ${ } ^ { 6 6 } \mathrm { T I N Y ^ { * 3 } }$ to represent using the 4-layer CNN and ResNet-18 (trained from scratch) on Tiny-ImageNet, respectively.

Label skew settings. We show the results regarding the label skew settings in Table 1. GPFL achieves superior performance in both pathological and practical settings. Concretely, in the practical setting on Cifar100, GPFL outperforms the best baseline Ditto by 8.99% with only 0.31% standard deviation. FedRep performs well on Tiny-ImageNet in the pathological setting, but GPFL outperforms it by 6.10% in the practical setting. Since there are many embeddings of tokens in NLP tasks, the feature guidance per category in GPFL and FedProto is beneficial for embedding learning in the label skew setting. GPFL achieves 1.63% improvement over FedProto and outperforms other baselines by 5.72% on AG News.

Next, we point out why GPFL outperforms other baselines with the experimental results. (1) GPFL v.s. Fed-Per & FedRep & FedRoD: On each client, the feature extractor trained in FedPer and FedRep only learns personalized feature information, while the feature extractor in FedRoD only learns the global one. Unlike them, GPFL locally learns both the global and personalized feature information, so GPFL outperforms FedPer/FedRep/FedRoD by 12.23%/9.47%/10.92% on Cifar100 in the practical setting. (2) GPFL v.s. FedProto & FedPHP: FedPHP/FedProto guides feature extraction with global features/prototypes throughout the FL process, but the large backbone ResNet-18 cannot learn to extract features well in early iterations. Then, poor global features and prototypes mislead local training, so FedPHP and FedProto achieve low accuracy on the large backbone ResNet-18. Besides, due to the classifica tion boundary intersection, FedProto performs worse on the dataset with more categories (e.g., Tiny-ImageNet with 200 categories). By sharing the feature extractor and guiding the feature extraction with the trainable global embeddings, GPFL outperforms FedProto by 17.32% on TINY\*.

Table 1. The test accuracy (%) on the CV and NLP tasks in label skew settings.
<table><tr><td>Settings</td><td colspan="3">Pathological Label Skew Setting</td><td colspan="5">Practical Label Skew Setting</td></tr><tr><td></td><td>FMNIST</td><td>Cifar100</td><td>TINY</td><td>FMNIST</td><td>Cifar100</td><td>TINY</td><td>TINY*</td><td>AG News</td></tr><tr><td>FedAvg</td><td>80.41±0.08</td><td>25.98±0.13</td><td>14.20±0.47</td><td>85.85±0.19</td><td>31.89±0.47</td><td>19.46±0.20</td><td>19.45±0.13</td><td>87.12±0.19</td></tr><tr><td>FedProx</td><td>78.08±0.15</td><td>25.94±0.16</td><td>13.85±0.25</td><td>85.63±0.57</td><td>31.99±0.41</td><td>19.37±0.22</td><td>19.27±0.23</td><td>87.21±0.13</td></tr><tr><td>Per-FedAvg</td><td>99.18±0.08</td><td>56.80±0.26</td><td>28.06±0.40</td><td>95.10±0.10</td><td>44.28±0.33</td><td>25.07±0.07</td><td>21.81±0.54</td><td>87.08±0.26</td></tr><tr><td>pFedMe</td><td>99.35±0.14</td><td>58.20±0.14</td><td>27.71±0.40</td><td>97.25±0.17</td><td>47.34±0.46</td><td>26.93±0.19</td><td>33.44±0.33</td><td>87.08±0.18</td></tr><tr><td>Ditto</td><td>99.44±0.06</td><td>67.23±0.07</td><td>39.90±0.42</td><td>97.47±0.04</td><td>52.87±0.64</td><td>32.15±0.04</td><td>35.92±0.43</td><td>91.89±0.17</td></tr><tr><td>FedPer</td><td>99.47±0.03</td><td>63.53±0.21</td><td>39.80±0.39</td><td>97.44±0.06</td><td>49.63±0.54</td><td>33.84±0.34</td><td>38.45±0.85</td><td>91.85±0.24</td></tr><tr><td>FedRep</td><td>99.56±0.03</td><td>67.56±0.31</td><td>40.85±0.37</td><td>97.56±0.04</td><td>52.39±0.35</td><td>37.27±0.20</td><td>39.95±0.61</td><td>92.25±0.20</td></tr><tr><td>FedRoD</td><td>99.52±0.05</td><td>62.30±0.02</td><td>37.95±0.22</td><td>97.52±0.04</td><td>50.94±0.11</td><td>36.43±0.05</td><td>37.99±0.26</td><td>92.16±0.12</td></tr><tr><td>FedPHP</td><td>99.30±0.13</td><td>63.09±0.04</td><td>37.06±0.57</td><td>97.38±0.16</td><td>50.52±0.16</td><td>35.69±3.26</td><td>29.90±0.51</td><td>90.52±0.19</td></tr><tr><td>FedProto</td><td>99.49±0.04</td><td>69.18±0.03</td><td>36.78±0.07</td><td>97.40±0.02</td><td>52.70±0.33</td><td>31.21±0.16</td><td>26.38±0.40</td><td>96.34±0.58</td></tr><tr><td>GPFL</td><td>99.85±0.08</td><td>71.78±0.26</td><td>44.58±0.06</td><td>97.81±0.09</td><td>61.86±0.31</td><td>43.37±0.53</td><td>43.70±0.44</td><td>97.97±0.14</td></tr></table>

![](images/c9ccfb14d482f55d5498bfba6e1a3e617ed47b7b7190f9287c8d457b4de0efac.jpg)

(a) Test accuracy curves in the feature shift setting.  
![](images/2a4914340563d768f45b9b48f3124c3fda3b7d7d21f7e79c2d154052485f46a0.jpg)  
(b) Training loss curves in the feature shift setting.  
Figure 3. Curves (smoothed) on Amazon Review dataset.

Feature shift setting. Using the Amazon Review dataset, each client contains the data from one domain with the personalized task of classifying the samples into positive or negative emotions. In other words, the data on every client belong to two categories, which is different from the label skew settings. In Figure 3a, we find that the traditional FL methods FedAvg and FedProx achieve good performance with a little gap compared to the pFL methods. It means that traditional FL methods suffer more in the label skew setting than in thefeature shift setting. Besides, FedAvg can maintain its performance after reaching the best accuracy. In contrast, the accuracy curves of several pFL methods, including Ditto, pFedMe, FedRep, FedProto, and Per-FedAvg, exhibit a drop in accuracy after reaching the peak accuracy, despite their training having converged (Figure 3b), which means overfitting. FedProto and Per-FedAvg achieve poor performance with a significant drop. On the contrary, GPFL performs the best and maintains its performance when converged, as the global information in GCE mitigates the overfitting of personalized models. FedRoD does not show superiority here, as its BSM loss is only designed for the label skew settings to tackle statistical heterogeneity.

Real world setting. From Table 2, we find that the regularization-based pFL methods (pFedMe and Ditto) perform well on HAR and the regularization-based traditional FL method FedProx outperforms FedAvg by 1.14%. However, most other pFL methods, e.g., Per-FedAvg, FedPer, FedRep, FedPHP, and FedProto perform worse than traditional FL methods (FedAvg and FedProx). FedAvg outperforms Per-FedAvg and FedPer by 10.08% and 11.62%, respectively. The GCE stores the global embedding of each activity, which guides the feature extractor to extract the global characteristic of one kind of human activity. Meanwhile, the personalized task can also guide the feature extractor to extract user-specific characteristics. Thus, our GPFL outperforms all baselines by up to 2.19% in this scenario.

## 4.4. Scalability

Based on MOON [28], we split the Cifar100 dataset into 20/30/50/100/500 sub-datasets to form 20/30/50/100/500 clients, respectively, in the practical label skew settings (β =

Table 2. The test accuracy (%) on the IoT task regarding effectiveness and the CV task regarding scalability.
<table><tr><td>Tasks</td><td>Effectiveness</td><td colspan="6">Scalability</td></tr><tr><td>Clients</td><td> $N = 3 0$ </td><td> $N = 3 0$ </td><td> $N = 5 0$ </td><td> $N = 1 0 0$ </td><td> $N = 5 0 0$ </td><td>一  $N = 1 0 | 5 0$ </td><td> $N = 3 0 | 5 0$ </td></tr><tr><td>FedAvg</td><td> $8 7 . 2 0 { \pm } 0 . 2 7 $ </td><td> $3 1 . 1 5 { \pm } 0 . 0 5$ </td><td> $3 1 . 9 0 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $3 1 . 9 5 { \pm } 0 . 3 7 $ </td><td> $2 9 . 5 1 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $2 5 . 2 8 { \pm } 0 . 3 2$ </td><td> $2 9 . 0 4 { \scriptstyle \pm 0 . 2 1 }$ </td></tr><tr><td>FedProx</td><td> $8 8 . 3 4 { \pm } 0 . 2 4 $ </td><td> $3 1 . 2 1 { \pm } 0 . 0 8$ </td><td> $3 1 . 9 4 { \pm } 0 . 3 0$ </td><td> $3 1 . 9 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $2 9 . 8 4 { \pm } 0 . 8 1 $ </td><td> $2 5 . 6 5 { \pm } 0 . 3 4$ </td><td> $2 9 . 0 4 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td>Per-FedAvg</td><td> $7 7 . 1 2 { \pm } 0 . 1 7 $ </td><td> $4 1 . 5 7 { \pm } 0 . 2 1 $ </td><td> $4 4 . 3 1 { \pm } 0 . 2 0 $ </td><td> $3 6 . 0 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td>1</td><td> $4 0 . 2 0 { \pm } 0 . 2 1 $ </td><td> $4 2 . 9 6 { \pm } 0 . 4 2$ </td></tr><tr><td>pFedMe</td><td> $9 1 . 5 7 { \pm } 0 . 1 2 $ </td><td> $4 7 . 0 4 { \pm } 0 . 2 8 $ </td><td> $4 8 . 3 6 { \pm } 0 . 6 4$ </td><td> $4 6 . 4 5 { \pm } 0 . 1 8$ </td><td> $3 1 . 3 0 { \pm } 0 . 8 9$ </td><td> $4 0 . 2 7 { \pm } 0 . 5 4$ </td><td> $4 2 . 1 9 { \pm } 0 . 3 8$ </td></tr><tr><td>Ditto</td><td> $9 1 . 5 3 { \pm } 0 . 0 9$ </td><td> $5 2 . 5 3 { \pm } 0 . 4 2 $ </td><td> $5 4 . 2 2 { \pm } 0 . 0 4 $ </td><td> $5 2 . 8 9 { \pm } 0 . 2 2$ </td><td> $3 0 . 2 4 { \pm } 0 . 7 2 $ </td><td> $4 8 . 2 3 { \pm } 0 . 3 5 $ </td><td> $5 0 . 9 8 { \pm } 0 . 2 9 $ </td></tr><tr><td>FedPer</td><td> $7 5 . 5 8 { \pm } 0 . 1 3 $ </td><td> $4 4 . 9 8 { \pm } 0 . 2 0 $ </td><td> $4 4 . 2 2 { \pm } 0 . 1 8$ </td><td> $4 0 . 3 7 { \pm } 0 . 4 1$ </td><td> $3 0 . 5 6 { \pm } 0 . 5 9$ </td><td> $4 3 . 6 4 { \pm } 0 . 4 2$ </td><td> $4 3 . 5 4 { \pm } 0 . 4 3$ </td></tr><tr><td>FedRep</td><td> $8 0 . 4 4 { \pm } 0 . 4 2 $ </td><td> $5 0 . 2 4 { \pm } 0 . 0 1$ </td><td> $4 7 . 4 1 { \pm } 0 . 1 8$ </td><td> $4 4 . 6 1 { \pm } 0 . 2 0 $ </td><td> $3 1 . 9 2 { \pm } 0 . 7 1 $ </td><td> $4 6 . 8 5 { \pm } 0 . 1 2$ </td><td> $4 7 . 6 3 { \pm } 0 . 2 6 $ </td></tr><tr><td>FedRoD</td><td> $8 9 . 9 1 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $5 0 . 1 1 { \pm } 0 . 0 3$ </td><td> $4 9 . 3 8 { \pm } 0 . 0 1$ </td><td> $4 6 . 6 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $3 4 . 6 1 { \pm } 0 . 9 8 $ </td><td> $4 6 . 3 2 { \pm } 0 . 0 2$ </td><td> $4 9 . 1 5 { \pm } 0 . 1 2$ </td></tr><tr><td>FedPHP</td><td> $8 7 . 9 4 { \pm } 0 . 5 4 $ </td><td> $4 9 . 2 8 { \pm } 0 . 0 6$ </td><td> $5 2 . 4 4 { \pm } 0 . 1 6$ </td><td> $4 9 . 7 0 { \pm } 0 . 3 1$ </td><td> $3 0 . 2 6 { \pm } 0 . 8 4$ </td><td> $4 5 . 7 1 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $4 8 . 6 5 { \pm } 0 . 2 4 $ </td></tr><tr><td>FedProto</td><td> $8 4 . 7 3 { \pm } 0 . 3 2 $ </td><td> $5 2 . 3 2 { \pm } 0 . 1 8$ </td><td> $5 0 . 2 9 { \pm } 0 . 1 8 $ </td><td> $4 7 . 1 1 { \pm } 0 . 1 5 $ </td><td> $3 4 . 9 1 { \pm } 0 . 8 3 $ </td><td> $4 9 . 2 3 { \pm } 0 . 2 8$ </td><td> $5 0 . 4 9 { \pm } 0 . 3 2 $ </td></tr><tr><td>GPFL</td><td> $\mathbf { 9 3 . 7 6 { \pm } 0 . 1 6 }$ </td><td> $\mathbf { 6 0 . 9 6 { \scriptstyle \pm 0 . 6 5 } }$ </td><td> $\mathbf { 6 0 . 9 8 { \scriptstyle \pm 0 . 3 2 } }$ </td><td> ${ \bf 5 7 . 1 1 \pm 0 . 2 1 }$ </td><td> $\mathbf { 3 7 . 2 8 { \pm } 0 . 6 3 }$ </td><td> $\pm \mathbf { 5 } 3 . 2 \mathbf { 6 } \pm \mathbf { 0 . 3 9 }$ </td><td> ${ \bf 5 9 . 1 8 } \pm { \bf 0 . 5 3 }$ </td></tr></table>

0.1). We have shown the results for 20 clients in Table 1, so we only show the results for 30/50/100/500 clients in Table 2. The meta-learning in Per-FedAvg requires at least two batches of data, which is invalid on some clients in our unbalanced settings when $N = 5 0 0$ . Since the total sample amount is constant on Cifar100, the average sample amount per client decreases when N increases and it is unreasonable to compare results between different N.

In Table 2, GPFL still outperforms other baselines with various N. When $N = 5 0 0 .$ , it is hard to achieve personalization without enough client data for pFL methods. Many pFL baselines, $e . g . ,$ , Ditto, FedPer, and FedPHP perform similarly with FedAvg in this situation. Since GCE provides extra information and relieves the data shortage issue on each client, GPFL outperforms all baselines.

In practice, clients join the FL process with their data, and more clients means more total data in FL. To simulate a more realistic scenario, we randomly select 10/30/50 clients from the 50 clients generated above to form a setting with a total of 10/30/50 clients, denoted by $N = 1 0 | 5 0 / N = 3 0 | 5 0$ $/ \ N = 5 0 .$ . In these settings, as shown in Table 2, all the methods benefit from the client amount increase, and GPFL still outperforms other baselines, showing the scalability of GPFL in practice.

## 4.5. Fairness

According to the work [45], personalization may result in poorer performance on some devices despite improving the average. It is essential to improve both performance and fairness when designing a pFL method. Here, following Ditto [29], we evaluate the fairness of all the methods through the standard deviation of the local accuracy on clients when reaching the best-averaged accuracy $( i . e . ,$ the test accuracy mentioned above), as shown in Table 3. To weaken the effect of the test accuracy magnitude for a more fair comparison, we also follow the work [22] to use the coefficient of variation metric for fairness evaluation.

Table 3. The fairness, i.e., standard deviation (%, ↓) [the coefficient of variation $( \times 1 0 ^ { - 2 } , \downarrow ) ]$ of the local accuracy on clients when reaching the best test accuracy on the CV, NLP, and IoT tasks in pathological label skew (paLS), practical label skew (prLS, $\beta =$ 0.1),feature shift, and real world settings.
<table><tr><td>Settings</td><td>paLS</td><td>prLS</td><td>Feature Shift</td><td>Real World</td></tr><tr><td>Clients</td><td> $N = 2 0$ </td><td> $N = 1 0 0$ </td><td> $N = 4$ </td><td> $N = 3 0$ </td></tr><tr><td>Datasets</td><td>TINY</td><td>Cifar100</td><td>Amazon Review</td><td>HAR</td></tr><tr><td>FedAvg</td><td>3.57 [25.14]</td><td>7.06 [22.10]</td><td>1.62 [1.84]</td><td>17.10 [19.61]</td></tr><tr><td>FedProx</td><td>3.51 [25.34]</td><td>7.08 [22.15]</td><td>1.60 [1.81]</td><td>17.35 [19.64]</td></tr><tr><td>Per-FedAvg</td><td>3.27 [11.65]</td><td>8.13 [22.54]</td><td>2.82 [3.29]</td><td>14.15 [18.35]</td></tr><tr><td>pFedMe</td><td>3.36 [12.12]</td><td>8.19 [17.63]</td><td>1.99 [2.25]</td><td>12.65 [13.81]</td></tr><tr><td>Ditto</td><td>3.84 [9.62]</td><td>9.89 [18.70]</td><td>2.12 [2.40]</td><td>13.20 [14.42]</td></tr><tr><td>FedPer</td><td>3.39 [8.51]</td><td>8.91 [22.07]</td><td>2.18 [2.47]</td><td>19.49 [25.79]</td></tr><tr><td>FedRep</td><td>3.53 [8.64]</td><td>8.99 [20.15]</td><td>2.15 [2.43]</td><td>21.16 [26.30]</td></tr><tr><td>FedRoD</td><td>3.46 [9.12]</td><td>8.87 [19.01]</td><td>2.24 [2.54]</td><td>16.93 [18.83]</td></tr><tr><td>FedPHP</td><td>3.81 [10.28]</td><td>9.45 [19.01]</td><td>1.96 [2.22]</td><td>13.81 [15.70]</td></tr><tr><td>FedProto</td><td>4.13 [11.23]</td><td>9.98 [21.18]</td><td>1.82 [2.08]</td><td>11.77 [13.89]</td></tr><tr><td>GPFL</td><td>3.21 [7.20]</td><td>8.05 [14.10]</td><td>1.62 [1.80]</td><td>8.42 [8.98]</td></tr></table>

In Table 3, our GPFL outperforms other pFL methods, especially in the real world setting and the settings with many clients, because sharing global information among clients promotes fairness. By learning both the global and personalized feature information, GPFL achieves a higher accuracy with lower discrimination compared to FedPer, FedRep, and FedRoD, which focus on learning only one kind of feature information during local training. The traditional FL methods achieve a low standard deviation but a high coefficient of variation, as their personalization performance is poor. Clients in FedProto only share prototypes but keep the model parameters secret, which limits the capacity of global information and leads to low fairness.

## 4.6. Stability

Table 4. The test accuracy (%) of the pFL methods on Cifar100 with N = 50, $\beta = 0 . 1$ , and $\rho \leq 1$
<table><tr><td></td><td> $\rho = 1$ </td><td> $\rho \in [ 0 . 5 , 1 ]$ </td><td> $\rho \in [ 0 . 1 , 1 ]$ </td></tr><tr><td>Per-FedAvg</td><td> $4 4 . 3 1 { \pm } 0 . 2 0 $ </td><td> $4 3 . 6 6 { \pm } 1 . 3 8 $ </td><td> $4 3 . 6 3 { \pm } 1 . 0 7$ </td></tr><tr><td>pFedMe</td><td> $4 8 . 3 6 { \pm } 0 . 6 4$ </td><td> $4 3 . 2 8 { \pm } 0 . 8 5 $ </td><td> $4 1 . 7 1 { \pm } 1 . 0 2 $ </td></tr><tr><td>Ditto</td><td> $5 0 . 5 9 { \pm } 0 . 2 2$ </td><td> $4 9 . 7 8 { \pm } 0 . 3 6$ </td><td> $4 8 . 3 3 { \pm } 3 . 2 7 $ </td></tr><tr><td>FedPer</td><td> $4 4 . 2 2 { \pm } 0 . 1 8$ </td><td> $4 4 . 1 2 { \pm } 0 . 2 1 $ </td><td> $4 4 . 0 7 { \scriptstyle \pm 0 . 2 7 }$ </td></tr><tr><td>FedRep</td><td> $4 7 . 4 1 { \pm } 0 . 1 8$ </td><td> $4 6 . 9 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $4 6 . 6 1 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td>FedRoD</td><td> $4 9 . 3 8 { \pm } 0 . 0 1$ </td><td> $4 9 . 0 7 { \pm } 0 . 4 3 $ </td><td> $4 7 . 8 0 { \pm } 1 . 3 5 $ </td></tr><tr><td>FedPHP</td><td> $5 0 . 2 3 { \pm } 0 . 1 2 $ </td><td> $4 5 . 1 9 { \pm } 0 . 0 7$ </td><td> $4 4 . 4 3 { \pm } 0 . 1 2 $ </td></tr><tr><td>FedProto</td><td> $5 0 . 2 9 { \pm } 0 . 1 8 $ </td><td> $4 9 . 4 5 { \pm } 0 . 2 1 $ </td><td> $4 6 . 0 5 { \pm } 4 . 0 3$ </td></tr><tr><td>GPFL</td><td> $\mathbf { 6 0 . 9 8 { \scriptstyle \pm 0 . 3 2 } }$ </td><td> $\mathbf { 6 0 . 6 0 { \dot { \pm } } 0 . 5 1 }$ </td><td> ${ \bf 6 0 . 0 4 } \pm { \bf 0 . 2 8 }$ </td></tr></table>

In real world scenarios, some clients cannot join the whole FL process because of low battery, lack of computation resources, unstable network, etc. Here, we simulate this scenario by varying the client joining ratio $\rho$ in every iteration on the Cifar100 dataset. Specifically, we uniformly sample a value for $\rho$ within the given range in each iteration.

In Table 4, our GPFL still maintains its superiority among the pFL methods, while some baselines perform worse with a more extensive range of $\rho .$ For example, pFedMe and FedPHP drop 6.65% and 5.80% from $\rho = 1 \mathrm { t o } \rho \in [ 0 . 1 , 1 ] .$ respectively. Furthermore, Per-FedAvg, pFedMe, Ditto, FedRoD, and FedProto achieve erratic performance (large standard deviations) in dynamic settings.

## 4.7. Privacy

Table 5. PSNR (dB, ↓) values for privacy evaluation on Cifar100 in label skew setting. “Fed” is omitted for method names.
<table><tr><td>Per-fe</td><td>Rep-fe</td><td>Avg-fe</td><td>Avg</td><td>RoD-fe</td><td>RoD</td><td> $\mathbf { G P F L - } f e$ </td><td> $\mathbf { G P F L - } s f e$ </td><td>GPFL-sm</td></tr><tr><td>7.94</td><td>7.73</td><td>6.92</td><td>7.30</td><td>6.82</td><td>7.48</td><td>6.56</td><td>6.41</td><td>6.71</td></tr></table>

Following FedCG, we evaluate the privacy-preserving ability of GPFL with representative baselines in Peak Signalto-Noise Ratio (PSNR) [20]. For FedPer, FedRep, FedAvg, and FedRoD, we conduct DLG attack for feature extractor (with suffix $^ { 6 6 } - f e ^ { 7 } )$ . For FedAvg and FedRoD, we also conduct DLG attacks for the entire model. For GPFL, we conduct DLG attack for feature extractor, pseudo feature extractor (with suffix $^ { 6 6 } { - } s f e ^ { , 9 } )$ , and pseudo model (with suffix $^ { 6 6 } \mathrm { - } s m ^ { \prime \prime } )$

In Table 5, all the PSNR values that correspond to GPFL are smaller than other baselines, which supports the analysis in Sec. 3.3. Since the parameters of the last FC layer in a backbone also contain class information [42], the entire model is more susceptible to privacy leakage than the feature <sub>extractor in FedAvg and FedRoD. However, due to the global</sub>w.o. conditional valve characteristic of GCE, GPFL-sm still maintains the privacypreserving ability.

## 5. Ablation Study

![](images/e7831dd568d726a20a09b23e6d16bfb9a3af1bed5b65ea5423011204be483c3c.jpg)  
Figure 4. Illustration of variants for ablation study.

To further show the effectiveness of each proposed module, we remove them from GPFL and create five variants $( ^ { 6 6 } w / o ^ { 7 }$ is short for “without”): (a) w/o personalized conditional input (PCI), (b) w/o CoV, (c) w/o $\overline { { \mathcal { L } _ { i } ^ { m l g } } }$ , (d) w/o GCE (GCE <sup>[</sup> also disappears), and (e) w/o CoV & GCE (FedPer). As shown in Figure 4, we input the constant vector a and b instead of the dynamic $\mathbf { \nabla } _ { \mathbf { p } _ { i } }$ and g to remove PCI. We report the test accuracy of GPFL and its five variants in Table 6.

Table 6. The accuracy (%) of GPFL and its variants on TINY\*.
<table><tr><td>GPFL</td><td>w/o PCI</td><td>w/o CoV</td><td>w/o  $\mathcal { L } _ { i } ^ { m l g }$ </td><td>w/o GCE</td><td>FedPer</td></tr><tr><td>43.70</td><td>42.74</td><td>40.23</td><td>41.72</td><td>39.48</td><td>38.45</td></tr></table>

We analyze each module according to Table 6. (a) Like $^ { g , }$ a and b are identical across clients, so GPFL inputs local data distribution information for CoV while w/o PCI does not. With this personalized information, GPFL performs better than w/o PCI. Even with the identical $^ { a / b , }$ w/o PCI still performs well thanks to end-to-end training. (b) Removing CoV causes a 3.47% accuracy decrease, as guiding one feature vector to learn both global and personalized information simultaneously is confusing. (c) The accuracy gap between GPFL and w/o $\mathcal { L } _ { i } ^ { m l g }$ shows the effectiveness of the magnitude-level global guidance, since w/o $\mathcal { L } _ { i } ^ { m l g }$ only removes the $\mathcal { L } _ { i } ^ { m l g }$ objective. (d) The accuracy decreases further when we remove the angle-level global guidance. Without learning global information during local training, the accuracy drops by 4.22%. As the trainable affine mapping adaptively adjusts the original features, w/o GCE still slightly outperforms FedPer.

## 6. Conclusion

For the collaborative learning and personalization goals of pFL, we propose GPFL to simultaneously learn global and personalized information on the client. We show the superiority of GPFL through extensive experiments regarding effectiveness, scalability, fairness, stability, and privacy. GPFL outperforms SOTA pFL methods by a large margin. Besides, we show the effectiveness of each proposed module.

## Acknowledgements

This work was supported in part by the Shanghai Key Laboratory of Scalable Computing and Systems, National Key R&D Program of China (2022YFB4402102), Inter net of Things special subject program, China Institute of IoT (Wuxi), Wuxi IoT Innovation Promotion Center (2022SP-T13-C), Industry-university-research Cooperation Funding Project from the Eighth Research Institute in China Aerospace Science and Technology Corporation (Shanghai) (USCAST2022-17), and Intel Corporation (UFunding 12679). This work was also partially supported by the Program of Technology Innovation of the Science and Technology Commission of Shanghai Municipality (Granted No. 21511104700 and 22DZ1100103).

## References

[1] Zareen Alamgir, Farwa K Khan, and Saira Karim. Federated Recommenders: Methods, Challenges and Future. Cluster Computing, 25(6):4075–4096, 2022. 1

[2] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L Reyes-Ortiz. Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine. In Ambient Assisted Living and Home Care: 4th International Workshop, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings 4, pages 216–223. Springer, 2012. 4

[3] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated Learning with Personalization Layers. arXiv preprint arXiv:1912.00818, 2019. 1, 2, 4

[4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer Normalization. arXiv preprint arXiv:1607.06450, 2016. 3

[5] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation Learning: A Review and New Perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8):1798–1828, 2013. 3

[6] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating Or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv preprint arXiv:1308.3432, 2013. 2

[7] Fei Chen, Mi Luo, Zhenhua Dong, Zhenguo Li, and Xiuqiang He. Federated Meta-Learning With Fast Convergence and Efficient Communication. arXiv preprint arXiv:1802.07876, 2018. 2

[8] Hong-You Chen and Wei-Lun Chao. On Bridging Generic and Personalized Federated Learning for Image Classification. In ICLR, 2021. 1, 2, 4

[9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. In ICML, 2020. 3

[10] Yiqiang Chen, Wang Lu, Xin Qin, Jindong Wang, and Xing Xie. MetaFed: Federated Learning Among Federations With Cyclic Knowledge Distillation for Personalized Healthcare. arXiv preprint arXiv:2206.08516, 2022. 1

[11] Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A Downsampled Variant of Imagenet as an Alternative to the Cifar Datasets. arXiv preprint arXiv:1707.08819, 2017. 4

[12] Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai. Exploiting Shared Representations for Personalized Federated Learning. In ICML, 2021. 1, 2, 4

[13] Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach. In NeurIPS, 2020. 2, 4

[14] Haozhe Feng, Zhaoyang You, Minghao Chen, Tianye Zhang, Minfeng Zhu, Fei Wu, Chao Wu, and Wei Chen. KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation. In ICML, 2021. 4, 5

[15] Yansong Gao, Minki Kim, Sharif Abuadbba, Yeonjae Kim, Chandra Thapa, Kyuyeon Kim, Seyit A Camtep, Hyoungshick Kim, and Surya Nepal. End-to-end evaluation of federated learning and split learning for internet of things. In 2020 International Symposium on Reliable Distributed Systems (SRDS), pages 91–100. IEEE, 2020. 2, 5

[16] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional Neural Processes. In ICML, 2018. 2

[17] Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, and Michael Moeller. Inverting Gradients-How Easy Is It to Break Privacy in Federated Learning? In NeurIPS, 2020. 4

[18] Yunhui Guo, Honghui Shi, Abhishek Kumar, Kristen Grauman, Tajana Rosing, and Rogerio Feris. Spottune: Transfer Learning through Adaptive Fine-Tuning. In CVPR, 2019. 2, 3

[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, 2016. 4

[20] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In ICPR, 2010. 8

[21] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Federated Visual Classification With Real-World Data Distribution. In ECCV, 2020. 1

[22] Rajendra K Jain, Dah-Ming W Chiu, William R Hawe, et al. A Quantitative Measure of Fairness and Discrimination. Eastern Research Laboratory, Digital Equipment Corporation, Hudson, MA, 21, 1984. 7

[23] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of Tricks for Efficient Text Classification. In EACL, 2017. 4

[24] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and Open Problems in Federated Learning. arXiv preprint arXiv:1912.04977, 2019. 1

[25] Georgios A Kaissis, Marcus R Makowski, Daniel Rückert, and Rickmer F Braren. Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging. Nature Machine Intelligence, 2(6):305–311, 2020. 1

[26] Alex Krizhevsky and Hinton Geoffrey. Learning Multiple Layers of Features From Tiny Images. Technical Report, 2009. 4

[27] Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Fed erated Learning on Non-IID Data Silos: An Experimental Study. In ICDE, 2022. 2, 5

[28] Qinbin Li, Bingsheng He, and Dawn Song. Model-Contrastive Federated Learning. In CVPR, 2021. 3, 5, 6

[29] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and Robust Federated Learning Through Personalization. In ICML, 2021. 2, 4, 7

[30] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated Optimization in Heterogeneous Networks. In MLSys, 2020. 2, 4

[31] Xiaoxiao Li, Meirui JIANG, Xiaofei Zhang, Michael Kamp, and Qi Dou. FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. In ICLR, 2021. 2, 5

[32] Xin-Chun Li, De-Chuan Zhan, Yunfeng Shao, Bingshuai Li, and Shaoming Song. FedPHP: Federated Personalization with Inherited Private Models. In ECML PKDD, 2021. 1, 4

[33] Yuanzhi Li and Yang Yuan. Convergence Analysis of Two-Layer Neural Networks with Relu Activation. In NeurIPS, 2017. 3

[34] Zexi Li, Qunwei Li, Yi Zhou, Wenliang Zhong, Guannan Zhang, and Chao Wu. Edge-cloud collaborative learning with federated and centralized features. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023. 1

[35] Zexi Li, Tao Lin, Xinyi Shang, and Chao Wu. Revisiting weighted aggregation in federated learning with neural networks. In International Conference on Machine Learning, 2023. 1

[36] Tao Lin, Lingjing Kong, Sebastian U Stich, and Martin Jaggi. Ensemble Distillation for Robust Model Fusion in Federated Learning. In NeurIPS, 2020. 2, 5

[37] Lanlan Liu and Jia Deng. Dynamic Deep Neural Networks: Optimizing Accuracy-Efficiency Trade-Offs by Selective Execution. In AAAI, 2018. 2

[38] Yang Liu, Anbu Huang, Yun Luo, He Huang, Youzhi Liu, Yuanyuan Chen, Lican Feng, Tianjian Chen, Han Yu, and Qiang Yang. Fedvision: An Online Visual Object Detection Platform Powered By Federated Learning. In AAAI, 2020. 1

[39] Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, and Jiashi Feng. No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID data. In NeurIPS, 2021. 4

[40] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-

Efficient Learning of Deep Networks from Decentralized Data. In AISTATS, 2017. 2, 4, 5

[41] Milad Nasr, Reza Shokri, and Amir Houmansadr. Compre hensive privacy analysis of deep learning. In Proceedings of the 2019 IEEE Symposium on Security and Privacy (SP), pages 1–15, 2018. 4

[42] Yifan Niu and Weihong Deng. Federated Learning for Face Recognition With Gradient Correction. In AAAI, 2022. 8

[43] Boris Oreshkin, Pau Rodríguez López, and Alexandre Lacoste. Tadam: Task Dependent Adaptive Metric for Improved Few-Shot Learning. In NeurIPS, 2018. 2, 3

[44] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment Matching for Multi-Source Domain Adaptation. In ICCV, 2019. 4

[45] Krishna Pillutla, Kshitiz Malik, Abdel-Rahman Mohamed, Mike Rabbat, Maziar Sanjabi, and Lin Xiao. Federated Learning with Partial Model Personalization. In ICML, 2022. 7

[46] Jiawei Ren, Cunjun Yu, Xiao Ma, Haiyu Zhao, Shuai Yi, et al. Balanced Meta-Softmax for Long-Tailed Visual Recognition. In NeurIPS, 2020. 2

[47] Aviv Shamsian, Aviv Navon, Ethan Fetaya, and Gal Chechik. Personalized Federated Learning using Hypernetworks. In ICML, 2021. 5

[48] Kihyuk Sohn. Improved Deep Metric Learning With Multi-Class N-Pair Loss Objective. In NeurIPS, 2016. 3

[49] Canh T Dinh, Nguyen Tran, and Tuan Dung Nguyen. Personalized Federated Learning with Moreau Envelopes. In NeurIPS, 2020. 1, 2, 4

[50] Alysa Ziying Tan, Han Yu, Lizhen Cui, and Qiang Yang. Towards Personalized Federated Learning. IEEE Transactions on Neural Networks and Learning Systems, 2022. 1

[51] Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, and Chengqi Zhang. Fedproto: Federated Prototype Learning across Heterogeneous Clients. In AAAI, 2022. 1, 3, 4

[52] Laurens Van der Maaten and Geoffrey Hinton. Visualizing Data Using T-SNE. Journal ofMachine Learning Research, 9(11), 2008. 5

[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All You Need. In NeurIPS, 2017. 3

[54] Chi Wang, Yang Hua, Zheng Lu, Jian Gao, and Neil Robertson. Temporal Meta-Adaptor for Video Object Detection. In BMVC, 2021. 3

[55] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor. Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. In NeurIPS, 2020. 5

[56] Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Lixin Fan, Rong Pan, and Qiang Yang. Fedcg: Leverage condi tional gan for protecting privacy and maintaining competitive performance in federated learning. In NeurIPS, 2022. 4

[57] Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S Davis, Kristen Grauman, and Rogerio Feris. Blockdrop: Dynamic Inference Paths in Residual Networks. In CVPR, 2018. 2

[58] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. 4

[59] Donggeun Yoo and In So Kweon. Learning loss for active learning. In CVPR, 2019. 4

[60] Ming Zeng, Le T Nguyen, Bo Yu, Ole J Mengshoel, Jiang Zhu, Pang Wu, and Joy Zhang. Convolutional neural networks for human activity recognition using mobile sensors. In 6th international conference on mobile computing, applications and services, pages 197–205. IEEE, 2014. 5

[61] Jie Zhang, Song Guo, Xiaosong Ma, Haozhao Wang, Wenchao Xu, and Feijie Wu. Parameterized Knowledge Transfer for Personalized Federated Learning. In NeurIPS, 2021. 3

[62] Jianqing Zhang, Yang Hua, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. FedALA: Adaptive Local Aggregation for Personalized Federated Learning. In AAAI, 2023. 1

[63] Jianqing Zhang, Yang Hua, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Fedcp: Separating feature information for personalized federated learning via conditional policy. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023. 1

[64] Jianqing Zhang, Dongjing Wang, and Dongjin Yu. TL-SAN: Time-aware Long-and Short-term Attention Network for Next-item Recommendation. Neurocomputing, 441:179– 191, 2021. 3

[65] Qiming Zhang, Jing Zhang, Wei Liu, and Dacheng Tao. Category anchor-guided unsupervised domain adaptation for semantic segmentation. In NeurIPS, 2019. 1

[66] Tuo Zhang, Lei Gao, Chaoyang He, Mi Zhang, Bhaskar Krishnamachari, and A Salman Avestimehr. Federated learning for the internet of things: applications, challenges, and opportunities. IEEE Internet ofThings Magazine, 5(1):24–29, 2022. 2, 5

[67] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-Level Convolutional Networks for Text Classification. In NeurIPS, 2015. 4

[68] Zhilu Zhang and Mert Sabuncu. Generalized Cross Entropy Loss for Training Deep Neural Networks With Noisy Labels. In NeurIPS, 2018. 4

[69] Ligeng Zhu, Zhijian Liu, and Song Han. Deep Leakage from Gradients. In NeurIPS, 2019. 4