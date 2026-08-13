# Scene-Aware Feature Matching

Xiaoyong Lu, Yaping Yan, Tong Wei, Songlin Du\* Southeast University, Nanjing, China {luxiaoyong, yan, weit, sdu}@seu.edu.cn

## Abstract

Current feature matching methods focus on point-level matching, pursuing better representation learning of individual features, but lacking further understanding of the scene. This results in significant performance degradation when handling challenging scenes such as scenes with large viewpoint and illumination changes. To tackle this problem, we propose a novel model named SAM, which applies attentional grouping to guide Scene-Aware feature Matching. SAM handles multi-level features, i.e., image tokens and group tokens, with attention layers, and groups the image tokens with the proposed token grouping module. Our model can be trained by ground-truth matches only and produce reasonable grouping results. With the sense-aware grouping guidance, SAM is not only more accurate and robust but also more interpretable than conventional feature matching models. Sufficient experiments on various applications, including homography estimation, pose estimation, and image matching, demonstrate that our model achieves state-of-the-art performance.

## 1. Introduction

Feature matching, which refers to finding the correct correspondence between two sets of features, is a fundamental problem for many computer vision tasks, such as object recognition [17], structure from motion (SFM) [28], and simultaneous localization and mapping (SLAM) [9]. But with illumination changes, viewpoint changes, motion blur and occlusion, it is challenging to find invariance and obtain robust matches from two images.

The classic image matching pipeline generally consists of four parts: (1) feature detection (2) feature description (3) feature matching (4) outlier filtering. For feature detection, keypoints that have distinguishable features to facilitate matching are detected. For feature description, descriptors are extracted based on keypoints and their neighborhoods. The keypoint positions and the corresponding descriptors are employed as features of the image. Then the feature matching algorithm is applied to find the correct correspondence in two sets of extracted features. Finally, the outlier filtering algorithm is applied to identify and reject outlier matches based on the obtained matches.

![](images/adf392552e429a9e5be78f6006deba01e3cf87ade2875718a47cde8b15ee50b2.jpg)  
Figure 1. An illustration of the grouping (top) and matching (bottom) result of our proposed method. Points in corresponding regions in the two images are correctly assigned to the same group, and the grouping information provides scene-aware guidance for point-level feature matching.

The current dominant approaches for image matching are learning-based descriptors with attention-based feature matching models. Learning-based descriptors extract local features with better representation capabilities by convolutional neural networks (CNN). Attention-based networks can enhance local features by perceiving global information and modeling contextual relationships between local features. While the above feature matching pipeline is the dominant method, the model performance still degrades significantly when dealing with extreme cases, such as scenes with large viewpoint changes and illumination changes. Because current methods only find correspondences at the low level, i.e., point-level textures, and do not incorporate sceneaware information, such as grouping information, semantic information, etc. Therefore, intra- and inter-image grouping is introduced to SAM to guide point-level attention-based matching.

In this work, we take point-level descriptors as image tokens and additionally introduce the concept of group tokens, which are selected from image tokens by the proposed group token selection module. Each group token represents a group of features shared by two images, and we intend to assign the corresponding points in both images to the same group, while the points that do not correspond to each other are assigned to different groups. We apply Transformer to model the relationship between image tokens and group tokens for intra- and inter-images, and propose a token grouping module to assign image tokens to different groups based on similarity. A novel multi-level score strategy is proposed to utilize the scene-aware grouping information to give guidance on point-level features, and to obtain reasonable grouping results relying only on ground-truth match supervision.

In summary, the contributions of this paper include:

• We propose a novel feature matching model SAM, which allows feature matching to rely on more than point-level textures by introducing group tokens to construct scene-aware features.

• The multi-level feature attention encoder and token grouping module are proposed to enable image tokens and group tokens to perceive global context information and assign image tokens to different groups.

• We are the first to utilize only ground-truth match supervision to enable the feature matching model to perform the scene-aware grouping and matching through the proposed multi-level score.

• SAM achieves state-of-the-art performance in multiple experiments while demonstrating outstanding robustness and interpretability.

## 2. Related Work

Local Feature Matching. For feature detection and description, there are many well-known works on handcrafted methods, such as SIFT [18], SURF [2], BRIEF [3] and ORB [26], which have good performance and are widely used in 3D computer vision tasks. With the rise of deep learning, many learning-based detectors have been proposed to further improve the robustness of descriptors under illumination changes and viewpoint changes, such as R2D2 [24], SuperPoint [7], D2-Net [8] and LF-Net [21].

In addition to detectors, other works have focused on how to get better matches with the obtained local features. SuperGlue [27] is the pioneering work to propose an attention-based feature matching network, where the selfattention layer utilizes global contextual information in one image to enhance features and the cross-attention layer finds potential matches in two images and performs information interaction on potential matching features. OETR [4] further detects the commonly visible region with an object detection algorithm to limit the attention-based feature matching to the overlap region. Besides matching the sparse descriptors generated by the detector, LoFTR [30] applies self- and cross-attention directly to the feature maps extracted by the convolutional neural network and generates matches in a coarse to fine manner. MatchFormer [33] further abandons the CNN backbone network and adopts a pure attention architecture, which can perform both feature extraction and feature matching. There are also methods named cost-volume-based methods [25, 15, 6], which find matches through 4D convolutional neural networks.

The above methods model the relationship between features at the point level and do not utilize higher-level scene information, resulting in non-robustness and significant performance degradation when handling large viewpoint changes and illumination changes. By introducing the concept of group tokens, we group the two image features based on the scene-aware information and construct multilevel features to make the model more robust when dealing with challenging scenes.

Vision Transformer. Inspired by the great success of Transformers in the field of Natural Language Processing, researchers have attempted to apply Transformer to Computer Vision. A. Dosovitskiy et al. [32] first proposed a pure Transformer named ViT directly to sequences of image patches for image classification. Many variants are subsequently proposed to solve various tasks. For feature matching, SuperGlue [27] and LoFTR [30] are the first sparse and dense matching methods to apply Transformer. SuperGlue applies classical scaled dot product attention, and LoFTR applies linear attention [29] to reduce runtime and memory consumption.

Our model is also an application of Transformer for the feature matching task. The global perceptual field of the attention mechanism facilitates the local features within and between images to perceive the global contextual information, making the matching more robust. And attention operations between image tokens and group tokens enable features to learn scene-aware grouping information.

Grouping. Most learning-based grouping models follow a pipeline of first learning representations with deep neural networks and then applying the grouping algorithms. For representation learning networks, common types of networks include multi-layer perceptron (MLP) [11], CNN [20] and Variational Autoencoder (VAE) [13]. Our model, on the other hand, applies Transformer as the representation learning network, globally perceiving image tokens and group tokens to learn deep representations.

For supervision, there are several commonly used grouping losses, which are K-means loss [37], Cluster assignment hardening [35], Cluster classification loss [10] and Localitypreserving loss [12]. Our grouping algorithm relies only on ground-truth matches to encourage the corresponding points of two images to be assigned to the same group.

![](images/92c59998737967ae80cb6100f6ec86b378d1d3bdb294f76a3c95ca2ebe86aee5.jpg)  
Figure 2. Proposed architecture. SAM consists of three parts, namely multi-level feature initialization, attention layers, and multi-level score construction. SAM first applies the position encoder to obtain position-aware descriptors, i.e. image tokens, and then selects group tokens to form multi-level features with the image tokens. The multi-level features are processed through attention layers, then the image tokens are assigned to different groups through the token grouping module, and the grouping information is employed to guide point-level matching by constructing multi-level scores.

![](images/9c563b9a7c316553673251f4334da23e0d039c4bcb3f64271464a12fc087e1d8.jpg)  
Figure 3. Group token selection module. The proper group tokens are chosen from image tokens based on the score projected by image tokens. And the gate signal computed from the score enables the block to be trained end-to-end.

## 3. Methodology

Assume that two sets of features $d _ { s } \in \mathbb { R } ^ { M \times C } , d _ { t } \in$ $\mathbb { R } ^ { N \times C }$ need to be matched which are descriptors extracted from two images. Subscripts s and t stand for the source image and target image, respectively. M, N are the number of descriptors, and C is the channels of descriptors. The keypoint positions are denoted as $p _ { s } \in \mathbb { R } ^ { M \times 3 } , \bar { p _ { t } } \in \mathbb { R } ^ { N \times 3 } ,$ which consist of two coordinates and detection confidence. Our objective is to find the correct correspondence between the two images utilizing descriptors and position information.

An overview of the network architecture is shown in Figure 2. The position information is first embedded into descriptors by the wave position encoder [19]. The positionaware descriptors are named image tokens, which are concatenated with the selected group tokens to form multi-level features. The self-attention layer and cross-attention layer are applied for L times to utilize the global context to enhance multi-level features, which are then re-split into image tokens and group tokens. The token grouping module is applied to assign image tokens to different groups based on similarity. Finally, the multi-level score is constructed from point-level features and group-level features to perform scene-aware matching.

## 3.1. Group Token Selection

The importance of clustering centers has been demonstrated by many clustering algorithms, without which the performance of the algorithm degrades or even appears as a trivial solution. To achieve the best scene-aware grouping effect, the group token selection module is proposed to select the proper group token among the image tokens.

As shown in Figure 3, we first apply a linear layer to compute the image tokens f as group scores s, which measure how effective each image token is as a group token. Then k image tokens with the highest group scores are selected as the group tokens g˜. The number of groups k is set to 2, representing overlapping and non-overlapping regions. To enable end-to-end training of the group token selection module, we apply the sigmoid function to calculate the group score s as a gating signal and multiply it with the selected image tokens to get the final group tokens $g .$ . The group token selection module is denoted as

![](images/3421f6cf0422b07d55bdd2b61802c467614e41cc112a459a864d8dccf8def058.jpg)  
Figure 4. Token Grouping Module. The pre-attention layer is employed to establish the relationship between image tokens and group tokens before assignment. The assign-attention layer is employed to assign image tokens to different groups based on the similarity between image tokens and group tokens. The pre-attention and the assign-attention apply Softmax and Gumbel-Softmax functions respectively.

$$
\begin{array} { l } { s = \mathrm { L i n e a r } ( f ) , } \\ { i d x = \mathrm { r a n k } ( s , k ) , } \\ { \tilde { g } = f ( i d x , : ) , } \\ { g a t e = \mathrm { s i g m o i d } ( s ( i d x ) ) , } \\ { g = \tilde { g } \odot g a t e , } \end{array}\tag{1}
$$

where  represents the element-wise matrix multiplication.

## 3.2. Multi-level Feature Attention

To perform information propagation between image tokens and group tokens, we concatenate them, $\mathrm { i } . \mathrm { e } . , f _ { s }$ and $g _ { s } ,$ $f _ { t }$ and $g _ { t } ,$ , to form the multi-level features $\hat { f } _ { s } , \hat { f } _ { t }$ , which are processed by the self-attention layer and the cross-attention layer. Specifically, the two sets of multi-level features are projected into two sets of $Q , K , V$ by three linear layers, and we compute attention as

$$
\begin{array} { r l } & { \mathcal { S } \mathcal { A } _ { s , t } = \mathrm { S o f t m a x } ( Q _ { s , t } K _ { s , t } ^ { T } / \sqrt { d } ) V _ { s , t } , } \\ & { \mathcal { C } \mathcal { A } _ { s } = \mathrm { S o f t m a x } ( Q _ { s } K _ { t } ^ { T } / \sqrt { d } ) V _ { t } , } \\ & { \mathcal { C } \mathcal { A } _ { t } = \mathrm { S o f t m a x } ( ( Q _ { s } K _ { t } ^ { T } ) ^ { T } / \sqrt { d } ) V _ { s } , } \end{array}\tag{2}
$$

where and denote self-attention and cross-attention output, and d is the feature dimension. The inputs of selfattention come from the feature of one image, such as $( Q _ { s } , K _ { s } , V _ { s } )$ or $( Q _ { t } , K _ { t } , V _ { t } )$ , while the inputs of the crossattention come from features of different images. To save computational costs and memory consumption, $Q _ { t } K _ { s } ^ { T }$ is replaced by $( Q _ { s } K _ { t } ^ { T } ) ^ { T }$ , since $Q _ { t } \dot { K } _ { s } ^ { T }$ and $\bar { Q _ { s } } K _ { t } ^ { T }$ are highly correlated.

Two MLPs are applied to adaptively fuse the self- and cross-attention outputs, and the fusion outputs are used to update the features as the input of the next attention layer.

$$
\begin{array} { r l } & { \hat { f } _ { s } ^ { l + 1 } = \hat { f } _ { s } ^ { l } + \mathrm { M L P } _ { s } ( [ \hat { f } _ { s } ^ { l } | S \mathcal { A } _ { s } ^ { l } | \mathcal { C } \mathcal { A } _ { s } ^ { l } ] ) , } \\ & { \hat { f } _ { t } ^ { l + 1 } = \hat { f } _ { t } ^ { l } + \mathrm { M L P } _ { t } ( [ \hat { f } _ { t } ^ { l } | S \mathcal { A } _ { t } ^ { l } | \mathcal { C } \mathcal { A } _ { t } ^ { l } ] ) . } \end{array}\tag{3}
$$

We stack $L \ = \ 9$ multi-level feature attention layers, which model the relationship between image tokens, and between image tokens and group tokens of two sets of features. Attention between image tokens finds potential matches at the point level between features, while attention between group tokens and image tokens allows group tokens to perceive global context and facilitates subsequent token grouping module.

## 3.3. Token Grouping Module

As shown in Figure 4, to assign image tokens to different groups based on similarity with group tokens in the embedding space, four parts are employed to form the token grouping module, namely spatial MLP, pre-attention layer, assign-attention layer and channel MLP.

Spatial MLP and channel MLP are introduced to the token grouping module to enhance the module capacity, which contain two fully-connected layers and an elementwise nonlinearity. Specifically, spatial MLP is applied to the transposed input $\mathbf { \bar { \Psi } } _ { g ^ { T } } \in \mathbb { R } ^ { \mathbf { \bar { C } } \times 2 }$ , mapping $\mathbb { R } _ { 2 } \mapsto \mathbb { R } _ { 2 }$ to interact between group tokens. And channel MLP is applied to $g \in \mathbb { R } ^ { 2 \times \check { C } }$ , mapping $\mathbb { R } _ { C } \mapsto \mathbb { R } _ { C }$ to interact between channels.

$$
\begin{array} { r l } & { \mathrm { s p a t i a l \ M L P : } O _ { * , i } = I _ { * , i } + W _ { 2 } ^ { s } \sigma ( W _ { 1 } ^ { s } I _ { * , i } ) , } \\ & { \mathrm { c h a n n e l \ M L P : } O _ { j , * } = I _ { j , * } + W _ { 2 } ^ { c } \sigma ( W _ { 1 } ^ { c } I _ { j , * } ) , } \end{array}\tag{4}
$$

where $W _ { 1 } ^ { s } , W _ { 2 } ^ { s } , W _ { 1 } ^ { c } , W _ { 2 } ^ { c }$ are learnable weight matrices, and σ is the GELU function.

As core parts of the token grouping module, the preattention layer and assign-attention layer perform the information propagation between image tokens and group tokens, and assign image tokens based on similarity, respectively. Both pre-attention layer and assign-attention layer apply three linear layers projecting group tokens as Q and image tokens as $K , V$ . The difference between the two attention is that Softmax is applied for pre-attention layer, and Gumbel-Softmax is applied for assign-attention layer.

Soft attention weight A for pre-attention layer and assign attention weight A<sup>˜</sup> for assign-attention layer are denoted as

$$
\begin{array} { r l } & { { \cal A } = \mathrm { S o f t m a x } ( { \cal Q } { \cal K } ^ { T } ) , } \\ & { { \cal \tilde { A } } _ { i , j } = \displaystyle \frac { \exp ( { \cal Q } _ { i } { \cal K } _ { j } ^ { T } + { \cal G } _ { i } ) } { \sum _ { k = 1 } ^ { 2 } \exp ( { \cal Q } _ { k } { \cal K } _ { j } ^ { T } + { \cal G } _ { k } ) } . } \end{array}\tag{5}
$$

G are i.i.d. samples drawn from Gumbel(0, 1) distribution.

After getting the assign attention weight, we decide the group to which the image tokens are assigned by the argmax over all group tokens. Since the argmax operation is not differentiable, the straight-through trick in [36] is applied to compute the assignment matrix as

$$
\hat { A } = \tilde { A } _ { a r g m a x } + \tilde { A } - \mathrm { s g } ( \tilde { A } ) ,\tag{6}
$$

where $\operatorname { s g } ( \cdot )$ is the stop gradient operation. By the straightthrough trick, assignment matrix A<sup>ˆ</sup> is numerically equal to the one-hot matrix $\tilde { A } _ { a r g m a x } ,$ and the gradient of A<sup>ˆ</sup> is equal to the gradient of assign attention weight A<sup>˜</sup>. Based on the assignment matrix A<sup>ˆ</sup>, all the image tokens of the same group are weighted summed to update the group tokens.

## 3.4. Multi-level Score

Conventional feature matching methods compute the dot product of two sets of features as the point-level score matrix, and then select matches based on it. We compute both point-level score matrix $S ^ { f } \in \mathbb { R } ^ { M \times N }$ and group-level score matrix $S ^ { g } \in \mathbb { R } ^ { 2 \times 2 }$ for the two sets of features based on image tokens $f$ and group tokens g.

$$
\begin{array} { c } { { S _ { i , j } ^ { f } = < f _ { i } ^ { s } , f _ { j } ^ { t } > , } } \\ { { S _ { i , j } ^ { g } = < g _ { i } ^ { s } , g _ { j } ^ { t } > . } } \end{array}\tag{7}
$$

To utilize the group information to guide the point-level matching, the group-level score matrix is expanded to pointlevel $\hat { S } ^ { g } \in \bar { \mathbb { R } ^ { M \times \bar { N } } }$ with the soft attention weights $A _ { s } \in$ $\mathbb { R } ^ { M \times 2 } , A _ { t } \in \mathbb { R } ^ { N \times 2 }$ of the two sets of features. The pointlevel score and the group-level score are weighted summed to obtain the multi-level score matrix $S ,$

$$
\begin{array} { l } { { \hat { S } ^ { g } = A _ { s } S ^ { g } A _ { t } ^ { T } , } } \\ { { \nonumber } } \\ { { S = \alpha S ^ { f } + \beta \hat { S } ^ { g } , } } \end{array}\tag{8}
$$

where α and β are learnable parameters. The multi-level score matrix S is taken as the cost matrix of the optimal transport problem. The Sinkhorn algorithm [5] is applied to iteratively obtain the optimal partial assignment matrix P. Based on P, matches with $P _ { i j }$ less than the matching threshold θ are filtered first, then the mutual nearest neighbor criterion is employed to select the final matches M.

For supervision, ground-truth matches $M _ { g t }$ and nonmatching point sets $I , J$ are computed from homography or camera pose and depth map. The first loss is the negative log-likelihood loss $L o s s _ { m }$ over the optimal partial assignment matrix P, which is denoted as

$$
\begin{array} { c } { { \displaystyle { L o s s _ { m } = - \frac { 1 } { | M _ { g t } | } \sum _ { ( i , j ) \in M _ { g t } } \log P _ { i , j } } } } \\ { { \displaystyle { - \frac { 1 } { | I | } \sum _ { i \in I } \log P _ { i , M + 1 } - \frac { 1 } { | J | } \sum _ { j \in J } \log P _ { N + 1 , j } } . } } \end{array}\tag{9}
$$

$P _ { * , M + 1 }$ and $P _ { N + 1 , }$ are learnable dustbins to which nonmatching points are assigned. $L o s s _ { m }$ provides supervision for matching and implicit supervision for grouping, as it encourages the assignment of corresponding points in two images to the same group, and the non-corresponding points to a different group.

To further enhance the ability to group precisely, we apply explicit supervision to assign attention weight A<sup>˜</sup> with ground-truth point sets $M _ { g t } , I$ and J.

$$
\begin{array} { r } { L o s s _ { g } = - \displaystyle \frac { 1 } { | M _ { g t } | } \sum _ { ( i , j ) \in M _ { g t } } ( \log \tilde { A } _ { i , 0 } ^ { s } + \log \tilde { A } _ { j , 0 } ^ { t } ) } \\ { - \displaystyle \frac { 1 } { | I | } \sum _ { i \in I } \log \tilde { A } _ { i , 1 } ^ { s } - \displaystyle \frac { 1 } { | J | } \sum _ { j \in J } \log \tilde { A } _ { j , 1 } ^ { t } . } \end{array}\tag{10}
$$

## 4. Experiments

## 4.1. Implementation Details

SAM is trained on the Oxford100k dataset [22] for homography estimation experiments and on the MegaDepth dataset [16] for pose estimation and image matching experiments. Our PyTorch implementation of SAM involves $L = 9$ attention layers, and all intermediate features have the same dimension $C = 2 5 6$ . The matching threshold θ is set to 0.2. For the homography estimation experiment, we employ the AdamW [14] optimizer for 10 epochs using the cosine decay learning rate scheduler and 1 epoch of linear warm-up. A batch size of 8 and an initial learning rate of 0.0001 are used. For the outdoor pose estimation experiment, we use the same AdamW optimizer for 30 epochs using the same learning rate scheduler and linear warm-up. A batch size of 2 and an initial learning rate of 0.0001 are used. All experiments are conducted on a single NVIDIA GeForce RTX 2060 SUPER GPU, 16GB RAM and Intel Core i7-10700K CPU.

<table><tr><td>Matcher</td><td>AUC</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>NN</td><td>39.47</td><td>21.7</td><td>65.4</td><td>32.59</td></tr><tr><td>NN + mutual</td><td>42.45</td><td>43.8</td><td>56.5</td><td>49.35</td></tr><tr><td>NN + PointCN</td><td>43.02</td><td>76.2</td><td>64.2</td><td>69.69</td></tr><tr><td>NN + OANet</td><td>44.55</td><td>82.8</td><td>64.7</td><td>72.64</td></tr><tr><td>SuperGlue</td><td>51.94</td><td>86.2</td><td>98.0</td><td>91.72</td></tr><tr><td>SAM</td><td>53.80</td><td>89.54</td><td>98.13</td><td>93.64</td></tr></table>

Table 1. Homography estimation on R1M dataset. AUC @10 pixels is reported. The best method is highlighted in bold.

## 4.2. Homography Estimation

Dataset. For the homography estimation, SAM is trained on the Oxford100k dataset [22] and evaluated on the 1M dataset [23]. To perform self-supervised training, we randomly sample ground-truth homography by limited parameters to generate image pairs. We resize images to 640 480 pixels and detect 512 keypoints in the image. When the detected keypoints are not enough, random keypoints are added for efficient batching.

Baselines. We employ the Nearest Neighbor (NN) matcher, NN matcher with outlier filtering methods [38, 39], and attention-based matcher SuperGlue [27] as baseline matchers. All matchers in Table 1 apply SuperPoint [7] as input descriptors for a fair comparison. The results of SuperGlue are from our implementation.

Metrics. The ground-truth matches are computed from the generated homography and the keypoint coordinates of the two images. A match is considered correct if the reprojection error is less than 3 pixels. We evaluate the precision and recall based on the ground-truth matches and compute the F1-score. We calculate reprojection error with the estimated homography and report the area under the cumulative error curve (AUC) up to 10 pixels.

Results. As shown in Table 1, SAM achieved the best performance in the homography estimation experiment. Benefiting from the powerful modeling capability of Transformer, the attention-based matcher is significantly ahead of other methods. Compared to the state-of-the-art outlier filtering method OANet, SAM achieves a +21% lead on F1-score. Both as attention-based methods, SAM is ahead of SuperGlue in both precision and recall because grouping information is introduced in addition to point-level matching, eliminating unreasonable matches and strengthening matches in the same group based on the grouping information. It ends up with a +1.92% advantage over SuperGlue on the F1-score. Qualitative results of matching and grouping are shown in Figure 6.

## 4.3. Outdoor Pose Estimation

Dataset. For the outdoor pose estimation experiment, the model is trained on the MegaDepth dataset [16] and evaluated on the YFCC100M dataset [31]. For training, 200 pairs of images in each scene are randomly sampled for each epoch. For evaluation, the YFCC100M image pairs and ground truth poses provided by SuperGlue are used. For training on the MegaDepth dataset, we resize the images to 960 720 pixels and detect 1024 keypoints.

<table><tr><td rowspan="2">Matcher</td><td colspan="3">Exact AUC</td><td colspan="3">Approx. AUC</td></tr><tr><td>@5°</td><td>@10°</td><td>@20°</td><td>@5°</td><td>@10°</td><td>@20°</td></tr><tr><td>NN + mutual</td><td>16.94</td><td>30.39</td><td>45.72</td><td>35.00</td><td>43.12</td><td>54.25</td></tr><tr><td>NN + OANet</td><td>26.82</td><td>45.04</td><td>62.17</td><td>50.94</td><td>61.41</td><td>71.77</td></tr><tr><td>SuperGlue</td><td>28.45</td><td>48.6</td><td>67.19</td><td>55.67</td><td>66.83</td><td>74.58</td></tr><tr><td>SAM</td><td>31.45</td><td>52.27</td><td>70.31</td><td>60.65</td><td>70.95</td><td>78.03</td></tr></table>

Table 2. Pose estimation on YFCC100M dataset. Our model lead other methods at all thresholds.

Baselines. The baseline method contains NN matchers with outlier filtering method [39] and attention-based matcher SuperGlue [27]. All matchers in Table 2 apply SuperPoint [7] as input descriptors for a fair comparison. The results of SuperGlue are from our implementation.

Metrics. The AUC of pose errors at the thresholds (5◦, 10◦, 20◦) are reported. Both approximate AUC [39] and exact AUC [27] are evaluated for a fair comparison.

Results. As shown in Table 2, for the outdoor pose estimation experiments, SAM achieves the best performance at all thresholds in both approximate AUC and exact AUC, demonstrating the robustness of our models. Compared to the attention-based matcher SuperGlue, which only considers point-level matching, our model can bring (+3.00%, +3.67%, +3.12%) improvement on exact AUC and (+4.98%, +4.12%, +3.45%) improvement on approximate AUC at three thresholds of (5◦, 10◦, 20◦), respectively. In outdoor scenes where large viewpoint changes and occlusions often occur, SAM provides scene understanding information for feature matching by utilizing grouping information to block out irrelevant interference.

## 4.4. Image Matching

Dataset. For the image matching experiment, the same model in the outdoor experiment is used. We follow the evaluation protocol as in D2-Net [8] and evaluate models on 108 HPatches [1] sequences, which contain 52 sequences with illumination changes and 56 sequences with viewpoint changes.

Baselines. Our model is compared with learning-based descriptors SuperPoint [7], D2Net [8], R2D2 [24] and advanced matchers SuperGlue [27], LoFTR [30], Patch2Pix [40], and CAPS [34].

Metrics. We compute the reprojection error from the ground truth homography and vary the matching threshold to plot the mean matching accuracy (MMA) curve, which is the average percentage of correct matches for each image.

Results. As shown in Figure 5, SAM achieves the best overall performance at reprojection error thresholds of 5 and above, demonstrating the robustness of our model in response to illumination changes and viewpoint changes. Experiments find that detector-based matching methods such as SuperGlue work well in viewpoint change scenes, while detector-free matching methods such as LoFTR work well in illumination change scenes. SAM performs well as a detector-based method in illumination change scenes, achieving an MMA of 100% at reprojection error thresholds of 8 and above, ahead of the detector-free matchers LoFTR and Patch2Pix and substantially ahead of the detector-based matcher SuperGlue. For viewpoint change scenes, our model is ahead of LoFTR at error thresholds of 6 and above.

![](images/5067284eb5cfd39ab5ab3eb8babef2c7794e0ac6b2569da8868060019c5ce338.jpg)

![](images/17262b2d56c6bdb764b0de0a82a07579b2ef1478e6998f6fc98de05f17389eb2.jpg)

![](images/ddf1653d0eb05170f0680a551eba5d88fa062d0bde239f38fc0b4aec6a8a93ac.jpg)  
Figure 5. Image Matching on HPatches dataset. MMA curves are plotted by changing the reprojection error threshold. Our model achieves the best overall performance at reprojection error thresholds of 5 and above.

<table><tr><td>Methods</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>(i) random</td><td>80.88</td><td>92.48</td><td>86.29</td></tr><tr><td>(ii) learnable parameters</td><td>81.22</td><td>95.10</td><td>87.61</td></tr><tr><td>(iii) selection</td><td>81.74</td><td>96.36</td><td>88.45</td></tr></table>

Table 3. Ablation study on group token.

<table><tr><td>Methods</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>(i) w/o pre-attention</td><td>80.85</td><td>95.84</td><td>87.71</td></tr><tr><td>(ii) w/o spatial MLP</td><td>81.35</td><td>96.16</td><td>88.14</td></tr><tr><td>(iii) w soft attention</td><td>81.60</td><td>96.15</td><td>88.28</td></tr><tr><td>(iv) w/o channel MLP</td><td>81.69</td><td>96.05</td><td>88.29</td></tr><tr><td>(v) full</td><td>81.74</td><td>96.36</td><td>88.45</td></tr></table>

Table 4. Ablation study on token grouping module.

## 4.5. Ablation Study

Comprehensive ablation experiments are conducted on the Oxford100k dataset to prove the validity of our designs. Group Token Selection. As shown in Table 3, we conduct ablation experiments on the methods of generating group tokens, containing random selection, learnable parameters as group tokens, and group token selection module. Firstly, as methods that can be learned end-to-end, learnable parameters group tokens and selection module outperform unlearnable random selection method, demonstrating the importance of proper group tokens. The token selection module performs better than the learnable parameter tokens because the token selection module can be adaptively selected based on the input image token whereas the learnable parameter tokens are static for all the images, so learnable parameter tokens have weaker generalizability when dealing with diverse scenes such as outdoor, indoor scenes.

<table><tr><td>Methods</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>(i) w/o group-level score</td><td>79.95</td><td>93.79</td><td>86.31</td></tr><tr><td>(ii) w group-level score</td><td>81.74</td><td>96.36</td><td>88.45</td></tr></table>

Table 5. Ablation study on multi-level score.

<table><tr><td>Methods</td><td>#Params (M)</td><td>Runtime (ms)</td></tr><tr><td>SuperGlue @2048 keypoints</td><td>12.0</td><td>104.37</td></tr><tr><td>LoFTR @800 × 800 resolution</td><td>11.6</td><td>373.50</td></tr><tr><td>ASpanFormer @800 × 800 resolution</td><td>15.8</td><td>500.28</td></tr><tr><td>SAM @2048 keypoints</td><td>14.8</td><td>111.42</td></tr></table>

Table 6. Efficiency analysis.

Token Grouping Module. As shown in Table 4, without the spatial MLP or channel MLP, the model performance degrades because MLP provides non-linearity and higher dimensional mapping enabling larger model capacity. And the performance also degrades without the pre-attention layer because the image tokens and the group tokens lose information propagation before assignment. When replacing hard attention with soft attention in the assign-attention, that is, replacing the one-hot assignment matrix with A obtained by Softmax, the performance of the token grouping module degrades because soft attention introduces more ambiguity to the group token while hard assignment makes the group tokens more separated. Stronger explicitness and weaker ambiguity in grouping make each group contain less information about weakly correlated image tokens.

Multi-level Score. Experiments on the multi-level score are conducted to demonstrate the guidance of group-level score to point-level matching. When the group-level term in the multi-level score is removed, our matcher degenerates to a conventional point-level matcher, and the performance degrades because the model loses the scene-aware information brought by grouping, demonstrating the effectiveness of our multi-level score design. Without the grouplevel term in score, the token grouping module also loses the supervision brought by ground-truth matches, so the token grouping module cannot be trained to produce reasonable grouping results.

![](images/7348443c437c9c32c9a4303369d76f146964968e22d7d1f0b04758b594400ff5.jpg)  
Figure 6. Qualitative results for grouping and matching. SAM can effectively assign the corresponding points of two images into the same group when dealing with indoor and outdoor scenes, thus guiding a more robust and accurate point-level matching.

## 5. Limitation and Discussion

Firstly, due to the additional group tokens and token grouping module, the computational complexity of SAM increases compared to point-level matchers, but only two groups do not harm the real-time capability of our model.

Since the token grouping module only utilizes groundtruth matching supervision, the trained model explicitly assigns overlapping regions in two images to one group. The model is unable to recognize the semantics of buildings or objects in the two scenes to guide matching. We believe that the model is able to group more complex semantic information if appropriate supervision is provided.

## 6. Conclusion

In this paper, we present a novel attention-based matcher SAM, which incorporates scene-aware group guidance. Compared to matching features at the point level, we introduce group tokens on the basis of the image tokens. Group tokens and image tokens are concatenated to model the global relationship through attention layers, and the token grouping module assigns image tokens to scene-aware groups. We build multi-level score which utilizes pointlevel and group-level information to generate matching. Benefiting from the scene-aware information, our model achieves state-of-the-art performance. SAM is also more interpretable than current feature matching models since grouping information can be visualized.

## 7. Acknowledgments

The authors would like to thank Prof. Min-Ling Zhang for insightful suggestions and fruitful discussions. This work was jointly supported by the National Natural Science Foundation of China under grants 62001110 and 62201142, the Natural Science Foundation of Jiangsu Province under grant BK20200353, and the Shenzhen Science and Technology Program under grant JCYJ20220530160415035.

## References

[1] V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk. Hpatches: A benchmark and evaluation of handcrafted and learned local descriptors. In Proceedings of the CVPR, pages 5173–5182, 2017.

[2] H. Bay, T. Tuytelaars, and L. Van G. SURF: Speeded Up Robust Features. In Proceedings of the ECCV, pages 404– 417, 2006.

[3] M. Calonder, V. Lepetit, C. Strecha, and P. Fua. BRIEF: Binary Robust Independent Elementary Features. In Proceedings ofthe ECCV, pages 778–792, 2010.

[4] Y. Chen, D. Huang, S. Xu, J. Liu, and Y. Liu. Guide local feature matching by overlap estimation. In Proceedings of the AAAI, pages 365–373, 2022.

[5] M. Cuturi. Sinkhorn Distances: Lightspeed Computation of Optimal Transport. In Proceedings of the NeurIPS, pages 2292–2300, 2013.

[6] Franc¸ois Darmon, Mathieu Aubry, and Pascal Monasse. Learning to guide local feature matches. In Proceedings of the 3DV, pages 1127–1136, 2020.

[7] D. DeTone, T. Malisiewicz, and A. Rabinovich. SuperPoint: Self-Supervised Interest Point Detection and Description. In Proceedings ofthe CVPRW, pages 224–236, 2018.

[8] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler. D2-Net: A Trainable CNN for Joint Description and Detection of Local Features. In Proceedings ofthe CVPR, pages 8092–8101, 2019.

[9] J. Engel, V. Koltun, and D. Cremers. Direct sparse odometry. Journal ofthe PAMI, 40(3):611–625, 2017.

[10] C. Hsu and C. Lin. Cnn-based joint clustering and representation learning with feature drift compensation for large-scale image data. IEEE Transactions on Multimedia, 20(2):421–429, 2017.

[11] W. Hu, T. Miyato, S. Tokui, E. Matsumoto, and M. Sugiyama. Learning discrete representations via information maximizing self-augmented training. In Proceedings of the ICML, pages 1558–1567, 2017.

[12] P. Huang, Y. Huang, W. Wang, and L. Wang. Deep embedding network for clustering. In Proceedings of the ICPR, pages 1532–1537, 2014.

[13] Z. Jiang, Y. Zheng, H. Tan, B. Tang, and H. Zhou. Variational deep embedding: A generative approach to clustering. Journal of the CoRR, 1, 2016.

[14] DP. Kingma and B. Jimmy. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[15] X. Li, K. Han, S. Li, and V. Prisacariu. Dual-resolution correspondence networks. In Proceedings of the NeurIPS, pages 17346–17357, 2020.

[16] Z. Li and N. Snavely. MegaDepth: Learning Single-View Depth Prediction from Internet Photos. In Proceedings of the CVPR, pages 2041–2050, 2018.

[17] C. Liu, J. Yuen, A. Torralba, J. Sivic, and W. T Freeman. Sift flow: Dense correspondence across different scenes. In Proceedings ofthe ECCV, pages 28–42, 2008.

[18] D. G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, pages 91–110, 2004.

[19] X. Lu, Y. Yan, B. Kang, and S. Du. Paraformer: Parallel attention transformer for efficient feature matching. arXiv preprint arXiv:2303.00941, 2023.

[20] Y. Lukic, C. Vogt, O. Durr, and T. Stadelmann. Speaker iden-¨ tification and clustering using convolutional neural networks. In Proceedings of the MLSP, pages 1–6, 2016.

[21] Y. Ono, E. Trulls, P. Fua, and K. M. Yi. LF-Net: Learning Local Features from Images. In Proceedings of the NeurIPS, pages 6237–6247, 2018.

[22] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman. Object retrieval with large vocabularies and fast spatial matching. In Proceedings of the CVPR, pages 1–8, 2007.

[23] F. Radenovic, A. Iscen, G. Tolias, Y. Avrithis, and O. Chum.´ Revisiting oxford and paris: Large-scale image retrieval benchmarking. In Proceedings of the CVPR, pages 5706– 5715, 2018.

[24] J. Revaud, C. De Souza, M. Humenberger, and P. Weinzaepfel. R2D2: Reliable and Repeatable Detector and Descriptor. In Proceedings of the NeurIPS, pages 12414–12424, 2019.

[25] I. Rocco, M. Cimpoi, R. Arandjelovic, A. Torii, T. Pajdla, ´ and J. Sivic. Neighbourhood consensus networks. In Proceedings of the NeurIPS, pages 1658–1669, 2018.

[26] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski. ORB: An efficient alternative to SIFT or SURF. In Proceedings of the ICCV, pages 2564–2571, 2011.

[27] P. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich. SuperGlue: Learning Feature Matching With Graph Neural Networks. In Proceedings of the CVPR, pages 4937–4946, 2020.

[28] J. L Schonberger and J. Frahm. Structure-from-motion revisited. In Proceedings of the CVPR, pages 4104–4113, 2016.

[29] Z. Shen, M.n Zhang, H. Zhao, S. Yi, and H. Li. Efficient attention: Attention with linear complexities. In Proceedings ofthe WACV, pages 3531–3539, 2021.

[30] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou. LoFTR: Detector-Free Local Feature Matching With Transformers. In Proceedings of the CVPR, pages 8922–8931, 2021.

[31] B. Thomee, B. Elizalde, D. A Shamma, K. Ni, G. Friedland, D. Poland, D. Borth, and L. J. Li. YFCC100M: The New Data in Multimedia Research. Communications ofthe ACM, pages 64–73, 2016.

[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N Gomez, L. Kaiser, and I. Polosukhin. Attention is All you Need. In Proceedings of the NeurIPS, pages 5998–6008, 2017.

[33] Q. Wang, J. Zhang, K. Yang, K. Peng, and R. Stiefelhagen. Matchformer: Interleaving attention in transformers for feature matching. In Proceedings of the ACCV, pages 2746– 2762, 2022.

[34] Q. Wang, X. Zhou, B. Hariharan, and N. Snavely. Learning feature descriptors using camera pose supervision. In Proceedings ofthe ECCV, pages 757–774, 2020.

[35] J. Xie, R. Girshick, and A. Farhadi. Unsupervised deep embedding for clustering analysis. In Proceedings ofthe ICML, pages 478–487, 2016.

[36] J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the CVPR, pages 18134–18144, 2022.

[37] B. Yang, X. Fu, N. D Sidiropoulos, and M. Hong. Towards k-means-friendly spaces: Simultaneous deep learning and clustering. In Proceedings of the ICML, pages 3861–3870, 2017.

[38] K. M. Yi, E. Trulls, Y. Ono, V. Lepetit, M. Salzmann, and P. Fua. Learning to Find Good Correspondences. In Proceedings ofthe CVPR, pages 2666–2674, 2018.

[39] J. Zhang, D. Sun, Z. Luo, A. Yao, L. Zhou, T. Shen, Y. Chen, L. Quan, and H. Liao. Learning two-view correspondences and geometry using order-aware network. In Proceedings of the ICCV, pages 5845–5854, 2019.

[40] Q. Zhou, T. Sattler, and L. Leal-Taixe. Patch2pix: Epipolarguided pixel-level correspondences. In Proceedings of the CVPR, pages 4669–4678, 2021.