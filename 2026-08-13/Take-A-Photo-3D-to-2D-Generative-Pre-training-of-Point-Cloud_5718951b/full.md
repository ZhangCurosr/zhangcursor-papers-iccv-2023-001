# Take-A-Photo: 3D-to-2D Generative Pre-training of Point Cloud Models

Ziyi Wang<sup>1,2\*</sup> Xumin Yu<sup>1,2\*</sup> Yongming Rao<sup>1,2</sup> Jie Zhou<sup>1,2</sup> Jiwen Lu<sup>1,2†</sup> <sup>1</sup>Department of Automation, Tsinghua University <sup>2</sup>BNRist

{wziyi22, yuxm20}@mails.tsinghua.edu.cn;

raoyongming95@gmail.com; {jzhou, lujiwen}@tsinghua.edu.cn

## Abstract

With the overwhelming trend of mask image modeling led by MAE, generative pre-training has shown a remarkable potential to boost the performance of fundamental models in 2D vision. However, in 3D vision, the over-reliance on Transformer-based backbones and the unordered nature of point clouds have restricted the further development of generative pre-training. In this paper, we propose a novel 3D-to-2D generative pre-training method that is adaptable to any point cloud model. We propose to generate view imagesfrom different instructed poses via the cross-attention mechanism as the pre-training scheme. Generating view images has more precise supervision than its point cloud counterpart, thus assisting 3D backbones to have afiner comprehension of the geometrical structure and stereoscopic relations of the point cloud. Experimental results have proved the superiority ofour proposed 3D-to-2D generative pre-training over previous pre-training methods. Our method is also effective in boosting the performance ofarchitecture-oriented approaches, achieving state-of-the-art performance when fine-tuning on ScanObjectNN classification and ShapeNet-Part segmentation tasks. Code is available at https: //github.com/wangzy22/TakeAPhoto.

## 1. Introduction

Nowadays, pre-training fundamental models with selfsupervised mechanisms has witnessed a thriving development in the computer vision community, given its low requirement in data annotation and its high transferability to downstream applications. Self-supervised pre-training aims to fully exploit the statistical and structural knowledge from abundant annotation-free data and empowers the fundamental models with robust representation ability. In 2D vision, self-supervised pre-training has shown strong potential and achieved remarkable progress on diverse backbones in various downstream tasks. Successful pre-training strategies in 2D domain such as contrastive learning [19, 7] and mask modeling [18, 3] have also been adopted to 3D point cloud analysis [58, 30, 23] in recent research.

![](images/6f0a77dc29e3b29fbd3b1d8c31320a3cc82125fb49e83e614ec05ec3d3678653.jpg)  
Figure 1: Principle illustration of 3D-to-2D generative pre-training. The photograph module explicitly encodes pose condition into 3D features from the backbone, and the 2D generator decodes pose-conditioned features into different view images.

However, the pre-training paradigm hasn’t become the prevailing trend in 3D learning and architectural design is still the mainstream to reach a new state-of-the-art performance, which is considerably different from the dominant status of pre-training in 2D domain. In object-level analysis, generative pre-training inspired by MAE [18] has been studied thoroughly, but their performances still lag behind architecture-based methods like PointNeXt [35]. Two factors mainly contribute to the inferior status of generative pretraining in 3D learning. Since point clouds are unordered sets of point coordinates, there is no direct and precise supervision like one-to-one MSE loss between generated point clouds and their corresponding ground truth. Chamfer Distance supervision for point clouds only calculates a rough set-to-set matching and its imprecision has been widely discussed in [24, 50, 20]. Additionally, existing advanced generative pre-training methods in object analysis are limited to the Transformer-based backbone, and fail to be extended to other sophisticated point cloud models.

To alleviate the aforementioned problems, we propose a 3D-to-2D generative pre-training method for point cloud analysis that has higher preciseness in supervision and broader adaptation to different backbones. Instead of reconstructing point clouds as previous literature [30, 60], we propose to generate view images of the input point cloud given the instructed camera poses. This is similar to taking photos of a realistic object from different perspectives to fully present its structure or internal relations. Therefore, we name our model Take-A-Photo, in short TAP. More specifically, we propose a pose-dependent photograph module that utilizes the cross-attention mechanism to explicitly encode pose conditions with 3D features from the backbone. Then a 2D generator decodes pose-conditioned features into view images that are supervised by rendered ground truth images. The principle illustration of TAP is shown in Figure 1. In the pose-dependent photograph module, we do not provide detailed projection relations from 3D points to 2D pixels, thus the cross-attention layers are encouraged to learn by themselves what those view images look like conditioned on given poses. Since the projection layout, part occlusion relation, faces colors that represent light reflections are largely distinct among view images, the proposed 3D-to-2D generative pre-training is a challenging pretext task that obliges 3D backbone to gain higher representation ability of geometrical knowledge and stereoscopic relations.

We conduct extensive experiments on various backbones and downstream tasks to verify the effectiveness and superiority of our proposed 3D-to-2D generative pre-training method. When pre-trained on synthetic ShapeNet [6] and transferred to real-world ScanObjectNN [45] classification, TAP brings consistent improvement to different backbone models and successfully outperforms previous point cloud generative pre-training methods based on Transformers backbone. With PointMLP [27] as the backbone, TAP achieves state-of-the-art performance on ScanObjectNN classification and ShapeNetPart [56] part segmentation among methods that do not include any pre-trained image or text model. We also conduct thorough ablation studies to discuss the architectural design of the TAP model and verify the individual contribution of each component.

In conclusion, the contributions of our paper can be summarized as follows: (1) We propose TAP, the first 3D-to-2D generative pre-training method that is adaptable to any point cloud model. TAP pre-training helps to exploit the potential of point cloud models on geometric structure comprehension and stereoscopic relation understanding. (2) We pro pose a Photograph Module where we derive mathematical formulations to encode pose conditions as query tokens in cross-attention layers. (3) TAP surpasses previous generative pre-training methods on the Transformers backbone and achieves state-of-the-art performance on ScanObjectNN classification and ShapeNetPart segmentation.

## 2. Related Work

Point Cloud Analysis. Point cloud analysis is a fundamental and important task in the realm of 3D vision. Current literature has developed two principal methodologies to extract structural representations from 3D point clouds, namely point-based and voxel-based methods. Point-based methods [32, 33, 48, 44, 35, 27] process unordered points directly, introducing various techniques for local information aggregation. Existing point-based methods can be categorized into three types: SetAbstraction-based [32, 33, 35], DynamicGraph-based [48], and Attention-based [57, 58, 30, 23, 60], all of which focus on modeling the relationships between points. Owing to their exceptional ability to effectively preserve fine-grained geometric information, point-based methods are frequently employed for object-level tasks. On the other hand, voxel-based methods [28, 21, 39] partition the 3D space into ordered voxels and employ 3D convolutions for feature extraction. Voxel-based methods are primarily based on SparseConvolution [10, 16], which enables efficient convolution operations in 3D space through sparse convolutions. In exchange for faster processing speeds, voxelbased methods sacrifice their capacity to capture detailed local structures, making them more suitable for large-scale scene-level tasks rather than object-level tasks.

Point Cloud Pre-training. Pre-training has always been a focal point of research in the field of deep learning. Generally speaking, we usually distinguish pre-training methods based on the amount of annotation required, namely fullsupervised pre-training [14, 59, 5], weakly-supervised pretraining [43, 4, 53], and unsupervised pre-training [19, 7, 8]. Among these methods, unsupervised pre-training has become the most popular approach, mainly due to its excellent transferability across tasks and its notable advantage of not relying on labeled data. Numerous researchers have proposed a variety of pretext tasks for unsupervised pre-training. Based on the type of pretext task employed, there are two prevailing pretext tasks for unsupervised pre-training. The first is contrastive learning, as exemplified by MoCo [19] and SimCLR [7]. The other method entails utilizing generative tasks to restore the data from partially or disrupted inputs, such as MAE [18] and BEiT [3].

Inspired by pre-training strategies in the image domain, there are more and more unsupervised pre-training methods being proposed for point cloud pre-training. PointContrast [54] embraces the principle of contrastive learning, whereas PointBERT [58] and PointMAE [30] integrate reconstruction pretext tasks. However, existing generativebased pre-training methods for point clouds only consider a single modality. In this paper, we propose a cross-modal generative-based pre-training strategy to achieve more effective pre-training.

Cross-Modal Learning. Recently, cross-modal learning has been a popular research topic, aiming at learning from multiple modalities such as images, audio and point clouds. It has the potential to enhance the performance of various tasks, including visual recognition, speech recognition, and point cloud analysis. A variety of methods have been proposed for cross-modal learning, including multi-task learning [11], conditional generation [40], pre-training [34, 13] and tuning [49, 55].

![](images/dfc1689e83ab563eec50862c30fa93725c73e0c21c8520b1e1503abddb859a88.jpg)  
Figure 2: The pipeline of TAP pre-training method. We design a query generator to encode pose conditions and implement cross-attention layers to transform 3D point cloud features $F _ { 3 \mathrm { d } }$ to 2D view image features $F _ { 2 \mathrm { d } }$ according to pose instruction. The predicted pose-dependent view image from 2D generator is supervised by ground truth view image via MSE loss.

In the realm of point cloud analysis, much previous work has explored this learning paradigm. Some literature leverages 2D data for 3D point cloud analysis, such as MVTN [17], MVCNN [42] and CrossPoint [2], proving that the multi-view images and the correspondence between images and points can be helpful for the 3D object understanding. Another line of the research, such as Image2Point [55] and P2P [49] take effort to adapt the models from 2D vision into 3D point cloud analysis, fully exploiting the relationship of 2D and 3D understanding. In this paper, we continue this learning paradigm in 3D vision domain, and for the first time propose the cross-modal generative pre-training scheme for point cloud models.

## 3. 3D-to-2D Generative Pre-training

## 3.1. Preliminary: Generative Pre-training

Generative pre-training is a fundamental branch of pretraining methods that aims at reconstructing integral and complete data given partial or disrupted input. Mathematically, suppose x is a sample from raw data with no annotation. The pre-processing step $T ( \cdot )$ either erases part of x randomly or splits x into pieces and intermingles them to get $\tilde { x } = T ( x )$ . The generative pre-training model M is designed to restore from those broken input ${ \hat { x } } = M ( T ( x ) )$ and the training loss function is designed to measure the reconstruction distance $\mathcal { L } = D ( \hat { x } , x )$ . In point cloud object analysis, earlier generative pre-training methods propose various pretext tasks as T, including deformation [1], jigsaw puzzles [41] and depth projection [47] to produce disarrayed or partial point clouds. Recently, inspired by MAE [18] in the image domain, generative pre-training in 3D domain mainly focuses on implementing random masking as $T$ and utilizing

Transformers model as M for reconstruction [58, 30, 23, 60]. The reconstruction distance D is usually measured by the classical $l _ { 2 }$ Chamfer Distance:

$$
D ( \hat { x } , x ) = \frac { 1 } { | \hat { x } | } \sum _ { a \in \hat { x } } \operatorname* { m i n } _ { b \in x } \lVert a - b \rVert _ { 2 } ^ { 2 } + \frac { 1 } { | x | } \sum _ { b \in x } \operatorname* { m i n } _ { a \in \hat { x } } \lVert a - b \rVert _ { 2 } ^ { 2 }\tag{1}
$$

Besides Chamfer Distance between point clouds, some methods also exploit feature distance between latents [58] or occupancy value distance [23] as the loss function.

The exact reason why generative pre-training would help enhance the representation ability of backbone models still remains an open question. However, abundant experimental results have conveyed that predicting missing parts according to known parts demands high reasoning ability and global comprehension capacity of the model. What’s more, generative pre-training is more efficient and suitable for point cloud object analysis than contrastive pre-training, given that contrastive pre-training typically requires a large amount of training data to avoid trivial overfitting solutions but there has always been a data-starvation problem in point cloud object research field.

## 3.2. Overall Pipeline

Different from the aforementioned generative pre-training methods that focus on uni-modal point cloud reconstruction, we propose a novel cross-modal pre-training approach of generating view images from instructed camera poses.

The overall architecture of our proposed TAP pre-training model is depicted in Figure 2. Our model takes as an input point cloud $\mathbf { \bar { \rho } } _ { P } \in \mathbb { R } ^ { N \times \mathbf { \bar { 3 } } }$ , where N is the number of points in the input point cloud. The basic building block of TAP mainly consists of: $1 ) \mathrm { ~ a ~ } 3 D$ Backbone that extracts 3D geometric features $F _ { 3 \mathrm { d } } \in \mathbb { R } ^ { n \times C _ { 3 \mathrm { d } } }$ , where n is the number of downsampled center points and $C _ { 3 \mathrm { d } }$ is the geometric feature dimension; 2) a pose-dependent Photograph Module that takes as inputs $F _ { 3 \mathrm { d } }$ and pose matrix $R \in \mathbb { R } ^ { 3 \times 3 }$ , and predicts view image features $\bar { F _ { 2 \mathrm { d } } ^ { R } } \in \mathbb { R } ^ { h \times w \times C _ { 2 \mathrm { d } } }$ conditioned on $R ,$ where $h ,$ w are height and width of predicted view image feature map; 3) an 2D Generator that decodes $F _ { \mathrm { 2 d } } ^ { R }$ into an RGB image $I _ { \mathrm { g e n } } ^ { \hat { R } } \in \mathbb { R } ^ { H \times W \times 3 }$ , where $H , W$ are height and width of the output view image.

As we place no restriction on $F _ { 3 \mathrm { d } }$ , the 3D Backbone can be arbitrarily chosen and adopted. Therefore, our TAP is more flexible and compatible than existing generative pre-training methods that are limited to Transformer-based architecture. Experimental results in Section 4 will later verify that TAP brings consistent improvement to all kinds of point cloud models. The technical designs of the pose-dependent Photograph Module will be thoroughly discussed in Section 3.3. The 2D Generator consists of four Transpose Convolution layers to progressively upsample image resolution and decode RGB colors of each pixel.

## 3.3. Photograph Module

Architectural Design. As illustrated in Figure 2, we leverage cross-attention mechanism from Transformers [46] to build our pose-dependent Photograph Module.

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } \left( { \frac { Q K ^ { T } } { \sqrt { d _ { k } } } } \right) V\tag{2}
$$

where $d _ { k }$ is the scaling factor, and $Q , K , V$ are quries, keys and values matrix. More specifically, we design a Query Generator Φ to encode camera pose conditions into query tokens: $Q = \Phi ( R ) \in \mathbb { R } ^ { h w \times C _ { 2 \mathrm { d } } }$ . We also design a Memory Builder Θ to construct K and V from 3D geometric features: $K = V = \Theta ( F _ { 3 \mathrm { d } } ) \in \mathbb { R } ^ { m \times C _ { 2 \mathrm { d } } }$ , where m is the number of memory tokens. The output sequence of the cross attention layers will be rearranged from $h w \times C _ { 2 \mathrm { d } }$ to $h \times w \times C _ { 2 \mathrm { d } }$ forming the predicted view image features $F _ { \mathrm { 2 d } } ^ { R }$

During the cross-attention calculation process, we do not explicitly provide any projection clues of which 3D points would project to which 2D pixel. Instead, the Photograph Module learns by itself how to arrange unordered 3D feature points to ordered 2D pixel grids, purely based on semantic similarities between 3D geometric features and our delicately-designed queries that reveal pose information. Since one sample will only have one set of memory tokens in 3D space but its view images from different poses are quite distinct from each other, learning to predict precise view images from instructed poses in a data-driven manner is not a trivial task. Therefore, during the end-to-end optimization process, the 3D backbone is trained to have a stronger perception of the object’s overall geometric structure and gain a higher representative ability of the stereoscopic relations. In this way, our proposed 3D-to-2D generative pre-training would help exploit the potential and enhance the strength of 3D backbone models.

Query Generator. The query generator $\Phi$ is designed to encode pose condition R into 2D grid of shape $h \times w$ . In object analysis, common practice is leveraging parallel light shading to project 3D objects onto 2D grids, and pose matrix R here is used to rotate objects into desired angles before projection. Therefore, each 2D grid actually represents an optical line that starts from infinity, passes through 3D objects and ends at the 2D plane. As a consequence, we choose the direction and the origin points that the optical line goes through as the delegate of the query grid.

Before deriving formulations of optical lines for each grid, let us first revisit the parallel light shading process for better comprehension. Given 3D coordinates $\mathbf { x } = ( x , y , z )$ of a point cloud $P$ and pose matrix $R ,$ rotation is first performed to align the object to the ideal pose position:

$$
\mathbf { x } ^ { \prime } = ( x ^ { \prime } , y ^ { \prime } , z ^ { \prime } ) = R \mathbf { x }\tag{3}
$$

Then we just omit the final dimension $z ^ { \prime }$ and evenly split the first two dimensions $( x ^ { \prime } , y ^ { \prime } )$ into 2D grids $( u , v )$

$$
u = \frac { x ^ { \prime } - x _ { 0 } } { g _ { h } } + o _ { h } , \quad v = \frac { y ^ { \prime } - y _ { 0 } } { g _ { w } } + o _ { w }\tag{4}
$$

where $( x _ { 0 } , y _ { 0 } )$ is the minimum value of $( x ^ { \prime } , y ^ { \prime } ) , ( g _ { h } , g _ { w } )$ is the grid size, $\left( o _ { h } , o _ { w } \right)$ is the offset value to place the projected object at the center of the image. $0 \leq u \leq h -$ $1 , 0 \leq v \leq w - 1$ and $( u , v )$ is a sampled pixel coordinate from the 2D grid.

Now let us begin to derive formulations of the optical line that passes through the query grid. We only know $( u , v )$ for each grid and we want to reversely trace which 3D points $( x , y , z )$ are on the same optical line during parallel light projection. According to Eq. 4:

$$
\begin{array} { r } { x ^ { \prime } = g _ { h } u + x _ { 0 } - o _ { h } = \Psi _ { h } ( u ) } \\ { y ^ { \prime } = g _ { w } v + y _ { 0 } - o _ { w } = \Psi _ { w } ( v ) } \end{array}\tag{5}
$$

If we denote $A = R ^ { - 1 }$ and $A _ { i j }$ as the element at $i ^ { t h }$ row and $j ^ { t h }$ column, then according to Eq. 3:

$$
\begin{array} { l } { x = A _ { 1 1 } \Psi _ { h } ( u ) + A _ { 1 2 } \Psi _ { w } ( v ) + A _ { 1 3 } z ^ { \prime } = \Omega _ { x } ( u , v ) + A _ { 1 3 } z ^ { \prime } } \\ { y = A _ { 2 1 } \Psi _ { h } ( u ) + A _ { 2 2 } \Psi _ { w } ( v ) + A _ { 2 3 } z ^ { \prime } = \Omega _ { y } ( u , v ) + A _ { 2 3 } z ^ { \prime } } \\ { z = A _ { 3 1 } \Psi _ { h } ( u ) + A _ { 3 2 } \Psi _ { w } ( v ) + A _ { 3 3 } z ^ { \prime } = \Omega _ { z } ( u , v ) + A _ { 3 3 } z ^ { \prime } } \end{array}\tag{6}
$$

According to the definition of line’s parametric equation, Eq. 6 represents a line passing through the origin point $O : ( \Omega _ { x } ( u , v ) , \Omega _ { y } ( u , v ) , \Omega _ { z } ( u , v ) )$ with optical line direction $\mathbf { d } = \left( A _ { 1 3 } , A _ { 2 3 } , A _ { 3 3 } \right)$ , where $\Omega _ { x } , \Omega _ { y } , \Omega _ { z }$ are xyz coordinates of O and their formulations are conditioned on u, v. Therefore, we concatenate the coordinate of origin point O, normalized direction $\mathbf { d } ^ { \dagger } = \mathbf { d } / \lVert \mathbf { d } \rVert _ { 2 }$ and normalized position $( u / h , v / w )$ as positional embedding together to be the initial state of our query. A multi-layer-perceptron (MLP) module is later leveraged to map the 8-dim initial query to higher dimensional space.

Memory Builder. The memory builder takes $F _ { 3 \mathrm { d } }$ as input to prepare for initial state of $K , V$ in cross-attention layers. We first concatenate aligned 3D coordinate $P _ { \mathrm { 3 d } }$ with 3D features to enhance the geometric knowledge of $F _ { 3 \mathrm { d } }$ :

$$
\hat { F } _ { 3 \mathrm { d } } = \mathrm { M L P } ( \cot ( F _ { 3 \mathrm { d } } , P _ { 3 \mathrm { d } } ) )\tag{7}
$$

Additionally, we initialize a learnable memory token $T _ { \mathrm { p a d } }$ as the pad token and concatenate it with $\hat { F } _ { 3 \mathrm { d } }$ to obtain the initial state of $K , V .$ The reason for concatenating a learnable pad token $T _ { \mathrm { p a d } }$ is that there are white background areas on the projected image (as shown in Figure 2). As $F _ { \mathrm { 3 d } }$ only encodes foreground objects, we further need a learnable pad token to represent background regions. Otherwise, the crossattention layers will be confused to learn how to combine foreground tokens into background features and this will inevitably diminish the pre-training effectiveness.

## 3.4. Objective Function

We perform per-pixel supervision with Mean Squared Error (MSE) loss between generated view image $I _ { \mathrm { g e n } } ^ { \hat { R } }$ and ground truth image $I _ { \mathrm { g t } } ^ { R } ,$ , aligned by camera pose R. For simplicity, we will omit R in later formulations. As the background of the rendered ground truth images is all white and reveals little information, we further design a compound loss to balance the weight between foreground regions and background regions:

$$
\mathcal { L } ( I _ { \mathrm { g e n } } , I _ { \mathrm { g t } } ) = w ^ { \mathrm { f g } } \mathcal { D } ^ { \mathrm { f g } } + w ^ { \mathrm { b g } } \mathcal { D } ^ { \mathrm { b g } }\tag{8}
$$

$$
\mathcal { D } ^ { k } ( I _ { \mathrm { g e n } } ^ { k } , I _ { \mathrm { g t } } ^ { k } ) = \frac { 1 } { H W } \sum _ { h , w } ( I _ { \mathrm { g e n } } ^ { k } ( h , w ) - I _ { \mathrm { g t } } ^ { k } ( h , w ) ) ^ { 2 }\tag{9}
$$

where $k = \mathrm { f g }$ (foreground), bg (background) and $w ^ { \mathrm { f g } }$ , w<sup>bg</sup> are loss weights for foreground and background, respectively. Such per-pixel supervision is more precise than the ambiguous set-to-set Chamfer Distance introduced in Eq. 1.

## 4. Experiments

In this section, we first introduce the setups of our pretraining scheme. Then we evaluate our pre-training method on various point cloud backbones by fine-tuning them on different downstream tasks, such as point cloud classification on ModelNet and ScanObjectNN datasets, and part segmentation on the ShapeNetPart dataset. Finally, we provide in-depth ablation studies for the architectural design of our proposed TAP pre-training pipeline.

## 4.1. Pre-training Setups

Data Setups To align with previous research practices [58, 30, 54], we choose ShapeNet [6] that contains more than 50 thousand CAD models as our pre-training datasets. We sampled 1024 points from each 3D CAD model to form the point clouds, consistent with previous work. Since ShapeNet does not provide images for each point cloud, we use the rendered image from 12 surrounding viewpoints generated by MVCNN [42]. During our pre-training, the models are exclusively pre-trained with the training split following the practice of previous work [58].

Table 1: Classification results on the ScanObjectNN dataset. We report the overall accuracy (%). The results with † are reproduced by PointNeXt [35] repository.
<table><tr><td>Method</td><td colspan="2">OBJ_BG OBJ_ONLY PB_T50_RS</td></tr><tr><td></td><td>Hierarchical Models with TAP Pre-training</td><td></td></tr><tr><td>†DGCNN [48]</td><td></td><td>86.1</td></tr><tr><td>+ TAP</td><td></td><td>86.6 (+0.5)</td></tr><tr><td>†PointNet++ [33]</td><td></td><td>86.2</td></tr><tr><td>+ TAP</td><td></td><td>86.8 (+0.6)</td></tr><tr><td>†PointMLP [27]</td><td></td><td>87.4</td></tr><tr><td>+ TAP</td><td></td><td>88.5 (+1.1)</td></tr><tr><td colspan="2">Standard Transformers with Generative Pre-training</td><td></td></tr><tr><td>w/o pre-training [46]</td><td>79.86 80.55</td><td>77.24</td></tr><tr><td>OcCo [47]</td><td>84.85 85.54</td><td>78.79</td></tr><tr><td>Point-BERT [58]</td><td>87.43 88.12</td><td>83.07</td></tr><tr><td>MaskPoint [23]</td><td>89.30 88.10</td><td>84.30</td></tr><tr><td>Point-MAE [30]</td><td>90.02 88.29</td><td>85.18</td></tr><tr><td>TAP (Ours)</td><td>90.36 89.50</td><td>85.67</td></tr></table>

Architecture Setups We conduct experiments on various point cloud encoders, including PointNet++ [33], DGCNN [48], PointMLP [27] and Transformers [58] for point cloud object classification. During the pre-training stage, the photograph module takes encoded point cloud features and pose conditions as inputs to generate a 32× downsampled view image feature map of size $7 \times 7$ from a specific viewpoint. Then the 2D generator progressively upsamples the image feature map to decode RGB view images of size 224 × 224. We do not alter the architecture of the point cloud backbone since the photograph module and the 2D generator are exclusively used during the pretraining phase and are dropped during the fine-tuning stage. In our experiment, the photograph module is a six-layer cross-attention block, with attention layer channels limited to 256 to enhance efficiency. During the pre-training task on ShapeNet, we utilized four simple transpose convolutions to upsample the reconstructed 2D feature map and predict the RGB value for each pixel.

Implementation Details The experiments of TAP pretraining and finetuning on various downstream tasks are implemented with PyTorch [31]. We utilize AdamW [26] optimizer and the CosineAnnealing learning rate scheduler [25] to pre-train the point cloud backbone for 100 epochs. We set the initial learning rate as $5 e ^ { - 4 }$ and weight decay as $5 e ^ { - 2 }$ In our experiment, we train various point cloud backbones with 32 batch sizes on a single Nvidia 3090Ti GPU. The drop path rate of the cross-attention layer is set to 0.1. The foreground and background loss weights w<sup>fg</sup>, w<sup>bg</sup> are set to 20 and 1. The detailed architecture of our simple 2D generator is: TConv(256, 128, 5, 4) → TConv(128, 64, 3, 2) → TConv(64, 32, 3, 2) → TConv(32, 3, 3, 2), where TConv stands for Transpose Convolution and the four numbers in the tuple denotes $( C _ { i n } , C _ { o u t }$ , Kernel, Stride) respectively. During the fine-tuning stage, we perform a learning rate warming up for point cloud backbones with 10 epochs, and keep other settings unchanged for a fair comparison.

Table 2: Results comparisons with previous methods on the ScanObjectNN and ModelNet40 datasets. The model parameters number (#Params) and overall accuracy (%) are reported. † denotes our reproduced results of PointMLP on the ModelNet40 dataset. Methods with ∗ introduce extra knowledge from pre-trained image models or pre-trained vision-language models. We do not compete with them for fair comparison and only list them for reference.
<table><tr><td>Method</td><td>#Params (M)</td><td>ScanObjectNN</td><td>ModelNet40</td></tr><tr><td colspan="4">Supervised Learning Only</td></tr><tr><td>PointNet [32]</td><td>3.5</td><td>68.0</td><td>89.2</td></tr><tr><td>PointNet++ [33]</td><td>1.5</td><td>77.9</td><td>90.7</td></tr><tr><td>Transformer [46]</td><td>22.1</td><td>77.24</td><td>91.4</td></tr><tr><td>DGCNN [48]</td><td>1.8</td><td>78.1</td><td>92.9</td></tr><tr><td>PointCNN [22]</td><td>0.6</td><td>78.5</td><td>92.2</td></tr><tr><td>DRNet [36]</td><td></td><td>80.3</td><td>93.1</td></tr><tr><td>SimpleView [15]</td><td></td><td>80.5±0.3</td><td>93.9</td></tr><tr><td>GBNet [37]</td><td>8.8</td><td>81.0</td><td>93.8</td></tr><tr><td>PRA-Net [9]</td><td>2.3</td><td>81.0</td><td>93.7</td></tr><tr><td>MVTN [17]</td><td>11.2</td><td>82.8</td><td>93.8</td></tr><tr><td>RepSurf-U [38]</td><td>1.5</td><td>84.3</td><td>94.4</td></tr><tr><td>PointMLP [27]</td><td>12.6</td><td>85.4±0.3</td><td>93.5†</td></tr><tr><td>PointNeXt [35]</td><td>1.4</td><td>87.7±0.4</td><td>93.7</td></tr><tr><td colspan="4">Transformers with Pre-training</td></tr><tr><td>OcCo [47]</td><td>22.1</td><td>78.8</td><td>92.1</td></tr><tr><td>Point-BERT [58]</td><td>22.1</td><td>83.1</td><td>93.2</td></tr><tr><td>MaskPoint [23]</td><td>22.1</td><td>84.3</td><td>93.8</td></tr><tr><td>Point-MAE [30]</td><td>22.1</td><td>85.2</td><td>93.8</td></tr><tr><td>Point-M2AE [60]</td><td>15.3</td><td>86.4</td><td>94.0</td></tr><tr><td colspan="4">With Pre-trained Image Model</td></tr><tr><td>ACT [13]*</td><td>22.1</td><td>88.2</td><td>93.7</td></tr><tr><td>I2P-MAE [61]*</td><td></td><td>90.1</td><td>93.7</td></tr><tr><td>ReCon [34]*</td><td>44.3</td><td>90.6</td><td>94.1</td></tr><tr><td colspan="4">Our Proposed TAP Pre-training</td></tr><tr><td>PointMLP + TAP</td><td>12.6</td><td>88.5</td><td>94.0</td></tr></table>

## 4.2. Downstream Tasks

In this section, we report the experimental results of various downstream tasks. We follow the previous work to conduct experiments of object classification on real-world ScanObjectNN and synthetic ModelNet40 datasets. We also verify the effectiveness of our pre-training method on the part segmentation task with the ShapeNetPart dataset.

![](images/9261570ed459ee91a865d9df9a383fcc107e5be2c213ea03970a183f7abd2a35.jpg)  
Figure 3: The visualization results of our proposed 3Dto-2D generative pre-training. The first row displays view images generated by our TAP pre-training pipeline and the second row shows ground truth images. Our TAP can produce view images with appropriate shapes and reflection colors, demonstrating its ability in capturing geometric structure and stereoscopic knowledge.

## 4.2.1 Object Classification

Main Results. To evaluate the effectiveness of our proposed TAP, we implement it with various point cloud architectures, including classical baselines such as PointNet++, DGCNN, and PointMLP, as well as widely used Standard Transformers backbone for existing generative pre-training methods. We follow the common practice to experiment our model on three variants of the ScanObjectNN dataset: 1) OBJ-ONLY: cropping object without any background; 2) OBJ-BG: containing the background and object; 3) PB-T50-RS: adopting various augmentations to the objects. We reported the results comparing with existing pre-training methods in Table 1.

As shown in the upper part of the table, we pretrain the graph-based architecture DGCNN, set-abstractionbased architecture PointNet++, and MLP-based architecture PointMLP with TAP, and observe consistent improvements across models. The results strongly convey that our proposed TAP can be successfully applied to various types of point cloud models and the proposed novel 3D-to-2D generative pre-training is effective regardless of the backbone architecture. Considering that nearly all existing generative pre-training methods are specially designed for Transformerbased architecture, our TAP is much superior in its wider adaptation and higher flexibility. Additionally, we also provide a detailed and fair comparison with previous work by implementing TAP with the Standard Transformers architecture in the lower part of the table, where no hierarchical designs or inductive bias is included. Our TAP outperforms previous pre-training methods in all three split settings, providing strong evidence that our 3D-to-2D generative pretraining strategy can also benefit attention-based architectures and surpass uni-modal generative pre-training competitors. It is worth noting that although TAP and many previous pre-training approaches have significantly improved the performance of Transformers in point cloud tasks, increasing the accuracy from 77.24 to 85.67, they still lag far behind advanced point cloud networks such as PointMLP. Therefore, TAP’s applicability to various models is an important characteristic that would benefit future research.

Table 3: Part segmentation results on the ShapeNetPart dataset. We report the mean IoU across all part categories mIoU<sub>C</sub>, the mean IoU across all instances mIoU<sub>I</sub>, and the IoU for each category.
<table><tr><td>Methods</td><td>mIoUc</td><td>mIoUI</td><td>aero</td><td>bag</td><td>cap</td><td>car</td><td>chair</td><td>earphone</td><td>guitar knife</td><td>lamp</td><td>laptop</td><td>motor</td><td>mug</td><td>pistol</td><td>rocket</td><td>skateboard</td><td>table</td></tr><tr><td colspan="10">Supervised Representation Learning Only</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PointNet [32]</td><td>80.4</td><td>83.7</td><td>83.4</td><td>78.7</td><td>82.5</td><td>74.9</td><td>73.0</td><td>91.5</td><td>85.9</td><td>80.8</td><td>95.3</td><td>65.2</td><td>93.0</td><td>81.2</td><td>57.9</td><td>72.8</td><td>80.6</td></tr><tr><td>PointNet++ [33]</td><td>81.9</td><td>85.1</td><td>82.4</td><td>79.0</td><td>87.7</td><td>77.3</td><td>71.8</td><td>91.0</td><td>85.9</td><td>83.7</td><td>95.3</td><td>71.6</td><td>94.1</td><td>81.3</td><td>58.7</td><td>76.4</td><td>82.6</td></tr><tr><td>DGCNN [48]</td><td>82.3</td><td>85.2</td><td>84.0</td><td>83.4</td><td>86.7</td><td>77.8</td><td>74.7</td><td>91.2</td><td>87.5</td><td>82.8</td><td>95.7</td><td>66.3</td><td>94.9</td><td>81.1</td><td>63.5</td><td>74.5</td><td>82.6</td></tr><tr><td>PointMLP [27]</td><td>84.6</td><td>86.1</td><td>83.5</td><td>83.4</td><td>87.5</td><td>80.5</td><td>78.2</td><td>92.2</td><td>88.1</td><td>82.6</td><td>96.2</td><td>77.5</td><td>95.8</td><td>85.4</td><td>64.6</td><td>83.3</td><td>84.3</td></tr><tr><td>KPConv [44]</td><td>85.1</td><td>86.4</td><td>84.6</td><td>86.3</td><td>87.2</td><td>81.1 91.1</td><td>77.8</td><td>92.6</td><td>88.4</td><td>82.7</td><td>96.2</td><td>78.1</td><td>95.8</td><td>85.4</td><td>69.0</td><td>82.0</td><td>83.6</td></tr><tr><td colspan="10">Transformers with Uni-modal Generative Pre-training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Point-BERT [58]</td><td>84.1</td><td>85.6</td><td>84.3</td><td>84.8</td><td>88.0</td><td>79.8 91.0</td><td>81.7</td><td>91.6</td><td>87.9</td><td>85.2</td><td>95.6</td><td>75.6</td><td>94.7</td><td>84.3</td><td>63.4</td><td>76.3</td><td>81.5</td></tr><tr><td>Point-MAE [30] MaskPoint [23]</td><td>84.2</td><td>86.1</td><td>84.3</td><td>85.0</td><td>88.3</td><td>80.5 91.3 91.2</td><td>78.5</td><td>92.1</td><td>87.4</td><td>86.1</td><td>96.1</td><td>75.2</td><td>94.6</td><td>84.7</td><td>63.5</td><td>77.1</td><td>82.4</td></tr><tr><td>Point-M2AE [60]</td><td>84.4</td><td>86.0</td><td>84.2</td><td>85.6</td><td>88.1</td><td>80.3</td><td>79.5</td><td>91.9</td><td>87.8</td><td>86.2</td><td>95.3</td><td>76.9</td><td>95.0</td><td>85.3</td><td>64.4</td><td>76.9</td><td>81.8</td></tr><tr><td></td><td>84.9</td><td>86.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Our Proposed 3D-to-2D Generative Pre-training</td><td></td><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td><td>78.1</td><td></td></tr><tr><td>PointMLP+TAP</td><td>85.2</td><td>86.9</td><td>84.8</td><td>86.1</td><td>89.5</td><td>82.5 92.1</td><td>75.9</td><td>92.3</td><td>88.7</td><td>85.6</td><td>96.5</td><td>79.8</td><td>96.0</td><td>85.9</td><td>66.2</td><td></td><td>83.2</td></tr></table>

Comparisons with Previous Methods. To clearly demonstrate the high performance of our proposed TAP pre-training in object classification tasks, we compare TAP with existing methods on both synthetic ModelNet and real-world ScanObjectNN (the hardest PB-T50-RS variant) datasets in Table 2. We categorize existing methods into two types: (1) architecture-oriented methods, which focus on developing novel model architectures for 3D point clouds and do not involve any pre-training techniques, and (2) pre-training methods, which pay more attention to the pre-training strategy and whose backbone model are mostly limited to Transformerbase architecture. It’s worth noticing that methods marked with an asterisk (∗) incorporate additional knowledge from pre-trained image models or pre-trained vision-language models. To ensure a fair and unbiased comparison, we refrain from directly comparing our method with these approaches. However, we include them in the listing for reference purposes, acknowledging their existence and potential relevance in related research.

From the experimental results, we can see that accompanied by PointMLP backbone model, our proposed TAP pre-training achieves the best classification accuracy on ScanObjectNN and ModelNet40 among existing models (with no pre-trained knowledge from image or language like P2P [49]), demonstrating the effectiveness of our approach and validating the superiority of our 3D-to-2D cross-modal generative pre-training method over previous generative pretraining methods. Moreover, we also note that our proposed method has brought higher performance improvements on the ScanObjectNN dataset than on ModelNet40. This may be attributed to the reason that the cross-modal generative pre-training has enhanced the network’s ability to understand point clouds from different views, which is beneficial for a more robust understanding of the real-scan data with more noise and disturbance in the ScanObjectNN dataset.

Visualization Results. Figure 3 shows the visualization results of TAP. The first row shows the generated view images while the second row displays the ground truth images for reference. The TAP method can successfully predict the accurate shape of the object and the RGB colors that represent light reflections in rendered images. Therefore, TAP is capable of capturing the geometric structure of 3D objects and reasoning occlusion relations from specific camera poses.

## 4.2.2 Part Segmentation

Performing dense prediction is always a more challenging task compared with classification. In this section, we evaluate the local distinguishability of our proposed TAP pre-training method, fine-tuning the pre-trained point cloud model on the ShapeNetPart dataset for the part segmentation task. Quantitative results are shown in Table 3. We imple ment PointMLP as the backbone model and compare our TAP results with two mainstreams of previous literature. The upper row displays classical architecture-oriented methods that focus on network design and are trained from scratch. The lower row shows members of the generative pre-training family that rely on Transformer-based architectures.

According to results comparisons, our TAP pre-training significantly improves the part segmentation performance of the PointMLP backbone, increasing class mIoU by 0.6 and instance mIoU by 0.8. More importantly, our TAP pretraining achieves state-of-the-art performance on both class mIoU and instance mIoU, surpassing leading works in both tracks. Specifically, TAP exceeds the performance of Point-M2AE on instance mIoU by 0.4. This satisfactory performance serves as strong evidence to convey that our proposed TAP pre-training is superior to previous uni-model generative pre-training mechanisms in dense prediction tasks. This may be attributed to the factor that our supervision in 2D with MSE loss is more precise than the ambiguous Chamfer Distance in 3D reconstruction. Therefore, models with TAP pre-training obtain more accurate comprehension of local geometry and detail awareness, which contributes to mIoU gain in dense prediction tasks. What’s more, TAP outperforms KPConv on instance mIoU by 0.5, demonstrating that the proposed 3D-to-2D generative pre-training method can fully exploit the potential of the point cloud model and help it have a better perception of objects’ geometric structure. As TAP is adaptable to any architecture, future improvements in architectural design can also benefit from TAP pre-training.

## 4.2.3 Few-shot Classification

Following Point-BERT [58], we conduct few-shot classification with Standard Transformers on ModelNet40 [52] dataset. As shown in Table 4, we report mean overall accuracy and standard deviation (mOA±std) on 10 pre-defined data folders for each few-shot setting. The way and shot in Table 4 specify the number of categories and the number of training examples per category, respectively.

From the results, TAP achieves the highest mean overall accuracy across all few-shot settings when compared to previous generative pre-training approaches. Furthermore, TAP exhibits significantly lower standard deviations than those reported in the existing literature for the majority of few-shot settings, which signifies its robust performance and consistent superiority. This indicates that TAP is not only capable of achieving high mean overall accuracy but also exhibits reliability and robustness across various few-shot settings. Such stability is crucial in real-world applications, where consistency and predictability are vital for practical deployment.

## 4.3. Scene-level Dense Predictions

To assess the effectiveness of TAP in handling scenelevel dense prediction tasks, we carry out experiments on more complicated scene-level object detection and semantic segmentation on the ScanNetV2 [12] dataset. For the object detection task, we adopt 3DETR [29] and pre-train its encoder on the object-level dataset ShapeNet [6] with TAP. Average precision at 0.25 and 0.5 IoU thresholds are reported. Regarding semantic segmentation, we employ the PointTransformerV2 [51](PTv2) model and pre-train it on the ScanNetV2 dataset with TAP. We report mean IoU for evaluation metric. It is worth mentioning that PTv2 repre sents the current state-of-the-art approach with open-source code availability.

Based on the results presented in Table 5, TAP consistently enhances the performance of all baselines, thereby showcasing its efficacy in tackling more intricate scene-level dense prediction tasks. Remarkably, even with the encoder solely pre-trained on an object-level dataset for scene-level detection task, significant improvements are observed in both $\mathrm { A P _ { 0 . 2 5 } }$ and $\mathsf { A P } _ { 0 . 5 }$ metrics. This suggests that the learned representations from TAP effectively capture relevant information and generalize well to complex scenes, even when the pre-training data is limited to object-level collections. Such generalization capabilities are valuable in scenarios where obtaining large-scale fully annotated scene-level datasets may be challenging or expensive.

Table 4: Few-shot Classification with Standard Transformers on ModelNet40 dataset. We report mean overall accuracy and standard deviation on 10 pre-defined data folders for each setting. Best results are marked bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">5-way</td><td colspan="2">10-way</td></tr><tr><td>10-shot</td><td>20-shot</td><td>10-shot</td><td>20-shot</td></tr><tr><td>w/o pre-training</td><td colspan="2"> $8 7 . 8 \pm 5 . 2 ~ 9 3 . 3 \pm 4 . 3 ~ 8 4 . 6 \pm 5 . 5 ~ 8 9 . 4 \pm 6 . 3$ </td><td colspan="2"></td></tr><tr><td>Point-BERT [58] MaskPoint [23]</td><td colspan="2"> $9 4 . 6 \pm 3 . 1 ~ 9 6 . 3 \pm 2 . 7 ~ 9 1 . 0 \pm 5 . 4 ~ 9 2 . 7 \pm 5 . 1$   $9 5 . 0 \pm 3 . 7 ~ 9 7 . 2 \pm 1 . 7 ~ 9 1 . 4 \pm 4 . 0 ~ 9 3 . 4 \pm 3 . 5$ </td><td></td><td></td></tr></table>

Table 5: Scene-level object detection and semantic segmentation on ScanNetV2 [12]. Average precision at 0.25 IoU thresholds $( \mathrm { A P _ { 0 . 2 5 } } )$ and 0.5 IoU thresholds $( \operatorname { A P } _ { 0 . 5 } )$ of detection and mean Intersection-over-Union (mIoU) of semantic segmentation are reported.
<table><tr><td rowspan="2">Method</td><td colspan="2">Det (3DETR [29])</td><td>Seg (PTv2 [51])</td></tr><tr><td> $\mathrm { A P } _ { 0 . 2 5 }$ </td><td> $\mathsf { A P } _ { 0 . 5 }$ </td><td>mIoU</td></tr><tr><td>Baseline</td><td>62.1</td><td>37.9</td><td>72.4</td></tr><tr><td>+TAP</td><td>63.0 (+0.9)</td><td>41.4 (+3.5)</td><td> $7 2 . 6 \ : ( + 0 . 2 )$ </td></tr></table>

## 4.4. Ablation Studies

To investigate the architectural design of our proposed Photograph Module in TAP pre-training pipeline, we conduct extensive ablation studies on the ScanObjectNN dataset with PointMLP as the backbone model.

Photograph Module Architectural Designs. In Photograph Module, we implement cross-attention layers to generate view image feature maps conditioned on pose instruction. We believe that letting the module learn by itself how to rearrange 3D point features in 2D grids will enhance the representation ability of the 3D backbone. Therefore, we conduct ablation studies to verify this hypothesis. As shown in Table 6a, we implement Model ${ \bf A } _ { 1 }$ with no attention layers, directly projecting 3D feature points to 2D grids based on Eq. 3 and Eq. 4. This results in a much simpler pretraining task, as the projection relation has been directly told. Additionally, in Model $\mathbf { A } _ { 2 }$ , we add self-attention layers after explicit projection to help the model capture longer-range correlations. Pose knowledge is encoded as a pose token that is concatenated to projected grids, similar to the CLS token in classification Transformers. According to quantitative results comparison with TAP that implements cross-attention layers, fine-tuning results of pre-training methods in $\mathbf { A } _ { 1 }$ and $\mathbf { A } _ { 2 }$ version show inferiority. Therefore, the cross-attention architecture we designed to entirely LEARN the projection relation is the most suitable choice for the proposed 3D-to-2D generative pre-training.

Table 6: Ablation studies on Photograph module in TAP pre-training pipeline. We choose PointMLP as the backbone model and conduct ablation studies on the ScanObjectNN dataset from two aspects: overall architectural designs and query designs. In Table(a), we first investigate the effectiveness of cross-attention design compared with direct projection (Model $\mathbf { A } _ { 1 } )$ and direct project with self-attention (Model A<sub>2</sub>). Then we analyze the influence of the different number of attention layers and feature channels in Model B and Model C. Finally we discuss whether pad token in memory builder is beneficial in Model D. In Table(b), we conduct further experiments to compare different approaches for query designing: (1) Using learnable query based on the given viewpoints (Model E) or use mathematical formulations derived in Eq. 6. (2) The information we need when we mathematically encode pose information into init query status: (i) Origin: the coordinate of origin point O that the optical line passes through. (ii) Direction: The normalized direction of the optical line. (iii) PE: The position embedding for each grid.  
(a) Overall Architectural Designs.
<table><tr><td></td><td></td><td>Model | Attention Type | LayerNum. Channels Mem.Pad</td><td></td><td>Acc.(%)</td></tr><tr><td> ${ \bf A } _ { 1 }$ </td><td>None</td><td>256</td><td>x</td><td>87.6 (-0.9)</td></tr><tr><td> $\mathbf { A } _ { 2 }$ </td><td>SelfAttn</td><td>6 layers 256</td><td>x</td><td>87.8 (-0.7)</td></tr><tr><td>B</td><td>CrossAttn</td><td>2 layers 256</td><td>√</td><td>87.9 (-0.6)</td></tr><tr><td>C D</td><td>CrossAttn</td><td>6 layers 512</td><td>√</td><td>87.8 (-0.7)</td></tr><tr><td></td><td>CrossAttn</td><td>6 layers 256</td><td>x</td><td>88.3 (-0.2)</td></tr><tr><td>TAP</td><td>CrossAttn</td><td>6 layers 256</td><td>√</td><td>88.5</td></tr></table>

What’s more, we discuss the number of cross-attention layers, the dimension of feature channels and whether to concatenate pad token in memory builder in Model B, C, D. According to the results, more cross-attention layers show stronger representation ability, while too large channel number will lead to performance decrease caused by over-fitting. The performance gain from Model D to TAP also verifies that the pad token design in the memory builder is essential.

Query Generator Designs. In the query generator, we derive the mathematical formulation of the optical lines passing through 2D grids. We propose to concatenate the coordinate of origin point O, normalized direction d<sup>†</sup> and position embedding $( u / h , v / w )$ as the initial state of queries. In Table 6b, we first compare this mathematical design with totally learnable queries that takes pose matrix R as input and implements MLP layers to predict query for each grid. As shown in Model E, learnable queries cannot satisfactorily encode pose information, while our derived formulation for query construction is both clearer in physical meaning and more competitive in fine-tuning accuracy.

(b) Query Designs.
<table><tr><td></td><td></td><td>Model | Query Type | Origin Direction PE</td><td></td><td>Acc.(%)</td></tr><tr><td>E</td><td>Learnable</td><td>x x</td><td>x</td><td>87.5 (-1.0)</td></tr><tr><td> $\mathrm { F _ { 1 } }$ </td><td>Formula</td><td>x √</td><td>√</td><td>86.5 (-2.0)</td></tr><tr><td> $\mathrm { F _ { 2 } }$ </td><td>Formula</td><td>√ x</td><td>x</td><td>87.8 (-0.7)</td></tr><tr><td> $\mathrm { F _ { 3 } }$ </td><td>Formula</td><td>√ √</td><td>X</td><td>88.0 (-0.5)</td></tr><tr><td> $\mathrm { F _ { 4 } }$ </td><td>Formula</td><td>√ x</td><td>√</td><td>88.1 (-0.4)</td></tr><tr><td>TAP</td><td>Formula</td><td>√</td><td>√ √</td><td>88.5</td></tr></table>

In ablation $\mathrm { F _ { 1 } }$ to $\mathrm { F _ { 4 } }$ , we progressively discuss the three components of query generation. Quantitative comparison with TAP verifies that every component is indispensable for query generation, where coordinates of origin points are of the most importance.

## 5. Conclusions

In this paper, we have proposed a novel 3D-to-2D generative pre-training method TAP that is adaptable to any point cloud model. We implemented the cross-attention mechanism to generate view images of point clouds from instructed camera poses. To better encode pose conditions and generate physically meaningful queries, we derived mathematical formulations of optical lines. The proposed TAP pre-training had higher preciseness in supervision and broader adaptation to different backbones, compared with directly reconstructing point clouds in previous methods. Experimental results conveyed that the TAP pre-training can help the backbone models better capture the structural knowledge and stereoscopic relations. Fine-tuning results of TAP pre-training achieve state-of-the-art performance on ScanObjectNN classification and ShapeNetPart segmentation, among methods that do not include any pre-trained image or text models. We believe the cross-modal generative pre-training paradigm will be a promising direction for future research.

## Acknowledgement

This work was supported in part by the National Key Research and Development Program of China under Grant 2022ZD0114903 and in part by the National Natural Science Foundation of China under Grant 62125603.

## References

[1] Idan Achituve, Haggai Maron, and Gal Chechik. Selfsupervised learning for domain adaptation on point clouds. In WACV, 2021.

[2] Mohamed Afham, Isuru Dissanayake, Dinithi Dissanayake, Amaya Dharmasiri, Kanchana Thilakarathna, and Ranga Rodrigo. Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding. In CVPR, pages 9902–9912, June 2022.

[3] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.

[4] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. NeruIPS, 2019.

[5] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.

[6] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An informationrich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[7] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. arXiv preprint arXiv:2006.10029, 2020.

[8] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Im proved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.

[9] Silin Cheng, Xiwu Chen, Xinwei He, Zhe Liu, and Xiang Bai. Pra-net: Point relation-aware network for 3d point cloud analysis. IEEE TIP, 30:4436–4448, 2021.

[10] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In CVPR, 2019.

[11] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In ICML, 2008.

[12] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richlyannotated 3d reconstructions of indoor scenes. In CVPR, 2017.

[13] Runpei Dong, Zekun Qi, Linfeng Zhang, Junbo Zhang, Jianjian Sun, Zheng Ge, Li Yi, and Kaisheng Ma. Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning? arXiv preprint arXiv:2212.08320, 2022.

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[15] Ankit Goyal, Hei Law, Bowei Liu, Alejandro Newell, and Jia Deng. Revisiting point cloud shape classification with a simple and effective baseline. In ICML, 2021.

[16] Benjamin Graham, Martin Engelcke, and Laurens Van Der Maaten. 3d semantic segmentation with submanifold sparse convolutional networks. In CVPR, 2018.

[17] Abdullah Hamdi, Silvio Giancola, and Bernard Ghanem. MVTN: multi-view transformation network for 3d shape recognition. In ICCV, 2021.

[18] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377, 2021.

[19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[20] Tianxin Huang, Zhonggan Ding, Jiangning Zhang, Ying Tai, Zhenyu Zhang, Mingang Chen, Chengjie Wang, and Yong Liu. Learning to measure the point cloud reconstruction loss in a representation space. In CVPR, 2023.

[21] Roman Klokov and Victor Lempitsky. Escape from cells: Deep kd-networks for the recognition of 3d point cloud models. In ICCV, 2017.

[22] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. NeurIPS, 2018.

[23] Haotian Liu, Mu Cai, and Yong Jae Lee. Masked discrimination for self-supervised learning on point clouds. In ECCV. Springer, 2022.

[24] Minghua Liu, Lu Sheng, Sheng Yang, Jing Shao, and Shi-Min Hu. Morphing and sampling network for dense point cloud completion. In AAAI, 2020.

[25] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2018.

[27] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In ICLR, 2022.

[28] Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In IROS. IEEE, 2015.

[29] Ishan Misra, Rohit Girdhar, and Armand Joulin. An endto-end transformer model for 3d object detection. In ICCV, 2021.

[30] Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In ECCV. Springer, 2022.

[31] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.

[32] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017.

[33] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++ deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017.

[34] Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining. arXiv preprint arXiv:2302.02318, 2023.

[35] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scal ing strategies. In NeurIPS, 2022.

[36] Shi Qiu, Saeed Anwar, and Nick Barnes. Dense-resolution network for point cloud classification and segmentation. In WACV, 2021.

[37] Shi Qiu, Saeed Anwar, and Nick Barnes. Geometric backprojection network for point cloud classification. IEEE TMM, 2022.

[38] Haoxi Ran, Jun Liu, and Chengjie Wang. Surface representation for point clouds. In CVPR, 2022.

[39] Gernot Riegler, Ali Osman Ulusoy, and Andreas Geiger. Octnet: Learning deep 3d representations at high resolutions. In CVPR, 2017.

[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In CVPR, 2022.

[41] Jonathan Sauder and Bjarne Sievers. Self-supervised deep learning on point clouds by reconstructing space. NeurIPS, 2019.

[42] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In ICCV, 2015.

[43] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. NeurIPS, 2017.

[44] Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J. Guibas. Kpconv: Flexible and deformable convolution for point clouds. ICCV, 2019.

[45] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In ICCV, 2019.

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[47] Hanchen Wang, Qi Liu, Xiangyu Yue, Joan Lasenby, and Matt J Kusner. Unsupervised point cloud pre-training via occlusion completion. In ICCV, 2021.

[48] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. TOG, 2019.

[49] Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, and Jiwen Lu. P2p: Tuning pre-trained image models for point cloud analysis with point-to-pixel prompting. In NeurIPS, 2022.

[50] Tong Wu, Liang Pan, Junzhe Zhang, Tai Wang, Ziwei Liu, and Dahua Lin. Balanced chamfer distance as a comprehensive metric for point cloud completion. In NeurIPS, 2021.

[51] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. In NeurIPS, 2022.

[52] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In CVPR, 2015.

[53] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In CVPR, 2020.

[54] Saining Xie, Jiatao Gu, Demi Guo, Charles R Qi, Leonidas Guibas, and Or Litany. Pointcontrast: Unsupervised pretraining for 3d point cloud understanding. In ECCV, 2020.

[55] Chenfeng Xu, Shijia Yang, Tomer Galanti, Bichen Wu, Xiangyu Yue, Bohan Zhai, Wei Zhan, Peter Vajda, Kurt Keutzer, and Masayoshi Tomizuka. Image2point: 3d point-cloud understanding with 2d image pretrained models. arXiv preprint arXiv:2106.04180, 2021.

[56] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for region annotation in 3d shape collections. ToG, 2016.

[57] Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, and Jie Zhou. Pointr: Diverse point cloud completion with geometry-aware transformers. In ICCV, 2021.

[58] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In CVPR, 2022.

[59] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, 2022.

[60] Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, and Hongsheng Li. Point-m2ae: multi-scale masked autoencoders for hierarchical point cloud pre-training. arXiv preprint arXiv:2205.14401, 2022.

[61] Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3d representations from 2d pre-trained models via image-to-point masked autoencoders. In CVPR, 2023.