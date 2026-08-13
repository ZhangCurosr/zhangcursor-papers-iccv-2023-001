# Spatial Self-Distillation for Object Detection with Inaccurate Bounding Boxes

Di Wu<sup>\*</sup>, Pengfei Chen<sup>∗</sup>, Xuehui Yu<sup>∗</sup>, Guorong Li, Zhenjun Han<sup>†</sup>, Jianbin Jiao

University of Chinese Academy of Sciences

## Abstract

Object detection via inaccurate bounding boxes supervision has boosted a broad interest due to the expensive high-quality annotation data or the occasional inevitability of low annotation quality (e.g. tiny objects). The previous works usually utilize multiple instance learning (MIL), which highly depends on category information, to select and refine a low-quality box. Those methods suffer from object drift, group prediction and part domination problems without exploring spatial information. In this paper, we heuristically propose a Spatial Self-Distillation based Object Detector (SSD-Det) to mine spatial information to refine the inaccurate box in a self-distillation fashion. SSD-Det utilizes a Spatial Position Self-Distillation (SPSD) module to exploit spatial information and an interactive structure to combine spatial information and category information, thus constructing a high-quality proposal bag. To further improve the selection procedure, a Spatial Identity Self-Distillation (SISD) module is introduced in SSD-Det to obtain spatial confidence to help select the best proposals. Experiments on MS-COCO and VOC datasets with noisy box annotation verify our method’s effectiveness and achieve state-of-the-art performance. The code is available at https://github.com/ucas-vg/ PointTinyBenchmark/tree/SSD-Det.

## 1. Introduction

Object detection [18, 38, 58, 30, 57] relying on largescale datasets like MS-COCO[31] has significantly progressed and achieved good performance. However, accurate bounding box annotations are expensive and challenging in natural contexts [12]. Especially in many professional scenarios, it is difficult to label accurate annotations without domain knowledge (e.g., agricultural crop observation and medical image processing) [12, 1]. As shown in Fig. 1a in some complex datasets, the human annotators may also annotate inaccurate bounding boxes due to the inherent ambiguities[20] of objects. In addition, labelling with a detector or weak signal[11] (e.g., point) is much cheaper but brings more inaccuracy. Therefore learning robust detectors with inaccurate bounding boxes[12, 4, 35, 53, 25] is a practical and meaningful task and has boosted a broad interest.

![](images/e1a3438d7f08ea81ef9208196e2c720a8235798e7bcfb56b6d390ca044769e53.jpg)

![](images/639aed5040c32aeedd4c3b7af8b27cb8e2a5c609a5dfb5845d1ccdee79e8c284.jpg)  
(b) Three problems during previous MIL refining. Because their selections depend solely on classification, (i,ii): the refined box (green box) drifts to another object or makes a group prediction (merging across multiple objects) due to neighbor disturbance. Yellow boxes are proposals in the middle person’s MIL bag. (iii): Local part may be more discriminative than the entire object and will be predicted.  
Figure 1: The sources of inaccurate box annotations and three problems caused by previous refinement methods.

To use the inaccurate annotations, most related methods [12, 11] refine the inaccurate annotations as Fig. 1a shows, and then train a detector head or re-train a detector with the refined box as the new supervision. There are two main steps during refining: 1) Bag Construction: For each object, obtaining some proposals around the inaccurate annotated bounding box to form the object-level proposal bag; 2) Proposal Selection: Selecting the top-k proposals with the highest classification confidence from each bag and then weighting average them to obtain the refined box.

During the proposal selection, they usually utilize multiple instance learning (MIL) [10] supervised by category information to choose the proposals with high classification confidence from the constructed bags. However, they pay less [12] or no [11] attention to mining spatial information, leading to the following problems as shown in Fig. 1b: (1) Object Drift: For each object, some proposals in the constructed bag do not have a high IoU with the original object but with another nearby object. These proposals are not spatially adjacent to the original object but still have high classification confidence, as the rightmost (the yellow box) proposal of $O _ { 2 }$ shows in the left-bottom corner of Fig. 1b. Only the category confidence is relied on for selecting proposals for $O _ { 2 }$ , and the rightmost proposal will be selected as the refined box. It means the refined box drifts to another object (Fig. 1b (i)), reducing the recall; (2) Group Prediction: Most works [7, 11, 56] select the top-k proposals by classification confidence and then weight average them as the refined box, causing the group prediction problem, as shown in Fig. 1b (ii); (3) Part Domination: The detector often focuses on the object’s semantic region, which can statistically represent the category (e.g. the face). As shown in Fig. 1b (iii), the high classification confidence of the animal is in the discriminant part (the face) rather than the entire object as mentioned by [44, 39].

To address these problems, we propose a Spatial Self-Distillation based detector (SSD-Det) to integrate the spatial cues into the bounding box refinement. SSD-Det has two important components: the Spatial Position Self-Distillation (SPSD) module for the bag construction step and the Spatial Identity Self-Distillation (SISD) module for the proposal selection step. To construct high-quality proposal bags, SPSD utilizes a neighborhood sampler to generate a balanced and flexible initial proposal bag for each object and then trains a regressor with the supervision of the annotated inaccurate bounding boxes. Finally, high-quality proposal bags are constructed with proposals corrected by the regressor. The mechanism behind SPSD is that the network learns the spatial information from the reliable samples, e.g. those low-noise annotations, in the dataset and then guides the noisy samples to produce high-quality proposals, as shown in Fig. 2. In addition, to further combine the category information and the spatial information, an interactive structure is implemented by alternately using SPSD to mine spatial cues and MIL to utilize the category information. With SPSD and the interactive structure, a high-quality proposal bag can be constructed. Experiments on MS-COCO show that SPSD can significantly improve the mean/max IoU between objects and proposals (about 18/10 points, Fig. 5) in the constructed bag. Instead of selecting proposals by classification confidence, we have proposed the SISD module in the proposal selection step. We use it to obtain each proposal’s spatial confidence by predicting the IoU with the object and combining the IoU with classification confidence to select the top-k proposals. It is worth mentioning that SISD is an object-related IoU predictor, which means that the predicted IoU may be different for the same proposal that appears in different objects’ bags. Accordingly, it guarantees that SISD can better handle object drift and group prediction problems. Experiments on MS-COCO and VOC datasets verify the effectiveness of our method and bring state-of-the-art performance. The contributions are as follows:

![](images/85ad35061a51c3913fff2421db133daad6cf92085e5fdb06d997de5c7d3f1fee.jpg)  
Figure 2: The mechanism of Spatial Self-Distillation. By assigning higher weight, the low-noise annotations can be seen as reliable samples to guide the training of proposals spatial position and identity learning in SPSD and SISD.

1) We further investigate the inaccurate-box supervised object detection tasks and propose an end-to-end training SSD-Det that combines the spatial and the category information in an interactive fashion.

2) We utilize an SPSD module to generate higher-quality proposals sampling through statistic-guide spatial position distillation, raising the upper bound of the refinement.

3) To add spatial cues to classification confidence, we also introduce an SISD module to select a proposal belonging to the object rather than the category.

4) The performance of our proposed SSD-Det improves the mean average precision (AP) of the best previous method (e.g. over 10 AP on 40% noisy MS-COCO) and achieves state-of-the-art under various noise rate box supervision on MS-COCO and VOC datasets.

## 2. Related Work

## 2.1. Object Detection

Classic object detection [15, 38, 37, 32, 30, 3, 43, 57] is supervised by an accurate bounding-box. One-stage detectors utilize anchors as the sliding-window, such as YOLO [37], SSD [32], and RetinaNet [30]. Two-stage detectors mine spatial information to predict proposals (e.g. selective search [47] in Fast R-CNN [15] or RPN in Faster R-CNN [38]) and conduct classification and bounding-box regression with filtered proposals sparsely. Transformerbased (i.e. DETR [3], Deformable-DETR [62], and Swin-Transformer [33]) detectors utilize global information for better representation. Sparse R-CNN [43] combines a transformer’s advantages and CNNs for detection.

## 2.2. Weakly-Supervised Object Detection (WSOD)

WSOD trains object detectors with image tag supervision. Only with the category annotation, the majority of previous methods treat each image as a bag and candidate proposals as instances. They follow the multiple instance learning (MIL) pipeline [1, 44, 46, 7, 48], which highly depends on category information. However, the MIL loss function leads to a non-convex optimization problem; thus, MIL solutions are usually stuck into the local minima. Context information [24, 51], spatial regularization [1, 9, 48], and optimization strategy [44, 48, 46] are proposed to address the problems. SPE [28] introduces Transformer into WSOD and uses attention to generate proposals. SD-LocNet [59] tackles the initialized noisy object locations in WSOD and proposes a self-directed localization network to identify the noisy object instances. [44, 46, 7] use the pseudo label for classification’s iterative refinement. However, we use the pseudo box as a better self-distillation teacher. [54, 39] conduct regression to move the proposals, whereas we conduct regression to distill for better bag construction. In this work, we also formulate box correction as a MIL problem.

## 2.3. Semi-Supervised Object Detection (SSOD)

Semi-supervised learning in object detection can be roughly categorized into two groups: consistency based [22, 45] and pseudo label based [27, 36, 41, 49, 64, 52]. [52] presents an end-to-end SSOD approach with two simple techniques named soft teacher and box jittering to facilitate the efficient leverage of the teacher model. Both Soft Teacher [52] and SSD-Det obtain pseudo-labels from candidate boxes, adopting distillation, box jitter and classification scores weighting policy. However, they are different: 1) SSOD selects candidate boxes from the teacher model’s detection results, while SSD-Det generates them with a generative approach, due to no any accurate supervised data for ensuring high-quality boxes in the detection results. 2) [52] is based on FixMatch [40] and requires maintaining two networks for teacher-student distillation structure. In contrast, our approach only needs a multi-head detector with a shared backbone for self-distillation. 3) Box Jitter: [52] calculates box variance from jittering for result selection, while we aim to generate candidate boxes that combine SISD-predicted IoU and classification scores, selecting boxes closer to the ground truth. 4) We use classification scores as weights for the next stage’s loss, while [52] employs them as a criterion for selecting reliable samples.

## 2.4. Learning with Noisy Annotations

Training CNNs under noisy labels has been an active research area. Previous research focuses on the classification task, and develops various techniques to deal with noisy labels, such as sample selection [17, 23] for training, label correction [42, 34], and robust loss functions [60, 14] against noisy labels. Recently, many efforts [4, 25, 35, 53, 11, 12] have been devoted to the object detection task. On the one hand, Simon et al. [4] first investigates the impact of different types of label noise on object detection. They propose a per-object co-teaching method to alleviate the effect of noisy labels. On the other hand, [53] proposes a meta-learning framework for noisy annotations consisting of noisy category labels and bounding boxes. [11, 12] utilize object-level MIL to refine the inaccurate box. OA-MIL [12] constructs proposal bags through label assignment in a discriminant style. P2BNet [11] originally conducts point-supervised object detection tasks. However, it can be seen as the box correction in its refinement stage. It uses hand-craft anchors to generate proposal bags. Our method inherits the generative style of P2BNet and conducts spatial distillation to mine spatial information.

## 2.5. Knowledge Distillation

Knowledge distillation (KD) [21] aims to learn compact and efficient student models guided by excellent teacher networks. It is first applied to object detection in [5], in which hint learning and KD are both used for multi-class object detection. Recently, many efforts [26, 50, 8, 16, 55] aim to mimic the feature. [61] shows that localization knowledge is more important and proposed a localization distillation method. We also transfer the spatial knowledge from reliable labeled instances to correct inaccurate bounding boxes (shown in Fig. 2) in a self-distillation manner.

## 3. Methodology

This work aims to learn a robust detector with inaccurate bounding boxes. Instead of training a detector with the original inaccurate bounding box, we follow most related works [11, 12] that design a branch to refine the inaccurate bounding box and then train the detector head or detector with the refined bounding box. The most important part is how to design a refining policy. We first design a two-stage basic box refiner (gray region in Fig. 3) as a naive solution that modified from [11]. Then, SPSD and SISD are proposed and added to further mine the spatial cues for box refinement, yielding SSD-Det. Therefore, the overall loss function is formulated as:

$$
\mathcal { L } = \mathcal { L } _ { B a s i c } + \alpha _ { 1 } \cdot \mathcal { L } _ { S P S D } + \alpha _ { 2 } \cdot \mathcal { L } _ { S I S D } + \alpha _ { 3 } \cdot \mathcal { L } _ { D e t }\tag{1}
$$

where $\alpha _ { 1 } , \alpha _ { 2 }$ and $\alpha _ { 3 }$ are set as 0.25, 0.25 and 4 respectively. $\mathcal { L } _ { D e t }$ denotes the loss of detector or detection head. During inference, only the detector or the detection head is used.

![](images/16ad4b3770b00d129314413ba06cce60d02231b027caf29642daf0f8703644c6.jpg)  
Figure 3: The framework of SSD-Det. It contains basic box refiner, SPSD module, SISD module and a detector head. Neighborhood sampler is adopted around the inaccurate annotation. Then, SPSD module generates better proposal bags which are fed into basic box refiner for MIL training. The selected proposals are average weighted as the refined box and supervise the next SPSD training. Meanwhile, the SISD module predicts the IoU between proposals and the object, and the estimated IoU is multiplied by classification score for better proposal selection to generate the refined box. SPSD shares backbone with the detector.

## 3.1. Basic Box Refiner

Motivated by [12] and WSOD [1], we design the basic box refiner (detailed structure figure is in supplementary) that leverages classification confidence to refine the inaccurate box annotation. Then the refine annotation is used to train a detection head or detector. Following [11], we design a two-stream structure as a MIL classifier to select the best proposal for box refinement.

Giving an image with inaccurate box annotation, for each object, B is a bag of proposals (bounding boxes) that are generated around its inaccurate annotation by a sampler policy (e.g., selective search[47], edge box[63], neighborhood sampler in Sec. 3.2). Meanwhile a feature map is extracted with a backbone network. And then through $7 \times 7$ RoIAlign [18] and two fully connected (fc) layers, features of proposal in B are extracted and denote as F. The basic box refiner takes proposal bag $\boldsymbol { B } \in \mathbb { R } ^ { P \times 4 }$ and features $\mathbf { F } \in \mathbb { R } ^ { P \times D }$ as inputs, where P, D are denoted as the number of proposals in B, feature dimension respectively.

Following [11] and [1], as Eq. 2 described, we apply the classification branch $f _ { c l s }$ to F yields $\mathbf { O } ^ { c l s }$ , which is then passed through the softmax function over classification dimension K to obtain the score $\mathbf { S } ^ { c l s } ~ \in ~ \mathbb { R } ^ { P \times K }$ , where K represents the number of instance categories. Likewise, instance selection branch $f _ { c l s }$ is applied to F to yield ${ \bf O } ^ { i n s }$ and instance score $\mathbf { S } ^ { i n s }$ is obtained through softmax function over P proposals. The proposal score S is obtained by computing the Hadamard product of the classification score and the instance score. The bag score $\widehat { \mathbf { S } }$ is obtained by the summating of the P proposal boxes’ proposal scores.

$$
\begin{array} { l } { { \displaystyle { \bf O } ^ { c l s } = f _ { c l s } ( { \bf F } ) \in { \mathbb R } ^ { P \times K } ; [ { \bf S } ^ { c l s } ] _ { p k } = e ^ { [ { \bf O } ^ { c l s } ] _ { p k } } \Big / \sum _ { k = 1 } ^ { K } e ^ { [ { \bf O } ^ { c l s } ] _ { p k } } . } \ ~ } \\ { { \displaystyle } } \\ { { \displaystyle { \bf O } ^ { i n s } = f _ { i n s } ( { \bf F } ) \in { \mathbb R } ^ { P \times K } ; [ { \bf S } ^ { i n s } ] _ { p k } = e ^ { [ { \bf O } ^ { i n s } ] _ { p k } } \Big / \sum _ { p = 1 } ^ { P } e ^ { [ { \bf O } ^ { i n s } ] _ { p k } } . } \ ~ } \\ { { \displaystyle } } \\ { { \displaystyle { \bf S } = { \bf S } ^ { c l s } \odot { \bf S } ^ { i n s } \in { \mathbb R } ^ { P \times K } ; \widehat { \bf S } = \sum _ { p = 1 } ^ { P } [ { \bf S } ] _ { p } \in { \mathbb R } ^ { K } } . } \end{array}
$$

where $[ \cdot ] _ { p k }$ is the value at row $p$ and column k in the matrix.

(2)

The basic box refiner has two similar stages. The loss of stage I(termed $\mathcal { L } _ { I } )$ adopt the MIL paradigm with the form of cross-entropy (CE) loss, defined as:

$$
\mathcal { L } _ { I } = C E ( \widehat { \mathbf { S } } , \mathbf { c } ) = - \sum _ { k = 1 } ^ { K } \mathbf { c } _ { k } \log ( \widehat { \mathbf { S } } _ { k } ) + ( 1 - \mathbf { c } _ { k } ) \log ( 1 - \widehat { \mathbf { S } } _ { k } )\tag{3}
$$

where $\mathbf { c } \in \{ 0 , 1 \} ^ { K }$ is the one-hot category label. And each object’s proposals with the top-k highest proposal score S are weighted to obtain the refined box.

The stage II takes the refined box of stage I as input and performs fine refining with a similar structure as stage I. Differently, the focal loss is adopted in stage II instead of cross entropy loss. In order to cooperate with focal loss, the classification branch uses the sigmoid $\sigma ( x )$ instead of sof tmax function and we sample some negative samples $\mathcal { N }$ to further suppress the background. With the bag score $\widehat { S }$ and the negative sample scores $\mathbf { S } _ { n e g } ^ { c l s }$ , the loss is:

$$
\mathcal { L } _ { I I } = \left. \mathbf { c } ^ { \mathrm { T } } , \widehat { \mathbf { S } } ^ { * } \right. \cdot \mathrm { F L } ( \widehat { \mathbf { S } } , \mathbf { c } ) + \sum _ { \mathcal { N } } \beta \cdot \mathrm { F L } ( \mathbf { S } _ { n e g } ^ { c l s } , c _ { n e g } ) \left. \right\}\tag{4}
$$

where FL is the focal loss [30], $\widehat { \mathbf { S } } _ { j } ^ { * }$ represents the bag score predicted by stage I. $\left. \mathbf { c } _ { j } ^ { \mathrm { { T } } } , \widehat { \mathbf { S } } _ { j } ^ { * } \right.$ represents the inner product of the two vectors, meaning the predicted bag score of the ground-truth category. $\beta$ is the average of $\left. \mathbf { c } _ { j } ^ { \mathrm { { T } } } , \widehat { \mathbf { S } } _ { j } ^ { * } \right.$ . They are used to weight each object’s FL for stable training. The overall loss function of the basic refiner here is:

$$
\mathcal { L } _ { B a s i c } = \mathcal { L } _ { I } + \alpha _ { I I } \cdot \mathcal { L } _ { I I }\tag{5}
$$

where $\alpha _ { I I }$ are the loss weights of the two stages.

During training, the refined box of stage II is used as supervision for a detection head or detector. After training, the basic box refiner will be removed, leaving a well-trained detection head or detector. In this way, we can train a detector under inaccurate annotations.

## 3.2. Spatial Position Self-Distillation (SPSD)

Like most MIL paradigm methods, basic box refiner has two main components: bag construction and proposal selection. And the main idea is to use classification information to guide the refining. In this paper, we add spatial information to improve refining. Specifically, SPSD is proposed to use spatial information to enhance bag construction.

Bag construction aims to obtain proposals for each object, while proposal selection is to select the proposals from the object bag. Then, the refined box is averaged over the selected proposals. Therefore, the quality of the proposals in constructed bag determines the upbound of refining. The bag construction can be implemented in a variety of ways. In this paper, the basic box refiner adopts a naive neighborhood sampler for bag construction. basic box refiner adopts a naive neighborhood sampler for bag construction.

Neighborhood Sampler. Proposals around the inaccurate box are sampled to construct an object bag. For each inaccurate box $b ^ { * } ~ = ~ ( b _ { x } ^ { * } , b _ { y } ^ { * } , b _ { w } ^ { * } , b _ { h } ^ { * } )$ , its scale and aspect ratio with s and v are adjusted and its positions $o _ { x } , o _ { y }$ are jittered to obtain the diverse proposal $\boldsymbol { b } = ( b _ { x } , b _ { y } , b _ { w } , b _ { h } ) \colon$

$$
\begin{array} { r } { b _ { w } = v \cdot s \cdot b _ { w } ^ { * } , \quad b _ { h } = 1 / v \cdot s \cdot b _ { h } ^ { * } , } \\ { b _ { x } = b _ { x } ^ { * } + b _ { w } \cdot o _ { x } , \quad b _ { y } = b _ { y } ^ { * } + b _ { h } \cdot o _ { y } . } \end{array}\tag{6}
$$

These proposals b are used to construct the positive proposal bag B to train the MIL classifier. Thanks to the handcraft sampling way, the number of proposals in different objects’ proposal bags is controllable and balanced. However, the hand-craft neighborhood sampler strategy is difficult to set hyper-parameters, and the sampling space is discrete. For example, when the jitter region is small, the optimization space of refining is limited, while when it is large, more background will be introduced. Hence, we propose the SPSD module to mine spatial information for higherquality proposal bag construction.

Statistically Guided Adaptive Sampling. Instead of simply using a neighborhood sampler, we adopt a statistically guided adaptive sampling by adding SPSD modules into the basic box refiner. Taking the constructed proposal bag B of hand-craft neighborhood sampler as input, the

RoI features of proposals in B are extracted and fed into the two shared fc layers to obtain F. Then a regression fc layer $f _ { d i s }$ , supervised by the inaccurate annotated bounding box $b ^ { * }$ , is introduced to predict the adaptive proposal bag $B ^ { d i s } = f _ { d i s } ( \mathbf { F } ) \in \mathbb { R } ^ { P \times { 4 } }$ , in which the proposals are closer to the object. Later, $B ^ { d i s }$ as the constructed proposal bag is fed into stage I of basic box refiner. In order to combine category and spatial information, we implement an interactive structure by alternately using SPSD to mine the spatial cues and using MIL in basic box refiner to utilize the category information. Specifically, the refined bounding box $\hat { b ^ { * } }$ of stage I that selected by the classification confidence is used to supervision of a new SPSD module for stage II. Similar as stage I, The new SPSD takes proposal bag $\bar { B } ^ { d i s }$ of handcraft neighborhood sampler as input. Through the RoI align and the two shared fc layers, the feature F<sup>ˆ</sup> is extracted. An extra fc layer $\hat { f _ { d i s } }$ is then utilized to conduct further regression. Different with stage I, the obtained $B ^ { \hat { d } i s }$ is supervised by the refined $\hat { b ^ { * } }$ . The loss function of the spatial distillation for adaptive sampling can be defined as $\mathcal { L } _ { S P S D }$ in Eq. 7.

$$
\mathcal { L } _ { S P S D } = \frac { 1 } { P } \big \{ \sum _ { p = 1 } ^ { P } \mathbf { L _ { 1 } } ( [ \mathcal { B } ^ { d i s } ] _ { p } , b ^ { * } ) + \sum _ { p = 1 } ^ { P } \mathbf { L _ { 1 } } ( [ \mathcal { B } ^ { \hat { d } i s } ] _ { p } , \hat { b ^ { * } } ) \big \}\tag{7}
$$

where the $\mathbf { L _ { 1 } }$ is the L1 loss function for loose restrictions.

The idea behind SPSD is that the dataset with inaccurate annotation still has many reliable, high-quality boxes and inaccurate boxes. Supervised by the high-quality boxes statistically, the network can guide those proposals sampled around the inaccurate bounding box to regress to the ground truth. With the self-distillation mechanism, SPSD learns the semantic-spatial correspondence knowledge from the reliable samples in the dataset and then propagates the knowledge to produce high-quality proposals.

Adaptive Negative Sampling. Negative samples are introduced in Stage II to better suppress the background. With the sampled $B ^ { \hat { d } i s }$ , we can adaptively sample the negative samples with a small IoU (set smaller than 0.3 by default) with all positive proposals in all bags, to compose the negative sample set $\mathcal { N }$ for stage II.

## 3.3. Spatial Identity Self-Distillation (SISD)

The basic box refiner selects the proposals only depending on classification confidence during proposal selection. To select the proposal which has high classification confi dence and is also spatially close to the object from the bag, we propose a SISD module to predict the IoU between proposals and their corresponding object. Afterwards, through the combination between the IoU and the classification confidence, top-k proposals are selected. In SISD, we design an Object Relevance Enhancement (ORE) module to distinguish different objects’ features with the same RoI region. And an identity predictor is designed to predict each proposal’s IoU with the object.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">AP  $A P _ { 5 0 }$ </td><td colspan="5">20% Box Noise Level</td><td rowspan="2"></td><td colspan="6">40% Box Noise Level</td></tr><tr><td> $A P _ { 7 5 }$ </td><td> $A P ^ { s }$ </td><td></td><td> $A P ^ { l }$ </td><td>AP</td><td> $A P _ { 5 0 }$ </td><td>AP75</td><td> $A P ^ { s }$ </td><td> $A P ^ { m }$ </td><td> $A P ^ { l }$ </td></tr><tr><td colspan="10">Val Set</td><td colspan="7"></td></tr><tr><td>Clean-FasterRCNN [38]</td><td>ResNet-50</td><td>37.9</td><td>58.1</td><td>40.9</td><td>21.6</td><td>41.6</td><td></td><td>48.7</td><td>37.9</td><td>58.1</td><td>40.9</td><td>21.6</td><td>41.6</td><td>48.7</td></tr><tr><td>Clean-FasterRCNN [38]</td><td>ResNet-101</td><td>39.4</td><td>60.1</td><td>43.1</td><td>22.4</td><td>43.7</td><td>51.1</td><td>39.4</td><td></td><td>60.1</td><td>43.1</td><td>22.4</td><td>43.7</td><td>51.1</td></tr><tr><td>Clean-Retinanet [30]</td><td>ResNet-50</td><td>36.7</td><td>56.1</td><td>39.0</td><td>21.6</td><td>40.4</td><td>47.4</td><td>36.7</td><td></td><td>56.1</td><td>39.0</td><td>21.6</td><td>40.4</td><td>47.4</td></tr><tr><td>Noisy-FasterRCNN [38]</td><td>ResNet-50</td><td>30.4</td><td>54.3</td><td>31.4</td><td>17.4</td><td>33.9</td><td>38.7</td><td></td><td>10.3</td><td>28.9</td><td>3.3</td><td>5.7</td><td>11.8</td><td>15.1</td></tr><tr><td>Noisy-Retinanet [30]</td><td>ResNet-50</td><td>30.0</td><td>53.1</td><td>30.8</td><td>17.9</td><td>33.7</td><td>38.2</td><td></td><td>13.3</td><td>33.6</td><td>5.7</td><td>8.4</td><td>15.9</td><td>18.0</td></tr><tr><td>FreeAnchor[58]</td><td>ResNet-50</td><td>28.6</td><td>53.1</td><td>28.5</td><td>16.6</td><td>32.2</td><td>37.0</td><td>10.4</td><td></td><td>28.9</td><td>3.3</td><td>5.8</td><td>12.1</td><td>14.9</td></tr><tr><td>Co-teaching[17]</td><td>ResNet-50</td><td>30.5</td><td>54.9</td><td>30.5</td><td>17.3</td><td>34.0</td><td>39.1</td><td>11.5</td><td></td><td>31.4</td><td>4.2</td><td>6.4</td><td>13.1</td><td>16.4</td></tr><tr><td>SD-LocNet[59]</td><td>ResNet-50</td><td>30.0</td><td>54.5</td><td>30.3</td><td>17.5</td><td>33.6</td><td>38.7</td><td></td><td>11.3</td><td>30.3</td><td>4.3</td><td>6.0</td><td>12.7</td><td>16.6</td></tr><tr><td>KL loss[20]</td><td>ResNet-50</td><td>31.0</td><td>54.3</td><td>32.4</td><td>18.0</td><td>34.9</td><td>39.5</td><td>12.1</td><td></td><td>36.7</td><td>3.7</td><td>6.2</td><td>13.0</td><td>17.4</td></tr><tr><td>OA-MIL[12]</td><td>ResNet-50</td><td>32.1</td><td>55.3</td><td>33.2</td><td>18.1</td><td>35.8</td><td>41.6</td><td>18.6</td><td></td><td>42.6</td><td>12.9</td><td>9.2</td><td>19.9</td><td>26.5</td></tr><tr><td>SSD-Det</td><td>ResNet-50</td><td>33.6</td><td>57.3</td><td>35.3</td><td>19.5</td><td>37.2</td><td>43.3</td><td></td><td>27.6</td><td>53.9</td><td>26.0</td><td>16.0</td><td>31.0</td><td>34.9</td></tr><tr><td>SSD-Det</td><td>ResNet-101</td><td>34.3</td><td>57.6</td><td>36.7</td><td>19.1</td><td>38.1</td><td>44.3</td><td></td><td>28.4</td><td>54.3</td><td>27.2</td><td>16.5</td><td>31.9</td><td>36.4</td></tr><tr><td>SSD-Det+FR</td><td>ResNet-50</td><td>34.4</td><td>57.3</td><td>36.8</td><td>20.0</td><td>38.2</td><td>44.0</td><td></td><td>29.3</td><td>54.8</td><td>29.0</td><td>17.1</td><td>32.9</td><td>36.9</td></tr><tr><td>SSD-Det+FR</td><td>ResNet-101</td><td>36.2</td><td>59.1</td><td>39.2</td><td>20.9</td><td>40.2</td><td>47.1</td><td>30.6</td><td></td><td>56.7</td><td>30.7</td><td>18.1</td><td>34.5</td><td>39.0</td></tr><tr><td colspan="10">Test Set</td><td colspan="7"></td></tr><tr><td>Clean-FasterRCNN [38]</td><td>ResNet-50</td><td>37.7</td><td>58.7</td><td>40.8</td><td>21.7</td><td>40.6</td><td></td><td>46.7</td><td>37.7</td><td>58.7</td><td>40.8</td><td>21.7</td><td>40.6</td><td>46.7</td></tr><tr><td>Noisy-FasterRCNN [38]</td><td>ResNet-50</td><td>30.7</td><td>54.9</td><td>31.3</td><td>18.0</td><td>33.7</td><td></td><td>37.7</td><td>10.4</td><td>29.0</td><td>3.3</td><td>6.0</td><td>11.3</td><td>14.6</td></tr><tr><td>OA-MIL[12]</td><td>ResNet-50</td><td>32.3</td><td>55.8</td><td>33.7</td><td>18.5</td><td>35.0</td><td>40.2</td><td></td><td>18.5</td><td>42.3</td><td>12.8</td><td>9.3</td><td>19.1</td><td>25.1</td></tr><tr><td>SSD-Det</td><td>ResNet-50</td><td>33.5</td><td>57.3</td><td>35.5</td><td>19.1</td><td>36.0</td><td>41.9</td><td></td><td>28.0</td><td>54.1</td><td>26.5</td><td>16.5</td><td>30.0</td><td>34.5</td></tr><tr><td>SSD-Det+FR</td><td>ResNet-50</td><td>34.7</td><td>57.9</td><td>37.2</td><td>20.0</td><td>37.7</td><td>42.7</td><td></td><td>29.7</td><td>55.6</td><td>29.3</td><td>17.5</td><td>32.4</td><td>36.2</td></tr></table>

Table 1: Performance comparison on COCO. FR is Faster R-CNN. \*-FR refers to a retrained Faster R-CNN (R50+FPN) using refined annotations from SSD-Det for improved performance.. Clean-\* and $\mathrm { N o i s y \mathrm { - } ^ { \mathrm { * } } }$ means original and noisy annotation.

Object Relevance Enhancement (ORE). ORE enhances object-relevant features, making SISD an objectrelevant IoU predictor. ORE allows the predicted IoU to be different for the same proposal in other objects’ bags. In addition, we integrate the feature of the bag’s corresponding object into the proposal feature, making the feature of different bags’ proposals distinct. That is the so-called ORE. For a proposal bag B, the feature F is obtained through the RoI align and two fc layers. It is worth mentioning that the two fcs do not share the parameters with those in the refiner since the optimization goals are contradictory. To represent the feature of the bag’s corresponding object, $\mathbf { F } ^ { + } \in \mathbb { R } ^ { 1 \times D }$ is calculated by averaging features of $P$ proposals in proposal bag B. The object feature $\mathbf { F } ^ { + }$ is broadcast into $\mathbb { R } ^ { P \times D }$ and then added to the proposal features to obtain the objectrelevant features $\mathbf { F } ^ { * } = \mathbf { F } + \mathbf { F } ^ { + }$

Spatial Identity Prediction. By a following identity fc layer, $U \in \mathbb { R } ^ { P \times 1 }$ is predicted. The pseudo label $T \in ( 0 , 1 )$ is IoU between proposals in $\boldsymbol { B }$ and the merged box $\hat { b ^ { * } }$ of stage I. For better optimization, the linear normalized $T ^ { \prime } =$ $( T - 0 . 5 ) / 0 . 5 \in ( - 1 , 1 )$ is utilized as supervision. The object function of the identity predictor is identified in:

$$
\mathcal { L } _ { S I S D } = s m o o t h _ { L 1 } ( U , T ^ { \prime } )\tag{8}
$$

where the $s m o o t h _ { L 1 }$ represents the smooth L1 loss. The predicted spatial confidence $U ^ { \prime }$ is obtained by normalizing the U. Finally, $S ^ { * } = U ^ { \prime } \cdot S$ is used to select the top-k proposals for merging as the refined boxes.

<table><tr><td>Method</td><td>Backbone</td><td colspan="3">Box Noise Level 10% 20% 30% 40%</td></tr><tr><td>Clean-FasterRCNN [38]</td><td>ResNet-50</td><td colspan="3">77.2 for clean</td></tr><tr><td>Clean-RetinaNet [30] Noisy-FasterRCNN [38]</td><td>ResNet-50 ResNet-50</td><td colspan="3">73.5 for clean</td></tr><tr><td>Noisy-RetinaNet [30]</td><td>ResNet-50</td><td>76.3 71.2 71.5 67.6</td><td>60.1 57.9</td><td>42.5 45.0</td></tr><tr><td>KL loss[20]</td><td>ResNet-50</td><td>75.8 72.7</td><td>64.6</td><td>48.6</td></tr><tr><td>Co-teaching[17]</td><td>ResNet-50</td><td>75.4 70.6</td><td>60.9</td><td>43.7</td></tr><tr><td>SD-LocNet[59]</td><td>ResNet-50</td><td>75.7 71.5</td><td>60.8</td><td>43.9</td></tr><tr><td>FreeAnchor[58]</td><td>ResNet-50</td><td>73.0 67.5</td><td>56.2</td><td>41.6</td></tr><tr><td>OA-MIL[12]</td><td>ResNet-50</td><td>77.4 74.3</td><td>70.6</td><td>63.8</td></tr><tr><td>SSD-Det</td><td>ResNet-50</td><td>77.1 74.8</td><td>71.5</td><td>66.9</td></tr></table>

Table 2: Performance comparison on the VOC 2007 test set. The evaluation metric is $A P _ { 5 0 }$ . The Clean-\* and Noisy-\* means original annotation and noisy annotation.

## 4. Experiment

## 4.1. Experimental Settings

Datasets and Evaluation Metrics. For experimental comparisons, two publicly available datasets are used for object detection with inaccurate bounding boxes: MS-COCO [31] and PASCAL VOC 2007 [13]. MS-COCO (2017 version) has 118k training and 5k validation images with 80 common object categories. PASCAL VOC 2007 is one of the most popular benchmarks in generic object detection with 20 classes.

Evaluation Metric. We use mean average precision

<table><tr><td rowspan="2">2-Ref</td><td rowspan="2">SPSD SISD Re-Train</td><td colspan="7">20% Box Noise Level</td><td colspan="7"></td><td rowspan="2"></td><td rowspan="2"> $\underline { { A P _ { 7 5 } ^ { t e s t } } }$ </td></tr><tr><td> $A P$ </td><td> $A P _ { 5 0 }$ </td><td> $A P _ { 7 5 }$ </td><td> $A P ^ { s }$ </td><td> $A P ^ { m }$ </td><td> $A P ^ { l }$ </td><td> $A P ^ { t e s t }$ </td><td> $A P _ { 7 5 } ^ { t e s t }$ </td><td> $A P$ </td><td> $A P _ { 5 0 }$ </td><td> $A P _ { 7 5 }$ </td><td> $A P ^ { s }$ </td><td> $A P ^ { m }$ </td><td> $A P ^ { l }$   $A P ^ { t e s t }$ </td></tr><tr><td></td><td></td><td>30.0</td><td>57.1</td><td>29.0</td><td>16.9</td><td>33.1</td><td>39.8</td><td></td><td></td><td>22.8</td><td>51.1</td><td>16.1</td><td>13.3</td><td>25.0</td><td>30.4</td><td></td><td></td></tr><tr><td>√</td><td></td><td>31.2</td><td>56.7</td><td>31.6</td><td>17.8</td><td>34.5</td><td>41.0</td><td>31.4</td><td>32.0</td><td>24.6</td><td>52.0</td><td>20.1</td><td>14.3</td><td>28.2</td><td>31.9</td><td>25.0</td><td>20.5</td></tr><tr><td>√</td><td>√</td><td>33.0</td><td>56.9</td><td>34.8</td><td>18.7</td><td>35.5</td><td>42.2</td><td>33.1</td><td>34.8</td><td>27.2</td><td>53.7</td><td>24.7</td><td>15.9</td><td>30.3</td><td>35.2</td><td>27.6</td><td>25.6</td></tr><tr><td>√</td><td>√ √</td><td>33.6</td><td>57.3</td><td>35.3</td><td>19.5</td><td>37.2</td><td>43.3</td><td>33.5</td><td>35.5</td><td>27.6</td><td>53.9</td><td>26.0</td><td>16.0</td><td>31.0</td><td>34.9</td><td>28.0</td><td>26.5</td></tr><tr><td>√</td><td></td><td>31.8</td><td>56.8</td><td>33.1</td><td>18.4</td><td>35.7</td><td>40.8</td><td>32.3</td><td>33.7</td><td>26.5</td><td>54.0</td><td>23.3</td><td>15.7</td><td>30.3</td><td>33.8</td><td>26.8</td><td>23.3</td></tr><tr><td>」</td><td>√</td><td>34.1</td><td>57.6</td><td>36.4</td><td>19.0</td><td>37.7</td><td>43.8</td><td>34.3</td><td>36.6</td><td>29.0</td><td>55.1</td><td>27.8</td><td>17.0</td><td>32.5</td><td>36.7</td><td>29.3</td><td>28.4</td></tr><tr><td>√</td><td>√</td><td>34.4</td><td>57.3</td><td>36.8</td><td>20.0</td><td>38.2</td><td>44.0</td><td>34.7</td><td>37.2</td><td>29.3</td><td>54.8</td><td>29.0</td><td>17.1</td><td>32.9</td><td>36.9</td><td>29.7</td><td>29.3</td></tr></table>

Table 3: Modules ablation of SPSD, SISD and Re-Train on MS-COCO validation set (without) and test set (with test). The Re-Train means we generate the pseudo label by SSD-Det and re-train a Faster R-CNN detector.

<table><tr><td>Methods (w/o SISD)</td><td>AP</td><td>AP50</td><td> $\overline { { \mathrm { A P } _ { 7 5 } } }$ </td><td>APs</td><td> $\overline { { \mathsf { A P } ^ { m } } }$ </td><td> $\overline { { \mathsf { A P } ^ { l } } }$ </td></tr><tr><td>Neighborhood Sampler</td><td>24.6</td><td>52.0</td><td>20.1</td><td>14.3</td><td>28.2</td><td>31.9</td></tr><tr><td>SPSD (II) w/o weighted</td><td>26.0</td><td>53.3</td><td>22.5</td><td>15.6</td><td>29.4</td><td>33.4</td></tr><tr><td>SPSD (II) w/ weighted</td><td>26.3</td><td>53.4</td><td>22.5</td><td>15.6</td><td>29.3</td><td>33.8</td></tr><tr><td>SPSD (I+II) w/ weighted</td><td>27.2</td><td>53.7</td><td>24.7</td><td>15.9</td><td>30.3</td><td>35.2</td></tr></table>

Table 4: Different setting of SPSD.

<table><tr><td>ORE Strategies of SISD</td><td>AP</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathrm { A P _ { 7 5 } }$ </td><td>APs</td><td> $\overline { { \mathsf { A P } ^ { m } } }$ </td><td> $\overline { { \mathsf { A P } ^ { l } } }$ </td></tr><tr><td>w/o SISD</td><td>27.2</td><td>53.7</td><td>24.7</td><td>15.9</td><td>30.3</td><td>35.2</td></tr><tr><td>SISD  $w / o \mathrm { O R E }$ </td><td>27.3</td><td>53.3</td><td>25.7</td><td>16.9</td><td>30.2</td><td>35.0</td></tr><tr><td>+ subtract</td><td>27.2</td><td>53.8</td><td>24.6</td><td>15.7</td><td>30.4</td><td>35.0</td></tr><tr><td>+ concatenate</td><td>27.2</td><td>54.0</td><td>24.5</td><td>16.1</td><td>30.2</td><td>34.8</td></tr><tr><td>+ add</td><td>27.6</td><td>53.9</td><td>26.0</td><td>16.0</td><td>31.0</td><td>34.9</td></tr><tr><td>+ add w/ shared fcs</td><td>23.0</td><td>49.9</td><td>17.6</td><td>12.6</td><td>25.3</td><td>30.4</td></tr></table>

Table 5: Different ORE strategies of SISD.

<table><tr><td rowspan=1 colspan=1>Num.</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { A P } / \mathrm { A P } _ { 5 0 } } }$ </td><td rowspan=1 colspan=1>24.6 / 52.0</td><td rowspan=1 colspan=1>26.3 / 53.4</td><td rowspan=1 colspan=1>27.2 / 53.7</td><td rowspan=1 colspan=1> $\overline { { 2 7 . 0 / 5 3 . 1 } }$ </td></tr></table>

## Table 6: Number of SPSD module.

mAP@[.5,.95] and $( \mathrm { m A P } @ . 5 )$ for MS-COCO and VOC. The $\left\{ A P , A P _ { 5 0 } , A P _ { 7 5 } , A P ^ { S m a l l } , A P ^ { M i d d l e } , A P ^ { L a r g e } \right\}$ is reported for MS-COCO and $A P _ { 5 0 }$ for VOC.

Synthetic Noisy Dataset. Following [12], We simulate noisy bounding boxes by perturbing clean boxes from the original annotations. The details are in the appendix. We simulate various box noise levels ranging from 10% to 40% for the VOC and {20%, 40%} for the MS-COCO.

Implementation Details. We implement our method on FasterRCNN [38] with ResNet50-FPN [19, 29] backbone, based on MMDetection [6]. All settings of our method and previous methods employ FPN for fair comparison. Similar to the default setting of object detection on MS-COCO, the stochastic gradient descent [2] algorithm is used to optimize on 1x training schedule. The batch size is two images per GPU on 8 GPUS. For the VOC dataset, the batch size is two images per GPU on 2 GPUS. The performance we report is on a single scale $( 1 3 3 3 ~ ^ { * } ~ 8 0 0$ for MS-COCO and 1000 \* 600 for VOC).

## 4.2. Comparison with State-of-the-Art

We compare our method with several state-of-the-art approaches [20, 17, 59, 58, 12] on MS-COCO and VOC 2007 datasets. We denote Clean-FasterRCNN and Noisy-FasterRCNN as FasterRCNN models trained under clean (original annotations) and noisy annotations with the default setting, respectively.

MS-COCO Dataset. Table 1 shows the comparison results on the MS-COCO. Inaccurate bounding box annotations significantly deteriorate the vanilla Faster R-CNN’s detection performance. Co-teaching and SD-LocNet only slightly improve the detection performance, especially under 40% box noise. That indicates that small-loss sample selection and sample weight assignment can not tackle noisy box annotations well. KL Loss slightly improves the performance under 20% and 40% box noise. By treating an object as a bag of instances, OA-MIL is somehow robust to noisy bounding boxes and performs better than other methods. Nevertheless, the previously-mentioned label assignment bag construction limits its ability to handle heavy noise. Our approach is more robust to noisy bounding boxes. It outperforms other methods by a large margin under high box noise levels and significantly boosts the detection performance across all metrics. For example, under 40% box noise, the end-to-end SSD-Det achieves 27.6 $A P$ and 53.9 $A P _ { 5 0 } ,$ , 9.0 and attains 11.3 point improvement compared with state-of-the-art method OA-MIL, respectively. Also, through re-training on FasterRCNN, the performance further reaches 29.3 AP and 54.8 $A P _ { 5 0 }$ . With the backbone of ResNet-101, the performance achieves consistent improvement. On MS-COCO test set, our method also achieves state-of-the-art performance.

VOC 2007 Dataset. Table 2 shows the comparison results on the VOC 2007 test set. Co-teaching, SD-LocNet and KL Loss, can not address inaccurate bounding box annotations well. OA-MIL improves the performance on different noisy datasets. Our approach obtains further improvements to 77.10, 74.80, 71.50, 66.90 $A P _ { 5 0 }$ on 10%, 20%, 30 % and 40 % noisy box datasets, respectively.

![](images/6933dc8e8067b79d2f7f70d2b3e9e14936574ec2acae40c9a527a9e4c19d8955.jpg)  
Figure 4: Qualitative detection results on COCO validation. Previous methods miss objects and face part prediction problems. Our method misses fewer objects, and the bounding box quality is better, especially for small or overlapped objects.

<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Box Refiner+Re-Train</td><td rowspan=1 colspan=1>SSD-Det</td><td rowspan=1 colspan=1>SSD-Det+Re-Train</td></tr><tr><td rowspan=1 colspan=1>AP/AP50</td><td rowspan=1 colspan=1>29.0 / 54.4</td><td rowspan=1 colspan=1>27.6/53.9</td><td rowspan=1 colspan=1>29.3 / 54.8</td></tr></table>

Table 7: Comparisons of end-to-end and Re-Train.

## 4.3. Ablation Study and Analysis

To further analyze SSD-Det’s effectiveness and robustness, we conduct more experiments on COCO val set if there are no other instructions. Except for Table 3, the noise level of these experiments is 40%.

Ablation of Modules. Ablation study of each component in our approach is given in Table 3, including: (i) Different stages of our basic box refiner. i.e. training object detector without the stage II (2-Ref), where the pseudo boxes predicted by the stage I are served as the supervision for training a parallel detector. (ii) SPSD, i.e. training without SPSD, where the object-bag is constructed directly by neighborhood sampling around the noisy ground-truth or the predicted pseudo boxes of the stage I. (iii) SISD. (iv) Re-Train with FasterRCNN (Re-Train).

Effectiveness of SPSD. SPSD further improves the detection performance on the MS-COCO, especially under high box noise levels, e.g. under 40% box noise level, SPSD boosts the performance from 24.6 to 27.2, as shown in Table 3 (row 3). In Table 4, we conduct further ablation on SPSD. With SPSD bag construction only in stage II, the performance increases by 1.4 AP. The performance further improves with the proposal score of stage I as weights. With SPSD in all stages, the AP reaches 27.2. Fig. 5 shows the bag quality. With SPSD, the mean IoU increases from 40.3 to 58.7 and the max and top-10 IoU increase to 78.3 and 75.1, which indicates a better upper bound of proposal selection. More high-quality proposals bring better optimization and easier proposals selection.

Number of SPSD. As shown in Table 6. When adding

<table><tr><td rowspan="3">Detectors</td><td colspan="2">Clean-supervised</td><td colspan="3">Noise-supervised</td></tr><tr><td>AP</td><td>AP50</td><td>AP AP50</td><td>(w AP</td><td>ours) AP50</td></tr><tr><td>FasterRCNN</td><td>37.9</td><td>58.1</td><td>10.3</td><td>28.9 29.3</td><td>54.8</td></tr><tr><td>SparseRCNN [43]†</td><td>45.0</td><td>64.1</td><td>6.0</td><td>20.3 34.3</td><td>60.2</td></tr><tr><td>De-DETR [62]†</td><td>46.8</td><td>66.3</td><td>5.0</td><td>16.9 35.2</td><td>60.9</td></tr></table>

Table 8: Experiments on advanced detectors. De-DETR is Deformable DETR. † uses multi-scale data augment. ’w/ ours’ means using our method under noisy supervision.

![](images/6deb20c3b97dc931295ebb6a171c57cf09517c15a83d975122804cf2378209c0.jpg)  
Figure 5: Bag quality (IoU of proposals with GT) of construction in SSD-Det. B.S. (blue) means neighborhood sampler. SPSD I (orange) denotes single SPSD adopted. SPSD II (yellow) is two SPSD and interactive structure adopted. SPSD II significantly improves the quality.

3 SPSD, performance drops slightly, probably due to the accumulation of errors outweighing the performance gain from extra stages. Hence, 2 SPSD is our default setting.

Effectiveness of SISD. SISD is designed to select object-aware proposals in box selection. Under 40% and 20% box noise, the detection performance improves from 27.2 to 27.6 and 33.0 to 33.6, which verifies the effectiveness of the module, shown in Table 3. We also study the strategies of ORE in SISD (Table 5). The minus or concat on object feature $\mathbf { F } _ { j } ^ { + }$ and proposal feature $\mathbf { F } _ { j }$ do not work. With add strategy, the performance is 27.60. If SISD shares the two fc layers, the performance drops to 22.99 since the optimization goals are contradictory (Identity distinguishes objects in the same category). If we directly use the RoI feature without ORE, the performance drops to 27.32 AP, verifying the effectiveness of the object relevance strategy.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Drift Rate (%)</td><td colspan="3">Group Rate (%)</td><td colspan="3">Part Rate (%)</td></tr><tr><td>all S</td><td>m</td><td>1</td><td></td><td>all s m</td><td></td><td>1</td><td>all s m 1</td><td></td></tr><tr><td>OA-MIL[12]</td><td>15.1 17.8 17.1 7.4</td><td></td><td></td><td></td><td></td><td>6.7 2.8 3.4 1.4</td><td></td><td>2.8 3.4 2.7 2.3</td><td></td></tr><tr><td>Ours</td><td>1.5</td><td>1.0</td><td>1.3 1.4</td><td></td><td></td><td>1.7 1.2 0.5 0.7</td><td></td><td>1.0 0.5 1.1 1.3</td><td></td></tr></table>

Table 9: Breakdown of different problems during refinement (COCO under 40% noise level). s, m and l mean small, middle and large scale.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">our</td><td colspan="3">Quality (Average IoU)</td><td colspan="4">Frequency (%)</td></tr><tr><td>All Part</td><td>Oversize</td><td>Shift</td><td>Reliable</td><td>Part</td><td>Oversize</td><td>Shift</td></tr><tr><td colspan="10">Detector as Annotator</td></tr><tr><td rowspan="2">Objects-F</td><td rowspan="2"></td><td>44.3</td><td>24.1</td><td>33.6</td><td>13.3</td><td>40.1</td><td>28.4</td><td>21.7</td></tr><tr><td>47.0</td><td>30.8</td><td>41.8</td><td>16.8</td><td>49.2</td><td>23.1</td><td>9.8 8.0</td><td>19.4</td></tr><tr><td rowspan="2">COCO-F</td><td>√</td><td>45.1</td><td>25.4</td><td>33.3</td><td>15.7</td><td>40.0</td><td>27.5</td><td>12.0</td><td>20.5</td></tr><tr><td></td><td>48.2</td><td>32.1</td><td>40.6</td><td>19.6</td><td>49.5</td><td>22.2</td><td>10.0</td><td>18.3</td></tr><tr><td colspan="10">Point-based Annotator</td></tr><tr><td rowspan="2">COCO-P</td><td rowspan="2">√</td><td>55.6</td><td>30.4</td><td>25.0</td><td>29.8</td><td>65.6</td><td>9.5</td><td>23.2</td><td>1.7</td></tr><tr><td>65.2</td><td>46.7</td><td>36.4</td><td>40.9</td><td>74.9</td><td>5.9</td><td>18.5</td><td>0.6</td></tr></table>

Table 10: Analysis of noisy annotations types and quality.

Affect of Re-Train. As most WSOD methods do, we re-run the experiments by training a fully supervised detector for better performance. We find that if the SSD-Det only trains the refiner and uses the pseudo label to train the FasterRCNN, the result is good but lower than re-train after the end-to-end training given in Table 7 (row 1). This is because joint training is beneficial for box refinement.

Experiments on Advanced Detectors. We re-train recent detectors, e.g. SparseRCNN and Deformable DETR, under the boxes refined by our method. Table 8 verifies that our method achieves consistency improvement.

The computational cost discussion. Similar to OA-MIL, our method adds SSD head to Faster R-CNN for auxiliary training to refine the noisy annotation and the head is not used during inference, the calculation cost during inference is same to standard Faster R-CNN.

## 4.4. Visualization and Discussion.

Fig. 4 shows that OA-MIL faces missing instances and grouping instances issues for small or overlapped objects (as mentioned in [12]), while our method still works well. For a better intuitive understanding of SISD and SPSD, we visualize the bag construction quality in Fig. 5 Then, we makes noise types breakdown of ’Drift’, ’Group’ and ’part dominance’ issues. We give the definition of IoU, IoG and

IoD:

$$
I o U = \frac { A ( I ) } { A ( D ) + A ( G ) - A ( I ) } , I o G = \frac { A ( I ) } { A ( G ) } , I o D = \frac { A ( I ) } { A ( D ) }\tag{9}
$$

where A(\*) is area of box \*, D and G are refined box and gt box respectively, and I is insertion between D and G. We statistically count the proportion of three noise types of ’bad’ refined boxes (having small IoU with gt) in Table 9: (i) Drift: ’bad’ refined box has a higher IoU with another nearby object. (ii) Group: ’bad’ refined box has high IoG with multiple objects. (iii) Part: ’bad’ refined box has a high IoD. Table 9 shows quantitative results for each noise type of baseline and ours. The drift, group, part problems reduce from 15.1%, 6.7%, 2.8% to 1.5%, 1.7%, 1.0% , respectively, demonstrating our improvement.

Experiments on Real-life Noisy Annotations. Reallife noisy annotations stem from: low-quality data (e.g., occlusion, blur), human annotator errors and automatic machine annotator limitations. Noise from human errors is quite subjective, since differences between annotators. For a more objective analysis, noisy annotations from machine annotator is used for experiments. Without loss of generality, Faster R-CNN well-trained on MS-COCO was applied to Objects365 images, yielding Objects-F dataset,and to COCO-val images, producing COCO-F dataset. P2BNet [11], a point-based annotator, was used on COCO-val images with point annotations, generating COCO-P dataset. SSD-Det effectively improves lowquality boxes. As shown in Table 10,with SSD-Det’s refinement, the average IoU increases for Objects-F (from 44.3 to 47.0), COCO-F (from 45.1 to 48.2) and COCO-P (from 55.6 to 65.2) datasets. Further, the proportion of reliable annotations increases, and noise categories’ frequency (Part, Oversize, and Shift) decreases for all datasets.

## 5. Conclusion

This paper investigates problems during refinement caused by solely using category information to select proposals. We also propose SSD-Det to mine spatial information in a self-distillation fashion. SSD-Det introduces the SPSD module to learn semantic-spatial correspondence knowledge with neighborhood sampler and an interactive structure to combine spatial information and category information, thus producing a high-quality proposal bag. SISD in SSD-Det is utilized to improve the proposal selection procedure by integrating object-relevant spatial confidence. Complete ablations on multiple datasets verify the effectiveness of SSD-Det.

## References

[1] Hakan Bilen and Andrea Vedaldi. Weakly supervised deep detection networks. In CVPR, 2016. 1, 3, 4

[2] Leon Bottou. Stochastic gradient descent tricks. In´ Neural Networks: Tricks of the Trade - Second Edition. Springer, 2012. 7

[3] Nicolas Carion, Francisco Massa, and Gabriel Synnaeve et al. End-to-end object detection with transformers. In ECCV, 2020. 2, 3

[4] Simon Chadwick and Paul Newman. Training object detectors with noisy data. In IV, 2019. 1, 3

[5] Guobin Chen, Wongun Choi, and Xiang Yu et al. Learning efficient object detection models with knowledge distillation. In NeurIPS, 2017. 3

[6] Kai Chen, Jiaqi Wang, and Jiangmiao Pang et al. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 7

[7] Ze Chen, Zhihang Fu, and Rongxin Jiang et al. SLV: spatial likelihood voting for weakly supervised object detection. In CVPR, 2020. 2, 3

[8] Xing Dai, Zeren Jiang, and Zhao Wu et al. General instance distillation for object detection. In CVPR, 2021. 3

[9] Ali Diba, Vivek Sharma, and Ali Mohammad Pazandeh et al. Weakly supervised cascaded convolutional networks. In CVPR, 2017. 3

[10] Thomas G. Dietterich, Richard H. Lathrop, and Tomas´ Lozano-Perez. Solving the multiple instance problem with´ axis-parallel rectangles. Artificial Intelligence, 1997. 2

[11] Chen. et al. Point-to-box network for accurate object detection via single point supervision. In ECCV, 2022. 1, 2, 3, 4, 9

[12] Liu. et al. Robust object detection with inaccurate bounding boxes. In ECCV, 2022. 1, 2, 3, 4, 6, 7, 9

[13] Mark Everingham, Luc Van Gool, and Christopher K. I. Williams et al. The pascal visual object classes (VOC) challenge. IJCV, 2010. 6

[14] Aritra Ghosh, Himanshu Kumar, and P. S. Sastry. Robust loss functions under label noise for deep neural networks. In AAAI, 2017. 3

[15] Ross B. Girshick. Fast R-CNN. In ICCV, 2015. 2

[16] Jianyuan Guo, Kai Han, and Yunhe Wang et al. Distilling object detectors via decoupled features. In CVPR, 2021. 3

[17] Bo Han, Quanming Yao, and Xingrui Yu et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NeurIPS, 2018. 3, 6, 7

[18] Kaiming He, Georgia Gkioxari, and Piotr Dollar´ et al. Mask R-CNN. In ICCV, 2017. 1, 4

[19] Kaiming He, Xiangyu Zhang, and Shaoqing Ren et al. Deep residual learning for image recognition. In CVPR, 2016. 7

[20] Yihui He, Chenchen Zhu, and Jianren Wang et al. Bounding box regression with uncertainty for accurate object detection. In CVPR, 2019. 1, 6, 7

[21] Geoffrey Hinton and Oriol et al. Vinyals. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2(7), 2015. 3

[22] Jisoo Jeong, Seungeui Lee, and Jeesoo Kim et al. Consistency-based semi-supervised learning for object detection. In NeurIPS, 2019. 3

[23] Lu Jiang, Zhengyuan Zhou, and Thomas Leung et al. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018. 3

[24] Vadim Kantorov, Maxime Oquab, and Minsu Cho et al. Contextlocnet: Context-aware deep network models for weakly supervised localization. In ECCV, 2016. 3

[25] Junnan Li, Caiming Xiong, Richard Socher, and Steven C. H. Hoi. Towards noise-resistant object detection with noisy annotations. CoRR, abs/2003.01285, 2020. 1, 3

[26] Quanquan Li, Shengying Jin, and Junjie Yan. Mimicking very efficient network for object detection. In CVPR, 2017. 3

[27] Yandong Li, Di Huang, and Danfeng Qin et al. Improving object detection with selective self-supervised self-training. In ECCV, 2020. 3

[28] Mingxiang Liao, Fang Wan, and Yuan Yao et al. End-toend weakly supervised object detection with sparse proposal evolution. In ECCV, 2022. 3

[29] Tsung-Yi Lin, Piotr Dollar, and Ross B. Girshick ´ et al. Feature pyramid networks for object detection. In CVPR, 2017. 7

[30] Tsung-Yi Lin, Priya Goyal, and Ross B. Girshick et al. Focal loss for dense object detection. In ICCV, 2017. 1, 2, 4, 6

[31] Tsung-Yi Lin and Michael et al. Maire. Microsoft coco: Common objects in context. In ECCV, 2014. 1, 6

[32] Wei Liu, Dragomir Anguelov, and Dumitru Erhan et al. SSD: single shot multibox detector. In ECCV, 2016. 2

[33] Ze Liu, Yutong Lin, and Yue Cao et al. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 3

[34] Xingjun Ma, Yisen Wang, and Michael E. Houle et al. Dimensionality-driven learning with noisy labels. In ICML, 2018. 3

[35] Jiafeng Mao, Qing Yu, and Kiyoharu Aizawa. Noisy localization annotation refinement for object detection. TIS, 2021. 1, 3

[36] Ilija Radosavovic, Piotr Dollar, and Ross B. Girshick ´ et al. Data distillation: Towards omni-supervised learning. In CVPR, 2018. 3

[37] Joseph Redmon, Santosh Kumar Divvala, and Ross B. Girshick et al. You only look once: Unified, real-time object detection. In CVPR, 2016. 2

[38] Shaoqing Ren, Kaiming He, and Ross B. Girshick et al. Faster R-CNN: towards real-time object detection with region proposal networks. TPAMI, 2017. 1, 2, 6, 7

[39] Zhongzheng Ren, Zhiding Yu, and Xiaodong Yang et al. Instance-aware, context-focused, and memory-efficient weakly supervised object detection. In CVPR, 2020. 2, 3

[40] Kihyuk Sohn, David Berthelot, and Nicholas Carlini et al. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPS, 2020. 3

[41] Kihyuk Sohn, Zizhao Zhang, and Chun-Liang Li et al. A simple semi-supervised learning framework for object detection. CoRR, 2020. 3

[42] Hwanjun Song, Minseok Kim, and Jae-Gil Lee. SELFIE: refurbishing unclean samples for robust deep learning. In ICML, 2019. 3

[43] Peize Sun, Rufeng Zhang, and Yi Jiang et al. Sparse R-CNN: end-to-end object detection with learnable proposals. In CVPR, 2021. 2, 3, 8

[44] Peng Tang and Xinggang Wang et al. Multiple instance detection network with online instance classifier refinement. In CVPR, 2017. 2, 3

[45] Peng Tang, Chetan Ramaiah, and Yan Wang et al. Proposal learning for semi-supervised object detection. In WACV, 2021. 3

[46] Peng Tang, Xinggang Wang, and Song Bai et al. PCL: proposal cluster learning for weakly supervised object detection. TPAMI, 2020. 3

[47] Koen E. A. van de Sande, Jasper R. R. Uijlings, and Theo Gevers et al. Segmentation as selective search for object recognition. In ICCV, 2011. 2, 4

[48] Fang Wan, Pengxu Wei, and Zhenjun Han et al. Min-entropy latent model for weakly supervised object detection. TPAMI, 2019. 3

[49] Keze Wang, Xiaopeng Yan, and Dongyu Zhang et al. Towards human-machine cooperation: Self-supervised sample mining for object detection. In CVPR, 2018. 3

[50] Tao Wang, Li Yuan, and Xiaopeng Zhang et al. Distilling object detectors with fine-grained feature imitation. In CVPR, 2019. 3

[51] Yunchao Wei, Zhiqiang Shen, and Bowen Cheng et al. TS ˆ2 2 C: tight box mining with surrounding segmentation context for weakly supervised object detection. In ECCV, 2018. 3

[52] Mengde Xu, Zheng Zhang, and Han Hu et al. End-to-end semi-supervised object detection with soft teacher. In ICCV. IEEE, 2021. 3

[53] Youjiang Xu, Linchao Zhu, and Yi Yang et al. Training robust object detectors from noisy category labels and imprecise bounding boxes. TIP, 2021. 1, 3

[54] Ke Yang, Dongsheng Li, and Yong Dou. Towards precise end-to-end weakly supervised object detection network. In ICCV, 2019. 3

[55] Zhendong Yang, Zhe Li, and Xiaohu Jiang et al. Focal and global knowledge distillation for detectors. In CVPR, 2022. 3

[56] Xuehui Yu, Pengfei Chen, and Di Wu et al. Object localization under single coarse point supervision. In CVPR, 2022. 2

[57] Xuehui Yu, Yuqi Gong, and Nan Jiang et al. Scale match for tiny person detection. In WACV, 2020. 1, 2

[58] Xiaosong Zhang, Fang Wan, and Chang Liu et al. Freeanchor: Learning to match anchors for visual object detection. In NeurIPS, 2019. 1, 6, 7

[59] Xiaopeng Zhang, Yang Yang, and Jiashi Feng. Learning to localize objects with noisy labeled instances. In AAAI, 2019. 3, 6, 7

[60] Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In NeurIPS, 2018. 3

[61] Zhaohui Zheng, Rongguang Ye, and Ping Wang et al. Localization distillation for dense object detection. In CVPR, 2022. 3

[62] Xizhou Zhu, Weijie Su, and Lewei Lu et al. Deformable DETR: deformable transformers for end-to-end object detection. In ICLR, 2021. 3, 8

[63] C. Lawrence Zitnick and Piotr Dollar. Edge boxes: Locating´ object proposals from edges. In ECCV, 2014. 4

[64] Barret Zoph, Golnaz Ghiasi, and Tsung-Yi Lin et al. Rethinking pre-training and self-training. In NeurIPS, 2020. 3