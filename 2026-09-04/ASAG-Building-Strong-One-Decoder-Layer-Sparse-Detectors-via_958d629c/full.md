# ASAG: Building Strong One-Decoder-Layer Sparse Detectors via Adaptive Sparse Anchor Generation

Shenghao Fu<sup>1,3,4</sup>, Junkai Yan<sup>1,4</sup>, Yipeng Gao<sup>1,4</sup>, Xiaohua Xie<sup>1,3,4∗</sup>, Wei-Shi Zheng<sup>1,2,3,4\*</sup>

<sup>1</sup>School of Computer Science and Engineering, Sun Yat-sen University, China, <sup>2</sup>Pengcheng Lab, China,

<sup>3</sup>Guangdong Province Key Laboratory of Information Security Technology, China,

<sup>4</sup>Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, China {fushh7, yanjk3, gaoyp23}@mail2.sysu.edu.cn, xiexiaoh6@mail.sysu.edu.cn, wszheng@ieee.org

## Abstract

Recent sparse detectors with multiple, e.g. six, decoder layers achieve promising performance but much inference time due to complex heads. Previous works have explored using dense priors as initialization and built one-decoderlayer detectors. Although they gain remarkable acceleration, their performance still lags behind their six-decoderlayer counterparts by a large margin. In this work, we aim to bridge this performance gap while retaining fast speed. We find that the architecture discrepancy between dense and sparse detectors leads to feature conflict, hampering the performance of one-decoder-layer detectors. Thus we propose Adaptive Sparse Anchor Generator (ASAG) which predicts dynamic anchors on patches rather than grids in a sparse way so that it alleviates the feature conflict problem. For each image, ASAG dynamically selects which feature maps and which locations to predict, forming afully adaptive way to generate image-specific anchors. Further, a simple and effective Query Weighting method eases the training instability from adaptiveness. Extensive experiments show that our method outperforms denseinitialized ones and achieves a better speed-accuracy tradeoff. The code is available at https://github.com/ iSEE-Laboratory/ASAG.

## 1. Introduction

Object detection is a fundamental and challenging computer vision task. Different from traditional CNN-based dense object detectors [27, 10, 20, 21, 32, 40] using slidingwindow paradigm, query-based sparse detectors [2, 43, 31, 9] use hundreds of object queries to search through the whole image, each representing an object or background. They get rid of some traditional hand-crafted components and procedures, e.g., anchors and Non-Maximum Suppression (NMS), which greatly simplifies the detection pipeline and makes the detector fully end-to-end trainable. With the help of powerful transformer encoder-decoder architecture, sparse detectors show promising performance.

![](images/5a725538633690a64d341b40f11b5eaf9aedf39d24d48a05165e0b27500eee30.jpg)  
Figure 1: Comparing one-decoder-layer detectors with sixdecoder-layer counterparts on FPS and AP with various decoder types. ASAGs achieve a better speed-accuracy tradeoff. Best viewed in color.

However, sparse detectors cost more inference time since they need n decoder layers (typically n = 6) to progressively refine bounding boxes, leading to a more complex head. Inference with only one decoder layer achieves a much faster speed, such as AdaMixer [9] (43 V.S. 22 FPS) and Sparse RCNN [31] (43 V.S. 25 FPS). The more complex the decoder is, the larger the FPS gap is. Unfortunately, simply discarding the extra decoder layers results in a significant performance drop.

Recently, several methods [36, 41] have attempted to bridge the performance gap between one- and six-decoderlayer detectors. Efficient DETR [36] finds that imageagnostic box and content queries should be blamed for a severe drop in performance when using one decoder layer.

Thus, both methods utilize an extra dense box prediction step before the decoder to provide proper query initialization. However, they fail to achieve comparable results to detectors with six decoder layers, for example Featurized Query RCNN [41] is lower than Sparse RCNN [31] by 1.5AP with 100 queries. We find that the features corresponding to dense and sparse detectors are significantly different. Such feature conflict hampers the performance of one-decoder-layer detectors, demonstrated in Section 3. Thus, although these methods achieve remarkable acceleration, it still has much room to narrow the performance gap.

In this work, we aim to build a fully sparse one-decoderlayer detector, narrowing the performance gap between oneand six-decoder-layer detectors and retaining the fast speed. The key difference between our method and other dense initialization methods is that we use patches as the basic prediction units, which can be the whole or part of an image. Sparsely predicting on patches alleviate the feature discrepancy caused by predicting on grids and enjoys global receptive fields. Further, we propose to loose the constraint that each image should use a fixed number of queries to detect objects so that more complex images detect objects with more queries and vice versa. Based on these two preconditions, we propose initializing image-specific queries using Adaptive Sparse Anchor Generator (ASAG), which is fully adaptive to each image in both anchors’ locations and numbers. We further design Adaptive Probing to adaptively crop patches on possible locations on different feature map levels. It runs in a top-down and coarse-to-fine way and greatly enhances the ability to detect small objects. Finally, an effective Query Weighting method is proposed to handle the instability coming from adaptiveness.

We conduct extensive experiments on the COCO [22] dataset with various decoder types. As shown in Figure 1, our model ASAG-S outperforms dense-initialized Query RCNN [41] by 2.6 AP with fewer FLOPs and the same decoder. We also retain the fast speed of one-decoder-layer detectors, thus achieving a better speed-accuracy trade-off.

## 2. Related Works

Improving Sparse Detectors. Recently, DETR [2] views object detection as a set-prediction problem and achieves promising performance. But it still has some apparent disadvantages and lacks interpretability. Many following works have solved the problems like slow convergence and relatively low performance on small objects by utilizing multi-scale feature pyramids [43, 9, 38], introducing more spatial priors [43, 8, 25, 34, 23, 9], stabilizing bipartite matching [16, 39], aligning feature space [37, 38], increasing positive samples [13, 4, 14, 44, 26], using knowledge distillation [12, 3, 5], initializing queries [43, 39, 11], etc. Although achieving outstanding performance and fast convergence speed, transformer-based sparse detectors still require more inference time than CNN-based dense detectors thus limiting their practical applications.

<table><tr><td>NO.</td><td>Detector</td><td>Init.</td><td>AP</td><td>APs</td><td> $\overline { { \mathsf { A P } _ { m } } }$ </td><td>AP_</td></tr><tr><td>(a) (b)</td><td>Deformable DETR+ [43] Deformable DETR+†</td><td>learned</td><td>46.2 37.9</td><td>28.3 23.1</td><td>49.2 41.9</td><td>61.5 49.1</td></tr><tr><td>(c) (d)</td><td>Deformable DETR++ [43] Deformable DETR++†</td><td>dense</td><td>46.9 33.7</td><td>29.4 23.1</td><td>50.1 38.4</td><td>61.6 41.1</td></tr></table>

Table 1: Effect of dense query initialization on one- and sixdecoder-layer detectors. The first decoder layer with dense initialization even underperforms the one without imagespecific initialization. †: Inference with the first layer.

Accelerating Sparse Detectors. As a well-known experience, most computation for a detector lies in the backbone due to high-resolution feature maps and dense computation. However, as sparse detectors have more complicated operators in neck and head, e.g., attention and grid sampling, the FLOPs cannot directly reflect the FPS. For example, the decoder of AdaMixer [9] takes 31% of the total FLOPs but nearly half of the FPS. Different from works [42, 33, 29] focusing on reducing computation in neck by approximating self-attention in the encoder, we aim to simplify the decoder. Recently, Li et al. [19] find that some unimportant queries are not worth being computed equally. However, decreasing the number of queries brings minor acceleration as Sparse RCNN [31] increasing the number of queries from 100 to 300 only uses an extra 1 FPS. Thus decreasing the number of decoder stages is a more promising way to speed up sparse detectors.

## 3. Why Should We Use Sparse Initialization for One-Decoder-Layer Detectors?

Some previous works [43, 39] have found that utilizing a dense box prediction step as a query initialization can be helpful to six-decoder-layer detectors. For example, Deformable DETR++ [43] outperforms Deformable DETR+ [43] by 0.7 AP as shown in Table 1. However, we surprisingly find that the first decoder layer with better initialization even performs worse than the one with imageagnostic initialization, as shown by (b) and (d) in Table 1. This phenomenon shows that the benefit from the dense box prediction step for six-decoder-layer detectors does not lie in better initialization but in more supervision signals to the encoder, as many works [4, 14, 44, 26] find that one-to-one matching is not sufficient for feature learning.

To better illustrate the phenomenon above, we visualized discriminability scores in the encoder in Figure 3 following [44], which are l<sup>2</sup>-norm of the corresponding grid features. Objects with higher discriminability scores can be better detected. Since dense detectors predict objects based on grid features while sparse detectors detect objects using queries, the architecture discrepancy makes the needed features totally different. As shown in Figure 3, sparse detectors pay more attention to background information than dense detectors [3]. Moreover, dense detectors prefer to activate the whole object uniformly since each grid has equal chances to predict the object, while sparse detectors tend to highlight some discriminative parts. The score maps of sparse detectors with dense initialization fall between dense and sparse detectors. Due to the powerful decoders with six layers, such detectors with dense initialization can tolerate the discrepancy of features and benefit from more positive signals. However, one-decoder-layer detectors with limited representation ability suffer from conflicting features. We hypothesize that this is why one-decoder-layer detectors with well-initialized queries still lag behind their sixdecoder-layer counterparts. This phenomenon also demonstrates that there is redundancy in six-decoder-layer detectors, which can be reduced for acceleration. Thus, we provide a sparse way to further narrow the performance gap between one- and six-decoder-layer detectors and retain the fast speed of one-decoder-layer detectors.

![](images/43b5186020d8947dd2686539f8e52c2d40b5152acbfa5d58ae2d959bdd7df418.jpg)  
Figure 2: Overview. Our model first uses Anchor Generator to predict dynamic anchors. Then, query features are extracted by RoIAlign [10] and refined by a Self-Attention layer (SA). One decoder layer of any kind is used for final prediction. The neck is changed following the decoder. We additionally add three auxiliary heads using the same proposals to provide more supervision signals, which are discarded during inference. Each component is supervised under one-to-one matching losses.

![](images/a1147506d3715d631a25cea8146269183ae4b21aa2ed2e97c99e4e4ffae8c7cb.jpg)  
Figure 3: Comparison on feature maps (discriminability scores [44]) of dense and sparse detectors. There exists a clear inconsistency between dense and sparse features.

## 4. Adaptive Sparse Anchor Generation for Query Initialization

## 4.1. Overview

In this section, we introduce our Adaptive Sparse Anchor Generator (ASAG for short), which initializes queries sparsely and is more suitable for sparse decoders while retaining fast speed. As shown in Figure 2, ASAG generates image-specific anchors adaptively from the aspect of locations and numbers, without using predefined spatial priors. Unlike complex decoders, our ASAG is lightweight, using only 0.06G FLOPs. With dynamic anchors, we use RoIAlign [10] to generate content queries since initializing both box and content queries is vital for one-decoderlayer detectors [36]. Further, an extra Self-Attention layer is utilized to model relationships between objects and reduce redundancy for NMS-free, similar to Featurized QR-CNN [41]. Lastly, a one-layer decoder of any kind [9, 31, 43] is used for final refinements.

Recent works [13, 4, 14, 44, 26] show that sparse detectors benefit from sufficient supervision signals. Considering that dense methods provide supervision to each grid and other six-decoder-layer detectors have more auxiliary losses, we additionally add three auxiliary parallel one-layer decoders for a fair comparison. Thus, our models are also supervised by six one-to-one matching losses. Different from Group DETR [4] that uses different groups of queries and the shared decoder by each group, we use different decoders with the same proposals.

![](images/b539e96d12d48c1250649ad9f1d635e019c912cfedcaa61fba5711d4fcd13ae8.jpg)  
Figure 4: Adaptive Sparse Anchor Generator (ASAG). ASAG starts predicting dynamic anchors from fixed feature maps and then adaptively explores larger feature maps using Adaptive Probing, which runs top-down and coarse-to-fine. Learnable position embeddings (PE) are added to patches after flattening to keep spatial structures.

## 4.2. Adaptive Sparse Anchor Generator

Patches are better prediction units. In this work, we use patches as the basic prediction units, from which possible anchors will be predicted. The patch is the whole or part of an image, which may contain lots of objects, and is much larger than grids or regions of interest. Thus, predicting from patches alleviates the feature discrepancy caused by predicting from grids and enjoys global receptive fields.

Inference with the fixed number of feature maps. We start demonstrating our method from using a fixed number of feature maps to predict all objects, i.e. $P _ { 5 } ^ { 1 }$ and $P _ { 6 }$ . As shown in the upper part of Figure 4, taking the $P _ { 5 }$ feature map as input, we first compress the feature map to a few channels along the channel dimension to save computation and parameters. To handle images with different sizes, we interpolate $P _ { 5 }$ to a fixed size and then evenly split it into four patches from top-left to bottom-right to narrow the predictor’s search space. If we directly view the whole $P _ { 5 }$ feature map as a single patch, the model tends to overlook some small objects. Considering that some large objects may appear across the patches, $P _ { 6 }$ feature map is used to handle the problem, which is downsampled from $P _ { 5 }$ after interpolating by a factor of 2, and viewed as a single patch (the brown patch). We use an MLP as the predictor to predict anchors of a fixed number on each patch simultaneously, each described by four coordinates and a location score. The location score can be seen as the class probability so that it is class-agnostic and is supervised by IoU [18] as a soft label.

Inference with a dynamic number of feature maps. Since $P _ { 5 }$ is not sufficient to predict small objects, using feature maps with larger sizes can obtain more precise anchors. Motivated by QueryDet [35], we propose to sparsely compute on large feature maps using Adaptive Probing to correct the anchors for small objects with low confidence.

As shown in the lower part of Figure 4, we select some anchors predicted from the fixed part, whose confidences fall in [η<sub>l</sub>, η<sub>h</sub>] and sizes are smaller than half of the patch size. Anchors with scores lower than $\eta _ { l }$ are seen as noisy anchors, and anchors with scores higher than $\eta _ { h }$ are accurate enough. Thus such anchors are not used in Adaptive Probing. With selected anchors, we crop some patches on the $P _ { 4 }$ feature map, whose centers are the corresponding anchors’ ones. Since patches may overlap with each other, we use NMS with IoU threshold $\eta _ { i o u }$ to reduce the redundancy. Anchors predicted from the higher resolution feature maps are more precise than the original ones, thus we replace them with the newly generated ones. Such probing iterations continue to perform on the following larger feature maps until the largest one $P _ { 3 }$ . We empirically find that additionally using $P _ { 2 }$ brings minor improvement but adds more inference time. Once no anchors are selected, the iterations break with an early-stop mechanism. Thus, the probing is adaptive to the number of iterations, the number of patches, and the locations of patches. Finally, all anchors with scores higher than $\eta _ { f }$ and not being selected for Adaptive Probing are gathered. Considering that the model needs to deal with pictures with different difficulties, the number of generated patches and anchors varies accordingly. We pad the output anchors to the max size for parallel processing.

Training ASAG. We first define three kinds of patches: 1) generated patch which is generated from selected anchors, 2) GTpatch which is gained by grouping ground truth boxes smaller than half of the patch size, and 3) random patch which is generated randomly. To ensure that predictors for each pyramid level are fully and equally trained, we define a minimal training patch number $N _ { T P } = 4$ for each level since the lower level tends to receive fewer supervision signals due to the early-stop mechanism. For feature maps from $P _ { 5 }$ to $P _ { 3 }$ , we use the generated patch, GT patch, and random patch in turn until the minimum patch number is met. For $P _ { 6 }$ , since we view the whole feature map as a single patch, we flip the patch horizontally and vertically to meet the minimum patch number. Only anchors generated from generated patches with confidence scores higher than $\eta _ { f }$ are gathered and sent to the following model parts. The targets for each patch are objects whose centers lie in the patch. Since the output anchors are unordered, bipartite matching is used to get one-to-one matching, similar to other parts of our model and other sparse detectors, except it is class-agnostic. Further, we use IoUs between ground truth boxes and the matched anchors as the soft labels [18].

Relationships with related works. The Adaptive Probing computes sparsely on large feature maps to save computation, similar to PointRend [15] and QueryDet [35]. However, both are dense prediction methods and it is clear where to explore on the larger feature map while we sparsely find the corresponding location. Further, QueryDet predicts objects in a divide-and-conquer way, but the Adaptive Probing is in a correct-and-replace manner thus we enjoy the early-stop mechanism. We can even discard large feature maps manually for efficient inference, as shown in Table 2.

## 4.3. Stablizing Training

Through the novel design above, we gain adaptive sparse anchors and proposals efficiently. However, as shown in Figure 5, we find that our dynamic anchors have two characteristics that are significantly different from the traditional hand-crafted anchors: 1) dynamic anchors may not be as precise as predefined anchors in the early training stage, 2) dynamic anchors change both in quality and numbers along the training process, making the detection head hard to optimize. It is unsuitable for treating two anchors with 0.1 and 0.9 IoU equally since it confuses the detectors about the definition of positive samples. Thus, we propose Query Weighting to ease the training difficulty by giving high-quality anchors with larger weights and vice versa. Soft labels make detectors pay more attention to precise predictions, stabilizing the training when dynamic anchors change, especially in the early training process. This simple weighting mechanism introduces no inference cost.

Motivated by DW [17] to give diverse positive and negative loss weights, our weighting functions are as follows:

$$
\mathrm { N o r m } ( x _ { 1 } , x _ { 2 } ) = \sigma ( ( x _ { 1 } \times x _ { 2 } - 1 / 3 ) \times 4 . 5 ) \div \sigma ( 3 ) ,\tag{1}
$$

$$
w _ { p o s } = \mathrm { N o r m } ( s ^ { \gamma _ { 1 } } , I o U ^ { \gamma _ { 2 } } ) ,\tag{2}
$$

$$
w _ { n e g } = \mathrm { N o r m } ( s ^ { \gamma _ { 1 } } , P _ { n e g } ( I o U ^ { \gamma _ { 2 } } ) ) - \sigma ( - 1 . 5 ) ,\tag{3}
$$

where s and IoU are classification scores s and IoUs, $P _ { n e g }$ is the same function as in [17], and σ denotes sigmoid function. After normalizing, the positive weights are roughly in [0.2, 1] and the negative weights are in [0, 0.8]. Since the sigmoid function is non-linear, it raises the small values while still keeping them within [0, 1]. Even if the matched anchors do not overlap with the targets, we cannot assign the positive weights to zeros as no other anchors will be assigned to the targets in the one-to-one label assignment.

![](images/14989eb15e18e98e7ca600baeb28caf6b76cee872cd38f060c699a4a4d895e95.jpg)  
(a) Epoch 0

![](images/90e473104e2651272fcb647dadd2cacd7fd27578b936f6469cd3166747962f58.jpg)  
(b) Epoch 11

![](images/90648b82482472827f936634d73fe018ebd1cb7354b21f2484a6924e83b9f18d.jpg)  
Figure 5: Visualizing dynamic anchors during the training process. The quality and number of anchors change along the training process making the model hard to optimize.

And we avoid assigning ones to negative weights of the only matched anchors. The weights only apply to losses not to matching costs.

Different from label weighting methods [18, 7, 17] in dense detectors, which align the separate regression and classification heads by giving larger weights to more suitable anchors within the candidate bag of each ground truth, Query Weighting is to tolerate dynamic anchors during the training process. Thus our Query Weighting is global-wise while label weighting is instance-wise. We show that sixdecoder-layer detectors do not benefit from Query Weighting due to regressing from fixed queries in Section 5.5.

## 5. Experiments

## 5.1. Settings

Dataset. We conduct ASAG experiments on the widely used detection dataset COCO2017 [22]. We train our model on train2017 (∼118k images) and report evaluation metrics on val2017 containing 5k images. We find each parallel decoder performs similarly (∼0.2 AP).

Configurations. To show the generalizability and make a fair comparison, we conduct experiments with well-known decoders, such as AdaMixer [9], Sparse RCNN [31], Deformable DETR [43], and our corresponding models are ASAG-A, ASAG-S, and ASAG-D, respectively. Featurized Query RCNN [41] and Efficient DETR [36] are denseinitialized one-decoder-layer detectors of Sparse RCNN and Deformable DETR, respectively. We compare with these methods using a similar average number of anchors fairly. For the 100 queries setting, we set the range of the number of anchors for each image as [5, 200], η<sub>f</sub> as 0.1, the number of predicted anchors for each patch on the fixed part and the adaptive part as 50 and 20. The corresponding configurations for the 300 queries setting are [50, 500], 0.05, 150, and 50.

<table><tr><td>Detector</td><td>FL</td><td>#An</td><td>#L</td><td>AP  $\overline { { \mathrm { A P } _ { 5 0 } \mathrm { A P } _ { 7 5 } } }$  APs</td><td> $\overline { { \mathsf { A P } _ { m } \ A P _ { l } } }$ </td><td></td><td>FPS</td></tr><tr><td>AdaMixer</td><td></td><td>100</td><td>6</td><td>42.7 61.5 45.9 24.7</td><td>45.4</td><td>59.2</td><td>23.8</td></tr><tr><td rowspan="6">ASAG-A</td><td></td><td>300</td><td>6</td><td>44.1 63.4 47.4 27.0</td><td>46.9</td><td>59.5</td><td>22.2</td></tr><tr><td> $\overline { { P _ { 3 - 6 } } }$ </td><td>107</td><td>1</td><td>42.6 60.5 45.8 25.9</td><td>45.8</td><td>56.9</td><td>28.9</td></tr><tr><td> $P _ { 4 - 6 }$ </td><td>97</td><td>1</td><td>42.1 59.9 45.7 24.8</td><td>46.0</td><td>56.9</td><td>30.3</td></tr><tr><td> $P _ { 5 - 6 }$ </td><td>87</td><td>1</td><td>40.3 57.5 43.3 21.2</td><td>44.4</td><td>57.1</td><td>31.3</td></tr><tr><td> $\overline { { P _ { 3 - 6 } } }$ </td><td>329</td><td>1</td><td>43.6 62.5 47.0 26.9</td><td>46.2</td><td>57.6</td><td>27.9</td></tr><tr><td> $P _ { 4 - 6 }$   $P _ { 5 - 6 }$ </td><td>313 280</td><td>1 1</td><td>43.4 62.1 46.8 26.4 41.9 60.5 44.9 23.5</td><td>46.4 45.5</td><td>57.7 57.9</td><td>29.0 30.5</td></tr></table>

Table 2: Comparison with AdaMixer [9] using the standard 1× schedule and R50. FL means feature map levels used in Anchor Generator. #An and #L denote the average number of anchors and the number of decoder layers, respectively. Models colored in yellow use efficient inference, and in blue use more than one decoder layer. Using only $P _ { 5 - 6 }$ means do not use Adaptive Probing totally.

Other implementation details. We keep the initial learning rate for the backbone and other parts as $2 \times 1 0 ^ { - 5 }$ and $2 \times 1 0 ^ { - 4 }$ . The batch size is 16. In the standard 1× schedule, the learning rate drops at epoch 8 and 11 by a factor of 0.1. In the 3× schedule, multi-scale training is utilized and the shorter side of images ranges from 640 to 800. The learning rate drops at epoch 24 and 33. No query patterns [34] is used. Following other DETR-like models, we use L1 loss, GIoU loss [28], and classification losses with coefficients 5, 2, 2. The Query Weighting applies to both regression and classification losses similar to [17]. AdamW [24] with weight decay 0.0001 is used as the optimizer.

As for other default hyper-parameters, the patch size in Anchor Generator is 15, thus the interpolate size for $P _ { 5 }$ is 30. $\gamma _ { 1 }$ and $\gamma _ { 2 }$ in Query Weighting are 0.4 and 0.6. For fast inference, we stop Adaptive Probing with an early stop if the number of selected anchors is less than 3. And we limit the number of patches for each level within 15. FPS are tested on a single NVIDIA 3090 GPU with batch size 1.

## 5.2. Main Results

Comparison under the standard 1× schedule. We first fairly compare ASAG-A with AdaMixer [9] with a few epochs. As shown in Table 2, although dynamic anchors are changing and imprecise in the early time, Query Weighting stabilizes the training and ASAG-A still converges in 12 epochs and achieves comparable results with six-decoderlayer AdaMixer with 1.25× speed-up. Since Adaptive Probing runs in an iterative and correct-and-replace way, we can stop at a specific level manually for efficient inference without re-training. Note that efficient inference will not hurt the performance of large objects.

Comparison with one-decoder-layer detectors. Since

<table><tr><td colspan="2">Detector</td><td>#An</td><td>#L</td><td>AP</td><td>AP50 AP75</td><td>AP APm 3</td><td>AP_</td><td>F</td><td>FPS</td></tr><tr><td colspan="2">F-QRCNN [41] F-QRCNN* [41]</td><td>100 300</td><td>1 1</td><td>41.3 41.7</td><td>59.4 44.9</td><td>26.7 44.2 27.7</td><td>52.4 44.3 52.0</td><td>140 143</td><td>38.2 37.1</td></tr><tr><td rowspan="4">ASAG-S (Ours)</td><td> $\overline { { P _ { 3 - 6 } } }$ </td><td>100</td><td>1</td><td>43.9</td><td>60.2 45.4 62.2 47.9</td><td>27.8</td><td>46.5 57.8</td><td>130</td><td>30.1</td></tr><tr><td> $P _ { 4 - 6 }$ </td><td>89</td><td>1</td><td>43.6</td><td>61.7</td><td></td><td>57.8</td><td>130</td><td>31.2</td></tr><tr><td> $P _ { 5 - 6 }$ </td><td>76</td><td>1</td><td>41.5 58.9</td><td>47.5 45.2</td><td>26.7 46.7 23.2</td><td>57.8</td><td>130</td><td></td></tr><tr><td> $\overline { { P _ { 3 - 6 } } }$ </td><td>312</td><td>1</td><td>45.0 64.1</td><td>49.1</td><td>45.4 47.4</td><td>57.8</td><td>136</td><td>33.0 29.2</td></tr><tr><td rowspan="2">ASAG-S (Ours)</td><td> $P _ { 4 - 6 }$ </td><td>292</td><td>1</td><td>44.8</td><td>63.9 48.8</td><td>29.5 28.9</td><td>47.5 57.8</td><td>136</td><td>30.5</td></tr><tr><td> $P _ { 5 - 6 }$ </td><td>256</td><td>1</td><td>43.2 61.9</td><td>47.1</td><td>25.8</td><td>46.7 57.8</td><td>136</td><td>31.7</td></tr><tr><td colspan="2">Effi-DETR [36]</td><td>300</td><td>1</td><td>45.1 63.1</td><td>49.1</td><td>28.3 48.4</td><td>59.0</td><td>210</td><td></td></tr><tr><td colspan="2">ASAG-D (Ours) ASAG-A (Ours)</td><td>253 102</td><td>1 1</td><td>45.8 64.1 45.3 63.3</td><td>49.4 48.9</td><td>27.3 49.6 27.3</td><td>61.0 48.5 59.7</td><td>182 131</td><td>19.7 28.9</td></tr></table>

Table 3: Comparison with other one-decoder-layer detectors with 36 epochs and R50. \*: Reimplement by us using official codes. F denotes GFLOPs.

<table><tr><td>Detector</td><td>#An</td><td>#L</td><td>AP AP50 AP75 APs</td><td>APm AP_l</td><td>F</td><td>FPS</td></tr><tr><td>Sparse RCNN [31]</td><td>100</td><td>6</td><td>42.8 61.2 45.7 26.7</td><td>44.6 57.6</td><td>134</td><td>26.0</td></tr><tr><td>Sparse RCNN [31]</td><td>300</td><td>6</td><td>45.0 63.4 48.2 26.9</td><td>47.2 59.5</td><td>152</td><td>25.4</td></tr><tr><td>CF-QRCNN [41]</td><td>100</td><td>2</td><td>43.0 61.3 46.8 28.3</td><td>45.7 55.5</td><td>142</td><td>34.1</td></tr><tr><td>CF-QRCNN [41]</td><td>300</td><td>2</td><td>44.6 63.1 48.9 29.5</td><td>47.4 57.5</td><td>148</td><td>33.6</td></tr><tr><td>ASAG-S (Ours)</td><td>100</td><td>1</td><td>43.9 62.2 47.9 27.8</td><td>46.5 57.8</td><td>130</td><td>30.1</td></tr><tr><td>ASAG-S (Ours)</td><td>312</td><td>1</td><td>45.0 64.1 49.1 29.5</td><td>47.4 57.8</td><td>136</td><td>29.2</td></tr><tr><td>Deform DETR↓: [43]</td><td>300</td><td>6</td><td>44.5 63.6 48.7 27.1</td><td>47.6 59.6</td><td>173</td><td>19.0</td></tr><tr><td>Deform DETR+↓: [43]</td><td>300</td><td>6</td><td>46.2 64.7 49.0 28.3</td><td>49.2 61.5</td><td>173</td><td>19.0</td></tr><tr><td>ASAG-D (Ours)</td><td>253</td><td>1</td><td>45.8 64.1 49.4 27.3</td><td>49.6 61.0</td><td>182</td><td>19.7</td></tr><tr><td>AdaMixer* [9]</td><td>100</td><td>6</td><td>45.6 64.8 49.3 28.8</td><td>48.5 60.9</td><td>103</td><td>23.8</td></tr><tr><td>AdaMixer [9]</td><td>300</td><td>6</td><td>47.0 66.0 51.1 30.1</td><td>50.2 61.8</td><td>125</td><td>22.2</td></tr><tr><td>AdaMixer† [9]</td><td>300</td><td>6</td><td>48.0 67.0 52.4 30.0</td><td>51.2 63.7</td><td>201</td><td>17.6</td></tr><tr><td>ASAG-A (Ours)</td><td>102</td><td>1</td><td>45.3 63.3 48.9 27.3</td><td>48.5 59.7</td><td>131</td><td>28.9</td></tr><tr><td>ASAG-A (Ours)</td><td>312</td><td>1</td><td>46.3 65.1 50.3 29.9</td><td>49.2 59.6</td><td>139</td><td>27.9</td></tr><tr><td>ASAG-A (Ours)†</td><td>296</td><td>1</td><td>47.5 66.1 51.2 30.4</td><td>50.6 62.6</td><td>206</td><td>21.3</td></tr></table>

Table 4: Comparison with six-decoder-layer detectors using 36 epochs and R50. \*: Reimplement by us using official codes. †: using R101. ‡: training with 50 epochs.

Anchor Generator alleviates the feature discrepancy caused by predicting from grids, ASAG-D outperforms Efficient DETR [36] by 0.7 AP and ASAG-S outperforms Featurized Query RCNN [41] by 2.6 AP, as shown in Table 3, showing the effectiveness of ASAG. Besides, different from computing densely on large feature maps, ASAG sparsely selects patches on different feature maps, saving much computation. Further, since the patch on $P _ { 6 }$ is the whole image, ASAG enjoys global receptive fields, bringing about much higher $\mathsf { A P } _ { l }$ compared with dense(grid)-initialized ones.

Comparison with six-decoder-layer detectors with the same decoder type. As shown in Table 4, while existing one-decoder-layer detectors with dense initialization fall behind their six-decoder-layer counterparts by a large margin, our model greatly narrows the performance gap. In particular, our ASAG-S even outperforms Sparse RCNN [31] in the 100 queries setting with faster speed and fewer FLOPs. More comparisons with other well-known detectors can be found in supplementary materials.

## 5.3. Ablation Studies

We conduct the following ablation studies with ASAG-A with R50, 100 queries, and 1× training schedule due to its fast convergence speed.

<table><tr><td>Dynamic #Query</td><td>Query Weighting</td><td>Auxiliary Replace</td><td></td><td>AP  $\mathrm { A P } _ { 5 0 } \mathrm { \ A P } _ { 7 5 } \mathrm { \ A P } _ { s } \mathrm { \ A P } _ { m } \mathrm { \ A P } _ { l }$  #An</td></tr><tr><td></td><td></td><td>Head</td><td>Anchor √</td><td>36.9 54.6 39.6 21.3 39.7 49.5 100</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>38.9 56.7 42.122.7 41.7 51.5 102</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>41.6 59.2 44.9 24.5 44.355.6103</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>42.6 60.5 45.8 25.9 45.8 56.9 107</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>42.4 60.2 46.0 25.2 45.5 56.9 121</td></tr></table>

(a) Ablation studies on each component. Replace Anchor denotes replace the selected anchors with newly generated ones in Adaptive Probing.

<table><tr><td rowspan=1 colspan=1>ηh</td><td rowspan=1 colspan=1>ηl</td><td rowspan=1 colspan=1> $\mathrm { A P \ A P _ { 5 0 } \ A P _ { 7 5 } \ A P _ { \it s } \ A P _ { \it m } \ A P _ { { \it l } } \# A n N _ { { \it P 4 } } N _ { \it P 3 } } $ </td></tr><tr><td rowspan=3 colspan=1>0.7</td><td rowspan=1 colspan=1>0.30.2</td><td rowspan=1 colspan=1>41.458.944.722.945.057.099 1.30.641.959.745.324.845.256.91042.21.6</td></tr><tr><td rowspan=2 colspan=1>0.10.05</td><td rowspan=1 colspan=1>42.660.545.825.945.856.9 1074.94.5</td></tr><tr><td rowspan=1 colspan=1>42.660.646.125.945.856.81249.910.9</td></tr><tr><td rowspan=4 colspan=1>0.80.70.60.4</td><td rowspan=4 colspan=1>0.1</td><td rowspan=1 colspan=1>42.560.445.825.945.756.91074.94.5</td></tr><tr><td rowspan=1 colspan=1>42.660.545.825.945.856.91074.94.5</td></tr><tr><td rowspan=1 colspan=1>42.560.445.825.945.7 56.91074.94.5</td></tr><tr><td rowspan=1 colspan=1>42.560.345.725.245.856.91084.9 4.5</td></tr></table>

(b) Ablation studies on Confidence Threshold for Adaptive Probing.

<table><tr><td rowspan=1 colspan=1>Size</td><td rowspan=1 colspan=1> $\mathrm { A P \ A P _ { 5 0 } \ A P _ { 7 5 } \ A P _ { { s } } \ A P _ { { m } } \ A P _ { { l } } \# A n N _ { { P } 4 } N _ { { P } 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=2 colspan=1>42.159.845.624.844.756.61064.43.642.360.145.925.345.256.8 1084.94.3</td></tr><tr><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>1542.6</td><td rowspan=1 colspan=1>60.545.8 25.9 45.8 56.9 1074.94.5</td></tr><tr><td rowspan=1 colspan=1>1842.2</td><td rowspan=1 colspan=1>60.145.7 24.6 45.3 56.5 1054.84.6</td></tr></table>

(c) Ablation studies on Patch Size for Adaptive Probing.

<table><tr><td rowspan=1 colspan=1> $\eta _ { i o u } \</td><td rowspan=1 colspan=1>Big | \mathrm { A P } \mathrm { \ A P { } _ { 5 0 } \ A P { } _ { 7 5 } \ A P { } _ { s } \ A P { } _ { m } \ A P { } _ { l } \# \mathrm { A n N } _ { P 4 } \mathrm { N } _ { P 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>42.560.945.525.345.457.21177.47.3</td></tr><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>42.6 60.545.8 25.9 45.8 56.9 1074.94.5</td></tr><tr><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>42.460.345.6 25.3 45.7 56.811044.33.9</td></tr><tr><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>42.259.945.7 24.7 45.856.9983.32.9</td></tr></table>

(d) Ablation studies on NMS Threshold for Adap tive Probing.

<table><tr><td>Method</td><td> $\mathbf { \Delta A R _ { 1 0 0 } ^ { 1 0 0 0 } A R _ { 3 0 0 } ^ { 1 0 0 0 } }$ </td><td></td></tr><tr><td>RPN [27] CF-QRCNN [41]</td><td>= 57.31</td><td>61.93 63.42</td></tr><tr><td>Dynamic Anchors</td><td>45.22</td><td>50.21</td></tr><tr><td>Dynamic Proposals</td><td>59.28</td><td>64.90</td></tr></table>

(e) Comparison on AR with the 3× recipe.

Table 5: Experiment results of ASAG-A with the 1× training recipe and 100 anchors on COCO val except for AR. The settings in our default model are colored in gray . #An denotes the average number of dynamic anchors. $\Nu _ { P 4 }$ and $\mathrm { N } _ { P 3 }$ denote the number of patches used in Adaptive Probing on corresponding feature maps.  
![](images/c8130fa5bbb15b49442e2cddd5777c78c29ed8d6af494506eec14fc87b51e2cb.jpg)

![](images/a1538b077928dc19dfcae80ac4141737249e23c386936e069b65298cf7979fdf.jpg)  
Figure 6: Statistic for Anchor Generator. Left: Correlation between the number of ground truth boxes and the generated anchors. Right: Histogram of used feature map level in Anchor Generator and the number of images.

Main components. In Table 5a, we ablate the main components newly introduced in our model. First, since anchors predicted from different feature maps using different predictors can not be treated equally, using dynamic number of anchors rather than fixed topk anchors based on scores is more appropriate, adding 2.0 AP. From the last row, we also find that preserving the selected anchors in Adaptive Probing brings no help but with more anchors, showing that the generated ones on larger feature maps are better than the selected ones and anchors cannot be sorted by scores equally. Second, Query Weighting stabilizes training and brings 2.7 AP gains. Further, providing adequate supervision signals is crucial for sparse detectors and using extra auxiliary parallel one-layer decoders increases 1.0 AP.

Patch size for Adaptive Probing. Since patches cropped on different feature maps predict anchors independently, patch sizes for each level can be different. For simplicity, we use the same size for each level in Adaptive Probing and share the MLP predictor. In Table 5c, large and small patch sizes are both inappropriate for small objects. For small patch sizes, fewer anchors are selected in Adaptive Probing due to length constraints, resulting in fewer patches. And relatively larger context is needed to detect small and middle objects, similar to the findings in [6]. As for large patch sizes, the predictor overlooks some small objects since we predict sparsely and without using spatial priors.

Confidence threshold for Adaptive Probing. In the upper part of Table 5b, relatively high η<sub>l</sub> ignores some possible small objects thus resulting in fewer anchors and patches and poor performance on small objects. Threshold η<sub>l</sub> lower than 0.1 is not necessary since low confidence anchors tend to be noisy anchors. The lower part shows Adaptive Probing is robust to $\eta _ { h }$ . Note that Adaptive Probing will not affect $\mathsf { A P } _ { l }$

NMS threshold for Adaptive Probing. To reduce redundancy, we use NMS to reduce overlapped patches and the threshold affects the number of patches a lot. In Table 5d, a low threshold reduces some useful patches mistakenly.

Comparison on AR. To validate the quality of our dynamic anchors and dynamic proposals, in Table 5e, we compare $\mathbf { A } \mathbf { R } ^ { 1 0 0 0 }$ with well-known RPN [27] and QGN used in [41]. $\mathbf { A R } _ { 1 0 0 } ^ { 1 0 0 0 }$ means $\mathbf { A } \mathbf { R } ^ { 1 0 0 0 }$ under the 100 queries setting. Although our model lacks dense priors, like anchor boxes or anchor points, dynamic proposals still outperform RPN and QGN in terms of $\mathbf { A } \mathbf { R } ^ { 1 0 0 0 }$ . And our Anchor Generator is lightweight using only 0.06 GFLOPs.

<table><tr><td>ηh</td><td>ηl</td><td>AP(↑)</td><td>mMR(↓)</td><td>R(↑)</td><td>ηiou</td><td>AP(↑)</td><td>mMR(↓)</td><td>R(↑)</td></tr><tr><td rowspan="4">0.7</td><td>0.3</td><td>90.8</td><td>44.0</td><td>96.4</td><td>0.4</td><td>90.6</td><td>43.7</td><td>96.0</td></tr><tr><td>0.2</td><td>91.2</td><td>43.7</td><td>96.7</td><td>0.25</td><td>91.3</td><td>43.5</td><td>96.9</td></tr><tr><td>0.1</td><td>91.3</td><td>43.5</td><td>96.9</td><td>0.2</td><td>91.2</td><td>43.8</td><td>96.8</td></tr><tr><td>0.05</td><td>91.4</td><td>43.5</td><td>96.9</td><td>0.1</td><td>90.2</td><td>44.4</td><td>95.6</td></tr><tr><td>0.8</td><td rowspan="4">0.1</td><td>91.3</td><td>43.5</td><td>96.9</td><td>Sparse*</td><td>89.2</td><td>48.3</td><td></td></tr><tr><td>0.7</td><td>91.3</td><td>43.5</td><td>96.9</td><td>RCNN</td><td></td><td></td><td>95.9</td></tr><tr><td>0.6</td><td>91.3</td><td>43.5</td><td>96.9</td><td>Deformable*</td><td>86.7</td><td>54.0</td><td>92.5</td></tr><tr><td>0.5</td><td>91.2</td><td>43.6</td><td>96.7</td><td>DETR</td><td></td><td></td><td></td></tr></table>

Table 6: CrowdHuman results on different Confidence Thresholds and NMS Thresholds for Adaptive Probing using ASAG-S. Rows in gray denote the default settings on COCO. \*: Results are taken from Sparse RCNN[31].

Distribution of the number of dynamic anchors. As shown in the left part of Figure 6, there is a clear positive correlation between the number of ground truth boxes and the generated anchors, showing that the Anchor Generator is adaptive to different images by generating more queries for difficult images and vice versa.

Distribution of the number of used feature map levels. Our Adaptive Probing method enjoys the early-stop mechanism. As shown in the right part of Figure 6, roughly 40% of the images in the validation set do not use all the feature maps, saving some computation.

Robustness of hyper-parameters in Adaptive Probing. In Adaptive Probing, we only select anchors whose confidences fall in [η<sub>l</sub>, η<sub>h</sub>] to crop patches in the larger feature maps and the patches are filtered by NMS with IoU threshold $\eta _ { i o u }$ for reducing redundancy. To show the robustness of these hyper-parameters, we follow Sparse RCNN [31] to conduct experiments on CrowdHuman [30] dataset, which is significantly different from COCO. Following Sparse RCNN, we run ASAG-S with 50 epochs and the average number of anchors within 500. As shown in Table 6, the default hyper-parameters of Adaptive Probing on COCO still work on CrowdHuman. And ASAG-S outperforms Sparse RCNN and Deformable DETR by a large margin.

## 5.4. Visualization

Comparison on feature maps. In this work, we provide a sparse way to initialize object queries by using patches as the prediction units thus alleviating the feature map discrepancy caused by predicting on grids. As shown in Figure 7, feature maps from our method are more similar to six-decoder-layer sparse detectors, which activate objects in an adaptive way rather than uniformly.

Visualization for Adaptive Probing. We visualize some results to understand how Anchor Generator works. More pictures will be displayed in the supplementary materials. In Figure 8, we draw dynamic anchors in white and patches in red. As shown in the first column, the anchors for different images are different and have covered most foreground objects, showing them truly adaptive and precise. Anchor Generator tends to generate more anchors for small objects through Adaptive Probing, which greatly enhances the ability to detect small objects. Although the red boxes, i.e. patches, sparsely locate on the images, they precisely cover small objects, such as the things on the table and the spectators in the stands, greatly increasing the recall rate.

![](images/5c2c018f2c3fbb616f11a1517501d5b9fabeb77b0f31150a72c2823ef5b8aa2a.jpg)  
Figure 7: Visualization of feature maps of different methods. Our models with sparse initialization are more consistent with sparse decoders.

![](images/ff683592d4687e8000ca05e625f68c7ec0e583cfe3bfeed28fc3e99dd98c6310.jpg)  
Figure 8: Visualization of dynamic anchors and Adaptive Probing. The white and red boxes represent anchors and patches, respectively.

<table><tr><td>Detector</td><td>Query Weighting</td><td>AP</td><td> $\overline { { \mathbf { A P 5 0 } } }$ </td><td>AP75</td><td>AP_s</td><td>APm</td><td>AP_{</td></tr><tr><td rowspan="2">AdaMixer [9]</td><td>√</td><td>39.2</td><td>47.7</td><td>42.1</td><td>21.5</td><td>42.2</td><td>55.3</td></tr><tr><td></td><td>42.7</td><td>61.5</td><td>45.9</td><td>24.7</td><td>45.4</td><td>59.2</td></tr></table>

Table 7: Query Weighting for six-decoder-layer detectors.

<table><tr><td>Detector</td><td>#L</td><td>AP</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathrm { A P _ { 7 5 } }$ </td><td> $\boldsymbol { \mathrm { A P } _ { s } }$ </td><td> $\mathsf { A P } _ { m }$ </td><td> $\overline { { \mathsf { A P } _ { l } } }$ </td></tr><tr><td>ASAG-A</td><td>1</td><td>42.6</td><td>60.5</td><td>45.8</td><td>25.9</td><td>45.8</td><td>56.9</td></tr><tr><td>ASAG-A</td><td>2</td><td>43.2</td><td>61.6</td><td>46.9</td><td>25.6</td><td>46.4</td><td>59.0</td></tr><tr><td>AdaMixer</td><td>6</td><td>42.7</td><td>61.5</td><td>45.9</td><td>24.7</td><td>45.4</td><td>59.2</td></tr></table>

Table 8: Effect of different number of decoder layers.

## 5.5. Discussion

Is Query Weighting beneficial to six-decoder-layer detectors? Most six-decoder-layer detectors and dense ones detect objects from image-agnostic and fixed queries (or anchors). The quality of initial queries can not be improved during the training process. Thus each decoder layer should be trained under difference IoU level [1] and Query Weighting can not benefit them, as shown in Table 7.

Is it equal to increase the same number of anchors for one- and six-decoder-layer detectors? Since six-decoderlayer detectors work well with image-agnostic and fixed queries, query initialization brings minor help to them. But more queries for them narrow the search space for each query. However, query initialization is vital to one-decoderlayer detectors and directly affects their performance [36]. As shown in Table 5e, increasing queries from 100 to 300 only brings 5.6 AR since the top100 proposals already lie in the top300 ones. Thus the benefit of increasing queries for one-decoder-layer detectors is less than for six-decoderlayer ones, shown in Table 2 and Table 4. How to scale up one-decoder-layer detectors needs future work.

More analysis on $\mathbf { A P } _ { l } .$ A core advantage of DETR-like detectors is that they reason globally so that the $\mathsf { A P } _ { l }$ is significantly higher than traditional detectors. However, although our ASAGs are comparable on AP with corresponding sixdecoder-layer counterparts, the $\mathsf { A P } _ { l }$ still lags behind them, as shown in Table 4. And it seems that the $\mathsf { A P } _ { l }$ of ASAGs cannot increase when using more queries just as the counterparts. Here we provide some analysis.

The $\mathsf { A P } _ { l }$ is affected by two factors: the number of decoder layers and query initialization. Regarding the number of decoder layers, we run ASAG-A with two decoder layers in Table 8 and get 43.2AP and 59.0AP<sub>l</sub> (vs one layer 42.6AP and 56.9AP<sub>l</sub>), in which $\mathsf { A P } _ { l }$ is already comparable with AdaMixer. As for query initialization, ASAG views the whole image in $P _ { 6 }$ as a patch and predicts anchors from it, enjoying global receptive fields during initialization, and thus shows superiority to dense-initialized onedecoder-layer detectors on $\mathsf { A P } _ { l }$ in Table 3.

Besides, the phenomenon that $\mathsf { A P } _ { l }$ does not improve with more queries also occurred in another one-decoderlayer detector (see F-QRCNN in Table 3), showing it is not specific to ASAG. Since one-decoder-layer detectors use image-specific queries and large objects are relatively easy to detect, the initial queries are accurate enough, and increasing queries brings a little recall for large objects. In contrast, queries of six-decoder-layer detectors are randomly initialized. Thus, the gains on $\mathsf { A P } _ { l }$ for one-decoderlayer detectors are smaller than six-decoder-layer counterparts when using more queries.

## 6. Conclusion

In this work, we find that dense initialization is not optimal for one-decoder-layer sparse detectors since predicting on grids leads to feature conflict, which hampers their performance. To tackle this problem, we propose ASAG and predict dynamic anchors based on patches in a sparse way, thus alleviating feature conflict. We further design Adaptive Probing to generate patches on different levels, which is adaptive to the number of used feature maps, the number of patches, and the locations of patches. Finally, simple but effective Query Weighting stabilizes the training. With the novel design, our dynamic anchors and proposals are better than dense ones without using predefined spatial priors. Experiments show that we greatly increase the performance of one-decoder-layer detectors and narrow the performance gap while retaining the fast speed.

Acknowledgments. This work was supported partially by the NSFC (U21A20471, U1911401, 62072482), Guangdong NSF Project (No. 2023B1515040025, 2020B1515120085).

## References

[1] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In CVPR, 2018. 9

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 1, 2

[3] Jiahao Chang, Shuo Wang, Guangkai Xu, Zehui Chen, Chenhongyi Yang, and Feng Zhao. Detrdistill: A universal knowledge distillation framework for detr-families. arXiv preprint arXiv:2211.10156, 2022. 2, 3

[4] Qiang Chen, Xiaokang Chen, Gang Zeng, and Jingdong Wang. Group detr: Fast training convergence with decoupled one-to-many label assignment. arXiv preprint arXiv:2207.13085, 2022. 2, 3

[5] Xiaokang Chen, Jiahui Chen, Yan Liu, and Gang Zeng. D ˆ3 etr: Decoder distillation for detection transformer. arXiv preprint arXiv:2211.09768, 2022. 2

[6] Yukang Chen, Yanwei Li, Tao Kong, Lu Qi, Ruihang Chu, Lei Li, and Jiaya Jia. Scale-aware automatic augmentation for object detection. In CVPR, 2021. 7

[7] Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R Scott, and Weilin Huang. Tood: Task-aligned one-stage object detection. In ICCV, 2021. 5

[8] Peng Gao, Minghang Zheng, Xiaogang Wang, Jifeng Dai, and Hongsheng Li. Fast convergence of detr with spatially modulated co-attention. In ICCV, 2021. 2

[9] Ziteng Gao, Limin Wang, Bing Han, and Sheng Guo. Adamixer: A fast-converging query-based object detector. In CVPR, 2022. 1, 2, 3, 5, 6, 9

[10] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-´ shick. Mask r-cnn. In ICCV, 2017. 1, 3

[11] Qinghang Hong, Fengming Liu, Dong Li, Ji Liu, Lu Tian, and Yi Shan. Dynamic sparse r-cnn. In CVPR, 2022. 2

[12] Linjiang Huang, Kaixin Lu, Guanglu Song, Liang Wang, Si Liu, Yu Liu, and Hongsheng Li. Teach-detr: Better training detr with teachers. arXiv preprint arXiv:2211.11953, 2022. 2

[13] Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, and Han Hu. Detrs with hybrid matching. arXiv preprint arXiv:2207.13080, 2022. 2, 3

[14] Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, and Han Hu. Detrs with hybrid matching. arXiv preprint arXiv:2207.13080, 2022. 2, 3

[15] Alexander Kirillov, Yuxin Wu, Kaiming He, and Ross Girshick. Pointrend: Image segmentation as rendering. In CVPR, 2020. 5

[16] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and Lei Zhang. Dn-detr: Accelerate detr training by introducing query denoising. In CVPR, 2022. 2

[17] Shuai Li, Chenhang He, Ruihuang Li, and Lei Zhang. A dual weighting label assignment scheme for object detection. In CVPR, 2022. 5, 6

[18] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. In NeurIPS, 2020. 4, 5

[19] Yunsheng Li, Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Pei Yu, Ying Jin, Lu Yuan, Zicheng Liu, and Nuno Vasconcelos. Should all proposals be treated equally in object detection? In ECCV, 2022. 2

[20] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He,´ Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 1

[21] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ ICCV, 2017. 1

[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2, 5

[23] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. In ICLR, 2022. 2

[24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 6

[25] Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, and Jingdong Wang. Conditional detr for fast training convergence. In ICCV, 2021. 2

[26] Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, and Philipp Krahenb¨ uhl. Nms strikes back.¨ arXiv preprint arXiv:2212.06137, 2022. 2, 3

[27] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In NeurIPS, 2015. 1, 7

[28] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In CVPR, 2019. 6

[29] Byungseok Roh, JaeWoong Shin, Wuhyun Shin, and Saehoon Kim. Sparse detr: Efficient end-to-end object detection with learnable sparsity. In ICLR, 2022. 2

[30] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123, 2018. 8

[31] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In CVPR, 2021. 1, 2, 3, 5, 6, 8

[32] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In ICCV, 2019. 1

[33] Tao Wang, Li Yuan, Yunpeng Chen, Jiashi Feng, and Shuicheng Yan. Pnp-detr: towards efficient visual analysis with transformers. In ICCV, 2021. 2

[34] Yingming Wang, Xiangyu Zhang, Tong Yang, and Jian Sun. Anchor detr: Query design for transformer-based detector. In AAAI, 2022. 2, 6

[35] Chenhongyi Yang, Zehao Huang, and Naiyan Wang. Querydet: Cascaded sparse query for accelerating high-resolution small object detection. In CVPR, 2022. 4, 5

[36] Zhuyu Yao, Jiangbo Ai, Boxun Li, and Chi Zhang. Efficient detr: improving end-to-end object detector with dense prior. arXiv preprint arXiv:2104.01318, 2021. 1, 3, 5, 6, 9

[37] Gongjie Zhang, Zhipeng Luo, Yingchen Yu, Kaiwen Cui, and Shijian Lu. Accelerating detr convergence via semanticaligned matching. In CVPR, 2022. 2

[38] Gongjie Zhang, Zhipeng Luo, Yingchen Yu, Jiaxing Huang, Kaiwen Cui, Shijian Lu, and Eric P Xing. Semantic-aligned matching for enhanced detr convergence and multi-scale feature fusion. arXiv preprint arXiv:2207.14172, 2022. 2

[39] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 2

[40] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In CVPR, 2020. 1

[41] Wenqiang Zhang, Tianheng Cheng, Xinggang Wang, Qian Zhang, and Wenyu Liu. Featurized query r-cnn. arXiv preprint arXiv:2206.06258, 2022. 1, 2, 3, 5, 6, 7

[42] Minghang Zheng, Peng Gao, Renrui Zhang, Kunchang Li, Xiaogang Wang, Hongsheng Li, and Hao Dong. End-toend object detection with adaptive clustering transformer. In BMVC, 2021. 2

[43] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. In ICLR, 2021. 1, 2, 3, 5, 6

[44] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. arXiv preprint arXiv:2211.12860, 2022. 2, 3