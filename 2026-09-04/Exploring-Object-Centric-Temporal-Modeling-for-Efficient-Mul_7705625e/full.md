# Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection

Shihao Wang<sup>1†</sup> Yingfei Liu<sup>2</sup> Tiancai Wang<sup>2‡</sup> Ying Li<sup>1‡</sup> Xiangyu Zhang<sup>2</sup> <sup>1</sup>Beijing Institute of Technology <sup>2</sup>MEGVII Technology

{wangshihao, ying.li}@bit.edu.cn {liuyingfei, wangtiancai, zhangxiangyu}@megvii.com

![](images/93a8365dfe32286252dd7dba77366daabe2c68f30e6094c5f8d46c64ff5e22b6.jpg)  
(a) Dense BEV Method

![](images/d882043a5c6f38f05c418e0c13416c5594e3ae946840e0ba6ffe01a181b5adb6.jpg)  
(b) Perspective Method

![](images/a2ad47ce4cc55252a8826a3e289b76fc898669b3c7fc3b439ba70bbd4d244884.jpg)  
(c) Object-Centric Method  
Figure 1. Different temporal fusion methods from bird-eye-view (BEV) space, perspective view, and our proposed object-centric. RF indicates receptive field. The solid lines and dotted lines represent spatial and temporal operations respectively.

## Abstract

In this paper, we propose a long-sequence modeling framework, named StreamPETR, for multi-view 3D object detection. Built upon the sparse query design in the PETR series, we systematically develop an object-centric temporal mechanism. The model is performed in an online manner and the long-term historical information is propagated through object queries frame by frame. Besides, we introduce a motion-aware layer normalization to model the movement of the objects. StreamPETR achieves significan performance improvements only with negligible computation cost, compared to the single-frame baseline. On the standard nuScenes benchmark, it is the first online multiview method that achieves comparable performance (67.6% NDS & 65.3% AMOTA) with lidar-based methods. The lightweight version realizes 45.0% mAP and 31.7 FPS, outperforming the state-of-the-art method (SOLOFusion) by 2.3% mAP and 1.8× faster FPS. Code has been available at https://github.com/exiawsh/StreamPETR.git.

## 1. Introduction

Camera-only 3D detection is crucial for autonomous driving because of the low deployment cost and ease of detecting road elements. Recently, multi-view object detection has made remarkable progress by leveraging temporal information [27, 16, 31, 25, 39, 29]. The historical features facilitate the detection of occlusion objects and greatly improve the performance. According to the differences between temporal representations, existing methods can be roughly divided into BEV temporal and perspective temporal methods.

BEV temporal methods [27, 16, 25, 39] explicitly warp BEV features from historical to current frame, as illustrated in Fig. 1 (a), where BEV features serve as an efficient intermediate representation for temporal modeling. However, the highly structured BEV features limit the modeling of moving objects. This paradigm requires a large receptive field to alleviate this problem [16, 39, 27].

Different from these approaches, perspective temporal methods [31, 29] are mainly based on DETR [4, 61]. The sparse query design facilitates the modeling of moving objects [29]. However, the sparse object queries need to interact with multi-frame image features for long-term temporal dependence (see Fig. 1 (b)), leading to multiple computations. Thus, existing works are either stuck in solving the moving objects or introducing multiple computation costs.

Based on the above analysis, we suppose it is possible to employ sparse queries as the hidden states of temporal propagation. In this way, we can utilize object queries to model moving objects while keeping high efficiency. Therefore, we introduce a new paradigm: object-centric temporal modeling and design an efficient framework, termed StreamPETR, as shown in Fig. 1 (c). StreamPETR directly performs frame-by-frame 3D predictions on streaming video. It is effective for motion modeling and is able to build long-term spatial-temporal interaction.

![](images/b3e1808fdecd18736c884c9ae9f349802ebb0de95d95109b4d4c304b5b129e84.jpg)  
Figure 2. The speed-accuracy trade-off of different models on nuScenes val set. The inference speed is calculated on RTX3090 GPU in online streaming video. T indicates the model with temporal modeling.

Specifically, a memory queue is first built to store the historical object queries. Then a propagation transformer conducts long-range temporal and spatial interaction with current object queries. The updated object queries are used to generate 3D bounding boxes and pushed into the memory queue. Besides, a motion-aware layer normalization (MLN) is introduced to implicitly encode the motion of the ego vehicle and surrounding objects at different time stamps.

Compared with existing temporal methods, the proposed object-centric temporal modeling brings several advantages. StreamPETR only processes a small number of object queries instead of dense feature maps at each time stamp, consuming negligible computational burden (as shown in Fig. 2). For moving objects, MLN alleviates the cumulative error in video streaming. Except for the location prior used in previous methods, StreamPETR additionally considers the semantic similarity by global attention, which facilitates the detection in motion scenes. To summarize, our contributions are:

• We pull out the key of streaming multi-view 3D detection and systematically design an object-centric temporal modeling paradigm. The long-term historical information is propagated through object queries frame by frame.

• We develop an object-centric temporal modeling framework, termed StreamPETR. It models moving objects and long-term spatial-temporal interaction simultaneously, consuming negligible storage and computation costs.

• On the standard nuScenes dataset, StreamPETR outperforms all online camera-only algorithms. Extensive experiment shows that it can be well generalized to other sparse query-based methods, e.g. DETR3D [47].

## 2. Related Work

## 2.1. Multi-view 3D Object Detection

Multi-view 3D detection is an important task in autonomous driving, which needs to continuously process multi-camera images and predict 3D bounding boxes over time. Pioneer’s works [47, 30, 18, 27, 19, 48] focus on the efficient transformation from multiple perspective views to a unified 3D space in a single frame. The transformation can be divided into BEV-based methods [18, 27, 50, 15, 25, 19] and sparse query based methods [47, 30, 29, 5, 48]. To alleviate the occlusion problem and ease the difficulty of speed prediction, recent works additionally introduce temporal information to extend these two paradigms.

It is relatively intuitive to extend the single-frame BEV methods for temporal modeling. BEVFormer [27] first introduces sequential temporal modeling into multi-view 3D object detection and applies temporal self-attention. BEVDet series [16, 25, 23] use concatenate operation to fuse the adjacent BEV features and achieve remarkable results. Furthermore, SOLOFusion [39] extends BEVStereo [23] to long-term memory and reaches a promising performance. Without an intermediate feature representation, the temporal modeling of query-based methods is more challenging. PETRv2 [31] performs the global cross-attention, while DETR4D [34] and Sparse4D [29] apply sparse attention to model the interaction between multiframes, which introduce multiple computations. However, the sparse query design is convenient to model the moving objects [29]. In order to combine the advantages of the two paradigms, we utilize sparse object queries as the intermediate representation, which can model moving objects and efficiently propagate long-term temporal information.

## 2.2. Query Propagation

Since DETR [4] is proposed in 2D object detection, the object query has been applied in many downstream tasks [55, 35, 58, 57, 12] to model the temporal interaction. For video object detection, LWDN [20] adopts a braininspired memory mechanism to propagate and update the memory feature. QueryProp [12] performs query interaction to reduce the computational cost on non-key frames. It achieves significant improvements and maintains high efficiency. 3D-MAN [53] has a similar idea and extends a single-frame Lidar detector to multi-frames, which effectively combines the features coming from different perspectives of a scene. In object tracking, MOTR [55] and Track-Former [35] propose the track query to model the object association across frames. MeMOT [2] employs a memory bank to build long temporal dependence, which further boosts performance. MOTRv2 [58] eases the conflict between the detection and association tasks by incorporating an extra detector. MUTR [57] and PF-Track [36] extend MOTR [55] into multi-view 3D object tracking and achieve a promising result.

![](images/9a9ae6494ed9b8e9dd3c988be9eefaf8b9fbd7ff0f395ef248efe8c326e384bf.jpg)  
Figure 3. Overall architecture of the proposed StreamPETR. The memory queue stores the historical object queries. In the propagation transformer, recent object queries successively interact with historical queries and current image features to obtain temporal and spatial information. The output queries are further used to generate detection results and the top-K foreground queries are pushed into the memory queue. Through the recurrent update of the memory queue, the long-term temporal information is propagated frame by frame.

## 3. Delving into Temporal Modeling

To facilitate our study, we present a generalized formulation for various temporal modeling designs. Given the perspective view features $F _ { 2 d } = \{ F _ { 2 d } ^ { 0 } \cdot \cdot \cdot F _ { 2 d } ^ { t } \}$ , dense BEV features $F _ { b e v } = \{ F _ { b e v } ^ { 0 } \cdot \cdot \cdot F _ { b e v } ^ { t } \}$ and sparse object features $F _ { o b j } = \{ F _ { o b j } ^ { 0 } \cdot \cdot \cdot F _ { o b j } ^ { \bar { t } ^ { - } } \}$ . The dominant temporal modeling methods can be formulated as:

$$
\tilde { F } _ { o u t } { = } { \varphi } ( F _ { 2 d } , F _ { b e v } , F _ { o b j } )\tag{1}
$$

where $\varphi$ is the temporal fusion operation, $\tilde { F } _ { o u t }$ is the output feature that includes temporal information. We first describe the existing temporal modeling from BEV and perspective view. After that, the proposed object-centric temporal modeling is elaborated.

BEV Temporal Modeling uses the grid-structured BEV features to perform the temporal fusion. To compensate for the ego vehicle motion, the last frame feature $F _ { b e v } ^ { \dot { t } - 1 }$ is usually aligned to the current frame.

$$
\tilde { F } _ { b e v } ^ { t } = \varphi ( F _ { b e v } ^ { t - 1 } , F _ { b e v } ^ { t } )\tag{2}
$$

Then a temporal fusion function $\varphi$ (concatenation [16, 25] or deformable attention [27]) can be applied for intermediate temporal representation $\tilde { F } _ { b e v } ^ { t }$ . Extending the above process to long temporal modeling, there are two main routes.

The first one is to align the historical k BEV features and concatenate them with the current frame.

$$
\tilde { F } _ { b e v } ^ { t } = \varphi ( F _ { b e v } ^ { t - k } , \cdots , F _ { b e v } ^ { t - 1 } , F _ { b e v } ^ { t } )\tag{3}
$$

For another one, the long-term historical information is propagated through the hidden states of BEV features $\tilde { F } _ { b e v } ^ { t - 1 }$ in a recurrent manner.

$$
\tilde { F } _ { b e v } ^ { t } = \varphi ( \tilde { F } _ { b e v } ^ { t - 1 } , F _ { b e v } ^ { t } )\tag{4}
$$

However, the BEV temporal fusion only considers the static BEV features and ignores the movement of the objects, leading to spatial dislocation.

Perspective Temporal Modeling is mainly performed via interactions between object queries and perspective features. The temporal function $\varphi$ is usually achieved by the spatial cross-attention [31, 29, 34]:

$$
\tilde { F } _ { o b j } ^ { t } = \varphi ( F _ { 2 d } ^ { t - k } , F _ { o b j } ^ { t } ) \cdot \cdot \cdot + \varphi ( F _ { 2 d } ^ { t } , F _ { o b j } ^ { t } )\tag{5}
$$

The cross-attention between object query and multiframe perspective view requires repeated feature aggregation. Simply extending to long-term temporal modeling greatly increases the computation cost.

Object-centric Temporal Modeling is our proposed object-centric solution, which models the temporal interaction by object queries. Through object queries, the motion compensation can be conveniently applied based on estimated states $F _ { o b j } ^ { t - 1 }$

$$
\tilde { F } _ { o b j } ^ { t - 1 } = \mu ( F _ { o b j } ^ { t - 1 } , M )\tag{6}
$$

![](images/0fe23d51edadf53c1867637c0f6895d6af16afdb96038c4507b532a1c9d69d05.jpg)  
Figure 4. The details of the propagation transformer and motion-aware layer normalization. In the propagation Transformer [43], object queries interact with hybrid queries and image features iteratively. The motion-aware layer normalization encodes the motion attributes (ego pose, timestamps, velocity) and performs a compensation implicitly. Rectangles of varying hues symbolize queries from distinct frames, gray rectangles represent initialized queries of current frame, dashed rectangles correspond to background queries.

where $\mu$ is an explicit linear velocity model or implicit function to encode motion attributes M (including the relative time interval $\triangle t ,$ estimated velocity v, and ego-pose matrix E, which are the same definition in Sec. 4). Further, a global attention $\varphi$ is constructed to propagate temporal information through object queries frame by frame:

$$
\tilde { F } _ { o b j } ^ { t } = \varphi ( \tilde { F } _ { o b j } ^ { t - 1 } , F _ { o b j } ^ { t } )\tag{7}
$$

## 4. Method

## 4.1. Overall Architecture

As illustrated in Fig. 3, StreamPETR is built upon endto-end sparse query-based 3D object detectors [30, 47]. It consists of an image encoder, a recursively updated memory queue, and a propagation transformer [43]. The image encoder is a standard 2D backbone, which is applied to extract semantic features from multi-view images. Then the extracted features, information in the memory queue, and object queries are fed into the propagation transformer to perform the spatial-temporal interaction. The main difference between StreamPETR and single-frame baseline is the memory queue, which recursively updates the temporal information of object queries. Combined with the propagation transformer, the memory queue can propagate temporal priors from previous to current frames efficiently.

## 4.2. Memory Queue

We design a memory queue of $N \times K$ for effective temporal modeling. N is the number of stored frames and K is the number of objects stored per frame. According to the experience, we set $N = 4$ and $K = 2 5 6$ (ensuring high recall in complex scenarios). After the preset time interval τ , the relative time interval $\triangle t .$ , context embedding $Q _ { c } ,$ , object center $Q _ { p } .$ , velocity v, and ego-pose matrix E of selected object queries are stored in memory queue. Specifically, the above information, corresponding to foreground objects (with top-K highest classification score), is selected and pushed into the memory queue. The entrance and exit of the memory queue follow the first-in, first-out (FIFO) rule. When information from a new frame is added to the memory queue, the oldest is discarded. Actually, the proposed memory queue is highly flexible and customized, users can freely control the maximal memory size $N \times K$ and saving interval τ during both training and inference.

## 4.3. Propagation Transformer

As illustrated in Fig. 4, the propagation transformer consists of three main components: (1) the motion-aware layer normalization module implicitly updates the object state according to the context embedding and motion information recorded in the memory queue; (2) the hybrid attention replaces the default self-attention operation. It plays the role of temporal modeling and removing duplicated predictions; (3) cross-attention is adopted for feature aggregation. It can be replaced with an arbitrary spatial operation to build the relationship between image tokens and 3D object queries, such as global attention in PETR [30] or sparse projective attention in DETR3D [47].

Motion-aware Layer Normalization is designed to model the movement of objects. For simplicity, we take the transformation process from the last frame t − 1 as the example and adopt the same operation for other previous frames. Given the ego pose matrix from the last frame $E _ { t - 1 }$ and current frame $E _ { t }$ , the ego transformation $E _ { t - 1 } ^ { t }$ can be calculated as:

$$
E _ { t - 1 } ^ { t } = \ E _ { t } ^ { i n v } \cdot \ E _ { t - 1 }\tag{8}
$$

Assume that objects are static, 3D centers $Q _ { p } ^ { t - 1 }$ in memory queue can be explicitly aligned to the current frame, which is formulated as:

$$
\tilde { Q } _ { p } ^ { t } = E _ { t - 1 } ^ { t } \cdot Q _ { p } ^ { t - 1 }\tag{9}
$$

where $\tilde { Q } _ { p } ^ { t }$ is the aligned centers. Motivated by the taskspecific control in generative model [8, 40, 56], we adopt a conditional layer normalization to model the movement of the objects. As shown in Fig. 4, the default affine transformation in layer normalization (LN) is closed. The motion attributes $( E _ { t - 1 } ^ { t } , v , \triangle t )$ are flattened and converted to affine vectors γ and β by two linear layers $( \xi _ { 1 } , \xi _ { 2 } )$ :

Table 1. Comparison on the nuScenes val set. <sup>∗</sup>Benefited from the perspective-view pre-training. <sup>‡</sup> 300 randomly initialized queries and 128 propagation queries. <sup>†</sup> Offline method using future frames. FPS is measured on RTX3090 with fp32.
<table><tr><td>Methods</td><td>Backbone</td><td>Image Size</td><td>Frames</td><td>mAP↑</td><td>NDS↑</td><td>mATE↓</td><td>mASE↓</td><td>mAOE↓</td><td>mAVE↓</td><td>mAAE↓</td><td>FPS↑</td></tr><tr><td>BEVDet [18]</td><td>ResNet50</td><td>256 × 704</td><td>1</td><td>0.298</td><td>0.379</td><td>0.725</td><td>0.279</td><td>0.589</td><td>0.860</td><td>0.245</td><td>16.7</td></tr><tr><td>BEVDet4D [16]</td><td>ResNet50</td><td>256 × 704</td><td>2</td><td>0.322</td><td>0.457</td><td>0.703</td><td>0.278</td><td>0.495</td><td>0.354</td><td>0.206</td><td>16.7</td></tr><tr><td>PETRv2 [31]</td><td>ResNet50</td><td>256 × 704</td><td>2</td><td>0.349</td><td>0.456</td><td>0.700</td><td>0.275</td><td>0.580</td><td>0.437</td><td>0.187</td><td>18.9</td></tr><tr><td>BEVDepth [25]</td><td>ResNet50</td><td>256 × 704</td><td>2</td><td>0.351</td><td>0.475</td><td>0.639</td><td>0.267</td><td>0.479</td><td>0.428</td><td>0.198</td><td>15.7</td></tr><tr><td>BEVStereo [23]</td><td>ResNet50</td><td>256 × 704</td><td>2</td><td>0.372</td><td>0.500</td><td>0.598</td><td>0.270</td><td>0.438</td><td>0.367</td><td>0.190</td><td>12.2</td></tr><tr><td> $\mathrm { B E V F o r m e r v 2 \ [ 5 1 ] ~ ^ { \dag } ~ } _ { \ast }$ </td><td>ResNet50</td><td></td><td></td><td>0.423</td><td>0.529</td><td>0.618</td><td>0.273</td><td>0.413</td><td>0.333</td><td>0.188</td><td></td></tr><tr><td>SOLOFusion [39]</td><td>ResNet50</td><td>256 × 704</td><td>16+1</td><td>0.427</td><td>0.534</td><td>0.567</td><td>0.274</td><td>0.511</td><td>0.252</td><td>0.181</td><td>11.4</td></tr><tr><td>BEVPoolv2 [17]</td><td>ResNet50</td><td>256 × 704</td><td>8+1</td><td>0.406</td><td>0.526</td><td>0.572</td><td>0.275</td><td>0.463</td><td>0.275</td><td>0.188</td><td>16.6</td></tr><tr><td>StreamPETR</td><td>ResNet50</td><td>256 × 704</td><td>8</td><td>0.432</td><td>0.540</td><td>0.581</td><td>0.272</td><td>0.413</td><td>0.295</td><td>0.195</td><td>27.1</td></tr><tr><td> $\mathrm { S t r e a m P E T R } ^ { \ast } \ddag$ </td><td>ResNet50</td><td>256× 704</td><td>8</td><td>0.450</td><td>0.550</td><td>0.613</td><td>0.267</td><td>0.413</td><td>0.265</td><td>0.196</td><td>31.7</td></tr><tr><td> $\mathrm { D E T R 3 D } \left[ 4 7 \right] ^ { \ast }$ </td><td>ResNet101-DCN</td><td>900 × 1600</td><td>1</td><td>0.349</td><td>0.434</td><td>0.716</td><td>0.268</td><td>0.379</td><td>0.842</td><td>0.200</td><td>3.7</td></tr><tr><td>Focal-PETR [44]</td><td>ResNet101-DCN</td><td>512 × 1408</td><td>1</td><td>0.390</td><td>0.461</td><td>0.678</td><td>0.263</td><td>0.395</td><td>0.804</td><td>0.202</td><td>6.6</td></tr><tr><td>PETR [30]*</td><td>ResNet101-DCN</td><td>512 × 1408</td><td>1</td><td>0.366</td><td>0.441</td><td>0.717</td><td>0.267</td><td>0.412</td><td>0.834</td><td>0.190</td><td>5.7</td></tr><tr><td>BEVFormer [27]*</td><td>ResNet101-DCN</td><td>900 × 1600</td><td>4</td><td>0.416</td><td>0.517</td><td>0.673</td><td>0.274</td><td>0.372</td><td>0.394</td><td>0.198</td><td>3.0</td></tr><tr><td>PolarDETR [5]-T*</td><td>ResNet101-DCN</td><td>900 × 1600</td><td>2</td><td>0.383</td><td>0.488</td><td>0.707</td><td>0.269</td><td>0.344</td><td>0.518</td><td>0.196</td><td>3.5</td></tr><tr><td>Sparse4D [29]*</td><td>ResNet101-DCN</td><td>900 × 1600</td><td>4</td><td>0.436</td><td>0.541</td><td>0.633</td><td>0.279</td><td>0.363</td><td>0.317</td><td>0.177</td><td>4.3</td></tr><tr><td>BEVDepth</td><td>ResNet101</td><td>512 × 1408</td><td>2</td><td>0.412</td><td>0.535</td><td>0.565</td><td>0.266</td><td>0.358</td><td>0.331</td><td>0.190</td><td></td></tr><tr><td>SOLOFusion</td><td>ResNet101</td><td>512 × 1408</td><td>16+1</td><td>0.483</td><td>0.582</td><td>0.503</td><td>0.264</td><td>0.381</td><td>0.246</td><td>0.207</td><td></td></tr><tr><td>StreamPETR*</td><td>ResNet101</td><td>512 × 1408</td><td>8</td><td>0.504</td><td>0.592</td><td>0.569</td><td>0.262</td><td>0.315</td><td>0.257</td><td>0.199</td><td>6.4</td></tr></table>

$$
\begin{array} { r } { \gamma = \xi _ { 1 } ( E _ { t - 1 } ^ { t } , v , \triangle t ) , } \\ { \beta = \xi _ { 2 } ( E _ { t - 1 } ^ { t } , v , \triangle t ) } \end{array}\tag{10}
$$

Afterward, the affine transformation is performed to get the motion-aware context embedding $\tilde { Q } _ { c } ^ { t }$ and motion-aware position encoding $\tilde { Q } _ { p \epsilon } ^ { t }$ .

$$
\begin{array} { c } { \tilde { Q } _ { p e } ^ { t } = \gamma \cdot L N ( \psi ( \tilde { Q } _ { p } ^ { t } ) ) + \beta , } \\ { \tilde { Q } _ { c } ^ { t } = \gamma \cdot L N ( Q _ { c } ^ { t } ) + \beta } \end{array}\tag{11}
$$

where ψ is a multi-layer perceptron (MLP) that converted the 3D sampled points $\tilde { Q } _ { p } ^ { t }$ into position encoding $\tilde { Q } _ { p e } ^ { t }$ . For the sake of unification, the MLN is also adopted into current object queries. The velocity v and time interval △t of the current frame are zero-initialized.

Hybrid Attention layer. The self-attention in DETR [4] contributes to duplicated prediction removal. We replace it with hybrid attention, which additionally introduces temporal interaction. As shown in Fig. 4, all stored object queries in the memory queue are concatenated with current queries to obtain the hybrid queries. The hybrid queries are regard as the key and value in multi-head attention. Since the number of hybrid queries is small (about 2k, which is far less than image tokens in the cross-attention), the hybrid attention layer brings negligible computation cost.

Following PETR [30], the query can be defined as a randomly initialized 3D anchor. To fully utilize the spatial and context priors in streaming video, some object queries in the memory queue are directly propagated into the current frame. In our implementation, queries from the last frame are concatenated with randomly initialized queries. For a fair comparison, the number of randomly initialized queries and propagated queries are set to 644 and 256 respectively.

## 5. Experiments

## 5.1. Dataset and Metrics

We evaluate our approach on the large-scale NuScenes dataset [1] and Waymo Open dataset [42].

The nuScenes Dataset includes 1000 scenes, which are 20 seconds in length and annotated at 2Hz. The camera rig covers the full 360° field of view (FOV). The annotations contain up to 1.4M 3D bounding boxes, and 10 common classes are used for evaluation: car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, barrier, and traffic cone. We compare the methods with the following metrics, the nuScenes Detection Score (NDS), mean Average Precision (mAP), and 5 kinds of True Positive (TP) metrics including average translation error (ATE), average scale error (ASE), average orientation error (AOE), average velocity error (AVE), average attribute error (AAE). Following the standard evaluation metrics, we report the average multi-object tracking accuracy (AMOTA), average multi-object tracking precision (AMOTP), recall (RECALL), multi-object tracking accuracy (MOTA) and ID switch (IDS) for 3D object tracking tasks [49, 52].

Waymo Open Dataset collects camera data only spanning a horizontal FOV of 230 degrees. The ground truth bounding boxes are annotated to a maximum range of 75 meters. The longitudinal error tolerant metrics LET-3D-AP, LET-3D-AP-H and LET-3D-APL are used for evaluation. Noting that we only use 20% of training data for fair comparison according to common practice.

Table 2. Comparison on the nuScenes test set. TTA is test time augmentaion.
<table><tr><td>Methods</td><td>Modality</td><td>Backbone</td><td>Image / Voxel</td><td>TTA</td><td>mAP↑</td><td>NDS↑</td><td>mATE↓</td><td>mASE↓</td><td>mAOE↓</td><td>mAVE↓</td><td>mAAE↓</td></tr><tr><td>CenterPoint [54]</td><td>L</td><td></td><td>0.075×0.075×0.2</td><td>x</td><td>0.603</td><td>0.673</td><td>0.262</td><td>0.239</td><td>0.361</td><td>0.288</td><td>0.136</td></tr><tr><td>FCOS3D [46]</td><td>C</td><td>R101-DCN</td><td>900 × 1600</td><td>V</td><td>0.358</td><td>0.428</td><td>0.690</td><td>0.249</td><td>0.452</td><td>1.434</td><td>0.124</td></tr><tr><td>DETR3D [47]</td><td>C</td><td>V2-99</td><td>900 × 1600</td><td>V</td><td>0.412</td><td>0.479</td><td>0.641</td><td>0.255</td><td>0.394</td><td>0.845</td><td>0.133</td></tr><tr><td>MV2D [48]</td><td>C</td><td>V2-99</td><td>640 × 1600</td><td>x</td><td>0.463</td><td>0.514</td><td>0.542</td><td>0.247</td><td>0.403</td><td>0.857</td><td>0.127</td></tr><tr><td>UVTR [24]</td><td>C</td><td>V2-99</td><td>900 × 1600</td><td>x</td><td>0.472</td><td>0.551</td><td>0.577</td><td>0.253</td><td>0.391</td><td>0.508</td><td>0.123</td></tr><tr><td>BEVFormer [27]</td><td>C</td><td>V2-99</td><td>900 × 1600</td><td>x</td><td>0.481</td><td>0.569</td><td>0.582</td><td>0.256</td><td>0.375</td><td>0.378</td><td>0.126</td></tr><tr><td>PETRv2 [31]</td><td>C</td><td>V2-99</td><td>640 × 1600</td><td>x</td><td>0.490</td><td>0.582</td><td>0.561</td><td>0.243</td><td>0.361</td><td>0.343</td><td>0.120</td></tr><tr><td>PolarFormer [19]</td><td>C</td><td>V2-99</td><td>900 × 1600</td><td>x</td><td>0.493</td><td>0.572</td><td>0.556</td><td>0.256</td><td>0.364</td><td>0.439</td><td>0.127</td></tr><tr><td>BEVStereo [23] StreamPETR</td><td>C</td><td>V2-99 V2-99</td><td>640 × 1600</td><td>x</td><td>0.525</td><td>0.610</td><td>0.431</td><td>0.246</td><td>0.358</td><td>0.357</td><td>0.138</td></tr><tr><td></td><td>C</td><td></td><td>640×1600</td><td>x</td><td>0.550</td><td>0.636</td><td>0.479</td><td>0.239</td><td>0.317</td><td>0.241</td><td>0.119</td></tr><tr><td>BEVDet4D [16]</td><td>C</td><td>Swin-B [32]</td><td>900 × 1600</td><td>V</td><td>0.451</td><td>0.569</td><td>0.511</td><td>0.241</td><td>0.386</td><td>0.301</td><td>0.121</td></tr><tr><td>BEVDepth [25]</td><td>C</td><td>ConvNeXt-B</td><td>640 × 1600</td><td>x</td><td>0.520</td><td>0.609</td><td>0.445</td><td>0.243</td><td>0.352</td><td>0.347</td><td>0.127</td></tr><tr><td>AeDet [10]</td><td>C</td><td>ConvNeXt-B</td><td>640 × 1600</td><td>V</td><td>0.531</td><td>0.620</td><td>0.439</td><td>0.247</td><td>0.344</td><td>0.292</td><td>0.130</td></tr><tr><td>PETRv2</td><td>C</td><td>RevCol-L [3]</td><td>640 × 1600</td><td>x</td><td>0.512</td><td>0.592</td><td>0.547</td><td>0.242</td><td>0.360</td><td>0.367</td><td>0.126</td></tr><tr><td>SOLOFusion [39]</td><td>C</td><td>ConvNeXt-B</td><td>640 × 1600</td><td>x</td><td>0.540</td><td>0.619</td><td>0.453</td><td>0.257</td><td>0.376</td><td>0.276</td><td>0.148</td></tr><tr><td>StreamPETR</td><td>C</td><td>ViT-L [9]</td><td>800 × 1600</td><td>x</td><td>0.620</td><td>0.676</td><td>0.470</td><td>0.241</td><td>0.258</td><td>0.236</td><td>0.134</td></tr></table>

Table 3. Comparison of 3D object tracking on nuScenes test set.
<table><tr><td rowspan=3 colspan=1>Methods</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>AMOTA↑</td><td rowspan=2 colspan=1>AMOTP↓</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>RECALL↑</td><td rowspan=1 colspan=1>IDS↓</td></tr><tr><td rowspan=1 colspan=1>CenterPoint [54]SimpleTrack [37]</td><td rowspan=1 colspan=1>0.6380.668</td><td rowspan=1 colspan=1>0.5550.550</td><td rowspan=1 colspan=1>67.5%70.3%</td><td rowspan=1 colspan=1>760575</td></tr><tr><td rowspan=4 colspan=1>QD3DT [14]MUTR3D [57]CC-3DT [11]</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>1.550</td><td rowspan=1 colspan=1>37.5%</td><td rowspan=1 colspan=1>6856</td></tr><tr><td rowspan=1 colspan=1>0.270</td><td rowspan=1 colspan=1>1.494</td><td rowspan=1 colspan=1>41.1%</td><td rowspan=1 colspan=1>6018</td></tr><tr><td rowspan=1 colspan=1>0.410</td><td rowspan=1 colspan=1>1.274</td><td rowspan=1 colspan=1>57.8%</td><td rowspan=1 colspan=1>3334</td></tr><tr><td rowspan=3 colspan=1>PolarDETR [5]UVTR [24]QTrack [52]</td><td rowspan=1 colspan=1>0.273</td><td rowspan=1 colspan=1>1.185</td><td rowspan=1 colspan=1>40.4%</td><td rowspan=1 colspan=1>2170</td></tr><tr><td rowspan=1 colspan=1>0.519</td><td rowspan=1 colspan=1>1.125</td><td rowspan=1 colspan=1>59.9%</td><td rowspan=2 colspan=1>22041484</td></tr><tr><td rowspan=1 colspan=1>0.480</td><td rowspan=1 colspan=1>1.100</td><td rowspan=1 colspan=1>59.7%</td></tr><tr><td rowspan=1 colspan=1>Sparse4D [29]</td><td rowspan=1 colspan=1>0.519</td><td rowspan=1 colspan=1>1.078</td><td rowspan=1 colspan=1>63.3%</td><td rowspan=1 colspan=1>1090</td></tr><tr><td rowspan=1 colspan=1>ByteTrackv2 [59]</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>1.005</td><td rowspan=1 colspan=1>63.5%</td><td rowspan=1 colspan=1>704</td></tr><tr><td rowspan=1 colspan=1>PF-Track [36]</td><td rowspan=1 colspan=1>0.434</td><td rowspan=1 colspan=1>1.252</td><td rowspan=1 colspan=1>53.8%</td><td rowspan=1 colspan=1>249</td></tr><tr><td rowspan=1 colspan=1>StreamPETR</td><td rowspan=1 colspan=1>0.653</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=1>73.3%</td><td rowspan=1 colspan=1>1037</td></tr></table>

## 5.2. Implementation Details

We conduct experiments with ResNet50 [13], ResNet101, V2-99 [21] and ViT [7] backbones under different pre-training. Following previous methods [27, 30, 39], the performance of ResNet50 and ResNet101 models with pre-trained weights ImageNet [6] and nuImages [1] are provided on the nuScenes val set. To scale up our method, we also report results on the nuScenes test set with V2-99 initialized from DD3D [38] checkpoint and ViT-Large [7]. Following BEVFormerv2 [51], the ViT-Large [7] is pre-trained on Objects365 [41] and COCO [28] dataset.

StreamPETR is trained by AdamW [33] optimizer with a batch size of 16. The base learning rate is set to 4e-4 and the cosine annealing policy is employed. Only key frames are used during both training and inference. All experiments are conducted without CBGS [60] strategy. Our implementation is mainly based on Focal-PETR [44], which introduces auxiliary 2D supervision. The models in the ablation study are trained for 24 epochs, while trained for 60 epochs when compared with others. In particular, we only train

Table 4. Comparison on the Waymo val set. ∗ The saving interval τ is set to 5 during testing. ‡ The saving interval τ is set to 1.
<table><tr><td>Methods</td><td>Backbone</td><td>mAPL↑</td><td>mAP↑</td><td>mAPH↑</td></tr><tr><td>BEVFormer++ [26]</td><td>ResNet101-DCN</td><td>0.361</td><td>0.522</td><td>0.481</td></tr><tr><td>MV-FCOS3D++ [45]</td><td>ResNet101-DCN</td><td>0.379</td><td>0.522</td><td>0.484</td></tr><tr><td>PETR-DN [30]</td><td>ResNet101</td><td>0.358</td><td>0.502</td><td>0.462</td></tr><tr><td>PETRv2 [31]</td><td>ResNet101</td><td>0.366</td><td>0.519</td><td>0.479</td></tr><tr><td>StreamPETR* 兴</td><td>ResNet101</td><td>0.399</td><td>0.553</td><td>0.517</td></tr><tr><td>StreamPETR‡</td><td>ResNet101</td><td>0.395</td><td>0.551</td><td>0.518</td></tr></table>

24 epochs for ViT-L [7] to prevent over-fitting. For image and BEV data augmentation, we adopt the same methods as PETR [18, 30]. We randomly skip 1 frame during the training sequence for temporal data augmentation [27].

## 5.3. Main Results

NuScenes Dataset. We compare the proposed Stream-PETR with previous state-of-the-art vision-based 3D detectors on the nuScenes val and test set. As shown in Tab. 1, StreamPETR shows superior performance on mAP, NDS, mASE, and mAOE metrics when adopting ResNet101 backbone with nuImages pretraining. Compared with the single frame baseline Focal-PETR, StreamPETR has considerable improvements of 11.4% mAP and 13.1% NDS. The mATE of StreamPETR is 10.9% better than Focal-PETR, indicating that our object-centric temporal modeling is able to improve both the accuracy of localization. With image resolutions of 256×704 and adopting ResNet50 backbone, StreamPETR exceeds the state-of-the-art method (SOLOFusion) by 0.5 % mAP and 0.6 % NDS. When we reduce the number of queries and apply nuImages pretraining, our method has 2.3 % and 1.6 % advantages in mAP and NDS. At the same time, the inference speed of StreamPETR is 1.8× faster.

When we compare the performance on the test set in Tab. 2 and adopt a smaller V2-99 backbone, StreamPETR can surpass SOLOFusion with ConvNext-Base backbone by 1.0% mAP and 1.7% NDS. Scaling up the backbone to ViT-Large [7, 9], StreamPETR achieves 62.0% of mAP, 67.6% of NDS, and 25.8% of mAOE. Note that it is the first online multi-view method that achieves comparable performance with CenterPoint.

![](images/a85cee541df9605b49ec60be6efccfa46a736774843fc77bf7ad43ed74077ee1.jpg)  
Figure 5. The mAP results with different distance thresholds (Dist TH) on the nuScenes val set. ∗ indicates StreamPETR without the proposed motion-aware layer normalization. Top: Boxes with a velocity lower than 1m/s are maintained for analysis. Down: Boxes with a velocity higher than 1m/s are maintained for analysis.

For 3D multi-object tracking task, we simply extend the multi-object tracking of CenterPoint [54] to the multi-view 3D setting. Owing to the exceptional detection and velocity estimation performance, StreamPETR significantly outperforms ByteTrackv2 [59] with an impressive margin of +8.9% AMOTA in Tab. 3. Furthermore, StreamPETR excels over CenterPoint [54] in AMOTA, and demonstrates superior benefits in RECALL.

Waymo Open Dataset. In this section, we provide experimental results on the Waymo val set, as shown in Tab. 4. Our model has trained 24 epochs and the saving interval of the memory queue is set to 5. It can be seen that our method shows superiority in official metrics compared with the dense BEV methods, e.g. BEVFormer++ [26] and MV-FCOS3D++ [45]. The Waymo open dataset has a larger evaluation range than nuScenes, our object-centric modeling method still shows obvious advantages in localization capability and longitudinal prediction. We also reimplemented PETR-DN and PETRv2 (all with query denoising [22]) as baseline models. StreamPETR outperforms the single-frame PETR-DN with a margin of 4.1% mAPL, 5.1% mAP, and 5.5% mAP-H. The Waymo open dataset covers part of the horizontal FOV, while object-centric temporal modeling still brings significant improvement. When we adopt the checkpoint and adjust saving interval τ to 1 during testing, StreamPETR has slight performance degradation, proving the adaptability on sensor frequency.

Table 5. Training frames for long-term fusion. W indicates testing in the sliding window, and V indicates testing in online video.
<table><tr><td rowspan=3 colspan=1>Training frames</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Test</td><td rowspan=2 colspan=1>mAP↑</td><td rowspan=2 colspan=1>NDS↑</td><td rowspan=2 colspan=1>mATE↓</td><td></td></tr><tr><td rowspan=1 colspan=1>mAVE↓</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>0.317</td><td rowspan=1 colspan=1>0.372</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.885</td></tr><tr><td rowspan=2 colspan=1>22</td><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.410</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.726</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.315</td><td rowspan=1 colspan=1>0.401</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.767</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=1>0.483</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.385</td></tr><tr><td rowspan=2 colspan=1>48</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.366</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.685</td><td rowspan=1 colspan=1>0.392</td></tr><tr><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=1>0.396</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.324</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.505</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.316</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1>0.507</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=1>0.325</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.645</td><td rowspan=1 colspan=1>0.316</td></tr></table>

## 5.4. Ablation Study & Analysis

Impact of Training Sequence Length. StreamPETR is trained in local sliding windows and tested in online streaming video. To analyze the inconsistency between training and testing, we conduct experiments with varying numbers of training frames and show results in Tab. 5. When adding more training frames, the performance of StreamPETR continues to grow, and the performance gap between sliding windows and online video decreases obviously. It is worth noting that when the number of training frames increases to 8, video testing (40.2% mAP, 50.5% NDS) shows superior performance than the sliding window (39.6% mAP, 50.1% NDS), which proves that our method has a good potential to build long-term temporal dependency. Expanding to 12 frames brings limited performance improvement, so we train our models on 8 frames for experimental efficiency. Effect of Motion-aware Layer Normalization. We compare the different designs for decoupling the ego vehicle and moving objects in Tab. 6. It can be seen that the performance does not improve when adopting explicit motion compensation (MC). We argue that the explicit way may cause error propagation in the early training phase. The MLN implicitly encodes and decouples the movements of the ego vehicle and moving objects. Specifically, implicit encoding of ego poses has achieved significant improvements, among which mAP increases by 2.0% and NDS increases by 1.8%. Besides, the encoding of relative time offset △t and object velocity v can further boost the performance. Both mAP and NDS are increased by 0.4%, which indicates that dynamic properties have a beneficial effect on the temporal interaction between object queries.

Number of Frames for Long-term Fusion. In Tab. 7, we analyze the impacts of memory size on hybrid attention. We can find that the mAP and NDS are improved with the increase of the memory size and begin to saturate when reaching 2 frames (nearly 1 second). The object query in StreamPETR is propagated and updated recursively, so even without a large-capacity memory queue, our method can still build a long-term spatial-temporal dependency. Since increasing the memory queue brings negligible computing costs, we use 4 frames to alleviate forgetting and obtain more stable results.

![](images/f5130188e67a52360574a6bcd75c05679b89dc1f506cb6959d4b4482f6e1489c.jpg)  
Figure 6. Visualization results of StreamPETR. On the BEV plane (right), the groud-truth and predictions are drawn in green and blue rectangles respectively. The failure cases are marked by red circles.

Table 6. Ablation of motion-aware layer normalization. MC is explicit motion compensation. LN is layer normalization.
<table><tr><td>MC |</td><td>|LN |</td><td>Ego Pose</td><td>Time</td><td>Velocity</td><td>mAP↑</td><td>NDS↑</td><td>mATE↓</td><td>mAVE↓</td></tr><tr><td rowspan="4">V</td><td>V V</td><td rowspan="4"></td><td rowspan="4">V</td><td rowspan="4"></td><td rowspan="4">0.378 0.380 0.375 0.398</td><td rowspan="4">0.483 0.481 0.481 0.501</td><td rowspan="4">0.697 0.693 0.702 0.667</td><td rowspan="4">0.354 0.379 0.370 0.316</td></tr><tr><td></td></tr><tr><td>V</td></tr><tr><td>V</td><td>0.488 0.697</td></tr><tr><td rowspan="2"></td><td>V</td><td></td><td></td><td>V</td><td>0.381 0.386</td><td>0.489</td><td>0.690</td><td>0.354 0.373</td></tr><tr><td>V</td><td>V</td><td>V</td><td>V</td><td>0.402</td><td>0.505</td><td>0.660</td><td>0.316</td></tr></table>

Table 7. Number of frames (N) for long-term fusion.
<table><tr><td>number frames</td><td>mAP↑</td><td>NDS↑</td><td>mATE↓</td><td>mAVE↓</td><td>FPS↑</td></tr><tr><td>0</td><td>0.317</td><td>0.372</td><td>0.770</td><td>0.885</td><td>27.7</td></tr><tr><td>1</td><td>0.394</td><td>0.501</td><td>0.669</td><td>0.324</td><td>27.7</td></tr><tr><td>2</td><td>0.401</td><td>0.505</td><td>0.660</td><td>0.314</td><td>27.4</td></tr><tr><td>3</td><td>0.400</td><td>0.504</td><td>0.663</td><td>0.322</td><td>27.3</td></tr><tr><td>4</td><td>0.402</td><td>0.505</td><td>0.660</td><td>0.316</td><td>27.1</td></tr></table>

Perspective v.s. Object-Centric. StreamPETR achieves efficient temporal modeling through the interaction of sparse object queries. An alternative solution is to build temporal interaction via the perspective memory [31]. As shown in Tab. 8, the query-based temporal modeling has superior performance than perspective-based both on speed and accuracy. The combination of the query and perspective memory does not further improve the performance, implying that the temporal propagation of global query interaction is sufficient to achieve leading performance. Besides, concatenating current object queries with the queries of the last frame improves 0.7% mAP and 0.9% NDS.

Analysis of Moving Objects. In this section, we detailed analyze the performance of StreamPETR on perceiving static and moving objects respectively. For fair comparisons, all models are trained with 24 epochs without CBGS [60] and evaluated on the nuScenes [1] val set. The detection performance of moving objects still lags behind that of static objects to a large margin even with temporal modeling. Compared with dense BEV paradigms [23, 17], StreamPETR∗ has reached promising performance on both static and moving objects. This proves the superiority of object-centric temporal modeling, which has global temporal and spatial receptive fields. Applying the implicit encoding for motion information, the performance of Stream-PETR can be further improved.

Table 8. From of the temporal propagation. ’Perspective and Object’ mean propagating temporal information via image features and object queries respectively. ’Propagated’ indicates concatenating the propagated queries from last frame.
<table><tr><td rowspan=2 colspan=1>Perspective |</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>| Object |</td><td rowspan=1 colspan=1>Propagated</td><td rowspan=1 colspan=1>mAP↑</td><td rowspan=1 colspan=1>NDS↑</td><td rowspan=1 colspan=1>mATE↓</td><td rowspan=1 colspan=1>mAVE↓</td><td rowspan=1 colspan=1>FPS↑</td></tr><tr><td rowspan=5 colspan=1>V</td><td rowspan=5 colspan=1>V</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=4 colspan=1>V</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.3170.361</td><td rowspan=2 colspan=1>0.3720.459</td><td rowspan=2 colspan=1>0.770</td><td rowspan=2 colspan=1>0.8850.374</td><td rowspan=2 colspan=1>27.718.9</td></tr><tr><td rowspan=1 colspan=1>0.731</td></tr><tr><td rowspan=1 colspan=1>0.395</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.703</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>27.1</td></tr><tr><td rowspan=2 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.505</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.316</td><td rowspan=1 colspan=1>27.1</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.341</td><td rowspan=1 colspan=1>18.6</td></tr></table>

## 5.5. Failure Cases

We show the detection results of a challenging scene in Fig. 6. StreamPETR shows impressive results on crowded objects within the detection range of 30m. However, our method has many False Positives on remote objects. It is a common phenomenon of camera-based methods. In a complex urban scene, the duplicated predictions on remote objects can be tolerable and cause relatively little impact.

## 6. Conclusion

In this paper, we propose StreamPETR, an effective long-sequence 3D object detector. Different from the previous works, our method explores an object-centric paradigm that propagates temporal information through object queries frame by frame. In addition, a motion-aware layer normalization is adopted to introduce the motion information. StreamPETR achieves leading performance improvements while introducing negligible storage and computation cost. It is the first online multi-view method that achieves comparable performance with lidar-based methods. We hope StreamPETR can provide some new insights into longsequence modeling for the community.

Acknowledgements. This work was supported by the National Natural Science Foundation of China (52102449), the China Postdoctoral Science Foundation (2021M690394), and the Beijing Institute of Technology Research Fund Program for Young Scholars.

## References

[1] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 5, 6, 8

[2] Jiarui Cai, Mingze Xu, Wei Li, Yuanjun Xiong, Wei Xia, Zhuowen Tu, and Stefano Soatto. Memot: multi-object tracking with memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8090–8100, 2022. 2

[3] Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiangwen Kong, Jun Li, and Xiangyu Zhang. Reversible column networks. arXiv preprint arXiv:2212.11696, 2022. 6

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 1, 2, 5

[5] Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Chang Huang, and Wenyu Liu. Polar parametrization for vision-based surround-view 3d detection. arXiv preprint arXiv:2206.10965, 2022. 2, 5, 6

[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6, 7

[8] Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. arXiv preprint arXiv:1610.07629, 2016. 4

[9] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. arXiv preprint arXiv:2303.11331, 2023. 6, 7

[10] Chengjian Feng, Zequn Jie, Yujie Zhong, Xiangxiang Chu, and Lin Ma. Aedet: Azimuth-invariant multi-view 3d object detection. arXiv preprint arXiv:2211.12501, 2022. 6

[11] Tobias Fischer, Yung-Hsu Yang, Suryansh Kumar, Min Sun, and Fisher Yu. Cc-3dt: Panoramic 3d object tracking via cross-camera fusion. arXiv preprint arXiv:2212.01247, 2022. 6

[12] Fei He, Naiyu Gao, Jian Jia, Xin Zhao, and Kaiqi Huang. Queryprop: Object query propagation for high-performance video object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 834–842, 2022. 2

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[14] Hou-Ning Hu, Yung-Hsu Yang, Tobias Fischer, Trevor Darrell, Fisher Yu, and Min Sun. Monocular quasi-dense 3d object tracking. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):1992–2008, 2022. 6

[15] Bin Huang, Yangguang Li, Enze Xie, Feng Liang, Luya Wang, Mingzhu Shen, Fenggang Liu, Tianqi Wang, Ping Luo, and Jing Shao. Fast-bev: Towards real-time on-vehicle bird’s-eye view perception. arXiv preprint arXiv:2301.07870, 2023. 2

[16] Junjie Huang and Guan Huang. Bevdet4d: Exploit temporal cues in multi-camera 3d object detection. arXiv preprint arXiv:/2203.17054, 2021. 1, 2, 3, 5, 6

[17] Junjie Huang and Guan Huang. Bevpoolv2: A cutting-edge implementation of bevdet toward deployment. arXiv preprint arXiv:2211.17111, 2022. 5, 8

[18] Junjie Huang, Guan Huang, Zheng Zhu, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790, 2021. 2, 5, 6

[19] Yanqin Jiang, Li Zhang, Zhenwei Miao, Xiatian Zhu, Jin Gao, Weiming Hu, and Yu-Gang Jiang. Polarformer: Multicamera 3d object detection with polar transformers. arXiv preprint arXiv:2206.15398, 2022. 2, 6

[20] Zhengkai Jiang, Peng Gao, Chaoxu Guo, Qian Zhang, Shiming Xiang, and Chunhong Pan. Video object detection with locally-weighted deformable neighbors. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8529–8536, 2019. 2

[21] Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae, and Jongyoul Park. An energy and gpu-computation efficient backbone network for real-time object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019. 6

[22] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and Lei Zhang. Dn-detr: Accelerate detr training by introducing query denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13619–13627, 2022. 7

[23] Yinhao Li, Han Bao, Zheng Ge, Jinrong Yang, Jianjian Sun, and Zeming Li. Bevstereo: Enhancing depth estimation in multi-view 3d object detection with dynamic temporal stereo. arXiv preprint arXiv:2209.10248, 2022. 2, 5, 6, 8

[24] Yanwei Li, Yilun Chen, Xiaojuan Qi, Zeming Li, Jian Sun, and Jiaya Jia. Unifying voxel-based representation with transformer for 3d object detection. arXiv preprint arXiv:2206.00630, 2022. 6

[25] Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. Bevdepth:

Acquisition of reliable depth for multi-view 3d object detection. arXiv preprint arXiv:2206.10092, 2022. 1, 2, 3, 5, 6

[26] Zhiqi Li, Hanming Deng, Tianyu Li, Yangyi Huang, Chonghao Sima, Xiangwei Geng, Yulu Gao, Wenhai Wang, Yang Li, and Lewei Lu. Bevformer ++ : Improving bevformer for 3d camera-only object detection: 1st place solution for waymo open dataset challenge 2022. 2023. 6, 7

[27] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. arXiv preprint arXiv:2203.17270, 2022. 1, 2, 3, 5, 6

[28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6

[29] Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581, 2022. 1, 2, 3, 5, 6

[30] Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. arXiv preprint arXiv:2203.05625, 2022. 2, 4, 5, 6

[31] Yingfei Liu, Junjie Yan, Fan Jia, Shuailin Li, Qi Gao, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petrv2: A unified framework for 3d perception from multi-camera images. arXiv preprint arXiv:2206.01256, 2022. 1, 2, 3, 5, 6, 8

[32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. 6

[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6

[34] Zhipeng Luo, Changqing Zhou, Gongjie Zhang, and Shijian Lu. Detr4d: Direct multi-view 3d object detection with sparse attention. arXiv preprint arXiv:2212.07849, 2022. 2, 3

[35] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multi-object tracking with transformers. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8844–8854, 2022. 2

[36] Ziqi Pang, Jie Li, Pavel Tokmakov, Dian Chen, Sergey Zagoruyko, and Yu-Xiong Wang. Standing between past and future: Spatio-temporal modeling for multi-camera 3d multiobject tracking. arXiv preprint arXiv:2302.03802, 2023. 3, 6

[37] Ziqi Pang, Zhichao Li, and Naiyan Wang. Simpletrack: Understanding and rethinking 3d multi-object tracking. In Computer Vision–ECCV 2022 Workshops: Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part I, pages 680–696. Springer, 2023. 6

[38] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is pseudo-lidar needed for monocular 3d

object detection? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3142–3152, 2021. 6

[39] Jinhyung Park, Chenfeng Xu, Shijia Yang, Kurt Keutzer, Kris Kitani, Masayoshi Tomizuka, and Wei Zhan. Time will tell: New outlooks and a baseline for temporal multi-view 3d object detection. arXiv preprint arXiv:2210.02443, 2022. 1, 2, 5, 6

[40] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Gaugan: semantic image synthesis with spatially adaptive normalization. In ACM SIGGRAPH 2019 Real-Time Live!, pages 1–1. 2019. 4

[41] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 6

[42] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020. 5

[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4

[44] Shihao Wang, Xiaohui Jiang, and Ying Li. Focal-petr: Embracing foreground for efficient multi-camera 3d object detection. arXiv preprint arXiv:2212.05505, 2022. 5, 6

[45] Tai Wang, Qing Lian, Chenming Zhu, Xinge Zhu, and Wenwei Zhang. Mv-fcos3d++: Multi-view camera-only 4d object detection with pretrained monocular backbones. arXiv preprint arXiv:2207.12716, 2022. 6, 7

[46] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Fcos3d: Fully convolutional one-stage monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 913–922, 2021. 6

[47] Yue Wang, Guizilini Vitor Campagnolo, Tianyuan Zhang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In In Conference on Robot Learning, pages 180–191, 2022. 2, 4, 5, 6

[48] Zitian Wang, Zehao Huang, Jiahui Fu, Naiyan Wang, and Si Liu. Object as query: Equipping any 2d object detector with 3d detection ability. arXiv preprint arXiv:2301.02364, 2023. 2, 6

[49] Dongming Wu, Wencheng Han, Tiancai Wang, Xingping Dong, Xiangyu Zhang, and Jianbing Shen. Referring multiobject tracking. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14633– 14642, 2023. 5

[50] Enze Xie, Zhiding Yu, Daquan Zhou, Jonah Philion, Anima Anandkumar, Sanja Fidler, Ping Luo, and Jose M Alvarez. Mˆ 2bev: Multi-camera joint 3d detection and segmentation with unified birds-eye view representation. arXiv preprint arXiv:2204.05088, 2022. 2

[51] Chenyu Yang, Yuntao Chen, Hao Tian, Chenxin Tao, Xizhou Zhu, Zhaoxiang Zhang, Gao Huang, Hongyang Li, Yu Qiao,

Lewei Lu, et al. Bevformer v2: Adapting modern image backbones to bird’s-eye-view recognition via perspective supervision. arXiv preprint arXiv:2211.10439, 2022. 5, 6

[52] Jinrong Yang, En Yu, Zeming Li, Xiaoping Li, and Wenbing Tao. Quality matters: Embracing quality clues for robust 3d multi-object tracking. arXiv preprint arXiv:2208.10976, 2022. 5, 6

[53] Zetong Yang, Yin Zhou, Zhifeng Chen, and Jiquan Ngiam. 3d-man: 3d multi-frame attention network for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1863–1872, 2021. 2

[54] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3d object detection and tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11784–11793, 2021. 6, 7

[55] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end multipleobject tracking with transformer. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII, pages 659–675. Springer, 2022. 2, 3

[56] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 4

[57] Tianyuan Zhang, Xuanyao Chen, Yue Wang, Yilun Wang, and Hang Zhao. Mutr3d: A multi-camera tracking framework via 3d-to-2d queries. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4537–4546, 2022. 2, 3, 6

[58] Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. Motrv2: Bootstrapping end-to-end multi-object tracking by pretrained object detectors. arXiv preprint arXiv:2211.09791, 2022. 2, 3

[59] Yifu Zhang, Xinggang Wang, Xiaoqing Ye, Wei Zhang, Jincheng Lu, Xiao Tan, Errui Ding, Peize Sun, and Jingdong Wang. Bytetrackv2: 2d and 3d multi-object tracking by associating every detection box. arXiv preprint arXiv:2303.15334, 2023. 6, 7

[60] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu. Class-balanced grouping and sampling for point cloud 3d object detection. arXiv preprint arXiv:1908.09492, 2019. 6, 8

[61] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 1