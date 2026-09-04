# DVIS: Decoupled Video Instance Segmentation Framework

Tao Zhang<sup>1</sup> Xingye Tian<sup>2</sup> Yu Wu<sup>1</sup> Shunping Ji<sup>1</sup>\* Xuebo Wang<sup>2</sup> Yuan Zhang<sup>2</sup> Pengfei Wan<sup>2</sup>

<sup>1</sup>Wuhan University <sup>2</sup>Y-tech, Kuaishou Technology

## Abstract

Video instance segmentation (VIS) is a critical task with diverse applications, including autonomous driving and video editing. Existing methods often underperform on complex and long videos in real world, primarily due to two factors. Firstly, offline methods are limited by the tightly-coupled modeling paradigm, which treats all frames equally and disregards the interdependencies between adjacent frames. Consequently, this leads to the introduction of excessive noise during long-term temporal alignment. Secondly, online methods suffer from inadequate utilization of temporal information. To tackle these challenges, we propose a decoupling strategy for VIS by dividing it into three independent sub-tasks: segmentation, tracking, and refinement. The efficacy of the decoupling strategy relies on two crucial elements: 1) attaining precise long-term alignment outcomes via frame-by-frame association during tracking, and 2) the effective utilization of temporal information predicated on the aforementioned accurate alignment outcomes during refinement. We introduce a novel referring tracker and temporal refiner to construct the Decoupled VIS framework (DVIS). DVIS achieves new SOTA performance in both VIS and VPS, surpassing the current SOTA methods by 7.3 AP and 9.6 VPQ on the OVIS and VIPSeg datasets, which are the most challenging and realistic benchmarks. Moreover, thanks to the decoupling strategy, the referring tracker and temporal refiner are super light-weight (only 1.69% of the segmenter FLOPs), allowing for efficient training and inference on a single GPU with 11G memory. The code is available at https://github.com/zhang-tao-whu/DVIS.

## 1. Introduction

Video Instance Segmentation (VIS) is a critical computer vision task that involves identifying, segmenting, and tracking all interested instances in a video simultaneously. This task was first introduced in [31]. The importance of VIS lies in its ability to facilitate many downstream computer vision applications, such as online autonomous driving and offline video editing.

![](images/42db6f8056083ea7ee83ea5ff4345e754c2751c3945c21d5186952f494e9a769.jpg)  
Figure 1. Pipelines of previous offline (a), online (b), and proposed DVIS (c) frameworks. Unlike previous methods that rely on tightly coupled networks, DVIS consists of independent components, including a segmenter, a referring tracker, and a temporal refiner.

Previous studies [12, 6, 28, 10] have demonstrated successful performance validation on videos with short durations and simple scenes [31]. However, in real-world scenarios, videos often present highly complex scenes, severe instance occlusions, and prolonged durations [21]. As a result, these approaches [12, 6, 28, 10] have exhibited poor performance on videos [21] that are more representative of real-world scenarios.

We believe that the fundamental reason for the failure of the aforementioned methods [12, 6, 28, 10] lies in the assumption that a coupled network can effectively predict the video segmentation results for any video, irrespective of its length, scene complexity, or instance occlusion levels. In the case of lengthy videos (e.g. 100 seconds and 500 frames), with intricate scenes, the same instance may exhibit significant variations in position, shape, and size between the first and last frames [21]. Even for experienced humans, accurately associating the same instance in two frames that are separated by a considerable interval is challenging without observing its gradual transformation trajectory over time. Therefore, the alignment/tracking difficulty is significantly increased in complex scenarios and lengthy videos, and even cutting-edge methods such as [5] face challenges in achieving convergence [11].

To tackle the aforementioned challenges, we propose to decouple the VIS task into three sub-tasks that are independent of video length and complexity: segmentation, tracking, and refinement. Segmentation aims to extract all appearing objects and obtain their representations from a single frame. Tracking aims to link the same object between adjacent frames. Refinement utilizes all temporal information of the object to optimize both segmentation and association results. Thus we have our decoupled VIS framework, as illustrated in Figure 1 (c). It contains three separate and independent components, i.e., a segmenter, a tracker, and a refiner. Given the extensive research on the segmenter in the field of image instance segmentation, our focus is to design an effective tracker for robustly associating objects across adjacent frames and a refiner for improving the quality of segmentation and tracking.

To achieve effective instance association, we propose the following principles: (1) encourage sufficient interaction between instance representations of adjacent frames to fully exploit their similarity for better association. (2) avoid mixing their information during the interaction process to prevent introducing indistinguishable noise that may interfere with the association results. Current SOTA methods, such as [29, 11], violate principle 1 by utilizing heuristic algorithms to match adjacent frame instance representations without any interaction, resulting in a significant performance gap compared to our method. While [9, 35] achieve interaction between instance representations of adjacent frames by passing instance representations, they violate principle 2. Following both principles, we designed the Referring Cross Attention (RCA) module, which serves as the core component of our highly effective referring tracker. RCA is a modified version of standard cross-attention [4] that introduces identification to avoid the blending of instance representations in consecutive frames and efficiently utilize their similarities. We further propose a novel temporal refiner that leverages 1D convolution and self-attention to effectively integrate temporal information, and crossattention to correct instance representations.

An decoupled VIS framework, called DVIS, is then naturally constructed by combining the segmenter, the referring tracker, and the temporal refiner. DVIS achieves new

SOTA performance on all the VIS datasets, surpassing previous SOTA method [29] by 7.3 AP on the most challenging OVIS dataset [21]. Additionally, DVIS can be seamlessly extended to other video segmentation tasks, such as video panoptic segmentation (VPS) [13], without any modification. DVIS also achieves new SOTA performance on the video panoptic segmentation dataset VIPSeg [20], surpassing previous SOTA method [1] by 9.6 VPQ. DVIS achieved 1st place in the VPS Track of the PVUW challenge at CVPR 2023.

Our decoupling strategy not only significantly improves the performance of video segmentation, but also dramatically reduces hardware resource requirements. Specifically, our proposed tracker and refiner operate exclusively on the instance representations output by the segmenter, avoiding the significant computational cost associated with interacting with image features. As a result, the computation cost of the tracker and refiner is negligible (only 5.18%/1.69% of the segmenter with R50/Swin-L backbone). Thanks to the decoupling design of the VIS task and framework, the tracker and refiner can be trained separately while keeping other components frozen. These advantages allow DVIS to be trained on a single GPU with 11G memory.

In summary , our main contributions are:

• We investigate the failure reasons of current methods on complex and lengthy real-world videos, and we address these challenges by introducing a novel decoupling strategy for VIS, which involves decomposing it into three decoupled sub-tasks: segmentation, tracking, and refinement.

• Following the decoupling strategy, we propose DVIS, which includes a simple yet effective referring tracker and temporal refiner to produce precision alignment results and efficiently utilize temporal information, respectively.

• DVIS achieves new SOTA performance in both VIS and VPS, as validated on five major benchmarks: OVIS [21], YouTube-VIS [31] 2019, 2021, and 2022, as well as VIPSeg [20]. Notably, DVIS significantly reduces the resources required for video segmentation, enabling efficient training and inference on a single GPU with 11G memory.

## 2. Related Works

Online Video Instance Segmentation. Most mainstream online VIS methods follow a pipeline of segmenting and associating instances. MaskTrack R-CNN [31] in corporates a tracking head based on [8] and associates instances in adjacent frames using multiple cues such as similarity score, semantic consistency, spatial correlation, and detection confidence. [3] replaces the segmenter in the above pipeline with a one-stage instance segmentation network. [32] proposes a crossover learning scheme that segments the same instances in another frame using the instance features of the current frame. With stronger segmenters and the widespread application of transformers in vision tasks [6, 4, 33], recent works such as [11, 29, 34] have achieved outstanding performance. [11] proposes a minimal VIS framework based on [6] that achieves instance association by measuring the similarity between the same instances in adjacent frames. [29, 34] introduce contrastive learning in VIS to obtain a more discriminative instance representation. [35, 9] completely remove heuristic matching algorithms by delivering instance representations and modeling inter-frame association. Inspired by [30, 9, 11, 19], DVIS also performs tracking based on instance representations, which significantly reduces memory requirements. Our proposed DVIS introduces a novel component called the referring tracker, which models interframe association by denoising current instance representations with the help of previous frame instance representations.

Offline Video Instance Segmentation. Previous offline video instance segmentation (VIS) methods have used various approaches to model the spatio-temporal representations of instances in the video. In [2], instance spatiotemporal embeddings are modeled using a 3D CNN. The first transformer-based VIS architecture proposed in [26] uses learnable initial embeddings for each instance of each frame, making it challenging to model instances with complex motion trajectories. [12] introduces inter-frame communication, which reduces computation and memory overhead while improving performance. By directly modeling a video-level representation for each instance, [5] achieves impressive results. [28] constructs a VIS framework based on deformable attention [36] to separate temporal and spatial interactions between instance representations and videos. To significantly reduce memory consumption and enable offline methods to handle long videos, [10] constructs the video-level instance representation from the instance representations of each frame. While [9] implements a semi-online VIS framework by replacing frames with clips, no significant gains were observed compared to the online version. The current SOTA methods for VIS have been demonstrated to overlook the importance of the refinement sub-task. Specifically, the refinement process has been neglected by [29, 11, 35, 9], while [12, 5, 28, 10] exhibit a lack of clear separation between refinement and other aspects of the segmentation and tracking sub-tasks. Our proposed DVIS achieves SOTA performance by decoupling the VIS task and designing an efficient temporal refiner to fully utilize the information of the overall video.

## 3. Method

By reflecting on and summarizing the shortcomings of [11, 10], we have proposed DVIS, a novel decoupled framework for VIS that consists of three independent components: a segmenter, a referring tracker, and a temporal refiner, illustrated in Figure 1(c). Specifically, we use Mask2Former [6] as the segmenter in DVIS. The referring tracker is introduced in Section 3.1, while the temporal refiner is presented in Section 3.2.

## 3.1. Referring Tracker

The referring tracker models the inter-frame association as a referring denoising task. The referring cross-attention is the core component of the referring tracker that effectively utilizes the similarity between instance representations of adjacent frames while avoiding their mixture.

Architecture. Figure 2 illustrates the architecture of the referring tracker. It takes in the instance queries $\{ Q _ { s e g } ^ { i } | i \in$ $[ 1 , T ] \}$ generated by the segmenter and outputs the instance queries $\{ Q _ { T r } ^ { i } | i \in [ 1 , T ] \}$ corresponding to the instances in the previous frame for the current frame, where $T$ is the length of the video. Firstly, the hungarian matching algorithm [14] is employed to match $Q _ { s e g }$ of adjacent frames, as is done in [11]:

$$
\left\{ \begin{array} { l l } { \tilde { Q } _ { s e g } ^ { i } = \mathrm { H u n g a r i a n } ( \tilde { Q } _ { s e g } ^ { i - 1 } , Q _ { s e g } ^ { i } ) , } & { i \in [ 2 , T ] } \\ { \tilde { Q } _ { s e g } ^ { i } = Q _ { s e g } ^ { i } , } & { i = 1 } \end{array} \right. ,\tag{1}
$$

where $\tilde { Q } _ { s e g }$ is the matched intance queries of the segmenter. The hungarian matching algorithm is not strictly necessary and omitting it results in only a slight performance degradation, as shown in Section $4 . 2 . \tilde { Q } _ { s e g }$ can be considered as the tracking result with noise and serves as the initial query for the referring tracker. To denoise the initial query $\tilde { Q } _ { s e g } ^ { i }$ of the current frame, the online tracker uses the denoised instance queries $Q _ { T r } ^ { i - 1 }$ from the previous frame as a reference.

The objective of the referring tracker is to refine the initial value with noise, which may contain incorrect tracking results, and produce accurate tracking results. The referring tracker comprises a sequence of L transformer denoising blocks, each of which consists of a referring cross-attention, a standard self-attention, and a feedforward network (FFN).

The referring cross-attention (RCA) is a crucial component of the denoising block, designed to capture the correlation between the current frame and its historical frames. Since the instance representations in adjacent frames are highly similar but differ in position, shape, size, etc., using the previous frame’s instance representation as the initial instance representation for the current frame (as done by [35, 9]) can introduce ambiguous information that makes the denoising task more difficult. RCA overcomes this issue by introducing identification (ID), while still effectively utilizing the similarity between the query (Q) and key (K) to generate the correct output. As shown in Figure 2, RCA is inspired by [23] and differs only slightly from the standard cross-attention:

![](images/4c42dd7d9c0843cb4415cbddb9f7557c2a909a590b36fc1b004467d5121f0ce9.jpg)  
Figure 2. The framework of the referring tracker. The instance representations output by the segmenter $( Q _ { s e g } )$ and referring tracker $( Q _ { T r } )$ are represented by squares and triangles, respectively. Instances with the same ID are assigned the same color.

$$
R C A ( I D , Q , K , V ) = I D + M H A ( Q , K , V ) .\tag{2}
$$

MHA refers to Multi-Head Attention [25], while ID, Q, K, and V denote identification, query, key, and value, respectively.

Finally, the denoised instance query $Q _ { T } ,$ is utilized as an input for the class head and mask head, which produce the category and mask coefficient output, respectively.

Losses. The referring tracker tracks instances frame by frame, and as such, the network is supervised using a loss function that aligns with this paradigm. Specifically, the instance label and prediction $\hat { y } _ { T } ,$ are only matched on the frame where the instance first appears. To expedite convergence during the early training phase, the prediction of the frozen segmenter $\hat { y } _ { s e g }$ is used for matching instead of the referring tracker’s prediction.

$$
\left\{ \begin{array} { l } { { \hat { \boldsymbol { \sigma } } = \arg \operatorname* { m i n } _ { \boldsymbol { \sigma } } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { m a t c h } ( \boldsymbol { y } _ { i } ^ { f ( i ) } , \boldsymbol { \hat { y } } _ { \boldsymbol { \sigma } ( i ) } ^ { f ( i ) } ) } } \\ { { \hat { \boldsymbol { y } } = \hat { y } _ { T r } \ i f \ I t e r \geq \frac { M a x \ - I t e r } { 2 } \ e l s e \ \hat { y } _ { s e g } } } \end{array} \right. ,\tag{3}
$$

where $f ( i )$ represents the frame in which the i-th instance first appears. $\mathsf { \bar { L } } _ { m a t c h } ( y _ { i } ^ { f ( i ) } , \hat { y } _ { \sigma ( i ) } ^ { f ( i ) } )$ is a pair-wise matching cost, as used in [5], between the ground truth y and the prediction yˆ having index $\sigma ( i )$ on the $f ( i )$ frame.

The loss function $\mathcal { L }$ is exactly the same as that in [5].

$$
\mathcal { L } _ { T r } = \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { N } \mathcal { L } ( y _ { i } ^ { t } , \hat { y } _ { \hat { \sigma } ( i ) } ^ { t } ) .\tag{4}
$$

## 3.2. Temporal Refiner

The failure of previous offline video instance segmentation methods can mainly be attributed to the challenge of effectively leveraging temporal information in highly coupled networks. Additionally, previous online video instance segmentation methods lacked a refinement step. To address these issues, we developed an independent temporal refiner to effectively utilize information from the entire video and refine the output of the referring tracker.

![](images/46612a50df50e4b384eee072b99fefc8c1ae2ccce6e5701e03eaeef22324e284.jpg)  
Figure 3. The framework of the temporal refiner. Instance representations for each frame $( Q _ { R f } )$ are denoted by pentagons, while the instance representations for the entire video $( { \hat { Q } } _ { R f } )$ are denoted by circles. Different colors indicate different instance IDs.

Architecture. Figure 3 shows the architecture of the temporal refiner. It takes the instance query $Q _ { T r }$ output from the referring tracker as input and outputs the instance query $Q _ { R f }$ after fully aggregating the overall information of the video. The temporal refiner is composed of L temporal decoder blocks that are cascaded together. Each temporal decoder block consists of two main components, namely the short-term temporal convolution block and the long-term temporal attention block. The short-term temporal convolution block exploits motion information while the long-term temporal attention block exploits information from the entire video. These blocks are implemented using 1D convolution and standard self-attention, respectively, and both operate in the time dimension.

Lastly, the mask coefficients for each instance in each frame are predicted by the mask head based on the refined instance query $Q _ { R f }$ . The class head predicts the category and score for each instance across the entire video, using the temporal weighting of $Q _ { R f }$ . The temporal weighting process can be defined as follows:

$$
\hat { Q } _ { R f } = \sum _ { t = 1 } ^ { T } S o f t M a x ( L i n e a r ( Q _ { R f } ^ { t } ) ) Q _ { R f } ^ { t } ,\tag{5}
$$

where $\hat { Q } _ { R f }$ is the temporal weighting of $Q _ { R f }$

Losses. The same matching cost and loss functions as [5] are used to supervise the temporal refiner during training. The segmenter and referring tracker are frozen during training, and therefore the referring tracker’s prediction results are used for matching in the early training phase to guide the network towards faster convergence.

$$
\left\{ \begin{array} { l l } { \displaystyle { \hat { \sigma } = \arg \operatorname* { m i n } _ { \sigma } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { m a t c h } ( y _ { i } , \hat { y } _ { \sigma ( i ) } ) } } & \\ { \displaystyle { \hat { y } = \hat { y } _ { R f } \hat { \sigma } \operatorname* { \ i f } _ { \mathbf { \varepsilon } } I t e r \geq \frac { M a x \_ I t e r } { 2 } \operatorname { e l s e } \hat { y } _ { T r } } } \end{array} \right. ,\tag{6}
$$

where $\hat { y } _ { R f }$ is the prediction result of the temporal refiner. The loss function is:

$$
\mathcal { L } _ { R f } = \sum _ { i = 1 } ^ { N } \mathcal { L } ( y _ { i } , \hat { y } _ { \hat { \sigma } ( i ) } ) .\tag{7}
$$

## 4. Experiments

We evaluate the performance of DVIS for VIS on the OVIS [21], YouTube VIS 2019, 2021, and 2022 [31] datasets, and for VPS on the VIPSeg [20] dataset. In Appendix, the descriptions of these datasets can be found in Section A, while implementation details, including network training and inference settings, are provided in Section B.

## 4.1. Main Results

We compare DVIS with current SOTA online and offline VIS methods on the OVIS, YouTube-VIS 2019, 2021, and 2022 datasets. When compared with online methods, DVIS will discard the temporal refiner as it utilizes information from future frames, in order to maintain a fair comparison. The results are reported in Tables 1, 2, and 3, respectively. We adopt MinVIS [11] as our baseline because DVIS is essentially identical to MinVIS after removing the referring tracker and temporal refiner. We also compare DVIS with current SOTA methods for VPS, and the results are shown in Table 4. The visualization of DVIS’s prediction results on these datasets is available in Figures I, II, and III of the Appendix.

<table><tr><td></td><td>Method</td><td>AP</td><td>OVIS</td><td></td><td> $\mathrm { A P _ { 5 0 } \ A P _ { 7 5 } \ A R _ { 1 } \ A R _ { 1 0 } }$ </td></tr><tr><td></td><td>MaskTrack R-CNN [31] CMaskTrack R-CNN [21] CrossVIS [32] VISOLO [7] [MinVIS [11] [MinVIS† [11] [IDOL [29]</td><td rowspan="2">10.8 15.4 14.9 15.3 25.0 26.4 30.2 30.2 31.0 54.8</td><td rowspan="2">25.3 33.9 32.7 31.0 45.5 49.6 28.2 51.0 30.2 51.3 53.9 30.1 55.0 30.5</td><td rowspan="2">8.5 7.9 13.1 9.3 12.1 10.3 13.8 11.1 24.0 13.9 25.2 13.3 31.1 28.0 14.5 38.6 30.0 15.037.5</td></tr><tr><td>Rest0 [IDOL† [29] ROVIS [35] Online Ours Ours† MinVIS [11] [MinVIS† [11]</td><td></td></tr><tr><td></td><td>IDOL [29] [IDOL† [29] [ROVIS [35] SwWn-l ROVIS† [35] GenVis* [9] Ours Ours†</td><td>41.6 40.0 42.6 41.6 42.6 45.2 45.9 47.1 13.1</td><td>39.4 61.5 65.2 63.1 65.7 65.0 42.9 64.7 69.1 71.1 71.9 27.8</td><td>41.3 18.1 43.3 42.8 19.3 45.1 40.5 17.6 46.4 45.2 17.9 49.6 18.7 46.9 42.6 18.4 49.1 48.4 19.1 48.6 48.3 18.5 51.5 49.2 19.4 11.6 9.4</td></tr><tr><td>Rest50 Oine</td><td>IFC [12] SeqFormer [28] Mask2Former-VIS [5] [VITA* [10] Ours Ours†</td><td>15.1 31.9 17.3 37.3 19.6 41.2 33.8 60.4 34.1 59.8 27.7 51.9</td><td>13.8 15.1 17.4 33.5 32.3</td><td>52.5 23.9 10.4 27.1 10.5 23.5 11.7 26.0 15.3 39.5 15.9 41.1 14.9 33.0 13.732.2 49.0 46.5</td></tr><tr><td>VITA* [10] [GenVIS* [9] Swin-l [MDQE† [15] Ours Ours†</td><td>Mask2Former-VIS [5]</td><td>25.846.5 45.4 69.2 42.6 67.8 48.6 74.7 49.9 75.9</td><td>24.9 24.4 47.8 18.9 44.3 18.3 50.5 18.8 53.8 53.0 19.4 55.3</td></tr></table>

Table 1. Results on the OVIS validation set. denotes training and evaluation at 720px. - denotes using COCO pseudo videos. The best metrics in each group are bolded.

Performance on the OVIS Dataset. In online mode, DVIS achieves 31.0 AP with ResNet50 and 47.1 AP with Swin-L on the OVIS validation set, outperforming the baseline MinVIS [11] by 4.6 AP and 5.5 AP, respectively. The referring tracker has shown significant performance gains, especially for medium and heavily occluded objects, as discussed in Section 4.2. DVIS outperforms the current SOTA online VIS methods IDOL [29] and RO-VIS [35] by 4.5 AP. This demonstrates the successful design of the referring tracker for robust tracking results particularly in heavily occluded scenarios.

<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Backbone</td><td colspan="4">Youtube-VIS 2019</td><td rowspan="2"></td><td colspan="4">Youtube-VIS 2021</td><td rowspan="2">AR10</td></tr><tr><td>AP AP50</td><td></td><td>AP75</td><td>AR1</td><td>AP  $\mathrm { A R } _ { 1 0 }$ </td><td>AP50</td><td> $\mathrm { A P _ { 7 5 } }$ </td><td>AR1</td></tr><tr><td rowspan="11">Oonliine</td><td>MaskTrack R-CNN [31]</td><td>ResNet-50</td><td>30.3</td><td>51.1</td><td>32.6</td><td>31.0</td><td>35.5</td><td>28.6</td><td>48.9</td><td>29.6</td><td>26.5</td><td>33.8</td></tr><tr><td>SipMask [3]</td><td>ResNet-50</td><td>33.7</td><td>54.1</td><td>35.8</td><td>35.4</td><td>40.1</td><td>31.7</td><td>52.5</td><td>34.0</td><td>30.8</td><td>37.8</td></tr><tr><td>CrossVIS [32]</td><td>ResNet-50</td><td>36.3</td><td>56.8</td><td>38.9</td><td>35.6</td><td>40.7</td><td>34.2</td><td>54.4</td><td>37.9</td><td>30.4</td><td>38.2</td></tr><tr><td>VISOLO [7]</td><td>ResNet-50</td><td>38.6</td><td>56.3</td><td>43.7</td><td>35.7</td><td>42.5</td><td>36.9</td><td>54.7</td><td>40.2</td><td>30.6</td><td>40.9</td></tr><tr><td>MinVIS [11]</td><td>ResNet-50</td><td>47.4</td><td>69.0</td><td>52.1</td><td>45.7</td><td>55.7</td><td>44.2</td><td>66.0</td><td>48.1</td><td>39.2</td><td>51.7</td></tr><tr><td>IDOL [29]</td><td>ResNet-50</td><td>49.5</td><td>74.0</td><td>52.9</td><td>47.7</td><td>58.7</td><td>43.9</td><td>68.0</td><td>49.6</td><td>38.0</td><td>50.9</td></tr><tr><td>Ours</td><td>ResNet-50</td><td>51.2</td><td>73.8</td><td>57.1</td><td>47.2</td><td>59.3</td><td>46.4</td><td>68.4</td><td>49.6</td><td>39.7</td><td>53.5</td></tr><tr><td>MinVIS [11]</td><td>Swin-L</td><td>61.6</td><td>83.3</td><td>68.6</td><td>54.8</td><td>66.6</td><td>55.3</td><td>76.6</td><td>62.0</td><td>45.9</td><td>60.8</td></tr><tr><td>IDOL [29]</td><td>Swin-L</td><td>64.3</td><td>87.5</td><td>71.0</td><td>55.6</td><td>69.1</td><td>56.1</td><td>80.8</td><td>63.5</td><td>45.0</td><td>60.1</td></tr><tr><td>Ours</td><td>Swin-L</td><td>63.9</td><td>87.2</td><td>70.4</td><td>56.2</td><td>69.0</td><td>58.7</td><td>80.4</td><td>66.6</td><td>47.5</td><td>64.6</td></tr><tr><td rowspan="8">IFC [12] Oine</td><td>EfficientVIS [30]</td><td>ResNet-50</td><td>37.9</td><td>59.7</td><td>43.0</td><td>40.3</td><td>46.6</td><td>34.0</td><td>57.5</td><td>37.3</td><td>33.8</td><td>42.5</td></tr><tr><td></td><td>ResNet-50</td><td>41.2</td><td>65.1</td><td>44.6</td><td>42.3</td><td>49.6</td><td>35.2</td><td>55.9</td><td>37.7</td><td>32.6</td><td>42.9</td></tr><tr><td>Mask2Former-VIS [5]</td><td>ResNet-50</td><td>46.4</td><td>68.0</td><td>50.0</td><td></td><td></td><td>40.6</td><td>60.9</td><td>41.8</td><td></td><td></td></tr><tr><td>SeqFormer [28]</td><td>ResNet-50</td><td>47.4</td><td>69.8</td><td>51.8</td><td>45.5</td><td>54.8</td><td>40.5</td><td>62.4</td><td>43.7</td><td>36.1</td><td>48.1</td></tr><tr><td>VITA [10]</td><td>ResNet-50</td><td>49.8</td><td>72.6</td><td>54.5</td><td>49.4</td><td>61.0</td><td>45.7</td><td>67.4</td><td>49.5</td><td>40.9</td><td>53.6</td></tr><tr><td>Ours</td><td>ResNet-50</td><td>52.6</td><td>76.5</td><td>58.2</td><td>47.4</td><td>60.4</td><td>47.4</td><td>71.0</td><td>51.6</td><td>39.9</td><td>55.2</td></tr><tr><td>SeqFormer [28]</td><td>Swin-L</td><td>59.3</td><td>82.1</td><td>66.4</td><td>51.7</td><td>64.4</td><td>51.8</td><td>74.6</td><td>58.2</td><td>42.8</td><td>58.1</td></tr><tr><td>Mask2Former-VIS [5]</td><td>Swin-L</td><td>60.4</td><td>84.4</td><td>67.0</td><td></td><td></td><td>52.6</td><td>76.4</td><td>57.2</td><td></td><td></td></tr><tr><td>VITA [10]</td><td>Swin-L</td><td>63.0</td><td>86.9</td><td>67.9</td><td></td><td>56.3</td><td>68.1</td><td>57.5</td><td>80.6</td><td>61.0</td><td>47.7</td><td>62.6</td></tr><tr><td>Ours</td><td>Swin-L</td><td>64.9</td><td>88.0</td><td></td><td>72.7</td><td>56.5</td><td>70.3</td><td>60.1</td><td>83.0</td><td>68.4</td><td>47.7</td><td>65.7</td></tr></table>

Table 2. Results on the validation set of YouTube-VIS 2019 & 2021. The best metrics in each group are bolded.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="5">YouTube-VIS 2022</td></tr><tr><td>AP</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathsf { A P } _ { 7 5 }$ </td><td>AR1</td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td>Swi-l</td><td>VITA [10]</td><td>41.1</td><td>63.0</td><td>44.0</td><td>39.3</td><td>44.3</td></tr><tr><td rowspan="3"></td><td>MinVIS [11]</td><td>33.1</td><td>54.8</td><td>33.7</td><td>29.5</td><td>36.6</td></tr><tr><td>Ours</td><td>45.9</td><td>69.0</td><td>48.8</td><td>37.2</td><td>51.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3. Results on the YouTube-VIS 2022 long videos. The best metrics in each group are bolded.
<table><tr><td></td><td>Method</td><td colspan="3">VIPSeg VPQ VPQTh VPQSt STQ</td></tr><tr><td>R50</td><td>VPSNet [13] VPSNet-SiamTrack [27] VIP-Deeplab [22] Clip-PanoFCN [20] Video K-Net [17] TarVIS [1] Tube-Link [16] Video-kMax [24]</td><td>14.0 17.2 16.0 22.9 26.1 33.5 39.2 38.2</td><td>14.0 17.3 12.3 25.0 39.2 I 一</td><td>14.2 20.8 17.3 21.1 18.2 22.0 20.8 31.5 31.5 28.5 43.1 39.5 39.9</td></tr><tr><td>Swin-L</td><td>Ours TarVIS [1] Ours</td><td>43.2 48.0 57.6</td><td>43.6 58.2 59.9</td><td>42.8 42.8 39.0 52.9 55.5 55.3</td></tr></table>

Table 4. Results on the VIPSeg dataset. The best metrics in each group are bolded.

In offline mode, DVIS achieves 34.1 AP with ResNet50 and 49.9 AP with Swin-L on the OVIS validation set, surpassing DVIS running in online mode by 3.1 AP and 2.8 AP, respectively. More impressively, DVIS outperforms the baseline MinVIS by 8.3 AP. Additionally, DVIS surpasses the previous pure offline VIS methods Mask2Former-VIS [5] and VITA [10] by 24.1 AP and 22.2 AP, respectively. Thus, DVIS achieves a new SOTA performance, demonstrating the superiority of the decoupled framework in complex scenarios compared to the previous coupled framework.

Performance on the YouTube-VIS 2019 and 2021 Datasets. In online mode, DVIS achieved 51.2 AP with ResNet50 and 63.9 AP with Swin-L on the YouTube-VIS 2019 validation set, outperforming MinVIS [11] by 3.8 AP and 2.3 AP, respectively. For YouTube-VIS 2021 validation set, DVIS achieved 46.4 AP with ResNet50 and 58.7 AP with Swin-L in online mode, outperforming MinVIS [11] by 2.2 AP and 3.4 AP, respectively. On the YouTube-VIS datasets, DVIS running in online mode shows comparable performance with the current SOTA method IDOL [29].

In offline mode, DVIS achieved 52.6 AP with ResNet50 and 64.9 AP with Swin-L on the YouTube-VIS 2019 validation set, outperforming DVIS running in online mode by 1.4 AP and 1.0 AP. Similarly, on the YouTube-VIS 2021 validation set, DVIS achieved 47.4 AP with ResNet50 and 60.1 AP with Swin-L, outperforming DVIS (online mode) by 1.0 AP and 1.4 AP. DVIS achieves a new SOTA performance on the YouTube-VIS 2019 and 2021 datasets.

Performance on the YouTube-VIS 2022 Dataset. DVIS achieves a new SOTA performance of 52.8 AP on the validation set of the YouTube-VIS 2022 dataset, with 59.6 AP on short videos and 45.9 AP on long videos. Since the short videos of the YouTube-VIS 2022 dataset largely overlap with the YouTube-VIS 2021 dataset, we compare the performance of DVIS with other methods only on long videos, as shown in Table 3. DVIS outperforms the baseline method MinVIS [11] by 12.8 AP and the current SOTA method VITA [10] by 4.8 AP.

Performance on the VIPSeg Dataset. DVIS achieves

<table><tr><td></td><td> $\mathrm { A P _ { a l l } }$ </td><td> $\mathsf { A P } _ { 1 }$ </td><td> $\mathrm { A P _ { m } }$ </td><td> $\mathrm { A P _ { h } }$ </td></tr><tr><td>baseline</td><td> $\overline { { 4 1 . 6 } }$ </td><td> $\overline { { 6 4 . 8 } }$ </td><td> $\overline { { 4 9 . 0 } }$ </td><td> $2 0 . 5$ </td></tr><tr><td>+Tracker</td><td>47.1(+5.5) 64.7(-0.1)</td><td></td><td> $5 4 . 2 ( + 5 . 2 ) $ </td><td> $2 4 . 8 ( + 4 . 3 ) $ </td></tr><tr><td>+Refiner</td><td> $4 9 . 9 ( + 8 . 3 ) $ </td><td> $6 7 . 1 ( + 2 . 3 ) $ </td><td> $5 6 . 0 ( + 7 . 0 ) $ </td><td> $2 9 . 8 ( + 9 . 4 ) $ </td></tr></table>

Table 5. Ablation study of the proposed components. The baseline is MinVIS [11]. All models use Swin-L as the backbone and are evaluated on the OVIS validation set with 720p input. $\operatorname { A P } _ { 1 } .$ $\mathrm { A P _ { m } }$ and $\mathrm { A P _ { h } }$ refer to the AP of the light, medium, and heavily occluded instances, respectively.

43.2 VPQ and 57.6 VPQ on the VIPSeg validation set when using ResNet50 and Swin-L backbones, respectively, surpassing the current SOTA VPS method TarVIS [1] by 9.7 VPQ and 9.6 VPQ. These results demonstrate the outstanding performance of DVIS on video panoptic segmentation (VPS) and its potential to achieve SOTA performance on all video segmentation tasks.

## 4.2. Ablation Experiments

Ablation experiments were conducted on the OVIS dataset, with DVIS evaluated using ResNet50 and input resized to 360p unless otherwise specified.

Effectiveness of Referring Tracker and Temporal Refiner. We conducted ablation experiments on the OVIS dataset to evaluate the effectiveness of the referring tracker and temporal refiner. The results of the experiments are presented in Table 5. Our findings indicate that the referring tracker leads to significant performance gains when processing medium and heavily occluded objects, resulting in an increase of $5 . 2 \mathrm { A P _ { m } }$ and $4 . 3 \mathrm { A P _ { h } }$ , respectively. However, there is a slight decrease of $0 . 1 \ \mathrm { A P _ { l } }$ in the case of lightly occluded objects, indicating that the improvement in the referring tracker is primarily in tracking quality rather than segmentation quality. We further illustrate our findings by presenting an instance of a completely occluded panda with ID 1 in the third frame, which is tracked well by DVIS but not by MinVIS [11], in Figure 4.

The temporal refiner leads to performance gains in both segmentation quality and tracking quality, with improvements of 2.3 $\mathrm { A P _ { l } , 7 . 0 \ A P _ { m } , }$ and $9 . 4 ~ \mathrm { A P _ { h } }$ across the board. The temporal refiner effectively utilizes the entire video information, leading to more significant improvements for heavily occluded objects, as demonstrated in Figure 4 where the green rectangles highlight a highly occluded panda. Despite this challenge, the temporal refiner produces accurate segmentation results, while the referring tracker fails due to its inability to leverage the full video information.

Initial Instance Representation of Referring Tracker. Our proposed referring tracker represents the VIS as referring denoising, making it crucial to select an appropriate initial value with noise. We evaluate the performance with different initial values and report the results in Table 6. The best performance is achieved when using the $Q _ { s e g }$ obtained by matching with the hungarian algorithm as the initial value. When zero is used as the initial value, the denoising task becomes a more challenging reconstruction problem, leading to a drop of 1.6 AP. Using the $Q _ { T r }$ of the previous frame as the initial value results in a 2.2 AP performance degradation, as it contains too much interference information. The network also performs well when using the unmatched $Q _ { s e g } ,$ where the initial values of the instance queries of each frame are linked by the learnable prior information, demonstrating the robustness of the referring tracker.

<table><tr><td>Initial Value</td><td>AP</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathsf { A P } _ { 7 5 }$ </td><td> $\mathrm { \ A R _ { 1 } }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td>Zero</td><td>28.9</td><td>52.5</td><td>27.9</td><td>14.7</td><td>35.7</td></tr><tr><td> $Q _ { T r } ^ { p r e }$ </td><td>28.3</td><td>50.1</td><td>27.3</td><td>14.5</td><td>33.8</td></tr><tr><td> $Q _ { s e g }$ </td><td>29.8</td><td>54.3</td><td>28.3</td><td>14.8</td><td>36.5</td></tr><tr><td>Matched  $Q _ { s e g }$ </td><td>30.5</td><td>54.7</td><td>30.1</td><td>15.0</td><td>36.5</td></tr></table>

Table 6. Ablation study of the initial instance representation in the referring tracker. $Q _ { T r } ^ { p r e }$ denotes the instance representation in the previous frame, and $Q _ { s e g }$ denotes the instance representation output by the segmenter.
<table><tr><td>Cross Attn Type</td><td>AP</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathsf { A P } _ { 7 5 }$ </td><td> $\mathrm { \ A R _ { 1 } }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td>Standard</td><td>2.9</td><td>5.0</td><td>2.8</td><td>2.4</td><td>3.4</td></tr><tr><td>Referring</td><td>30.5</td><td>54.7</td><td>30.1</td><td>15.0</td><td>36.5</td></tr></table>

Table 7. Ablation study on the type of cross-attention in the referring tracker.

<table><tr><td></td><td>AP  $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathrm { A P _ { 7 5 } }$ </td><td> $\mathrm { \ A R _ { 1 } }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td>Baseline</td><td>32.2 57.9</td><td>31.3</td><td>15.1</td><td>38.7</td></tr><tr><td>w/o Long-term Attn.</td><td>31.0 56.2</td><td>29.3</td><td>14.9</td><td>37.8</td></tr><tr><td>w/o Short-term Conv.</td><td>31.8 56.6</td><td>30.2</td><td>14.8</td><td>37.9</td></tr><tr><td>w/o Cross Attn.</td><td>30.6 55.5</td><td>28.7</td><td>14.8</td><td>36.6</td></tr></table>

Table 8. Ablation study on the components of the temporal decoder block. Attn. denotes attention and Conv. denotes convolutional.“w/o” refer to without.

Referring Cross-Attention. The referring crossattention is a crucial component of the referring tracker, responsible for linking historical frames with the current frame. We evaluated the importance of the referring crossattention by comparing it to the standard cross-attention, where ID is set to Q in Equation 2. The results in Table 7 demonstrate that replacing the referring cross-attention with the standard cross-attention leads to an extreme drop in performance. This finding highlights the critical role of interframe associations modeled by the referring cross-attention in the success of the referring tracker.

Impact of Different Components of Temporal Decoder Block. To evaluate the impact of different components of the temporal decoder block, we conducted experiments by removing each component individually and reporting the corresponding performance in Table 8. Our results show that removing long-term temporal self-attention led to a performance degradation of 1.2 AP. Although the function of the long-term attention overrides the function of the short-term convolution, removing the short-term convolution still resulted in a performance degradation of 0.4 AP, suggesting that it is beneficial for utilizing information in adjacent frames. Moreover, the removal of cross-attention resulted in a significant drop of 1.6 AP since incorrect instance queries cannot be efficiently corrected without it, even though information from different frames can still be utilized.

![](images/b0e1a7dd9d4bcfdee35edde9ffc8080a207e82efd900780a36798d5628027305.jpg)  
Figure 4. Visualization results comparing DVIS with current SOTA online and offline VIS methods. VITA shows poor segmentation quality (highlighted with red circles) and tracking stability (highlighted with red rectangles). The referring tracker demonstrates strong tracking ability (highlighted with blue rectangles), while the temporal refiner effectively utilizes contextual information from previous and future frames (highlighted with green rectangles).

Performance of DVIS in Semi-Online Mode. In realworld scenarios, videos are often of infinite length, making it impossible to run VIS models in pure offline mode. We conduct experiments to measure the performance difference between semi-online and offline modes, as shown in Table II of Appendix. When videos are cut into clips of length 1 as input to DVIS, i.e., no other frame information available for the current frame, the performance was only comparable to that of DVIS without temporal refiner. However, as the clip length increased, the performance of the semi-online mode gradually approached that of the pure offline mode, and achieved comparable performance after the clip length exceeded 80 frames (33.8 vs. 33.8).

Computational Cost. The computational cost of DVIS components was measured by evaluating the parameters, MACs, and inference time of the segmenter, referring tracker, and temporal refiner. Table 9 presents the results. When using Mask2Former with ResNet50 and Swin-L as the segmenter, the referring tracker and temporal refiner combined only accounted for 5.18% and 1.69% of the segmenter’s computation, respectively. This demonstrates that the referring tracker and temporal refiner can efficiently achieve VIS with almost negligible computational cost.

<table><tr><td>Component</td><td>Inp.</td><td> $\Nu _ { \mathrm { Q } }$ </td><td colspan="3">Params(M) MACs(G) Time(ms)</td></tr><tr><td>M2F(R50)</td><td>480p</td><td>100</td><td>43.95</td><td>103.73</td><td rowspan="4">48.10</td></tr><tr><td>Tracker Refiner</td><td>480p</td><td>100 100</td><td>9.68</td><td>1.68</td><td>7.63 1.11</td></tr><tr><td>M2F(SwinL)</td><td>480p</td><td>200</td><td>14.41 215.30</td><td>3.69</td><td></td></tr><tr><td></td><td>720p 720p</td><td>200</td><td>9.68</td><td>851.00</td><td>275.19</td></tr><tr><td>Tracker Refiner</td><td>720p</td><td>200</td><td>14.41</td><td>5.13 9.27</td><td>7.97 2.00</td></tr></table>

Table 9. Computational cost of DVIS components. M2F refers to the Mask2Former used as the segmenter of DVIS. Inp. denotes the size of the input video, and $\Nu _ { \mathrm { Q } }$ denotes the number of queries. The inference time per frame is measured on a 1080Ti GPU.

## 5. Conclusion

In this paper, we propose DVIS, a decoupled VIS framework that separates the VIS task into three sub-tasks: segmentation, tracking, and refinement. Our contributions are three-fold: 1) we decouple the VIS task and introduce the DVIS framework, 2) we propose the referring tracker, which enhances tracking robustness by modeling inter-frame associations as referring denoising, and 3) we propose the temporal refiner, which utilizes information from the entire video to refine segmentation results, a capability that was missing in previous methods. Our results show that DVIS achieves SOTA performance on all VIS datasets, outperforming all existing methods, supporting the effectiveness of our decoupling standpoint and the design of DVIS. Additionally, DVIS’s SOTA performance on VPS demonstrates its potential and versatility. We believe that DVIS will serve as a strong and fundamental baseline, and our decoupling insights will inspire future works in both online and offline VIS.

## References

[1] Ali Athar, Alexander Hermans, Jonathon Luiten, Deva Ramanan, and Bastian Leibe. Tarvis: A unified approach for target-based video segmentation. arXiv preprint arXiv:2301.02657, 2023.

[2] Ali Athar, Sabarinath Mahadevan, Aljosa Osep, Laura Leal-Taixe, and Bastian Leibe. Stem-seg: Spatio-temporal em- ´ beddings for instance segmentation in videos. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 158–177. Springer, 2020.

[3] Jiale Cao, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, and Ling Shao. Sipmask: Spatial information preservation for fast image and video instance segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pages 1–18. Springer, 2020.

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 213–229. Springer, 2020.

[5] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G Schwing. Mask2former for video instance segmentation. arXiv preprint arXiv:2112.10764, 2021.

[6] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022.

[7] Su Ho Han, Sukjun Hwang, Seoung Wug Oh, Yeonchool Park, Hyunwoo Kim, Min-Jung Kim, and Seon Joo Kim. Visolo: Grid-based space-time aggregation for efficient online video instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2896–2905, 2022.

[8] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-´ shick. Mask r-cnn. In Proceedings ofthe IEEE international conference on computer vision, pages 2961–2969, 2017.

[9] Miran Heo, Sukjun Hwang, Jeongseok Hyun, Hanjung Kim, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. A generalized framework for video instance segmentation. arXiv preprint arXiv:2211.08834, 2022.

[10] Miran Heo, Sukjun Hwang, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. Vita: Video instance segmentation via object token association. arXiv preprint arXiv:2206.04403, 2022.

[11] De-An Huang, Zhiding Yu, and Anima Anandkumar. Minvis: A minimal video instance segmentation framework without video-based training. arXiv preprint arXiv:2208.02245, 2022.

[12] Sukjun Hwang, Miran Heo, Seoung Wug Oh, and Seon Joo Kim. Video instance segmentation using inter-frame communication transformers. Advances in Neural Information Processing Systems, 34:13352–13363, 2021.

[13] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9859–9868, 2020.

[14] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

[15] Minghan Li, Shuai Li, Wangmeng Xiang, and Lei Zhang. Mdqe: Mining discriminative query embeddings to segment occluded instances on challenging videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10524–10533, 2023.

[16] Xiangtai Li, Haobo Yuan, Wenwei Zhang, Guangliang Cheng, Jiangmiao Pang, and Chen Change Loy. Tube-link: A flexible cross tube baseline for universal video segmentation. arXiv preprint arXiv:2303.12782, 2023.

[17] Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Video k-net: A simple, strong, and unified baseline for video segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18847– 18857, 2022.

[18] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[19] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multi-object tracking with transformers. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8844–8854, 2022.

[20] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21033–21043, 2022.

[21] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. International Journal of Computer Vision, 130(8):2022–2039, 2022.

[22] Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Vip-deeplab: Learning visual perception with depth-aware video panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3997–4008, 2021.

[23] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 208–223. Springer, 2020.

[24] Inkyu Shin, Dahun Kim, Qihang Yu, Jun Xie, Hong-Seok Kim, Bradley Green, In So Kweon, Kuk-Jin Yoon, and Liang-Chieh Chen. Video-kmax: A simple unified approach for online and near-online video panoptic segmentation. arXiv preprint arXiv:2304.04694, 2023.

[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia

Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[26] Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. End-to-end video instance segmentation with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8741–8750, 2021.

[27] Sanghyun Woo, Dahun Kim, Joon-Young Lee, and In So Kweon. Learning to associate every segment for video panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2705–2714, 2021.

[28] Junfeng Wu, Yi Jiang, Song Bai, Wenqing Zhang, and Xiang Bai. Seqformer: Sequential transformer for video instance segmentation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII, pages 553–569. Springer, 2022.

[29] Junfeng Wu, Qihao Liu, Yi Jiang, Song Bai, Alan Yuille, and Xiang Bai. In defense of online models for video instance segmentation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII, pages 588–605. Springer, 2022.

[30] Jialian Wu, Sudhir Yarram, Hui Liang, Tian Lan, Junsong Yuan, Jayan Eledath, and Gerard Medioni. Efficient video instance segmentation via tracklet query and proposal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 959–968, 2022.

[31] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5188–5197, 2019.

[32] Shusheng Yang, Yuxin Fang, Xinggang Wang, Yu Li, Chen Fang, Ying Shan, Bin Feng, and Wenyu Liu. Crossover learning for fast online video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8043–8052, 2021.

[33] Kaining Ying, Zhenhua Wang, Cong Bai, and Pengfei Zhou. Isda: Position-aware instance segmentation with deformable attention. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2619–2623. IEEE, 2022.

[34] Kaining Ying, Qing Zhong, Weian Mao, Zhenhua Wang, Hao Chen, Lin Yuanbo Wu, Yifan Liu, Chengxiang Fan, Yunzhi Zhuge, and Chunhua Shen. Ctvis: Consistent training for online video instance segmentation. arXiv preprint arXiv:2307.12616, 2023.

[35] Zitong Zhan, Daniel McKee, and Svetlana Lazebnik. Robust online video instance segmentation with track queries. arXiv preprint arXiv:2211.09108, 2022.

[36] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.