# Beyond Single Path Integrated Gradients for Reliable Input Attribution via Randomized Path Sampling

Giyoung Jeon<sup>\*</sup> LG AI Research giyoung.jeon@lgresearch.ai

Haedong Jeong UNIST, KAIST haedong@unist.ac.kr

Jaesik Choi<sup>†</sup> KAIST, INEEJI jaesik.choi@kaist.ac.kr

## Abstract

Input attribution is a widely used explanation methodfor deep neural networks, especially in visual tasks. Among various attribution methods, Integrated Gradients (IG) [28] isfrequently used because ofits model-agnostic applicability and desirable axioms. However, previous work [24, 8, 9] has shown that such method often produces noisy and unreliable attributions during the integration of the gradients over the path defined in the input space. In this paper, we tackle this issue by estimating the distribution of the possible attributions according to the integrating path selection. We show that such noisy attribution can be reduced by aggregating attributionsfrom the multiple paths instead ofusing a single path. Inspired by Stick-Breaking Process [20], we suggest a random process to generate rich and various sampling of the gradient integrating path. Using multiple input attributions obtained from randomized path, we propose a novel attribution measure using the distribution of attributions at each input features. We identify proposed method qualitatively show less-noisy and object-aligned attribution and its feasibility through the quantitative evaluations.

## 1. Introduction

Along with the steep improvement and the real world application of the deep learning models [4, 31], discovering the evidence of the black-box model decision is considered to be important for debugging the malfunction [12] and promise the safety and the fairness [5] of the models. Within the vast literature of explaining the decision of the deep models, input attribution [22, 2, 21, 28] is one of widely used methods to quantify the relative contribution of each features to the model output. Input attribution provides the explanation in the form of heatmaps, which is useful to indicate the spatial existence of evidences, especially in visual tasks.

Among various approaches to compute the input attributions, Integrated Gradient (IG), one of widely used methods, and its variants [28, 15, 9] are of particular interest in our work. These methods explore the input space along the predefined path and integrate the gradients to provide the reliable attributions. The integration path of such methods consists of a baseline which represents the missingness of features and a connecting line between the input and the baseline. With different desired properties, various paths can be used to compute the attribution. For example, Guided IG [9] proposes the adaptive path to alleviate the high and noisy gradients unrelated to the prediction. The selection of baseline can also affect to the attribution results [27].

While the above methods address the importance of selecting appropriate integration path, in this paper, we claim that the single path is not reliable enough to interpret the decision of neural networks. We provide a simple example that the attribution computed by a single path provides high variance according to different path selection. For better reliability, we propose a novel attribution method to take the expectation of the path-integrated attribution over the distribution of possible paths. To sample from the distribution over the vast variety of possible paths, we adopt the notion of Stick-Breaking Process, which is one sort of stochastic processes that samples the probability distribution. The main contributions of our work are summarized as,

• Address the inconsistency of attribution according to the selection of the integration path, and propose a novel attribution method that takes the expectation over the distribution of random paths to retain the reliability of attribution.

• Propose a sampling method to generate a random integration path inspired by the Stick-Breaking Process. From the proposed method, we can generate the vast integration paths efficiently.

• Evaluate the attribution in qualitative and quantitative measure to validate the reliability of the proposed method on various of architecture of the networks.

![](images/e5faf7d74dcf33a3a136805fbcc7a847c8615d292baef31ac3aacd8087169517.jpg)  
Figure 1: An illustration of Stick-breaking Path Integration (SPI) for the given input x. Using the realized distribution G from SBP, we randomly generate the integration path in the input domain by taking CDF of each distribution (colored lines in the left-bottom). From the sampled paths, we apply the gradient integration along each path to gather the multiple attribution samples. By taking the average, the attribution of SPI can be obtained.

## 2. Related Work

Attribution methods The input attribution methods aims to measure the relative sensitivity of the model output with respect to the input features. Saliency method [22] is a simple approach to use the gradient as attribution. Then Grad\*Input method [21] is proposed to multiply the input with the gradient for better input alignment. FullGrad [26] proposes to use the bias gradient in addition to Grad\*Input. Guided Backpropagation [25] suggests to consider only features that positively contributes to the prediction by ignoring the negative backpropagated gradients. Layerwise Relevance Propagation (LRP) [2, 14] is another method to modify the backward propagation. LRP proposes the relevance propagation rules based on the Taylor decomposition. There exists a family of attribution methods which do not require the access to the internal properties (e.g., gradients, parameters). LIME [17] trains a surrogate linear model, which resembles the original model for the data that features are masked out from the input to be explained. RISE [16] computes the attribution by aggregating the model outputs from the multiple random masked inputs.

Path-based attribution methods Integrated Gradients (IG) [28] is one of widely used input attribution method. It is built upon the game-theoretic notion of pay-off distribution method, Aumann-Shapley value [1]. IG has several desirable properties, called axioms, which supports the reliability of the attribution. IG is calculated by integrating the gradients along the path from the baseline to the input. Based on this work, several extended research has been performed. To reduce the noise in the attribution, SmoothGrad [24] takes the average of the multiple random noise added inputs and NoiseGrad [3] inject noise to the weight parameters. From the observation that the sum of gradient of multiple features is more stable than the individual gradients, XRAI [8] proposes the method to merge the attribution of neighboring features using multi-level superpixels. Some researches try to modify the integration path with their own desired characteristics [9, 15].

Evalutation of attribution methods Even though a plethora of attribution methods are proposed, evaluating the reliability or the correctness are still challenging. Pixel flip is an causal metric between the input and the prediction confidence [19]. Pixel flip is calculated by setting the value of each feature to zero in order of increasing or decreasing order of attributions. As an extension, insertion and the deletion game is proposed by starting the insertion game from the blurred image to avoid the spurious effects when small number of pixels are inserted [16]. RemOve-And-Retrain (ROAR) measures the performance drop when the dataset is reorganized by perturbing the input images with the average pixel value in decreasing order of attribution [7]. Steep performance drop in ROAR indicates that the attribution method correctly points out important features which are actually used to train and inference.

## 3. Stick-breaking Path Integration

In this section, we propose our new path-based attribution method, Stick-breaking Path Integration (SPI). We first provide some backgrounds about Integrated Gradients (IG)

![](images/15fe59ccd985506c63de40c7bb30a6e3e400778f878b17e0b7be87497fae888e.jpg)

![](images/ae4ef8881661ababe272f7139768b0c81851f248f455eaf1e9cd196ca32d7e1b.jpg)

![](images/aa8e9d8d73f9870ee88ade41ba8f1162ea91592d82c36a070b492ecf91ae9c51.jpg)

![](images/e9bf14d0e1e3e654945268b7ae5f866dac3c21f99e38cc19530c740fce173a4c.jpg)

(a)  
![](images/ada857dc7a3ccd75c56083de7e5b5c75d6a306e7c99a8d8a2f84440ad4c4c9d7.jpg)  
(b)  
Figure 2: (a) An illustrative example of various path integrated attributions with difference path choices. Each colored region indicates the passed different decision region along the integrated path. We identify that the complexity of integration (e.g., the number of decision regions) can be determined by the shape of the integrated path. (b) An illustrative example of converting the realized partitions of a stick to the integration paths.

[28] and its family of attribution methods, path methods. From the insight that the attribution highly depends on the path selection, we propose to take the expectation over the distribution of random paths for reliability of attribution. Due to the difficulty in defining the distribution of paths, we propose a sampling process which is motivated by the Stick-Breaking Process (SBP) [20]. Using the multiple attributions computed by randomly sampled paths, we finally propose SPI. In addition, we propose a visualization method for SPI, SPI-P, which utilizes the stochastic property of the sampling process.

## 3.1. Path methods and Integrated Gradients

Integrated Gradient (IG) [28] is proposed as an explanation method for neural networks by adopting the Aumann-Shapley value [1], which is a credit allocation method developed for the cooperative game theory. Based on such allocation strategy, IG aims to compute the contribution of input features to the model prediction. For a differentiable model function $f ,$ the input x and the baseline ¯x, IG on i-th feature is given as

$$
\phi _ { i } ( \pmb { \gamma } ) = \int _ { 0 } ^ { 1 } \frac { \partial f ( \pmb { \gamma } ( t ) ) } { \partial \gamma _ { i } ( t ) } \frac { \partial \gamma _ { i } ( t ) } { \partial t } d t , \quad \pmb { \gamma } ( t ) = \bar { \bf x } + t ( \mathbf { x } - \bar { \bf x } )\tag{1}
$$

where $\gamma ( t )$ represents a continuous path from the baseline ¯x to the input x and $\gamma _ { i } ( t )$ refers to the value of i-th feature at step t along the path. Previous work has shown that integrating over different path γ yields different type of attribution method [15, 9]. We call this group of attribution methods with arbitrary selection of path γ as path methods. However, the intermediate gradient in the single path is likely to be noisy and it reduces the reliability and interpretability of the obtained attribution. Several work has shown such noise can be reduced by averaging IG over randomly perturbed inputs [24], randomly perturbed weights [3] or average pooling over spatial location (i.e., local pixels for image) [8].

In this work, we propose to aggregate the attribution from multiple paths with a fixed baseline sampled from the proposed distribution of paths to reduce the noise and improve the confidence of the attribution. We first define the integration path and its desired properties.

Definition 3.1 (Integration Path). The integration path is a mapping function from $t \in [ 0 , 1 ]$ to the input domain X, $\gamma : [ 0 , 1 ] \mapsto \mathcal { X }$ . The path should satisfy two properties; (1) the path starts from the baseline, $\gamma ( 0 ) { = } \bar { \mathbf { x } }$ , and ends at the input to be attributed, $\gamma ( 1 ) { = } \mathbf { x }$ and (2) the path monotonically proceeds from ¯x to x, i.e., $\begin{array} { r } { \frac { d { \gamma } _ { i } ( t ) } { d t } = C ( x _ { i } - \bar { x } _ { i } ) } \end{array}$ for $C \geq 0$

In the rest of the paper, we denote the bold symbols for the vector $( \mathrm { e . g . , \ x ) }$ . We use the subscript to represent the indexed component $( \mathrm { e . g . , ~ } x _ { i } )$ . We use the superscript to represent the different indexed instances (e.g., $\mathbf { \bar { w } } ^ { ( 1 ) } , \ldots , \mathbf { w } ^ { ( j ) } , \ldots )$

## 3.2. Relationship between Path and Attribution

A neural network equipped with the partial linear activation, such as ReLU, is known to have the form of the piece-wise linear function [13]. The piece-wise linear function is defined by multiple linear functions, where each linear function is only feasible in corresponding linear region, $\mathcal { R } ^ { ( j ) }$ . Such piece-wise linear function f can be formulated as follow with each linear region $\mathcal { R } ^ { ( j ) }$

$$
f ( \mathbf { x } ) = \left\{ \begin{array} { l l } { \mathbf { w } ^ { ( 1 ) T } \mathbf { x } + \mathbf { b } ^ { ( 1 ) } } & { \mathbf { x } \in \mathcal { R } ^ { ( 1 ) } } \\ { \cdot \cdot \cdot } \\ { \mathbf { w } ^ { ( j ) T } \mathbf { x } + \mathbf { b } ^ { ( j ) } } & { \mathbf { x } \in \mathcal { R } ^ { ( j ) } } \\ { \cdot \cdot \cdot } \end{array} \right.\tag{2}
$$

Then the path method provides the attribution in terms of weighted sum of corresponding weight vectors in each region,

$$
\phi _ { i } ( \pmb { \gamma } ) = \sum _ { j } \frac { w _ { i } ^ { ( j ) } \alpha _ { i } ^ { ( j ) } ( \pmb { \gamma } ) } { \sum _ { j ^ { \prime } } \alpha _ { i } ^ { ( j ^ { \prime } ) } ( \pmb { \gamma } ) } ,\tag{3}
$$

$$
\alpha _ { i } ^ { ( j ) } ( \pmb { \gamma } ) = \int _ { t = 0 } ^ { 1 } \frac { d \gamma _ { i } ( t ) } { d t } \delta ^ { ( j ) } ( \pmb { \gamma } ( t ) ) d t\tag{4}
$$

where $\alpha _ { i } ^ { ( j ) } ( \pmb { \gamma } )$ represents the projected length of the path $\gamma$ to the i-th feature that passes through the region $\mathcal { R } ^ { ( j ) }$ The delta function $\delta ^ { ( j ) } ( \gamma ( { \bar { t } } ) )$ returns 1 if $\gamma ( t ) \in \mathcal { R } ^ { ( j ) }$ and otherwise 0. Even though $\phi _ { j }$ in Equation 4 sums over every region, we note that it is equivalent to the summation over the regions that path γ passes through, because $\alpha _ { i } ^ { ( j ) } ( \pmb { \gamma } ) = 0$ for the region $\mathcal { R } ^ { ( j ) }$ that γ does not pass through.

It has been shown that to increase the expressivity of the DNNs, the number of linear regions should also increase [30]. With the high dimensional input space and the vast number of linear regions, using a single path for the path methods would induce high uncertainty and variance to the resulting attributions. Figure 2a illustrates that even in 2- dimensional space with several linear regions, different selection of path aggregates mostly different weight vectors of corresponding regions. To alleviate such uncertainty with selecting an integration path, we propose to compute the expectation of path method over the distribution of possible paths as follow,

$$
\mathbb { E } _ { \pmb { \gamma } } \left[ \phi _ { i } ( \pmb { \gamma } ) \right] = \int _ { \pmb { \gamma } } \int _ { 0 } ^ { 1 } \frac { \partial f ( \pmb { \gamma } ( t ) ) } { \partial \gamma _ { i } ( t ) } \frac { \partial \gamma _ { i } ( t ) } { \partial t } d t P ( \pmb { \gamma } ) d \pmb { \gamma } .\tag{5}
$$

## 3.3. Randomized Path Sampling

As the distribution of the path, $P ( \gamma )$ , is difficult to be defined, we propose an alternative approach to randomly generate the path with the variety. When generating a path, we have two choices to select; (1) whether proceed to the destination or not and (2) how far to proceed. We reduce this problem to the Stick-Breaking Process (SBP) [20]. SBP is a generative process to obtain random fractions of a stick with its initial length of 1. The fraction obtained by SBP can be regarded as how much portion from the baseline to the input should move at each step. In this configuration, the number of fraction can be regarded as the number of steps in the path and each length of fractions can be regarded as each length of steps. Figure 2b depicts how the sampled partitions of the unit length stick can be used to build the integration path. Each sampled fraction is represented a Probability Mass Function (PMF), and such PMF is defined as follow,

![](images/0083743ec27d81a0ff4f702c308b7ea402affa19bf5839b62daf2619e64619f8.jpg)  
Figure 3: Visualization of CDF of SBP realized distributions G with different concentration parameter α (colored lines). The base distribution is given as uniform distribution, U(0, 1), and its CDF is given as the straight line (black line). The realized CDF converges to the CDF of the base distribution when the concentration hyperparameter α increase.

$$
G ( t ) { = } \sum _ { k { = 1 } } ^ { { \infty } } { \pi } _ { k } \delta _ { t _ { k } } ( t ) \sim S B P ( H , \alpha ) ,\tag{6}
$$

$$
\pi _ { k } { = } \beta _ { k } \prod _ { i = 1 } ^ { k - 1 } ( 1 - \beta _ { i } ) ,\tag{7}
$$

$$
\beta _ { k } \sim B e t a ( 1 , \alpha ) , \quad t _ { k } \sim H\tag{8}
$$

where $\delta _ { t _ { k } } ( t )$ is a delta function that returns zero everywhere, except for $\delta _ { t _ { k } } ( t = t _ { k } ) = 1$ and H is a base distribution. The expectation of G sampled by SBP is desired to estimate the base distribution H. The hyperparameter $\alpha ,$ also known as concentration parameter, controls the realized distribution G to be more similar to the base distribution if α takes larger value.

From the sampled PMF G(t), we define the CDF $F _ { G } ( t )$ which can be used to build the integration path function. Let the integration path be

$$
\gamma _ { i } ( t ) = \bar { x } _ { i } + F _ { G } ( t ) ( x _ { i } - \bar { x } _ { i } ) .\tag{9}
$$

We note that the Equation 10 satisfies the properties of Definition $3 . 1 ; ( 1 ) \gamma _ { i } ( 0 ) = \bar { x _ { i } }$ and $\gamma _ { i } ( 1 ) { = } x _ { i }$ because $F _ { G } ( 0 ) = 0$ and $F _ { G } ( 1 ) = 1$ , and (2) $d F _ { G } ( t ) / d t = G ( t ) \geq 0$ . With taking the base distribution H to be the uniform distribution, $H = U ( 0 , 1 )$ , whose CDF is equivalent to the IG straight path. Figure 3 depicts the CDF of SBP realized distributions using different α. When the value of α increases, the realized CDF converges to the CDF of $U ( 0 , 1 )$ (black line), and we call this CDF as the base path.

Definition 3.2 (Stick-breaking Path (SP)). Given a hyperparameter $\alpha _ { i } > 0$ , the Stick-breaking Path (SP) of i-th feature from the baseline $\bar { x } _ { i }$ to the input $x _ { i }$ is defined as the multiplication of the CDF of the distribution $G _ { i }$ obtained by SBP and the difference between $\bar { x } _ { i }$ and $x _ { i }$

$$
\gamma _ { i } ( t ; \alpha _ { i } ) = \bar { x } _ { i } + F _ { G _ { i } } ( t ) ( x _ { i } - \bar { x } _ { i } )\tag{10}
$$

$$
G _ { i } \sim S B P ( U ( 0 , 1 ) , \alpha )\tag{11}
$$

With the sampling process of SP and hyperparameter $\alpha ,$ we finally propose a new attribution method, Stick-breaking Path Integration (SPI).

Definition 3.3 (Stick-breaking Path Integration). For a model f and a baseline ¯x, SPI is defined as the expectation of the path method over the path distribution,

$$
\begin{array} { l } { S P I _ { i } ( \mathbf { x } ; \boldsymbol { \alpha } ) = \mathbb { E } _ { \boldsymbol { \gamma } } [ \phi _ { i } ( \boldsymbol { \gamma } ) ] } \\ { \quad = \mathbb { E } _ { \mathbf { G } } [ ( x _ { i } - \bar { x } _ { i } ) \displaystyle \int _ { 0 } ^ { 1 } \frac { \partial f ( \bar { \mathbf { x } } + F _ { \mathbf { G } } ( t ) \odot ( \mathbf { x } - \bar { \mathbf { x } } ) ) } { \partial ( \bar { x } _ { i } + F _ { G _ { i } } ( t ) ( x _ { i } - \bar { x } _ { i } ) ) } \frac { d F _ { G _ { i } } ( t ) } { d t } d t } \end{array}\tag{12}
$$

where G is a vector with each index follows SBP, $G _ { i } \sim$ $S B P ( U ( 0 , 1 ) , \alpha )$ and $F _ { \mathbf { G } } : [ 0 , 1 ] \mapsto \mathbb { R } ^ { d }$ is a stack of SP that returns a vector in the input space according to the current step t.

## 3.4. Visualization with score based on statistics of attributions

As we sample SP from the random process SBP, the attribution computed using SP, $\phi _ { i } ( \gamma )$ , also can be regarded as random variables. Assuming that $\phi _ { i } ( \gamma )$ follows the Gaussian distribution, the estimated Gaussian distribution of $\phi _ { i } ( \gamma )$ is given as follow

$$
\phi _ { i } ( \pmb { \gamma } ) \sim N ( \mu , \sigma ) ,\tag{13}
$$

$$
\mu = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \phi _ { i } ( \boldsymbol { \gamma } ^ { ( k ) } ) ,\tag{14}
$$

$$
\sigma = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \phi _ { i } ( \pmb { \gamma } ^ { ( k ) } ) - \mathbb { E } _ { \pmb { \gamma } ^ { ( k ) } } [ \phi _ { i } ( \pmb { \gamma } ^ { ( k ) } ) ] \right) ^ { 2 } .\tag{15}
$$

We observe that the input attributions obtained by integrating gradient over each randomly sampled path deviates in their own distribution. We categorize such attributions to three; (1) the one with low variance and low mean, (2) the one with low variance with high mean and (3) the one with high variance. For the first and the second cases, we can identify that those features are consistently show low/high contribution with less dependent to the path. However, for the third case, we need to measure how probable that such feature would contribute much.

With this insights, we propose an additional method by assigning the probability of attribution at each feature is sufficiently positive along SBP. Assuming that the attributions for each feature are distributed normally, we propose SPI-P, which measures the probability of attribution to be larger than some threshold Θ. We use the top 5% quantile value of $S P I ( { \bf x } ; \alpha )$ as the threshold.

![](images/dc15637f23c735ca7713eed1cd43a9557b3ce12fe297b2248bdcaee0374cde0a.jpg)  
Figure 4: Input attribution obtained by proposed SPI method with varying the hyperparameter α. With increasing $\alpha ,$ , SPI-E becomes noisy and resembles the attribution of IG. With decreasing α, the noise is reduced on the objects, but the checkered pattern appear in the background of the image. SPI-P with low α also returns high probability for the background.

Definition 3.4 (SPI Probability (SPI-P)). Assume that the attribution at i-th feature follows the Gaussian distribution with estimated $\mu$ and σ in Equation 15. Then SPI-P is defined by the CDF of the Gaussian distribution,

$$
\begin{array} { l } { { \displaystyle { S P I - P _ { i } ( { \bf x } ; \alpha ) = P ( \Phi _ { i } > \Theta ) } } \ ~ } \\ { { \displaystyle ~ = 1 - \frac { 1 } { \sigma \sqrt { 2 \pi } } \int _ { - \infty } ^ { \Theta } \exp \left( - \frac { ( \theta - \mu ) ^ { 2 } } { \sigma ^ { 2 } } \right) } \ ~ } \end{array}\tag{dθ}
$$

(16)

## 3.5. Analysis on α

We note that α controls the variance of sampled paths in Section 3.3. In this section, we provide the analysis on how the attribution obtained by proposed method differs according to the value of α. Figure 4 shows the qualitative comparison of the proposed two methods, SPI-E and SPI-P. The first row in Figure 4 indicates that as α increases, the attribution becomes similar to the attribution of IG. As previously described, if α increase, the realized paths converges to the base path, which is the straight line of IG. The attribution with with high α shows noisy results that the positive and the negative values are alternatively placed nearby. Such noisy attribution has been raised as a problem in gradientbased attribution methods [9]. In contrast, the result with low α shows loss noisy attribution. Comparing the attribution from the high and the low α, With low α, the attribution shows less noisy and more consistent on the object.

## 4. Experiments and Results

In this section, we verify the effectiveness of SPI by the qualitative analysis and the quantitative comparison. We first provide the qualitative comparison among different attribution methods, by providing the attribution heatmaps on the randomly chosen examples. We note that the quantitative evaluation on the attribution methods is challenging because there is no ground truth of the attribution and the ground truth even changes if we train a new model. To alleviate the absence of the ground truths, we provide the quantitative comparison using two widely used metrics: (1) pixel insertion/deletion game [19, 16], and (2) RemOve-And-Retrain (ROAR) [7] to identify that the proposed method can assign attributions which are more relevant to model behavior. We select various gradient-based attribution methods for the comparison: Gradient\*Input [21], Guided Back-Propagation (GuidedBProp) [25], Integrated Gradients (IG) [28], FullGrad [26], and GuidedIG [9].

![](images/0fb4c6e8f2b51c1d09435bcc1e920aaf5d9791df9b51cc1eca56a9d929be822c.jpg)  
Figure 5: Qualitative comparison among various attribution methods for VGG-16 in the validation dataset of ImageNet. Each column describes the heatmaps obtained by each method. SPI generates more object-oriented and less noisy attribution heatmaps. With SPI-P, the heatmaps are more distinguishable between the important features and the irrelevant background.

<table><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>VGG-16  Inception-v3  RN-18</td></tr><tr><td rowspan=6 colspan=1>3Inserion</td><td rowspan=6 colspan=1>Grad*InputGuidedBPropIGFullGradGuidedIGSPI (Ours)</td><td rowspan=1 colspan=1>0.078        0.171       0.114</td></tr><tr><td rowspan=1 colspan=1>0.094        0.145       0.124</td></tr><tr><td rowspan=1 colspan=1>0.096        0.243       0.158</td></tr><tr><td rowspan=1 colspan=1>0.415        0.558       0.448</td></tr><tr><td rowspan=1 colspan=1>0.110       0.255       0.185</td></tr><tr><td rowspan=1 colspan=1>0.443       0.704      0.515</td></tr><tr><td rowspan=6 colspan=1>Dlelet (()</td><td rowspan=6 colspan=1>Grad*InputGuidedBPropIGFullGradGuidedIGSPI (Ours)</td><td rowspan=1 colspan=1>0.045        0.105       0.050</td></tr><tr><td rowspan=1 colspan=1>0.113        0.162       0.145</td></tr><tr><td rowspan=1 colspan=1>0.036        0.066       0.038</td></tr><tr><td rowspan=1 colspan=1>0.110       0.175       0.131</td></tr><tr><td rowspan=1 colspan=1>0.029        0.061       0.020</td></tr><tr><td rowspan=1 colspan=1>0.019        0.051       0.018</td></tr></table>

Table 1: Insertion/Deletion game of various attribution methods on three model architectures. Higher score is better for insertion and lower score is better for deletion.

## 4.1. Qualitative Comparison

For the visual inspection, which is the main application of the input attribution, we qualitatively compare the attribution heatmaps obtained by various methods. For the heatmap visualization, we take the color range to be bounded by top 5% absolute value of each attribution. Figure 5 provides the attribution heatmaps of randomly selected images from the validation set of ImageNet with the pre-trained VGG-16. We identify that the attributions obtained by SPI show more aligned to the target class object and less attributed to the unrelated background. For example, in the third row, SPI correctly focuses on the necklace, while other methods are distracted by the irrelevant pixels in the background. With SPI-P visualization, we also identify that the attribution better isolates the important features from the input image.

## 4.2. Pixel insertion and deletion game

Pixel flip is first proposed to benchmark if attribution methods correctly capture the relevance between the input features and the model output [19]. It is later introduced as evaluating the effect on the model prediction caused by the pixel perturbation [16]. To quantify the relevance between the input features and the model output, pixel flip modifies the pixel values in order of relevance obtained by attribution methods, from high to low. Then it measures the change of softmax output for the target class with the perturbation. Pixel flip consists of two evaluations, the insertion game and the deletion game. The insertion game inserts the original pixel value to the predefined baseline, and the deletion game deletes the original pixel value in the input. If the input attribution is correctly related to the model prediction, then the insertion game would give high score by steep increase of output with the insertion of highly attributed pixels. In the same manner, the deletion game would give low score with correctly related attribution. When modifying the pixel, several choices arise which value to replace. We follow the configuration of previous work [16], which use the blurred input for the insertion and the zero value for the deletion.

We use 50k images of the validation set provided by ImageNet [18]. We use three publicly available pre-trained models: VGG-16 [23], Inception-v3 [29], ResNet-18 (RN-18) [6]. Table 1 indicates the insertion and the deletion results for the various attribution methods and model architectures. We identify that SPI shows the best performance in both games on entire three architectures.

## 4.3. RemOve-And-Retrain (ROAR)

ROAR [7] is another metric to evaluate if the attribution method correctly indicates the features that are important in the perspective of the model training. ROAR is calculated by measuring the performance drop when the model is re-trained with modified training data. Each sample in the training dataset is modified by removing pixels with top k% attribution and replacing them with the average pixel value of the input. For the ROAR experiment, we use ResNet-18 architecture and train the model on 50k images of training set provided by CIFAR-10 dataset [11]. For the training, we use the Adam optimizer [10] with learning rate 3e-4 and 100 epochs. After training with each modified dataset, the performance of the trained model is measured with the standard test dataset with 10k images in CIFAR-10. We note that the attribution method captures more relevant features if the test accuracy is lower. Table 2 shows the test accuracy measure in the ROAR experiment for each attribution method. We repeatably conduct 10 trials of the experiments, where the parameters are random initialized at each trial and fixed between attribution methods. Table 2 shows the test accuracy measure in the ROAR experiment for each attribution method. The result indicates that the model trained on the modified dataset with SPI steeply decreases the test accuracy even with 10% removed. We conclude that SPI is effective in identifying the input features which are important to train the DNNs models.

## 4.4. Different Distribution for Path Sampling

We investigate the impact of using different distribution (hypergeometric distribution) on the attribution result to evaluate how effective the SBP is for path sampling. In Figure 6, we present two examples of paths that generated by the hypergeometric distribution with randomly sampled parameters. The first example in Figure 6 (a) is produced by using the uniform distribution for the parameter n. We demonstrate that SBP with $\alpha ~ = ~ 0 . 0 1$ can also approximately represent this distribution, as depicted in Figure 6 (b). The second example in Figure 6 (c) employs the Gaussian distribution instead of the uniform distribution, and we show that it can also be approximated by SBP, as illustrated in Figure 6 (d). In contrast, as depicted in Figure 6 (e), attempting to represent the distributed paths from SBP with $\alpha = 1 0$ using the hypergeometric distribution proves challenging, as controlling the parameters $( \mathrm { e } . \mathrm { g } . , n )$ becomes non-trivial. Consequently, we believe that SBP with the uniform distribution as the base distribution has greater representation power than the hypergeometric distribution. We also confirm that SBP with $\alpha = 1 0$ and the uniform distribution yields the best performance in the insertion/deletion game compared to other methods.

![](images/ca0ee0837ad149735b1408fe24827e869893b04dcef04f4ae073e8fbde6cb22e.jpg)

Figure 6: Comparison of paths generated by hypergeometric and Stick-Breaking Process (SBP) distributions with randomly sampled parameters. We identify that SBP has greater representation power to sample the multiple paths.
<table><tr><td>Removed %</td><td>10</td><td>30</td><td>50</td><td>70</td><td>90</td></tr><tr><td>Grad*Input</td><td> $5 2 . 7 6 { \pm } 0 . 9 5 \%$ </td><td> $3 9 . 7 4 { \scriptstyle \pm 1 . 0 1 \% }$ </td><td> $\overline { { 3 3 . 2 2 \pm 1 . 3 3 \% } }$ </td><td> $2 9 . 2 5 { \scriptstyle \pm 0 . 4 3 \% }$ </td><td> $\overline { { 2 4 . 9 2 { \pm } 1 . 3 9 \% } }$ </td></tr><tr><td>GuidedBProp</td><td> $6 7 . 5 1 { \pm } 0 . 4 0 \%$ </td><td> $6 3 . 7 0 { \scriptstyle \pm 0 . 8 0 \% }$ </td><td> $6 1 . 9 1 { \pm } 0 . 8 5 \%$ </td><td> $5 9 . 9 9 { \scriptstyle \pm 0 . 9 3 \% }$ </td><td> $5 2 . 7 3 { \scriptstyle \pm 0 . 5 2 \% }$ </td></tr><tr><td>IG</td><td> $3 9 . 7 0 { \scriptstyle \pm 0 . 7 9 \% }$ </td><td> $2 6 . 2 3 { \pm } 1 . 1 1 \%$ </td><td> $2 1 . 4 6 { \pm } 0 . 5 3 \%$ </td><td> $1 7 . 7 3 { \pm } 0 . 5 8 \%$ </td><td> $1 5 . 7 3 { \pm } 0 . 8 2 \%$ </td></tr><tr><td>FullGrad</td><td> $6 7 . 4 9 { \scriptstyle \pm 0 . 7 9 \% }$ </td><td> $5 6 . 8 4 { \scriptstyle \pm 0 . 9 7 \% }$ </td><td> $4 2 . 3 1 { \pm } 0 . 7 5 \%$ </td><td> $2 6 . 1 0 { \pm } 1 . 1 9 \%$ </td><td> $1 4 . 8 2 { \pm } 1 . 0 9 \%$ </td></tr><tr><td>GuidedIG</td><td> $3 8 . 9 0 { \pm } 1 . 6 3 \%$ </td><td> $2 4 . 2 3 { \pm } 1 . 0 1 \%$ </td><td> $2 0 . 6 4 { \pm } 1 . 2 2 \%$ </td><td> $1 8 . 1 1 { \pm } 0 . 7 5 \%$ </td><td> $1 5 . 6 6 { \pm } 0 . 8 1 \%$ </td></tr><tr><td>SPI (Ours)</td><td> $3 1 . 0 0 { \scriptstyle \pm 0 . 4 8 \% }$ </td><td> $\mathbf { 1 9 . 7 0 } \pm 0 . 4 0 \%$ </td><td> $1 6 . 0 5 { \scriptstyle \pm 0 . 9 1 \% }$ </td><td> ${ \bf 1 3 . 0 9 } \pm 0 . 7 2 \%$ </td><td> $1 0 . 8 8 { \scriptstyle \pm 1 . 0 6 \% }$ </td></tr></table>

Table 2: ROAR evaluation of various attribution methods on ResNet-18 trained with CIFAR-10.

## 5. Discussion

In this paper, we proposed the novel attribution method, Stick-breaking Path Integration (SPI), to provide more reliable explanation on the relationship between the input features and the model decision. Based on the path method, which is computed by integrating the gradients along the path in the input space, we first provide the affect of path selection to the computation of the input attribution. By raising the necessity of considering multiple paths for the reliable attribution, we propose to average the path-based attribution over the distribution of paths. We qualitatively show that our method provides attributions with more objectaligned and less noisy. We also provide the quantitative evaluations, pixel flip and ROAR, to identify that our method is well-aligned with the model behavior.

Our work also sheds light for several future works that would provide more reliable or meaningful attributions. The selection of the base distribution would be different to the uniform distribution. For example, one may use the Gaussian distribution to control the path to proceed at early steps or late steps by managing the mean of the distribution. Using different path sampling method would be another approach. Our method assumes the paths are distributed uniformly, but one may define a distribution with several modes. Randomizing the baseline would also diversify the sampled integration path. We expect our work would provide new aspect of attribution methods to inspect the black-box models.

## Acknowledgement

This work was conducted by Center for Applied Research in Artificial Intelligence (CARAI) grant funded by DAPA and ADD (UD230017TD) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2022-0-00984, Artificial Intelligence, Explainability, Personalization, Plug and Play, Universal Explanation Platform), (No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)), (No. 2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).

## References

[1] Robert J Aumann and Lloyd S Shapley. Values ofnon-atomic games. 2015.

[2] Sebastian Bach, Alexander Binder, Gregoire Montavon,´ Frederick Klauschen, Klaus-Robert Muller, and Wojciech¨ Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 2015.

[3] Kirill Bykov, Anna Hedstrom, Shinichi Nakajima, and Ma-¨ rina M-C Hohne. Noisegrad: enhancing explanations by¨ introducing stochasticity to model weights. arXiv preprint arXiv:2106.10185, 2021.

[4] Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In Proceedings ofthe 21th ACM SIGKDD international conference on knowledge discovery and data mining, pages 1721–1730, 2015.

[5] Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017.

[6] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[7] Sara Hooker, D. Erhan, Pieter-Jan Kindermans, and Been Kim. A benchmark for interpretability methods in deep neural networks. In NeurIPS, 2019.

[8] Andrei Kapishnikov, Tolga Bolukbasi, Fernanda Viegas, and´ Michael Terry. Xrai: Better attributions through regions. In ICCV, 2019.

[9] Andrei Kapishnikov, Subhashini Venugopalan, Besim Avci, Ben Wedin, Michael Terry, and Tolga Bolukbasi. Guided integrated gradients: An adaptive path method for removing noise. In CVPR, 2021.

[10] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[11] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[12] Sebastian Lapuschkin, Stephan Waldchen, Alexander¨ Binder, Gregoire Montavon, Wojciech Samek, and Klaus-´

Robert Muller. Unmasking clever hans predictors and as-¨ sessing what machines really learn. Nature Communications, 2019.

[13] Guido F Montufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear regions of deep neural networks. Advances in neural information processing systems, 27, 2014.

[14] Woo-Jeoung Nam, Shir Gur, Jaesik Choi, Lior Wolf, and Seong-Whan Lee. Relative attributing propagation: Interpreting the comparative contributions of individual units in deep neural networks. In AAAI, 2020.

[15] Deng Pan, Xin Li, and Dongxiao Zhu. Explaining deep neural network models with adversarial gradient integration. In IJCAI, 2021.

[16] Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421, 2018.

[17] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. Why should i trust you?: Explaining the predictions of any classifier. In ACM SIGKDD. ACM, 2016.

[18] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 2015.

[19] Wojciech Samek, Alexander Binder, Gregoire Montavon,´ Sebastian Lapuschkin, and Klaus-Robert Muller. Evaluating¨ the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems, 2017.

[20] Jay Sethuraman. A constructive definition of dirichlet priors. 1991.

[21] Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, and Anshul Kundaje. Not just a black box: Learning important features through propagating activation differences. arXiv preprint arXiv:1605.01713, 2016.

[22] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

[23] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. 2015.

[24] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas,´ and Martin Wattenberg. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.

[25] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

[26] Suraj Srinivas and Franc¸ois Fleuret. Full-gradient representation for neural network visualization. In NeurIPS, 2019.

[27] Erik Strumbelj and Igor Kononenko. Explaining prediction<sup>ˇ</sup> models and individual predictions with feature contributions. Knowledge and information systems, 2014.

[28] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In ICML. PMLR, 2017.

[29] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[30] Huan Xiong, Lei Huang, Mengyang Yu, Li Liu, Fan Zhu, and Ling Shao. On the number of linear regions of convolutional neural networks. In International Conference on Machine Learning, pages 10514–10523. PMLR, 2020.

[31] Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and Kazuya Takeda. A survey of autonomous driving: Common practices and emerging technologies. IEEE access, 8:58443– 58469, 2020.