# SVDiff: Compact Parameter Space for Diffusion Fine-Tuning

Ligong Han<sup>1,2\*</sup> Yinxiao Li<sup>2</sup> Han Zhang<sup>2</sup> Peyman Milanfar<sup>2</sup> Dimitris Metaxas<sup>1</sup> Feng Yang<sup>2</sup> <sup>1</sup>Rutgers University <sup>2</sup>Google Research

## Abstract

Diffusion models have achieved remarkable success in text-to-image generation, enabling the creation of highquality imagesfrom text prompts or other modalities. However, existing methodsfor customizing these models are limited by handling multiple personalized subjects and the risk of overfitting. Moreover, their large number ofparameters is inefficient for model storage. In this paper, we propose a novel approach to address these limitations in existing textto-image diffusion modelsfor personalization. Our method involves fine-tuning the singular values of the weight matrices, leading to a compact and efficient parameter space that reduces the risk of overfitting and language-drifting. We also propose a Cut-Mix-Unmix data-augmentation technique to enhance the quality of multi-subject image generation and a simple text-based image editingframework. Our proposed SVDiff method has a significantly smaller model size compared to existing methods (≈2,200 timesfewer parameters compared with vanilla DreamBooth), making it more practicalfor real-world applications.

## 1. Introduction

Recent years have witnessed the rapid advancement of diffusion-based text-to-image generative models [19, 53, 20, 47, 51], which have enabled the generation of highquality images through simple text prompts. These models are capable of generating a wide range of objects, styles, and scenes with remarkable realism and diversity. These models, with their exceptional results, have inspired researchers to investigate various ways to harness their power for image editing [31, 37, 79].

In the pursuit of model personalization and customization, some recent works such as Textual-Inversion [16], DreamBooth [52], and Custom Diffusion [32] have further unleashed the potential of large-scale text-to-image diffusion models. By fine-tuning the parameters of the pretrained models, these methods allow the diffusion models to be adapted to specific tasks or individual user preferences.

![](images/b8979007dc413de975b2a5f380bd1a23012c6ecf4e58e7021287b07845a097ac.jpg)  
Figure 1. Applications of SVDiff. Style-Mixing: mix styles from personalized objects and create novel renderings; Multi-Subject: generate multiple subjects in the same scene; Single-Image Editing: text-based editing from a single image.

Despite their promising results, there are still some limitations associated with fine-tuning large-scale text-to-image diffusion models. One limitation is the large parameter space, which can lead to overfitting or drifting from the original generalization ability [52]. Another challenge is the difficulty in learning multiple personalized concepts especially when they are of similar categories [32].

To alleviate overfitting, we draw inspiration from the efficient parameter space in the GAN literature [50] and propose a compact yet efficient parameter space, spectral shift, for diffusion model by only fine-tuning the singular values of the weight matrices of the model. This approach is inspired by prior work in GAN adaptation showing that constraining the space of trainable parameters can lead to improved performance on target domain [48, 36, 41, 64]. Comparing with another popular low-rank constraint [22], the spectral shifts utilize the full representation power of the weight matrix while being more compact (e.g. 1.7MB for

StableDiffusion [51, 15, 69], full weight checkpoint consumes 3.66GB of storage). The compact parameter space allows us to combat overfitting and language-drifting issues, especially when prior-preservation loss [52] is not applicable. We demonstrate this use case by presenting a simple DreamBooth-based single-image editing framework.

To further enhance the ability of the model to learn multiple personalized concepts, we propose a simple Cut-Mix-Unmix data-augmentation technique. This technique, together with our proposed spectral shift parameter space, enables us to learn multiple personalized concepts even for semantically similar categories $( e . g .$ . a “cat” and a “dog”).

In summary, our main contributions are:

• We present a compact (≈2,200× fewer parameters compared with vanilla DreamBooth [52], measured on StableDiffusion [51]) yet efficient parameter space for diffusion model fine-tuning based on singular-value decomposition of weight kernels.

• We present a text-based single-image editing framework and demonstrate its use case with our proposed spectral shift parameter space.

• We present a generic Cut-Mix-Unmix method for dataaugmentation to enhance the ability of the model to learn multiple personalized concepts.

This work opens up new avenues for the efficient and effective fine-tuning large-scale text-to-image diffusion models for personalization and customization. Our proposed method provides a promising starting point for further research in this direction.

## 2. Related Work

Text-to-image diffusion models Diffusion models [58, 61, 19, 62, 40, 59, 18, 60, 10] have proven to be highly effective in learning data distributions and have shown impressive results in image synthesis, leading to various applications [74, 45, 9, 49, 26, 23, 34, 56, 28, 24, 29]. Recent advancements have also explored transformer-based architectures [67, 44, 6, 7]. In particular, the field of text-guided image synthesis has seen significant growth with the introduction of diffusion models, achieving state-of-the-art results in large-scale text-to-image synthesis tasks [39, 47, 53, 51, 4]. Our main experiments were conducted using StableDiffusion [51], which is a popular variant of latent diffusion models (LDMs) [51] that operates on a latent space of a pretrained autoencoder to reduce the dimensionality of the data samples, allowing the diffusion model to utilize the wellcompressed semantic features and visual patterns learned by the encoder.

Fine-tuning generative models for personalization Recent works have focused on customizing and personalizing text-to-image diffusion models by fine-tuning the text embedding [16], full weights [52], cross-attention layers [32], or adapters [77, 38] using a few personalized images. Other works have also investigated training-free approaches for fast adaptation [17, 73, 11, 27, 55]. The idea of finetuning only the singular values of weight matrices was introduced by FSGAN [50] in the GAN literature and further advanced by NaviGAN [13] with an unsupervised method for discovering semantic directions in this compact parameter space. Our method, SVDiff, introduces this concept to the fine-tuning of diffusion models and is designed for fewshot adaptation. A similar approach, LoRA [14], explores low-rank adaptation for text-to-image diffusion fine-tuning, while our proposed SVDiff optimizes all singular values of the weight matrix, leading to an even smaller model checkpoint. Similar idea has also been explored in few-shot segmentation [64].

Diffusion-based image editing Diffusion models have also shown great potential for semantic editing [33, 3, 2, 31, 79, 37, 72, 68, 63, 42, 43, 5, 71]. These methods typically focus on inversion [59] and reconstruction by optimizing the nulltext embedding or overfitting to the given image [79]. Our proposed method, SVDiff , presents a simple DreamBoothbased [52] single-image editing framework that demonstrates the potential of SVDiff in single image editing and mitigating overfitting.

## 3. Method

## 3.1. Preliminary

Diffusion models StableDiffusion [51], the model we experiment with, is a variant of latent diffusion models (LDMs) [51]. LDMs transform the input images x into a latent code z through an encoder $\mathcal { E } ,$ where ${ \bf z } = \mathcal { E } ( { \bf x } )$ , and perform the denoising process in the latent space Z. Briefly, a LDM $\scriptstyle { \hat { \epsilon } } _ { \theta }$ is trained with a denoising objective:

$$
\mathbb { E } _ { \mathbf { z } , \mathbf { c } , \epsilon , t } \Big [ \big \| \hat { \mathbf { \epsilon } } _ { \boldsymbol { \theta } } \big ( \mathbf { z } _ { t } | \mathbf { c } , t \big ) - \boldsymbol { \epsilon } \big \| _ { 2 } ^ { 2 } \Big ] ,\tag{1}
$$

where $( \mathbf { z } , \mathbf { c } )$ are data-conditioning pairs (image latents and text embeddings), $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , t ∼ Uniform(1, T), and θ represents the model parameters. We omit t in the following for brevity.

Few-shot adaptation in compact parameter space of GANs The method of FSGAN [50] is based on the Singular Value Decomposition (SVD) technique and proposes an effective way to adapt GANs in few-shot settings. It takes advantage of the SVD to learn a compact update for domain adaptation in the parameter space of a GAN. Specifically, FSGAN reshapes the convolution kernels of a GAN, which are in the form of $W _ { t e n s o r } \in \mathbb { R } ^ { c _ { o u t } \times c _ { i n } \times }$ h×w into 2-D matrices W, which are in the form of $\begin{array} { r l } { W } & { { } = } \end{array}$ reshape $( W _ { t e n s o r } ) \ \in \ \mathbb { R } ^ { c _ { o u t } \times ( c _ { i n } \times h \times w ) }$ FSGAN then performs SVD on these reshaped weight matrices of both the generator and discriminator of a pretrained GAN and adapts their singular values to a new domain using a standard GAN training objective.

![](images/ba4f7d209958d6f1ffbd18d4463280a05769001d352b70b953204f1da86b8364.jpg)  
Figure 2. Performing singular value decomposition (SVD) on weight matrices. In an intermediate layer of the model, (a) the convolutional weights $W _ { t e n s o r }$ (b) serve as an associative memory [8]. (c) SVD is performed on the reshaped 2-D matrix W.

## 3.2. Compact Parameter Space for Diffusion Finetuning

Spectral shifts The core idea of our method is to introduce the concept of spectral shifts from FSGAN [50] to the parameter space of diffusion models. To do so, we first perform Singular Value Decomposition (SVD) on the weight matrices of the pre-trained diffusion model. The weight matrix (obtained from the same reshaping as FS-GAN [50] mentioned above) is denoted as W and its SVD is $W = U \Sigma V ^ { \top }$ , where $\Sigma = \mathrm { d i a g } ( \pmb { \sigma } )$ and $\pmb { \sigma } = [ \sigma _ { 1 } , \sigma _ { 2 } , . . . ]$ are the singular values in descending order. Note that the SVD is a one-time computation and can be cached. This procedure is illustrated in Fig. 2. Such reshaping of the convolution kernels is inspired by viewing them as linear associative memories [8]. The patch-level convolution can be expressed as a matrix multiplication, $\mathbf { f } _ { o u t } \ = \ W \mathbf { f } _ { i n } ,$ where $\mathbf { f } _ { i n } \in \mathbb { R } ^ { ( c _ { i n } \times h \times w ) \times 1 }$ is flattened patch feature and $\mathbf { f } _ { o u t } \in \mathbb { R } ^ { c _ { o u t } }$ is the output pre-activation feature corresponding to the given patch. Intuitively, the optimization of spectral shifts leverages the fact that the singular vectors correspond to the close-form solutions of the eigenvalue problem [54]: $\mathrm { m a x } _ { \mathbf { n } } \| W \mathbf { n } \| _ { 2 } ^ { 2 } \mathrm { s . t . } \| \mathbf { n } \| = 1$

Instead of fine-tuning the full weight matrix, we only update the weight matrix by optimizing the spectral shift [13], $\delta ,$ which is defined as the difference between the singular values of the updated weight matrix and the original weight matrix. The updated weight matrix can be re-assembled by

$$
W _ { \delta } = U \Sigma _ { \delta } V ^ { \top } \mathrm {  ~ \ w i t h ~ } \Sigma _ { \delta } = \mathrm { d i a g } ( \mathrm { R e L U } ( \sigma + \delta ) ) .\tag{2}
$$

Training loss The fine-tuning is performed using the same loss function that was used for training the diffusion model, with a weighted prior-preservation loss [52, 8]:

$$
\begin{array} { r } { \mathcal { L } ( \pmb { \delta } ) = \mathbb { E } _ { \mathbf { z } ^ { * } , \mathbf { c } ^ { * } , \mathbf { \epsilon } \epsilon , t } \left[ \| \hat { \mathbf { \epsilon } } _ { \theta _ { \delta } } ( \mathbf { z } _ { t } ^ { * } | \mathbf { c } ^ { * } ) - \mathbf { \epsilon } \| _ { 2 } ^ { 2 } \right] + \lambda \mathcal { L } _ { p r } ( \pmb { \delta } ) } \end{array}\tag{with}
$$

$$
\mathcal { L } _ { p r } ( \pmb { \delta } ) = \mathbb { E } _ { \mathbf { z } ^ { p r } , \mathbf { c } ^ { p r } , \epsilon , t } \Big [ \| \hat { \epsilon } _ { \theta _ { \delta } } ( \mathbf { z } _ { t } ^ { p r } | \mathbf { c } ^ { p r } ) - \epsilon \| _ { 2 } ^ { 2 } \Big ]\tag{3}
$$

![](images/a3e1fb57bd46851ce0f8e75e59d045dae99074f40f913ffef8d60952eac28bce.jpg)  
Figure 3. Cut-Mix-Unmix data-augmentation for multi-subject generation. The figure shows the process of Cut-Mix-Unmix data augmentation for training a model to handle multiple concepts. The method involves manually constructing image-prompt pairs where the image is created using a CutMix-like data augmentation [76] and the corresponding prompt is written as, for example, “photo of a [V<sub>2</sub>] sculpture and a [V<sub>1</sub>] dog”. The prior preservation image-prompt pairs are created in a similar manner. The objective is to train the model to separate different concepts by presenting it with explicit mixed samples. During inference, a different prompt, such as “photo of a [V ] dog sitting besides a [V ] sculpture”.

where $( \mathbf { z } ^ { \ast } , \mathbf { c } ^ { \ast } )$ represents the target data-conditioning pairs that the model is being adapted to, and $( \mathbf { z } ^ { p r } , \mathbf { c } ^ { p r } )$ represents the prior data-conditioning pairs generated by the pretrained model. This loss function extends the one proposed by Model Rewriting [8] for GANs to the context of diffusion models, with the prior-preservation loss serving as the smoothing term. We set $\lambda = 0$ for single image editing.

Combining spectral shifts Moreover, the individually trained spectral shifts can be combined into a new model to create novel renderings. This can enable applications including interpolation, style mixing, or multi-subject generation (Fig. 8). Here we consider two common strategies, addition and interpolation. To add $\delta _ { 1 }$ and $\delta _ { 2 }$ into $\delta ^ { \prime }$

$$
\Sigma _ { \delta ^ { \prime } } = \mathrm { d i a g } ( \mathrm { R e L U } ( \pmb { \sigma } + \pmb { \delta } _ { 1 } + \pmb { \delta } _ { 2 } ) ) .\tag{4}
$$

For interpolation between two models with $0 \leq \alpha \leq 1$

$$
\Sigma _ { \pmb { \delta } ^ { \prime } } = \mathrm { d i a g } ( \mathrm { R e L U } ( \pmb { \sigma } + \alpha \pmb { \delta } _ { 1 } + ( 1 - \alpha ) \pmb { \delta } _ { 2 } ) ) .\tag{5}
$$

This allows for smooth transitions between models and the ability to interpolate between different image styles.

## 3.3. Cut-Mix-Unmix for Multi-Subject Generation

We discovered that when training the StableDiffusion [51] model with multiple concepts simultaneously (randomly choosing one concept at each data sampling iteration), the model tends to mix their styles when rendering them in one image for difficult compositions or subjects of similar categories [32] (as shown in Fig. 6). To explicitly guide the model not to mix personalized styles, we propose a simple technique called Cut-Mix-Unmix. By constructing and presenting the model with “correctly” cut-and-mixed image samples (as shown in Fig. 3), we instruct the model to unmix styles. In this method, we manually create CutMixlike [76] image samples and corresponding prompts (e.g. “photo of a [V<sub>1</sub>] dog on the left and a [V<sub>2</sub>] sculpture on the right” or “photo of a [V<sub>2</sub>] sculpture and a [V<sub>1</sub>] dog” as illustrated in Fig. 3). The prior loss samples are generated in a similar manner. During training, Cut-Mix-Unmix data augmentation is applied with a pre-defined probability (usually set to 0.6). This probability is not set to 1, as doing so would make it challenging for the model to differentiate between subjects. During inference, we use a different prompt from the one used during training, such as “a [V<sub>1</sub>] dog sitting beside a [V ] sculpture”. However, if the model overfits to the Cut-Mix-Unmix samples, it may generate samples with stitching artifacts even with a different prompt. We found that using negative prompts can sometimes alleviate these artifacts, as detailed in appendix.

We further present an extension to our fine-tuning approach by incorporating an “unmix” regularization on the cross-attention maps. This is motivated by our observation that in fine-tuned models, the dog’s special token attends largely to the panda’s region. To enforce separation between the two subjects, we use MSE on the non-corresponding regions of the cross-attention maps. This loss encourages the dog’s special token to focus solely on the dog and vice versa for the panda. The results of this extension show a significant reduction in stitching artifact. Details of the crossattention regularization are presented in the appendix.

## 3.4. Single-Image Editing

In this section, we present a framework for single image editing, by fine-tuning a diffusion model with an imageprompt pair. The procedure is outlined in Fig. 4. The desired edits can be obtained at inference time by modifying the prompt. For example, we fine-tune the model with the input image and text description “photo of a crown with a blue diamond and a golden eagle on $i t ^ { \prime \prime }$ , and at inference time if we want to remove the eagle, we simply sample from the fine-tuned model with text “photo ofa crown with a blue diamond on $i t ^ { \prime \prime }$ . To mitigate overfitting during finetuning, we use the spectral shift parameter space instead of full weights, reducing the risk of overfitting and language drifting. The trade-off between faithful reconstruction and editability, as discussed in [35], is acknowledged, and the purpose of employing SVDiff here is to allow more flexible edits rather than exact reconstructions.

For edits that do not require large structural changes (like repose, “standing” → “lying down” or “zoom in”), results can be improved with DDIM inversion [59]. Before sampling, we run DDIM inversion with classifier-free guidance [21] scale 1 conditioned on the target text prompt c and encode the input image $\mathbf { z } ^ { \ast }$ to a latent noise map,

![](images/9b845c272dfa579b1ebb30882e14227cbe3ea243e10d622ab35dbb5586c57fbf.jpg)  
Figure 4. Pipeline for single image editing with a text-to-image diffusion model. (a) The model is fine-tuned with a single imageprompt pair, where the prompt describes the input image without a special token. (b) During inference, desired edits are made by modifying the prompt. For edits with no significant structural changes, the use of DDIM inversion [59] has been shown to improve the editing quality.

$$
\mathbf z _ { T } = \mathrm { D D I M I n v e r t } ( \mathbf z ^ { * } , \mathbf c ; \theta ^ { \prime } ) ,\tag{6}
$$

(θ<sup>′</sup> denotes the fine-tuned model parameters) from which the inference pipeline starts. As expected, large structural changes may still require more noise being injected in the denoising process. Here we consider two types of noise injection: i) setting $\eta > 0$ (as defined in DDIM [59], and ii) perturbing $\mathbf { z } _ { T }$ . For the latter, we interpolate between $\mathbf { z } _ { T }$ and a random noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ with spherical linear interpolation [57, 59],

$$
\widetilde { \mathbf { z } } _ { T } = \mathrm { s l e r p } ( \alpha , \mathbf { z } _ { T } , \epsilon ) = \frac { \sin ( ( 1 - \alpha ) \phi ) } { \sin ( \phi ) } \mathbf { z } _ { T } + \frac { \sin ( \alpha \phi ) } { \sin ( \phi ) } \epsilon ,\tag{7}
$$

with $\phi = \operatorname { a r c c o s } \left( \cos ( \mathbf { z } _ { T } , \epsilon ) \right)$ . For more results and analysis, please see the experimental section.

Other approaches, such as Imagic [31], have been proposed to address overfitting and language drifting in finetuning-based single-image editing. Imagic fine-tunes the diffusion model on the input image and target text description, and then interpolates between the optimized and target text embedding to avoid overfitting. However, Imagic requires fine-tuning on each target text prompt at test time.

## 4. Experiment

The experiments evaluate SVDiff on various tasks such as single-/multi-subject generation, single image editing,

![](images/f891dc5f74a37fb9c6ab6e84c7b628f0e763d25689b0011924e0718ac1c8cc8f.jpg)  
Figure 5. Results for single subject generation. DreamBooth [52] and Custom Diffusion [32] are implemented in StableDiffusion with Diffusers library [69]. Each subfigure consists 3 samples: a large one on the left and 2 small one on the right. The text prompt under input images are used for training and the text prompt under sample images are used for inference. We observe that SVDiff performs similarly as DreamBooth, and preserves subject identities better than Custom Diffusion for row 2, 3, 5.

and ablations. The DDIM [59] sampler with η = 0 is used for all generated samples, unless specified otherwise.

## 4.1. Single-Subject Generation

In this section, we present the results of our proposed SVDiff for customized single-subject generation proposed in DreamBooth [52], which involves fine-tuning the pretrained text-to-image diffusion model on a single object or concept (using 3-5 images). The original DreamBooth was implemented on Imagen [53] and we conduct our experiments based on its StableDiffusion [51] implementation [75, 69]. We provide visual comparisons of 5 examples in Fig. 5. All baselines were trained for 500 or 1000 steps with batch size 1 (except for Custom Diffusion [32], which used a default batch size of 2), and the best model was selected for fair comparison. As Fig. 5 shows, SVDiff produces similar results to DreamBooth (which fine-tunes the full model weights) despite having a much smaller parameter space. Custom Diffusion, on the other hand, tends to underfit the training images as seen in rows 2, 3, and 5 of Fig. 5. We assess the text and image alignment in Fig. 9. The results show that the performance of SVDiff is similar to that of DreamBooth, while Custom Diffusion tends to underfit as seen from its position in the upper left corner.

## 4.2. Multi-Subject Generation

In this section, we present the multi-subject generation results to illustrate the advantage of our proposed “Cut-Mix-Unmix” data augmentation technique. When enabled, we perform Cut-Mix-Unmix data-augmentation with probability of 0.6 in each data sampling iteration and two subjects are randomly selected without replacement. A comparison between using “Cut-Mix-Unmix” (marked as “w/ Cut-Mix-Unmix”) and not using it (marked as “w/o Cut-Mix-Unmix”, performing augmentation with probability 0) are shown in Fig. 6. Each row of images are generated using the same text prompt displayed below the images. Note that the Cut-Mix-Unmix data augmentation technique is generic and can be applied to fine-tuning full weights as well.

To assess the visual quality of images generated using the “Cut-Mix-Unmix” method with either SVD or full weights, we conducted a user study using Amazon MTurk [1] with 400 generated image pairs . The participants were presented with an image pair generated using the same random seed, and were asked to identify the better image by answering the question, “Which image contains both objects from the two input images with a consistent background?” Each image pair was evaluated by 10 different raters, and the aggregated results showed that SVD was favored over full weights 60.9% of the time, with a standard deviation of 6.9%. More details and analysis will be provided in the appendix.

Additionally, we also conducted experiments that involve training on three concepts simultaneously. During training, we still construct Cut-Mix samples with probability 0.6 by randomly sample two subjects. Interestingly, we observe that for concepts that are already semantically wellseparated, e.g. “dog/building” or “sculpture/building”, the model can successfully generate desired results even without using Cut-Mix-Unmix. However, it fails to disentangle semantically more similar concepts, e.g. “dog/panda” as shown in Fig. 6-g.

## 4.3. Single Image Editing

In this section, we present results for the single image editing application. As depicted in Fig. 7, each row presents three edits with fine-tuning of both spectral shifts (marked as “Ours”) and full weights (marked as “Full”). The text prompts for the corresponding edited images are given below the images. The aim of this experiment is to demonstrate that regularizing the parameter space with spectral shifts effectively mitigates the language drift issue, as defined in [52] (the model overfits to a single image and loses its ability to generalize and perform desired edits).

As previously discussed, when DDIM inversion is not employed, fine-tuning with spectral shifts can lead to sometimes over-creative results. We show examples and comparisons of editing results with and without DDIM inver-

![](images/74beb0218134c2fd4b01b36b8d4de24b243cbe61bbf41e9c1b9facf71e96bfce.jpg)

![](images/7264232e7ef07cc7b7be7c084e3f3be628a8e79ea679074b44cce038dc1274e6.jpg)

![](images/5b951e175ca179ccb5faa9d3f8c405fa733bf583fe4deb7fd0982eadd81dac8b.jpg)  
(e)

![](images/6fade8ef61426f98941d577f120d0f74d69a301fef95c084a5241ddac6eb0ff0.jpg)  
(d)  
sculpture  
�<sub>"</sub> plushy  
�<sub>"</sub> sculpture  
Input Images  
�<sub>!</sub> plushy

![](images/f7d53872b93aca89748e3aa9ad5513a6a2345a97c99845c13d34c19b376f8630.jpg)

![](images/a5e42dd10a5d7612d3dd6f450727953e4f5cde61ef7bab76a06cb30e57b2a3bd.jpg)

![](images/fb3d2d7d719c4dfa40f5b8b90777b786531de0fd186138004a998d46d4d94aa2.jpg)

![](images/87c3e7912f0625ad6a04c05581a4fb383500d96ac148eea6cb56421d4cce1a1d.jpg)

![](images/8258438e2b79274f5a06fddbc3ca52ff5023ed109d00b4b5e87b5c3ec92c3150.jpg)  
(a)

SVD

![](images/7569c4fb24b7d61287b75440d209269ccbb33479beb8e73c735681a4fa1e7742.jpg)

![](images/bcf0a725b63403076a89cd34edd151fe65e38378d28c25b80f87ef0ddfd5a58a.jpg)

![](images/5c8465aceb76e2b524ac08b9ca1ed2a7489436a953074820fba0b4fef0ad7846.jpg)

![](images/062098622918e01336f409fb9f5322fc5889b25b89a4fa485ce22c9c5d9f9197.jpg)  
“photo of a �<sub>!</sub> plushy sitting besides a �<sub>"</sub> plushy”

![](images/664fac8058a2e0406e7de084c340ce4b3778fe31b2bcb16348700a73b6970080.jpg)

![](images/b956c641c39a6583b2f7338d5bd9c092dadf12dfa147f1ea740856e214f2f80e.jpg)

![](images/13c0caf279c87cc6c9b7b297c2c2563c57f249632a001e7a3d31ec0196f4d00f.jpg)

![](images/43c5a2a3edb5f84103f7e97f472bc4e0f3435776136c51bbc736e87b578da092.jpg)  
�<sub>!</sub> dog  
�<sub>"</sub> cat

![](images/0429ca674342edaf90bf9e3502b39ebd530f094ba629e1aa448f2eddfe7fce0f.jpg)

![](images/f355841c7eabb87c7a3e1b7d1687ea965cfbc44330f4f6f3b758b12d03afcd3d.jpg)

(b)

![](images/805e22ce1bafc1f9b2b8415fdcf337efde9fdeddcd47edf5f811851bfaa43f57.jpg)

![](images/f1da7a91c8f9cc4b9cc5ee31372a893298f3602e826939537f0ba95a55b31662.jpg)

![](images/e9f2d12922bc8532590b2fb947ee83c15d683b49de2eba6b5155b4f014c28262.jpg)

![](images/cb26353758317bed82799e6356b1e95fc503d1d9017e4f7f801d419654892b48.jpg)  
“photo of a �<sub>!</sub> dog sitting besides a �<sub>"</sub> cat”

![](images/a98c192153083e0d0e04d508b63ba1dfe4b0d855f06d759b22f4d60264dc167d.jpg)

![](images/811850e8f19536837e07544759abdb8dff0ad2ed811b9f93230f9393136a8f3b.jpg)

![](images/fae1e594e49873befa1b188952d71ac38804118aa753aa98909d061dbfd9d7c7.jpg)

![](images/1a5e6534d70579dae35bbafe389276991ac8299afa8e465dac58d198bcac0900.jpg)  
“photo of a �<sub>!</sub> dog sitting besides a �<sub>"</sub> sculpture”

![](images/786f6e86f76a54f0e8d64b02df83e35d38263a3b3b05e44dcaf5b130fd9e54e4.jpg)

![](images/a9b51eecce7fd1d811ab7e907d418315e07b61ef458ad8a63f389cbeedc4113d.jpg)

![](images/b56eb93bf854b983761a92c219bfca164ed7faf2930c6e9dbcc2b7af750c79a2.jpg)

![](images/2f8fcb15aa6f9369605946d6cd9d4716948fc1c4bac8eb9ee69334bc2b8e2a78.jpg)

![](images/7abb56021a525a378c6f3b52c5ca4ba5d9daf9a033bfef0040c6906081a3c6f4.jpg)

![](images/036a8c643788066a171b0fa44b33201afa93587b86541d53323e698fb15b2aa6.jpg)

![](images/c97098634d630304d20437927311018c91f092b33e76b897ac907935deed6a33.jpg)  
“photo of a �<sub>!</sub> dog sitting besides a �<sub>"</sub> sculpture”

![](images/0fb6b0c3455b33c5e8e8228732e979807c9057e25b6ed59bc16e659b7b4801da.jpg)

![](images/1c1f99ec01321b3d19912959720d9b272e8c1dfad2513da08f7e2b0087f77420.jpg)

![](images/bb0a1de2f70f75d29ce191ebef6b865d0c0e09b18caba2227bd4d2c4f9b63461.jpg)  
�<sub>!</sub> dog

![](images/681aa1688a6b44270db0200e9cc20351f23017f622d8e89e7ec7b4b90748cb62.jpg)

![](images/7492ea53716af45dcf5bba923ee5a35054542e1f314214b62b0ea549420e58df.jpg)

![](images/ec2706391e9bfb94740ed9922d51d87512ed7fa0eff07feaf9e47ea18d17553f.jpg)

![](images/16a2787efc40c938ef4f37e1b415ead44417825e319c98a20b5f49c1bb6e776e.jpg)

![](images/8f170db2a8d945133c67d1baad7d087b8503182b1762000b2f2f629480645a5b.jpg)

![](images/e1a71937bbeeb3f3c8714450177836da3c9a505d308434b315364e956eb588e8.jpg)

![](images/4b3a25480c34cde930a9f321e0ce0d8faec718bc97604260abbada21f332fbb9.jpg)  
“a �<sub>!</sub> dog in front of a �<sub>#</sub> building”

![](images/8ec07930550277cea9aeddfe06ac8eecbb3d447f61a9d8033028f0b8d2767893.jpg)  
�<sub>"</sub> sculpture

![](images/2198d311d22257dacf4e4502d7d8d03c6afbd0bce61ed00a05caedb6d47ffd10.jpg)

![](images/90d49a8d62605270c1327dfca1748f8b242b163fa18c0cc9508293373301c6ad.jpg)

![](images/b3d661a4b212eb251c0a4adee460281ac0f6782eb52bf764c65990e717bc5b59.jpg)

![](images/42110254f57a998959a80e5afaeb29dcc5b5f8650300501ebe1ef2f28fa12053.jpg)

![](images/21330b0169be7c1aedb794440e49fbccc87c300bfbe5b954cb50ce3323b0ec9e.jpg)

![](images/f819b864083f7a2649515b618de28fc343985f6e41364f5be9dd0bbe9491df90.jpg)

![](images/e39b93f7c3961d361a2e47c7e51fc5d600a860c65f21671fd900d4cb2592ff80.jpg)  
“a �<sub>"</sub> sculpture in front of a �<sub>#</sub> building”

![](images/849b219f7463b87720a122fae62b47adb9d6ca6d8480eec00586294f0aea2d5f.jpg)  
�<sub>#</sub> building

![](images/2de947ef15646e8a1caa17834a302b5d0295edb237f4f0ab75ca49c3a7035e32.jpg)  
(g)

![](images/297a9c310546950f7f0aaa8b60380efc26ae47786b8d61435bdd5bcd857defdf.jpg)

![](images/0eb7084cdf05b5050946914db63b58b87e377451da62f5a68d0dc151a9d36f56.jpg)

![](images/89391c9f17c932e082d52894ffc794d0c3c0ebf396d8d7685f1fbfece55848a8.jpg)  
“a �<sub>!</sub> dog sitting besides a �<sub>"</sub> sculpture in front of a �<sub>#</sub> building”

![](images/33555c563d39b8d08e48eb3f39e6c7875be68f1049941629bf3d5648e5b2cd78.jpg)

![](images/ca3a8fc7ba901c5f364db5b96f7c6b8102097a526935e84bef18dde060ea1ed8.jpg)

Figure 6. Results for multi-subject generation. (a-d) show the results of fine-tuning on two subjects and (e-g) show the results of fine-tuning on three subjects. Both full weight (“Full”) fine-tuning and SVDiff (“SVD”) can benefit from the Cut-Mix-Unmix dataaugmentation. Without Cut-Mix-Unmix, the model struggles to disentangle subjects of similar categories, as demonstrated in the last two columns of (a,b,c,d,g).

Ours  
Input Images  
![](images/cc0fe3a23dd652a6fdd0c51d965105c3cdd41f8b028997702dc980f85e8a798f.jpg)  
(a)

![](images/3ffc177d12ee12d4166700d84cbc49f1ad45de0fa9a17c5f98e244c94815c17f.jpg)

Full  
![](images/c6775709bdbeafa459355f71406363137d1f9ed4e1736a310f750227c2d7f70f.jpg)  
“photo of a white green painted room with a bed, a lamp, and a picture on the wall”

![](images/d72d91e91455f2ce54f44635f13449e50c02924b96979dfd5f303d01e50cd58e.jpg)  
(b)

![](images/5c4edb953588719ad8cf5229678541e19ee67e4a5ecd468c6c2c1e0479aaeaf4.jpg)

![](images/72eda6f0b43741fa4104732dbfb3f020b5da3cd4421b9b515a94485c40ce4138.jpg)

![](images/b3e468bdc90cd8e5b907e4b373cf611820e871047b965efecb7ed992173b64c1.jpg)  
“photo of a pink red chair with black legs”

(c)  
![](images/1112d06d27d8611049c80370a41aea310640d4835da87f61b0410c90d52528ca.jpg)

![](images/e04386042760fc89d94cf5cbe09b606d4581ab58ae7d5618a7bccbd37052e87e.jpg)  
“photo of a yellow dog holding a yellow flower in mouth, with fence in background”

![](images/ba6935e976be56f8916e5842be649c6c9a645ca74b800b8ec046977b58b210c0.jpg)

![](images/779c1c581ca67dc61d7f7903786c92752e98d9c39069227ee4bef5b5cfff57a4.jpg)

![](images/b2b33ccda8252acddd26c313a2cc1510cf6e76cdb16fab9caaecf522b221cb63.jpg)  
“photo of a green statue of liberty holding a golden red torch in hand”

![](images/db0fd0c92b9095bb47306d42e07ffb53b467ecfc83b080dde778c03e59717188.jpg)  
Ours

Full  
![](images/97c48ae5ff73aafa6e33a1bf7a0a3a94dc3d0b3e48771aa6334c473accb1d032.jpg)  
Ours  
“photo of a white painted room with a bed, a lamp, and a picture on the wall”

![](images/ee3cac601a1d145735b53875e0917ce9c655545672158b298b9d30aa2f9ba902.jpg)

![](images/c3557db0e956c69673e66320a4a0525e13583f0e0c13723c4aac3698fa216620.jpg)

![](images/0e591643646374e7e49e0d824b374567f6868ca3e87dda96d6c602a612468697.jpg)  
“photo of a pink purple chair with black legs”  
“photo of a white painted empty room with …”

![](images/fd9a0f695b5e07b38791d3c8ceec0c2726d1b1ee2b88b08bfab1a959c217e7cf.jpg)

(d)  
![](images/e079c7fcae804818513de9ca5d7f09acd0ffec789838c8949a538f862a09a9f3.jpg)

![](images/4625082d97a5ae99727cebf8cfd6ecd122135d6556ff8a7b702e5a1d7242d33b.jpg)

![](images/8cf1fe13dc41fc74d9764852f32a38c51a79a61f405ea5e23fd2c62488fd7cf0.jpg)

![](images/f3df47b2628584ac0c10745303f18f1dd36927f4c6ef814d6441e64f952a79ec.jpg)  
“photo of a pink chair with black white legs”  
“photo of a yellow dog holding a yellow flower in mouth lying on the ground, with fence in background”

![](images/cf6deb9240ab55b21e75f2517cf99c521c5c3bf705118e144d591e18db43f3df.jpg)

![](images/772ab14f7f521b2afcd2679c95c9fbfbac4e070a1fda552e8cd0cc7c91995014.jpg)

![](images/32985340eb127983bff643c416361f8348b4d55ce0da20ea30506251ae5fd0cc.jpg)

![](images/4682ba1dffad44065b4d13a394b614fecb1105f0364185d663a48cdd2abc55a0.jpg)

![](images/c776c3694228cd465e46dd06fd45763230cb9c048ea0209197c6db0c4498e9d9.jpg)

![](images/3b9e4b882e0c2d86210f983b8d9ca7179fc70ce50ac2637e6e017ab09451ec92.jpg)  
“photo of a yellow black dog holding a yellow flower in mouth, with fence in background”

![](images/baa385d297778e279d4208c03ce4b07b08cbe91a4607252d7da5e2d26b94caa4.jpg)

![](images/4a27c77f76e8f69ecaddb34e1ed95558b1f4cbd1ca739402811226afc8ebb64f.jpg)  
“photo of a green statue of liberty holding a golden torch an apple in hand”

![](images/dce657627d89f2dee81934a65a983c6a50a4ccad55cb1478f60aead7bbd1101f.jpg)  
“photo of a grey purple Beetle car”  
“photo zoom in view of a green statue of liberty holding a golden torch in hand”

(e)  
![](images/75d539cc13165e38e6fb3f5af49c7e10fc00b1eb9624dc615d8fce4f665be6d7.jpg)

![](images/3dac5c74589acb0b890947d94cdc29c55867a95b6803fae02cbc6840252e04af.jpg)  
“photo Watercolor painting of a  
grey Beetle car”

![](images/d8045d75936e8c98a7c3d982ff109ee9c78974448c47c33eca07c80368a422a3.jpg)

![](images/012ee37775319832e13ca5b77c683221922d6b2887c2f5c28571039aaabb07f1.jpg)  
Benz car”

Figure 7. Results for single image editing. SVDiff (“Ours”) enables successful image edits despite slight misalignment with the original image. SVDiff performs desired modifications when full model fine-tuning (“Full”) fails, such as removing an object (2nd edit in (a)), adjusting pose (2nd edit in (c)), or zooming in (3rd edit in (d)). The backgrounds in some cases may be affected, however, the subject of the image remains well-preserved. For ours, we use DDIM inversion [59] for all edits in (a,c,e) and the first edit in (d).

sion [59] in the appendix. Our results show that DDIM inversion improves the editing quality and alignment with the input image for non-structural edits when using our spectral shift parameter space, but may worsen the results for full weight fine-tuning. For example, in Fig. 7, we use DDIM inversion for the edits in (a,c,e) and the first edit in (d). The second edit in (d) presents an interesting example where our method can actually make the statue hold an apple with its hand. Additionally, our fine-tuning approach still produces the desired edit of an empty room even with DDIM inversion, as seen in the third edit of Fig. 7-a. Overall, we see that SVDiff can still perform desired edits when full model fine-tuning exhibits language drift, i.e. it fails to remove the picture in the second edit of (a), change the pose of the dog in the second edit of (c), and zoom-in view in (d).

## 4.4. Analysis and Ablation

Due to space limitations, we present parameter subsets, weight combination, interpolation and style mixing analysis in this section and provide further analysis including rank, scaling, and correlation in the appendix.

Parameter subsets We explore the fine-tuning of spectral shifts within a subset of parameters in UNet. We consider 12 distinct subsets for our ablation study, as outlined in Tab. 1. Due to space limitations, we provide the visual samples and text-/image-alignment scores for each subset on 5 subjects in appendix. Our findings are as follows: (1) Optimizing the cross-attention (CA) layers generally results in better preservation of subject identity compared to optimizing key and value projections. (2) Optimizing the up-, down-, or mid-blocks of UNet alone is insufficient to maintain identity, which is why we did not further isolate subsets of each part. However, it appears that the up-blocks exhibit the best preservation of identity. (3) In terms of dimensionality, the 2D weights demonstrate the most influence, and offer better identity preservation than UNet-CA.

![](images/3fed0725a90ad3d6cf71356f4975bc0f9f5c09d8aec4da75032402835a0f91f6.jpg)

![](images/7236b021daad6d52c03085c01b1bec9ca8304bb2f318f193c21f4f9cf0dca4e0.jpg)

![](images/97eea09e296ce5becdf82841fbd61d7a04a3db513963e7367931e10f0d0a2335.jpg)  
(a) “photo of a �<sub>!</sub> dog”

![](images/1170fdb92a5985782c1a6a66c849ba6108bc319b70c26c3439ec30c0e6e421e8.jpg)

![](images/2dd1bfbd8912664c8f48cc5b353c84d753ebf59618520edceac7d9dbb36573aa.jpg)  
“photo of a �<sub>#</sub> sculpture”(b)

![](images/569d1b3c30d61e1133b51c35ae75db5fe2ce0d0d787963f80c0ab45c4a97e140.jpg)

![](images/b7a640305f436c21bfea20724f52776dd75b2de7bd24ed87bdded9a6710cd3d2.jpg)  
“photo of a �<sub>!</sub> dog in a(c) swimming pool”

![](images/ffb83b148fe898ec36817046155b91a50eadbda43c1fa8321882e134c1ceae80.jpg)

![](images/ccfa52fa8c590619110369c6e5fb43ec77c08455915f80e66d1bf72de220f958.jpg)  
“photo of a �<sub>#</sub> sculpture(d) in Minecraft style”

![](images/4edbbbc30d3c320146d9c7e1a2a3a63938adddc000906015c2b5e5a709d10da2.jpg)

![](images/895c8354f64d1cd436e93dd8106c13ec84ac0cff09b13dd8b0e0e2c672ee321c.jpg)  
“a �<sub>!</sub> dog sitting besides(e)

![](images/a74ff6d24931165b614ef3b845a0715479c153d15a82f9738955a74c151f4a5e.jpg)  
“photo of a �<sub>"</sub> building”(f)

a �<sub>#</sub> sculpture”  
![](images/78be4b48ed4b7423558b8877b2e2f3efa5e78c41ee18817ca6bd4e228b0c47a3.jpg)  
“photo of a �<sub>#</sub> sculpture” (g)

![](images/fe5152a557b21555ce6747626e32b44da882f2acd1b0a48ec27fea9488947d9b.jpg)

![](images/faab32c23f1f8fc34be22cd95776df7562deda89a41558f38ffd992201b09950.jpg)  
Minecraft style”

![](images/8342a079f62503265cc238a636d0e50623267ee1db70b53578b212b990aa1010.jpg)

![](images/7aca2b2e2be591872534e5417202e4ca79e72c71e2469c125c55c9d2612a4f24.jpg)  
“a �<sub>#</sub> sculpture made of(i)  
glass”

![](images/c1fd0d538d4517cd776bf1b10394a8fbe1c9883b5b26cf3f67a02144336948ef.jpg)

![](images/1ef8241f69161d707c6b98fa57d7f2ca5aac17e740af1e2745cc8fe8a93c0aad.jpg)  
“a �<sub>#</sub> sculpture in front(j)  
of a �<sub>"</sub> building”

Figure 8. Effects of combining spectral shifts $( \Sigma _ { \delta ^ { \prime } } = \mathrm { d i a g } ( \mathrm { R e L U } ( \pmb { \sigma } + \pmb { \delta } _ { 1 } + \pmb { \delta } _ { 2 } ) ) )$ ) and weight deltas $\left( W ^ { \prime } = W + \Delta W _ { 1 } + \Delta W _ { 2 } \right)$ in one model. The combined model retains individual subject features but may mix styles for similar subjects. The results also suggests that the task arithmetic property [25] of language models also holds in StableDiffusion.
<table><tr><td>Subset</td><td>SVDiff Parameters</td><td>Storage</td><td>Subset</td><td>SVDiff Parameters</td><td>Storage</td></tr><tr><td>UNet</td><td>all UNet layers</td><td>1404KB</td><td>Up-Blocks</td><td>up-blocks in UNet</td><td>789KB</td></tr><tr><td>UNet-CA</td><td>all CrossAttn layers in UNet</td><td>194KB</td><td>Down-Blocks</td><td>down-blocks in UNet</td><td>469KB</td></tr><tr><td>UNet-CA-KV</td><td>WK, WV in CrossAttn in UNet</td><td>84.8KB</td><td>Mid-Block</td><td>mid-blocks in UNet</td><td>135KB</td></tr><tr><td>UNet-1D</td><td>all 1-D weights in UNet</td><td>430KB</td><td>Up-CA</td><td>CrossAttn in up-blocks</td><td>106KB</td></tr><tr><td>UNet-2D</td><td>all 2-D weights in UNet</td><td>617KB</td><td>Down-CA</td><td>CrossAttn in down-blocks</td><td>70.4KB</td></tr><tr><td>UNet-4D</td><td>all 4-D weights in UNet</td><td>355KB</td><td>Mid-CA</td><td>CrossAttn in mid-block</td><td>17.7KB</td></tr></table>

Table 1. Fine-tuning 12 subsets of parameters in UNet, along with their corresponding model sizes.

![](images/afd36ecf93e41e3e78877962c873c8c9ad44dfbb0d041e134203fa1f4b5f633f.jpg)

![](images/a464d575a320ac3dbedb5ee20fd88a1e2946884e18dddb1d16e566c1bf278901.jpg)  
(a) Correlations of Spectral Shifts  
(b) Text- and Image-Alignment Scores

![](images/53bd5ec72cc06e34b1254635f5ee33626c2185082a725d8a1d5c3fd1875dd649.jpg)

![](images/22ecd7a2157b7694af5e3cfcbf9ae1570398f094514a345e7a0611d58da22f6a.jpg)

![](images/356db838bcbfe058100fd4d64613d6c577db62e7f3b1ed65e0dd909c66582ae6.jpg)

![](images/0c77ddb6e30eff09785ad0e210942973c9d0633d0d359aa6fbc080ab87ab377e.jpg)

![](images/12d23cf90da4433ad884851ad118e79fba9d53ee146f2e55826d49ec3ade54b5.jpg)  
Figure 9. (a) Correlation of individually learned spectral shifts for different subjects. The cosine similarities between the spectral shifts of two subjects are averaged across all layers and plotted. The diagonal shows average similarities between two runs with different learning rates. (b) Text- and image-alignment for singlesubject generation. The generated image is denoted as x˜. The text-alignment is measured by the CLIP score [46, 12] cos(x˜, c), and the image-alignment is defined as $1 - \mathcal { L } _ { \mathrm { L P I P S } } \big ( \tilde { \mathbf { x } } , \mathbf { x } ^ { * } \big )$ [78].

![](images/883e72e287dd4693e4f930862e72bf33547b846fab2a5d6c507d911e8be4bf8d.jpg)

![](images/11c1b09d71a3769bde500a68d8301970caccfa342a834880f1bc681c92333061.jpg)

![](images/ec253d2581688ced194918f0d03e3b3433ba04baa33b8e11e5a8a875dd5575f6.jpg)  
Figure 10. Style-Mixing results with SVDiff. Following Extended Textual Inversion [70], we utilize spectral shifts in layer (16, down’, 1) - (8, down’, 0) to provide geometry information, while the remaining layers contribute to the appearance.

Weight combination We analyze the effects of weight combination by Eq. (4). Fig. 8 shows a comparison between combining only spectral shifts (marked in “SVD”) and combining the full weights (marked in “Full”). The combined model in both cases retains unique features for individual subjects, but may blend their styles for similar concepts (as seen in (e)). For dissimilar concepts (such as the $[ V _ { 2 } ]$ sculpture and $\left[ V _ { 3 } \right]$ building in (j)), the models can still produce separate representations of each subject. Interestingly, combining full weight deltas can sometimes result in better preservation of individual concepts, as seen in the clear building feature in (j). We posit that this is due to the fact that SVDiff limits update directions to the eigenvectors, which are identical for different subjects. As a result, summing individually trained spectral shifts tends to create more “interference” than summing full weight deltas.

![](images/fbe5a23e3fad1f1f09acc75096c181f23d9de328e1393cb5497c12bb5febe342.jpg)  
Figure 11. Effects of interpolating spectral shifts or full weights.

Style transfer and mixing A simple and straightforward way to enable style-mixing is to combine individually trained spectral shifts using Eq. (4) and inference with the combined model. We show visual examples of this strategy in appendix and further explore a more challenging and controllable approach for style-mixing as follows. Inspired by the disentangling property observed in StyleGAN [30], we hypothesize that a similar property applies in our context. Following Extended Textual Inversion (XTI [70]), we conducted a style mixing experiment, as illustrated in Fig. 10. For this experiment, we fine-tuned SVDiff on the UNet-2D subset and employed the geometry information provided by $( { \bf \nabla } 1 6 , ~ \mathrm { d o w n } ^ { \prime } , ~ { \bf \nabla } 1 ) ~ - ~ ( { \bf \nabla } 8 $ down’, 0) (as described in XTI, Section 8.1). We observe that our spectral shift parameter space allows us to achieve a similar disentangled style-mixing effect, comparable to the P+ space in XTI.

Interpolation Fig. 11 shows the results of weight interpolation for both spectral shifts and full weights. The models are marked as “SVD” and “Full”, respectively. The first two rows of the figure demonstrate interpolating between two different classes, such as “dog” and “sculpture”, using the same abstract class word “thing” for training. Each column shows the sample from α-interpolated models. For spectral shifts $\mathrm { ( ^ { 6 6 } S V D ^ { , 9 } ) }$ , we use Eq. (5) and for full weights, we use $W ^ { \prime } = W + \alpha \Delta W _ { 1 } + ( 1 - \alpha ) \Delta W _ { 2 } = \alpha W _ { 1 } + ( 1 - \alpha ) W _ { 2 }$ . The images in each row are generated using the same random seed with the deterministic DDIM sampler [59] $( \eta = 0 )$ . As seen from the results, both spectral shift and full weight interpolation are capable of generating intermediate concepts between the two original classes.

## 4.5. Comparison with LoRA

In our comparison of SVDiff and LoRA [14, 22] for single image editing, we find that while LoRA tends to underfit, SVDiff provides a balanced trade-off between faithfulness and realism. Additionally, SVDiff results in a significantly smaller delta checkpoint size, being 1/2 to 1/3 that of LoRA. However, in cases where the model requires extensive fine-tuning or learning of new concepts, LoRA’s flexibility to adjust its capability by changing the rank may be beneficial. Further research is needed to explore the potential benefits of combining these approaches. A comparison can be found in the appendix.

It is noteworthy that, with rank one, the storage and update requirements for the W matrix of shape $M \times N$ in SVDiff are min(M, N) floats, compared to $( M + N )$ floats for LoRA. This may be useful for amortizing or developing training-free approaches for DreamBooth [52]. Additionally, exploring functional forms [65, 66] of spectral shifts is an interesting avenue for future research.

## 5. Conclusion and Limitation

In conclusion, we have proposed a compact parameter space, spectral shift, for diffusion model fine-tuning. The results of our experiments show that fine-tuning in this parameter space achieves similar or even better results compared to full weight fine-tuning in both single- and multisubject generation. Our proposed Cut-Mix-Unmix dataaugmentation technique also improves the quality of multisubject generation, making it possible to handle cases where subjects are of similar categories. Additionally, spectral shift serves as a regularization method, enabling new use cases like single image editing.

Limitations Our method has certain limitations, including the decrease in performance of Cut-Mix-Unmix as more subjects are added and the possibility of an inadequatelypreserved background in single image editing. Despite these limitations, we see great potential in our approach for fine-tuning diffusion models and look forward to exploring its capabilities further in future research, such as combining spectral shifts with LoRA or developing training-free approaches for fast personalizing concepts.

Acknowledgments This research has been partially funded by research grants to D. Metaxas through NSF: IUCRC CARTA 1747778, 2235405, 2212301, 1951890, 2003874, and NIH-5R01HL127661.

## References

[1] Amazon mechanical turk. https://www.mturk.com/, 2005. 5

[2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. arXiv preprint arXiv:2206.02779, 2022. 2

[3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 2

[4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2

[5] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. arXiv preprint arXiv:2302.07121, 2023. 2

[6] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 2

[7] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. ICML, 2023. 2

[8] David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model. In European conference on computer vision, pages 351–369. Springer, 2020. 3

[9] Manuel Brack, Patrick Schramowski, Felix Friedrich, Dominik Hintersdorf, and Kristian Kersting. The stable artist: Steering semantics in diffusion latent space. arXiv preprint arXiv:2212.06013, 2022. 2

[10] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2

[11] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Rui, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186, 2023. 2

[12] Yuxiao Chen, Jianbo Yuan, Yu Tian, Shijie Geng, Xinyu Li, Ding Zhou, Dimitris N Metaxas, and Hongxia Yang. Revisiting multimodal representation in contrastive learning: from patch and token embeddings to finite discrete tokens. arXiv preprint arXiv:2303.14865, 2023. 8

[13] Anton Cherepkov, Andrey Voynov, and Artem Babenko. Navigating the gan parameter space for semantic image editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3671–3680, 2021. 2, 3

[14] cloneofsimo. cloneofsimo/lora: Low-rank adaptation for fast text-to-image diffusion fine-tuning. https://github. com/cloneofsimo/lora. 2, 9

[15] CompVis. Compvis/stable-diffusion. https://github. com/CompVis/stable-diffusion. 2

[16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1, 2

[17] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2023. 2

[18] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696–10706, 2022. 2

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2

[20] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. J. Mach. Learn. Res., 23:47–1, 2022. 1

[21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 4

[22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1, 9

[23] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 2

[24] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin C.K. Chan, and Ziwei Liu. ReVersion: Diffusion-based relation inversion from images. arXiv preprint arXiv:2303.13495, 2023. 2

[25] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022. 8

[26] Shir Iluz, Yael Vinker, Amir Hertz, Daniel Berio, Daniel Cohen-Or, and Ariel Shamir. Word-as-image for semantic typography. arXiv preprint arXiv:2303.01818, 2023. 2

[27] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 2

[28] Jindong Jiang, Fei Deng, Gautam Singh, and Sungjin Ahn. Object-centric slot diffusion. arXiv preprint arXiv:2303.10834, 2023. 2

[29] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. arXiv preprint arXiv:2303.17606, 2023. 2

[30] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020. 9

[31] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. arXiv preprint arXiv:2210.09276, 2022. 1, 2, 4

[32] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 2, 3, 5

[33] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVII, pages 423–439. Springer, 2022. 2

[34] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125, 2023. 2

[35] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 4

[36] Sangwoo Mo, Minsu Cho, and Jinwoo Shin. Freeze the discriminator: a simple baseline for fine-tuning gans. arXiv preprint arXiv:2002.10964, 2020. 1

[37] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022. 1, 2

[38] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2

[39] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2

[40] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021. 2

[41] Atsuhiro Noguchi and Tatsuya Harada. Image generation from small datasets via batch statistics adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2750–2758, 2019. 1

[42] Hadas Orgad, Bahjat Kawar, and Yonatan Belinkov. Editing implicit assumptions in text-to-image diffusion models. arXiv preprint arXiv:2303.08084, 2023. 2

[43] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. arXiv preprint arXiv:2302.03027, 2023. 2

[44] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 2

[45] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2

[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 8

[47] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1, 2

[48] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. Advances in neural information processing systems, 30, 2017. 1

[49] Mengwei Ren, Mauricio Delbracio, Hossein Talebi, Guido Gerig, and Peyman Milanfar. Image deblurring with domain generalizable diffusion models. arXiv preprint arXiv:2212.01789, 2022. 2

[50] Esther Robb, Wen-Sheng Chu, Abhishek Kumar, and Jia-Bin Huang. Few-shot adaptation of generative adversarial networks. arXiv preprint arXiv:2010.11943, 2020. 1, 2, 3

[51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022. 1, 2, 3, 5

[52] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022. 1, 2, 3, 5, 9

[53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1, 2, 5

[54] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1532–1540, 2021. 3

[55] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 2

[56] Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, and Sungroh Yoon. Edit-a-video: Single video editing with object-aware consistency. arXiv preprint arXiv:2303.07945, 2023. 2

[57] Ken Shoemake. Animating rotation with quaternion curves. In Proceedings of the 12th annual conference on Computer graphics and interactive techniques, pages 245–254, 1985. 4

[58] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2

[59] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 2, 4, 5, 7, 9

[60] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 2

[61] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019. 2

[62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2

[63] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image translation. In International Conference on Learning Representations, 2022. 2

[64] Yanpeng Sun, Qiang Chen, Xiangyu He, Jian Wang, Haocheng Feng, Junyu Han, Errui Ding, Jian Cheng, Zechao Li, and Jingdong Wang. Singular value fine-tuning: Fewshot segmentation requires few-parameters fine-tuning. In Advances in Neural Information Processing Systems, 2022. 1, 2

[65] Hossein Talebi and Peyman Milanfar. Global image denoising. IEEE Transactions on Image Processing, 23(2):755– 768, 2013. 9

[66] Hossein Talebi and Peyman Milanfar. Nonlocal image editing. IEEE Transactions on Image Processing, 23(10):4460– 4473, 2014. 9

[67] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV, pages 459–479. Springer, 2022. 2

[68] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for textdriven image-to-image translation. arXiv preprint arXiv:2211.12572, 2022. 2

[69] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 2, 5

[70] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 8, 9

[71] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhi Naik. End-to-end diffusion latent optimization improves classifier guidance. arXiv preprint arXiv:2303.13703, 2023. 2

[72] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. arXiv preprint arXiv:2211.12446, 2022. 2

[73] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2

[74] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022. 2

[75] Xavierxiao. Xavierxiao/dreambooth-stable-diffusion: Implementation of dreambooth with stable diffusion. https://github.com/XavierXiao/ Dreambooth-Stable-Diffusion. 5

[76] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 6023–6032, 2019. 3, 4

[77] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 2

[78] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 8

[79] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. arXiv preprint arXiv:2212.04489, 2022. 1, 2