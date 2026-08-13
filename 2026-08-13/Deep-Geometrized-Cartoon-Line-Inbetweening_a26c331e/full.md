# Deep Geometrized Cartoon Line Inbetweening

Li Siyao<sup>1</sup> Tianpei Gu<sup>2∗</sup> Weiye Xiao<sup>3</sup> Henghui Ding<sup>1</sup> Ziwei Liu<sup>1</sup> Chen Change Loy<sup>1</sup> <sup>B</sup> <sup>1</sup>S-Lab, Nanyang Technological University <sup>2</sup>Lexica <sup>3</sup>Southeast University

{siyao002, henghui.ding, ziwei.liu, ccloy}@ntu.edu.sg, gutianpei@ucla.edu, 230189776@seu.edu.cn

![](images/1fece6931eeaaa27b6c80acbdbbb40c99d6216838f2e47cc8e1428960440c3b7.jpg)  
Figure 1: Inbetweening on two source cartoon line drawings of Monkey D. Luffy extracted from ONE PIECE. We compare our proposed AnimeInbet with state-of-the-art frame interpolation methods VFIformer [14], EISAI [5], FILM [23] and RIFE [6].

## Abstract

We aim to address a significant but understudied problem in the anime industry, namely the inbetweening of cartoon line drawings. Inbetweening involves generating intermediate frames between two black-and-white line drawings and is a time-consuming and expensive process that can benefit from automation. However, existing frame interpolation methods that rely on matching and warping whole raster images are unsuitable for line inbetweening and of ten produce blurring artifacts that damage the intricate line structures. To preserve the precision and detail of the line drawings, we propose a new approach, AnimeInbet, which geometrizes raster line drawings into graphs of endpoints and reframes the inbetweening task as a graph fusion problem with vertex repositioning. Our method can effectively capture the sparsity and unique structure of line drawings while preserving the details during inbetweening. This is made possible via our novel modules, i.e., vertex geometric embedding, a vertex correspondence Transformer, an effective mechanism for vertex repositioning and a visibility predictor. To train our method, we introduce Mixamo-Line240, a new dataset ofline drawings with ground truth vectorization and matching labels. Our experiments demonstrate that AnimeInbet synthesizes high-quality, clean, and

complete intermediate line drawings, outperforming existing methods quantitatively and qualitatively, especially in cases with large motions. Data and code are available at https: //github.com/lisiyao21/AnimeInbet.

## 1. Introduction

Cartoon animation has undergone significant transformations since its inception in the early 1900s, when consecutive frames were manually drawn on paper. Although automated techniques now exist to assist with some specific procedures during animation production, such as colorization [22, 32, 10, 39, 4] and special effects [38], the core element – the line drawings of characters – still needs handdrawing each frame individually, making 2D animation a labor-intensive industry. Developing an automated algorithm that can produce intermediate line drawings from two input key frames, commonly referred to as “inbetweening”, has the potential to significantly improve productivity.

Line inbetweening is not a trivial subset of general frame interpolation, as the structure of line drawings is extremely sparse. Unlike full-textured images, line drawings contain only around 3% black pixels, with the rest of the image being white background. As illustrated in Figure 2, this poses two significant challenges for existing raster-image-based frame interpolation methods. 1) The lack of texture in line drawings makes it challenging to compute pixel-wise correspondence accurately in frame interpolation. One pixel can have many similar matching candidates, leading to inaccurate motion prediction. 2) The warping and blending used in frame interpolation can blur the salient boundaries between the line and the background, leading to a significant loss of detail.

![](images/8a04bc786d5154f1af09df58e8dc3f33b1bc247e9070f9f2c30c2b40eace6992.jpg)  
Figure 2: Raster vs geometrized inbetweening. Top: search space of a pixel (left) vs a vertex (right) in matching. Bottom: pixel warping/sampling (left) vs vertex repositioning (right).

To address the challenges posed by line inbetweening, we propose a novel deep learning framework called AnimeInbet, which inbetweens line drawings in a geometrized format instead of raster images. Specifically, the source images are transformed into vector graphs, and the goal is to synthesize an intermediate graph. This reformulation can overcome the challenges discussed earlier in this paper. As illustrated in Figure 2, the matching process in the geometric domain is conducted on concentrated geometric endpoint vertices, rather than all pixels, reducing potential ambiguity and leading to more accurate correspondence. Moreover, the repositioning does not change the topology of the line drawings, enabling preservation of the intricate and meticulous line structures. Compared to existing methods, our proposed AnimeInbet framework can generate clean and complete intermediate line drawings, as demonstrated in Figure 1.

The core idea of our proposed AnimeInbet framework is to find matching vertices between two input line drawing graphs and then reposition them to create a new intermediate graph. To achieve this, we first design a vertex encoding strategy that embeds the geometric features of the endpoints of sparse line drawings, making them distinguishable from one another. We then apply a vertex correspondence Transformer to match the endpoints between the two input line drawings. Next, we propagate the shift vectors of the matched vertices to unmatched ones based on the similarities of their aggregated features to realize repositioning for all endpoints. Finally, we predict a visibility mask to erase the vertices and edges occluded in the inbetweened frame, ensuring a clean and complete intermediate frame.

To facilitate supervised training on vertex correspondence, we introduce MixamoLine240, the first line art dataset with ground truth geometrization and vertex matching labels. The

![](images/8957977c7b6c416d1bea2b69486b4926dc6dbd92591e1e9a6e3c9c7b579e7af0.jpg)  
Figure 3: Geometrized line art in MixamoLine240. 2D endpoints and connected lines are projected from vertices and edges of orinal 3D mesh. Endpoints indexed to unique 3D vertices are matched (marked in the same colors).

2D line drawings in our dataset are selectively rendered from specific edges of a 3D model, with the endpoints indexed from the corresponding 3D vertices. By using 3D vertices as reference points, we ensure that the vertex matching labels in our dataset are accurate and consistent at the vertex level.

In a conclusion, our work contributes a new and challenging task of line inbetweening, which could facilitate one of the most labor-intensive art production processes. We also propose a new method that outperforms existing solutions, and introduce a new dataset for comprehensive training.

## 2. Related Work

Frame Interpolation. Frame interpolation is a widely studied task in recent years, involving synthesizing intermediate frames from existing ones. Many approaches have been proposed [13, 19, 20, 7, 17, 34, 18, 21, 26, 6, 23, 5, 14, 11], such as those that use optical flows or deep networks to search for matching areas and warp them to proper intermediate locations. Among the most recent algorithms, RIFE [6] directly predicts intermediate flows to warp the input frames and blends the warped frames into intermediate ones by a visible mask. VFIformer [14] adopts the same idea to predict the intermediate flows but proposes a Transformer to synthesize the intermediate from both warped images and features. Reda et al. [23] design a scale-agnostic feature pyramid to predict the intermediate flows and warp frames in a hierarchical manner to handle extreme large motions. Siyao and Zhao et al. [30] propose a frame interpolation pipeline specific for 2D cartoon in the wild, while Chen and Zwicker [5] improves the perceptual quality by embedding an optical-flow based line aggregator. While these methods achieve impressive performance on raster natural or cartoon videos, their pixel-oriented nature are not suitable for inbetweening concise and sparse line arts, which can yield severe artifacts and are not feasible for real usage in anime creation. Research on Anime. There has been increasing research interest in techniques to facilitate 2D cartoon creation, including sketch simplification [28, 27], vectorization [40, 36, 15, 12], colorization [22, 32, 10, 39, 4], shading [38], head reenactment [8] and line-art-based cartoon generation [37]. While these studies may improve specific aspects of animation creation, the core line arts still rely on manual frame-by-frame drawing. Some sporadic rule-based methods have been developed for stroke inbetweening under strict conditions, but these methods lack the flexibility required for wider applications [35, 3]. Our work is the first to propose a deep learning-based method for inbetweening geometrized line arts. Additionally, we introduce vertex-wise correspondence datasets on line arts. It is noteworthy that existing datasets are not sufficiently ‘clean’ for our task since cartoon contour lines can cross the boundaries of motion, leading to incorrect corresponding labels at the vertex level [25, 29].

![](images/978cc349590aeda6cd40d4653e22da568ea61902d6757ff8185cd35165c9aa16.jpg)

<table><tr><td>Train actions</td><td>break dance climb</td><td>capoeira run</td><td>chap giratoria shove</td><td>fist fight magic attack</td><td>flying knee tripping</td></tr><tr><td>Test</td><td>chip</td><td>evade</td><td>flair</td><td>sword slash</td><td>hip hop</td></tr><tr><td></td><td></td><td></td><td>actions hurricane kick soccer tackle standing death</td><td>swim</td><td>stand up</td></tr></table>

Figure 4: Data composition. Training and test sets are separately composed by 10 characters × 10 actions. First & second rows are training & test characters, respectively. Shaded are for validation.

## 3. Mixamo Line Art Dataset

To facilitate training and evaluation of geometrized line inbetweening, we develop a large-scale dataset, named MixamoLine240, which consists of 240 sequences of consecutive line drawing frames, with 100 sequences for training and 140 for validation and testing. To obtain this vast amount of cartoon line data, we utilize a “Cel-shading” technique, i.e., to use computer graphics software (Blender in this work) to ren der 3D resources into an anime-style appearance that mimics the hand-drawn artistry. Unlike previous works [25, 29] that only provide raster images, MixamoLine240 also provides ground-truth geometrization labels for each frame, which include the coordinates of a group of vertices (V) and the connection topology (T). Additionally, we assign an index number $( R [ i ] )$ to each 2D endpoint $( V [ i ] )$ that refers to a unique vertex in the 3D mesh of the character, as illustrated in Figure 3, which can be further used to deduce the vertexlevel correspondence. Specifically, given two frames $I _ { 0 }$ and

Table 1: Difficulty statistics with various frame gaps.
<table><tr><td>Frame gap→</td><td></td><td>0 (60 fps) 1 (30 fps) 5 (10 fps) 9 (6 fps)</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Trrain</td><td>Occlusion rate (%)</td><td>14.8</td><td>21.5</td><td>37.8</td><td>46.6</td></tr><tr><td>Avg. vtx shift</td><td>8.6</td><td>16.4</td><td>42.6</td><td>62.8</td></tr><tr><td>Avg. max vtx shift</td><td>26.0</td><td>48.9</td><td>129.7</td><td>192.3</td></tr><tr><td>Occlusion rate (%)</td><td>18.4</td><td>26.5</td><td>44.2</td><td>53.5</td></tr><tr><td rowspan="4">Test</td><td>Avg. vtx shift</td><td>7.8</td><td>14.9</td><td>38.9</td><td>57.0</td></tr><tr><td> $\operatorname { A v g } .$  max vtx shift</td><td>23.8</td><td>45.0</td><td>119.3</td><td>173.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

$I _ { 1 }$ in a sequence, the 3D reference IDs reveal the vertex correspondence $\{ ( i , j ) \}$ for those vertices i in $I _ { 0 }$ and j in $I _ { 1 }$ having $R _ { 0 } [ i ] = R _ { 1 } [ j ]$ , while the rest unmatched vertices are marked as occluded. This strategy allows us to produce correspondence pairs with arbitrary frame gaps to flexibly adjust the input frame rate during training. Next, we discuss the construction and challenges inherent in the data.

Data Construction. In Blender, the mesh structure of a 3D character remains stable, i.e., the number of 3D vertex and the edge topology keep constant, when moving without additional subdivision modifier. We employ this property to achieve consistent line art rendering and accurate annotations for geometrization and vertex matching. As shown in Figure 3, the original 3D mesh contains all the necessary line segments required to represent the character in line art. During rendering, the visible outline from the camera’s perspective is selected based on the material boundary and the object’s edge. This process ensures that every line segment in the resulting raster image corresponds to an edge in the original mesh. The 2D endpoints of each line segment are simply the relevant 3D vertices projected onto the camera plane, referenced by the unique and consistent index of the corresponding 3D vertex. Meanwhile, since the 3D mesh naturally defines the vertex connections, the topology of the 2D lines can be transferred from the selective edges used for rendering. To prevent any topological ambiguity that may be caused by overlapped vertices in 3D space, we merge the endpoints that are within a Euclidean distance of 0.1 in the projected 2D space. This enables us to obtain both the raster line drawings and the accurate labels of each frame.

To create a diverse dataset, we used the open-source 3D material library Mixamo [1] and selected 20 characters and 20 actions, as shown in Figure 4. Each action has an average of 191 frames. We combined 10 characters and 10 actions to render 100 sequences, with a total of 19,930 frames as the training set. We then used the remaining 10 characters and 10 actions to render an 18,230-frame test set, ensuring that the training and testing partitions are exclusive. We also created a 44-sequence validation set, consisting of 20 unseen characters, 20 unseen actions, and 4 with both unseen character and action. To create this set, we combined the test characters “Swat”and “Warrok” and actions “sword slash” and “hip hop” with the training characters and actions. The validation set contains 11,102 frames and was also rendered at 1080p resolution with a frame rate of 60 fps. To ensure consistency across all frames, we cropped and resized each frame to a unified $7 2 0 \times 7 2 0$ character-centered image.

![](images/b590bd71b8ce8bacd4b7b9009b4648c0bdb8f42781f1cf8864cb124ed3cda6f0.jpg)  
Figure 5: Pipeline of proposed AnimeInbet. Our framework is composed of four main parts: the vertex geometric embedding, the vertex correspondence Transformer, repositioning propagation and graph fusion. Given a pair of line images $I _ { 0 }$ and $I _ { 1 }$ and their vector graphs $G _ { 0 }$ and $G _ { 1 }$ , our method generates the intermediate frame $G _ { t }$ in geometrized format.

Challenges. Table 1 summarizes the statistics that reflect the difficulty of the line inbetweening task under various input frame rates. With an increase in frame gaps, the inbetweening task becomes more challenging with larger motion magnitudes and higher occlusion percentages. For instance, when the frame gap is 9, the input frame rate becomes 6 fps, and the average vertex shift is 62.8 pixels. The mean value of the maximum vertex shift in a frame (“Avg. max vtx shift”) reaches 192.3 pixels, which is 27% of the image width. Additionally, nearly half of the vertices are unmatched in such cases, making line inbetweening a tough problem. Furthermore, the image composition of the test set is more complex than that of the training set. A training frame has an average of 1,256 vertices and 1,753 edges, while a test frame has an average of 1,512 vertices and 2,099 edges since the test set has more complex characters such as “Maw”.

## 4. Our Approach

An overview of the proposed line inbetweening framework, AnimeInbet, is depicted in Figure 5. Unlike existing frame interpolation methods that use raw raster images $I _ { 0 }$ and $I _ { 1 } ,$ , we process vector graphs $G _ { 0 } = \{ V _ { 0 } , T _ { 0 } \}$ and $G _ { 1 } = \{ V _ { 1 } , T _ { 1 } \}$ instead. The vertex coordinates in the images are represented by $V \in \mathbb { R } ^ { K \times 2 }$ , and the binary adjacency matrix is denoted by $T \in { 0 , 1 ^ { K \times K } }$ , where K denotes the number of vertices. The goal is to generate the intermediate graph $G _ { t }$ at time $t \in ( 0 , 1 )$ . To this end, we first design a CNN-based vertex geometric embedding to encode $V _ { 0 }$ and $V _ { 1 }$ to features $F _ { 0 }$ and $F _ { 1 }$ , respectively, as detailed in Section 4.1. Along with the embeddings, a vertex correspondence Transformer is proposed to aggregate the mutuality of vertex features to $\hat { F } _ { 0 }$ and $\hat { F } _ { 1 }$ by alternating self- and cross-attention layers (Section 4.2). The aggregated features are used to compute the correlation matrix $\bar { \mathcal { C } } \in \mathbb { R } ^ { K _ { 0 } \times K _ { 1 } }$ and to induce the vertex matching by row-wise and column-wise argmax.

![](images/60176b16c18e88323b0a7cd1444ed4cad3a32bbbb43b711e7e5f5d0e5c98d9a0.jpg)  
Figure 6: Vertex Geometric Embedding. The goal is to obtain discriminative and meaningful features to describe each vertex.

In cases where vertices are occluded during large motion, we adopt a self-attention-based layer to propagate the vertex shifts from matched vertices to the unmatched, and obtain repositioning vectors $r _ { 0 } ~ \in ~ \mathbb { R } ^ { K _ { 0 } \times 2 }$ and $r _ { 1 } ~ \in ~ \mathbb { R } ^ { K _ { 1 } \times 2 }$ for all vertices (Section 4.3). Finally, we superpose the two input graphs based on the predicted correspondence, and we further refine the output by predicting visibility maps $m _ { 0 } \in \{ 0 , 1 \} ^ { K _ { 0 } }$ and $m _ { 1 } \in \{ 0 , 1 \} ^ { K _ { 1 } }$ to mask off those vertices of $V _ { 0 }$ and $V _ { 1 }$ that disappear in the intermediate frame, respectively, to obtain the final inbetweened line drawing $G _ { t } ,$ as explained in Section 4.5.

Geometrizing Line Drawings. The process of creating artwork has become largely digital, allowing for direct export in vectorized format. However, for line drawings that only appear in raster images, there are various commercial software and open-source research projects available [40, 36, 15, 12] that can be used to convert the raster images into the required vectorized input format. We will ablate the performance of line vectorization in our experiments.

![](images/cdc415184d8555d922557e0cb27919f5226648b197d434293f4d50fd48196719.jpg)  
Figure 7: Vertex Correspondence Transformer. SA and CA represent self-attention and cross-attention, respectively.

## 4.1. Vertex Geometric Embedding

Discriminative features for each vertex are desired to achieve accurate graph matching. Line graphs are different from general graphs as the spatial position of endpoint vertices, in addition to the topology of connections, determines the geometric shape of the line. The geometric graph embedding for line art is hence designed to comprise three parts: 1) image contextual embedding, 2) positional embedding, and 3) topological embedding, as shown in Figure 6.

For image contextual embedding, we use a 2D CNN $\mathcal { E } _ { I }$ to extract deep contextual features within the same size of the input raster image I. Then, for each vertex $V _ { 0 } [ i ] : = ( x , y )$ we index feature $\mathcal { E } _ { I } ( I ) \left[ ( x , y ) \right]$ as the image embedding for the i-th vertex. As to the positional embedding, we employ a 1D CNN $\mathcal { E } _ { P }$ to map each vertex coordinate $( x , y )$ to a C-dimensional feature. To include the topological information into a lower dimensional feature, we first conduct spectral embedding [2]  on the binary adjacency matrix $T ,$ which involves an eigenvector decomposition on the Laplacian matrix of the graph, then feed the spectral embedding to a subsequent 1D CNN ${ \mathcal { E } } _ { T }$ . The final geometric graph embedding is formulated as

$$
F _ { 0 } = \mathcal { E } _ { I } \left( I _ { 0 } \right) \left[ V _ { 0 } \right] + \mathcal { E } _ { P } \left( V _ { 0 } \right) + \mathcal { E } _ { T } \left( S \left( T _ { 0 } \right) \right) .\tag{1}
$$

We obtain F<sub>1</sub> in the same way. $F _ { 1 }$

## 4.2. Vertex Correspondence Transformer

We use geometric features $F _ { 0 }$ and $F _ { 1 }$ to establish a vertexwise correspondence between $G _ { 0 }$ and $G _ { 1 }$ . Specifically, we compute a correlation matrix between vertex features and identify the matching pair as those with the highest value across both the row and the column of the matrix. Prior to this step, we apply a Transformer that aggregates the mutual consistency both intra- and inter-graph.

Mutual Aggregation. Following [24, 31], we employ a cascade of alternating self- and cross-attention layers to aggregate the vertex feature. In a self-attention layer, all queries, keys and values are derived from the single source feature,

$$
S A ( F _ { 0 } ) = \mathrm { s o f t m a x } \left( { \frac { \mathcal { Q } ( F _ { 0 } ) K ^ { T } ( F _ { 0 } ) } { \sqrt { C } } } \right) \mathcal { V } ( F _ { 0 } ) ,\tag{2}
$$

where $\mathcal { Q } ,$ and  represent MLPs for query, key and value, respectively; while in the cross-attention layer, the keys and values are computed from another feature:

$$
C A ( F _ { 0 } , F _ { 1 } ) = \mathrm { s o f t m a x } \left( \frac { \mathcal { Q } ( F _ { 0 } ) K ^ { T } ( F _ { 1 } ) } { \sqrt { C } } \right) \mathcal { V } ( F _ { 1 } ) .\tag{3}
$$

After N layers of rotating self- and cross-attention layers as shown in Figure 7, we obtain aggregated feature $\hat { F } _ { 0 }$ and $\hat { F } _ { 1 }$ In the aggregation, each vertex is represented as an attentional pooling of all other vertices within the same graph and across the two graphs achieving a full fusion of information with mutual dependencies.

Correlation Matrix and Vertex Matching. We compute the correlation matrix $\mathcal { P }$ as $\begin{array} { r } { \mathcal { P } = \frac { \hat { F } _ { 0 } \hat { F } _ { 1 } ^ { T } } { \sqrt { C } } } \end{array}$ . We further apply a differentiable optimal transport $( O T )$ [24] to improve the dual selection consistency and obtain $\hat { \mathcal { P } } = O T ( \mathcal { P } )$ . Then, we predict the one-way matching from $G _ { 0 }$ to $G _ { 1 }$ and vice versa as arg max indices across rows and columns:

$$
\left\{ \begin{array} { l l } { \mathcal { M } _ { 0 \to 1 } = \{ ( i , j ) | j = \arg \operatorname* { m a x } \hat { \mathcal { P } } _ { i , : } , i = 0 , . . . , K _ { 0 } - 1 \} } \\ { \mathcal { M } _ { 1 \to 0 } = \{ ( i , j ) | i = \arg \operatorname* { m a x } \hat { \mathcal { P } } _ { : , j } , j = 0 , . . . , K _ { 1 } - 1 \} . } \end{array} \right.\tag{4}
$$

A vertex pair is selected into the final correspondence if it is mutually consistent and its correlation value is larger than $\theta \colon$

$$
\begin{array} { r } { \hat { \mathcal { M } } = \{ ( i , j ) | ( i , j ) \in \mathcal { M } _ { 0  1 } \cap M _ { 1  0 } , \hat { \mathcal { P } } _ { i , j } > \theta \} . } \end{array}\tag{5}
$$

Otherwise, vertices will be considered to be occluded.

## 4.3. Repositioning Propagation

Fused vertices $( i , j )$ from vertex correspondence can be linearly relocated to $t V _ { 0 } [ i ] + ( 1 - t ) V _ { 1 } [ j ]$ in intermediate graph $G _ { t }$ based on time t. However, the positions of the unmatched vertices in $G _ { t }$ are still unknown. To reposition these vertices, we design an attention-based scheme similar to Xu et al. [33] to predict bidirectional shift vectors $r _ { 0 \to 1 }$ and $r _ { 1  0 }$ for $V _ { 0 }$ and $V _ { 1 }$ , respectively. Formally,

$$
\left\{ \begin{array} { l l } { r _ { 0 \to 1 } = \mathrm { s o f t m a x } \left( \frac { \hat { F } _ { 0 } \hat { F } _ { 0 } ^ { T } } { \sqrt { C } } \right) \left( \mathrm { s o f t m a x } ( \hat { \mathcal { P } } ) V _ { 1 } - V _ { 0 } \right) } \\ { r _ { 1 \to 0 } = \mathrm { s o f t m a x } \left( \frac { \hat { F } _ { 1 } \hat { F } _ { 1 } ^ { T } } { \sqrt { C } } \right) \left( \mathrm { s o f t m a x } ( \hat { \mathcal { P } } ^ { T } ) V _ { 0 } - V _ { 1 } \right) . } \end{array} \right.\tag{6}
$$

We then compute the final repositioning vectors as follows:

$$
r _ { 0 } [ i ] = \left\{ \begin{array} { l l } { V _ { 1 } [ j ] - V _ { 0 } [ i ] , } & { \mathrm { i f } ~ \exists ~ j ~ s . t . ~ ( i , j ) \in \hat { \mathcal { M } } , } \\ { r _ { 0 \to 1 } [ i ] , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{7}
$$

while $r _ { 1 }$ is computed in a similar way.

In this step, the motion vector $r _ { 0 \to 1 }$ of an unmatched vertex $V _ { 0 } [ i ]$ is computed as a softmax average of shifts to all vertices in $G _ { 1 } , i . e .$ , softmax $( \hat { \mathcal { P } } _ { i , : } ) V _ { 1 } - V _ { 0 }$ . It is then refined by attention pooling from matched vertices, based on self-similarity given by $\hat { F } _ { 0 } \mathbf { \bar { { F } } } _ { 0 } ^ { T } / \sqrt { C }$ . Vertices are reasonably repositioned in the new vector graph after this step.

Table 2: Quantitative evaluations of state-of-the-art frame interpolation methods using Chamfer Distance (reported in units of $\times 1 0 ^ { - 5 }$ , with lower values indicating better performance). The first place and runner-up are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="4">Validation Set</td><td colspan="4">Test Set</td></tr><tr><td> $\mathrm { g a p } = 1$ </td><td> $\mathrm { g a p } = 5$ </td><td> $\mathrm { g a p } = 9$ </td><td> $\operatorname { A v g } .$ </td><td> $\mathrm { g a p } = 1$ </td><td> $\mathrm { g a p } = 5$ </td><td> $\mathrm { g a p } = 9$ </td><td> $\operatorname { A v g } .$ </td></tr><tr><td>VFIformer [14]</td><td>7.82</td><td>26.04</td><td>50.71</td><td>28.19</td><td>7.62</td><td>27.55</td><td>50.68</td><td>28.62</td></tr><tr><td>RIFE [6]</td><td>5.02</td><td>27.79</td><td>49.81</td><td>27.54</td><td>5.85</td><td>28.91</td><td>51.08</td><td>28.61</td></tr><tr><td>EISAI [5]</td><td>5.66</td><td>27.64</td><td>49.43</td><td>27.57</td><td>6.02</td><td>29.14</td><td>52.36</td><td>29.17</td></tr><tr><td>FILM [23]</td><td>3.18</td><td>16.84</td><td>30.74</td><td>16.92</td><td>3.50</td><td>17.94</td><td>33.51</td><td>18.31</td></tr><tr><td>AnimeInbet (ours)</td><td>2.20</td><td>11.12</td><td>21.27</td><td>11.53</td><td>2.80</td><td>12.69</td><td>23.21</td><td>12.90</td></tr><tr><td>AnimeInbet-VS(ours)</td><td>2.62</td><td>11.43</td><td>22.36</td><td>12.14</td><td>3.44</td><td>13.41</td><td>23.67</td><td>13.51</td></tr></table>

## 4.4. Visibility Prediction and Graph Fusion

To handle occlusions in the source line arts, we use a three-layer MLP to predict binary visibility maps $m _ { 0 }$ and $m _ { 1 }$ for the input graphs, obtained as $m _ { 0 } =  { \mathrm { M L P } } ( \hat { F } _ { 0 } )$ and $m _ { 1 } = \mathbf { M L P } ( \hat { F } _ { 1 } )$ . Then, we merge the vertices to $V _ { t }$ in the two graphs according to the following rule:

$$
\begin{array} { r l } & { V _ { t } = \left\{ ( 1 - t ) V _ { 0 } [ i ] + t V _ { 1 } [ j ] \Big | ( i , j ) \in \hat { \mathcal { M } } \right\} } \\ & { \quad \cup \left\{ V _ { 0 } [ i ] + t \cdot r _ { 0 } [ i ] \Big | i \notin \hat { \mathcal { M } } , m _ { 0 } [ i ] = 1 \right\} } \\ & { \quad \cup \left\{ V _ { 1 } [ j ] + ( 1 - t ) r _ { 1 } [ j ] \Big | j \notin \hat { \mathcal { M } } , m _ { 1 } [ j ] = 1 \right\} , } \end{array}\tag{8}
$$

where we implement the repositioning that is compatible with arbitrary time $t \in ( 0 , 1 )$ ). As to $T _ { t } .$ , we union all original connections if both endpoint vectors are both visible in $G _ { t }$ . Or formally, $T _ { t } [ \widetilde { i } ] [ \widetilde { j } ] \stackrel { \cdot } { = } T _ { t } [ \widetilde { j } ] [ \widetilde { i } ] = 1 \mathrm { i f } T _ { 0 } [ i ] [ j ] = 1$ or $T _ { 1 } [ i ] [ j ] = 1$ , where $( i , j )$ and $( \widetilde { i } , \widetilde { j } )$ are the vertex indices in the original graph and the merged one.

## 4.5. Learning

The training objective of AnimeInbet composes of three terms: $\mathcal { L } = \mathcal { L } _ { c } + \mathcal { L } _ { r } + \mathcal { L } _ { m }$ , where the $\mathcal { L } _ { c } , \mathcal { L } _ { \ i }$ and $\mathcal { L } _ { m }$ are used to supervise the learning of vertex matching $\hat { \mathcal { M } }$ repositioning vectors $r _ { 0 }$ and $r _ { 1 }$ , and visibility masks $m _ { 0 }$ and $m _ { 1 }$ , respectively. $\mathcal { L } _ { c }$ is to enlarge the correlation values of ground truth pairs and is defined as:

$$
\mathcal { L } _ { c } = - \frac { 1 } { \vert \mathcal { M } ^ { G T } \vert } \sum _ { ( i , j ) \in \mathcal { M } ^ { G T } } \log \hat { \mathcal { P } } _ { i , j } ,\tag{9}
$$

where $\mathcal { M } ^ { G T }$ is the ground truth matching labels. For $\mathcal { L } _ { r }$ and $\mathcal { L } _ { m } ,$ we regress $r _ { 0 \to 1 } , r _ { 1 \to 0 } , m _ { 0 }$ , and $m _ { 1 }$ as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { r } = \displaystyle \frac { 1 } { K _ { 0 } } \| r _ { 0 \to 1 } - r _ { 0 \to 1 } ^ { G T } \| _ { 1 } + \frac { 1 } { K _ { 1 } } \| r _ { 1 \to 0 } - r _ { 1 \to 0 } ^ { G T } \| _ { 1 } } \\ & { \mathcal { L } _ { m } = \mathrm { B C E } ^ { w } \left( \sigma ( m _ { 0 } ) , m _ { 0 } ^ { G T } \right) + \mathrm { B C E } ^ { w } \left( \sigma ( m _ { 1 } ) , m _ { 1 } ^ { G T } \right) , } \end{array}\tag{10}
$$

where σ represents the sigmoid function, and $\mathrm { B C E } ^ { w }$ is the binary cross-entropy loss with bias weight w. However, since the shift vectors of occluded vertices cannot be obtained directly by subtracting the matched vertices, we conduct a frame-by-frame backtrack to generate pseudo labels to support the point-wise supervision of the repositioning vector and visibility maps.

Pseudo Labels of Repositioning and Visibility. Assume $G ^ { ( 0 ) }$ and $G ^ { ( Z ) }$ are the 0-th and the Z-th frames in a training sequence, which are used for two input line sources. $_ \mathrm { A l } .$ though there can exist many unmatched vertices in the two graphs when the gap Z is large, the matching rate between adjacent frames $( \mathrm { g a p } = 0 )$ is relatively high according to Table 1. Based on this, we iteratively backtrack a shift vector $r ^ { ( z ) }$ from the $G ^ { ( Z ) }$ to $G ^ { ( 0 ) }$

$r ^ { ( z ) } [ i ] = \left\{ \begin{array} { l } { V ^ { ( z + 1 ) } [ j ] - V _ { ( z ) } [ i ] + r ^ { ( z + 1 ) } , } \\ { \frac { 1 } { | \mathcal { N } _ { i } | } \sum _ { k \in \mathcal { N } _ { i } } r ^ { ( z ) } [ k ] , \ \mathrm { o t h e r w i s } } \end{array} \right.$ if $i , j$ is matched se

(11)

where ${ \mathcal { N } } _ { i }$ regards to the neighbors of the i-th vertex in $G ^ { ( z ) }$ and $r ^ { ( Z ) }$ is initialized to be 0. The termination $r ^ { ( 0 ) }$ of the backtrack is regarded as the pseudo repositioning label $r _ { 0  1 } ^ { G T }$ As to the visibility labels, we first deuce $r _ { 0  t } ^ { G T }$ as above and compute $m _ { 0 } ^ { G T }$ as

$$
m _ { 0 } ^ { G T } [ i ] = \{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f } V _ { 0 } [ i ] + r _ { 0  t } ^ { G T } \in \widetilde { I } _ { t } , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } , } } \end{array}\tag{12}
$$

where $\widetilde { I } _ { t }$ is $I _ { t }$ dilated by a $3 \times 3$ kernel. $r _ { 1  0 } ^ { G T }$ and $m _ { 1 } ^ { G T }$ are computed in reversed order.

## 5. Experiments

Implementation Details. In the vertex geometric embedding module, the image encoder $\mathcal { E } _ { I }$ is implemented as a three-layer 2D CNN, while the positional encoder $\mathcal { E } _ { P }$ and the topological encoder ${ \mathcal { E } } _ { T }$ are 1D CNNs with a kernel size of 1. Encoding feature C is 128 in our experiments. Before feeding vertex coordinates V into $\mathcal { E } _ { P } , V$ are first normalized to the scale between ( 1, 1); the dimension of the spectral embedding feature is 64. Threshold θ in Equation 5 is 0.2. In both training and evaluation, intermediate time t is 0.5, which regards the center frame between $I _ { 0 }$ and $I _ { 1 }$ . The detailed network structures are provided in the supplementary file. We use Adam [9] optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ to train the AnimeInbet for 70 epochs, where we first solely supervise the network using the correspondence loss $\mathcal { L } _ { c }$ for the 50 epochs, and then adopt the full loss for the rest 20 epochs. Bias weight w in ${ \mathcal { L } } _ { m }$ is 0.2. Since vertex numbers differ in frames, we feed one pair of input frames each time but adopt gradient accumulation for a mini-batch size of 8. The model is trained with an NVIDIA Tesla V100 GPU for about five days. During the test, $G _ { t }$ is visualized as a raster image by cv2.line function with a line width of 2 pixels. We evaluate our model on both ground truth vectorization labels (noted as “AnimeInbet”) and those vectorized from VirtualSketcher [15] (noted as “AnimeInbet-VS”, to simulate the cases when input anime drawing are vector and raster, respectively.

![](images/3a3b0f58c1f26cd785a9952175b6072fc2bfb9389c82b89594085de3dc723534.jpg)  
Figure 8: Inbetweening results on MixamoLine240 test set. Examples are arranged from small (top) to large (bottom) motion magnitudes.

Evaluation Metric. Following [16, 5], we adopt the chamfer distance (CD) as the evaluation metric, which has been initially introduced to measure the similarity between two point clouds. Formally, CD is computed as:

$$
C D ( I _ { t } , I _ { t } ^ { G T } ) = \frac { 1 } { H W d } \sum ( I _ { t } D T ( I _ { t } ^ { G T } ) + I _ { t } ^ { G T } D T ( I _ { t } ) ) ,\tag{13}
$$

where $I _ { t }$ and $I _ { t } ^ { G T }$ are predicted binary lines and ground truth, while H, W and d are image height, width, and a search diameter [5], respectively. DT denotes the Euclidean distance transform. To transfer predicted raster images into binary sketches, we threshold pixels smaller than 0.99 times the maximum value to 0.

## 5.1. Comparison to Existing Methods

Since there is no existing geometrized line inbetweening study that we can directly compare our proposed model with, we set several state-of-the-art raster-image-based frame interpolation methods as baselines, including VFIformer [14], RIFE [6], EISAI [5], FILM [23]. Specifically, EISAI is originally intended for 2D animation and embeds an optical flow-based contour aggregator. We test each model’s performance on frame pairs within frame gaps of 1, 5 and 9, respectively. For fairness, we finetune each compared method on the training set of MixiamoLine240 with relative frame gaps using a learning rate of $1 \times 1 0 ^ { - 6 }$ for five epochs.

User study on different frame gaps (ours wins)  
![](images/716548da56238cf274494db33fdc0751a03a9413f3b1d1e2eecd30573b1e29d1.jpg)  
Figure 9: Statistics of user study. In the boxplot, triangles and colored lines represent mean and median values, respectively. Circles are outliers beyond 1.5× interquartile range (3σ in a normal distribution).

As shown in Table 2, our AnimeInbet favorably outperforms all compared methods on both the validation set and the test set of MixamoLine240. On the validation set, our approach achieves an average CD value of 11.53, representing a significant improvement over the best-performing compared method, FILM, with over 30% enhancement. Upon closer inspection, the advantage of AnimeInbet becomes more pronounced as the frame gap increases (0.98, 5.72 and 9.47 for gaps of 1, 5, and 9, respectively), indicating that our method is more robust in handling larger motions. On the test set, our method maintains its lead over the other compared methods, with improvements of 0.70 (20%), 5.25 (29%), and 10.30 (31%) from the best-performing compared method FILM for the frame gaps of 1, 5, and 9, respectively. Given that both the characters and actions in the test set are new, our method’s superiority on the test set provides more convincing evidence of its advantages over the existing frame interpolation methods.

To illustrate the advantages of our method, we present several inbetweening results in Figure 8. We arranged these examples in increasing levels of difficulty from top to bottom. When the motion is simple, compared methods can interpolate a relatively complete shape of the main body of the drawing. However, they tend to produce strong blurring (RIFE) or disappearance (VFIformer, EISAI, and FILM) of noticeable moving compositions (indicated by red arrows). In contrast, our method maintains a concise line structure in these key areas. When the input frames involve the whole body’s movement within large magnitudes, the intermediate frames predicted by the compared methods become indistin guishable and patchy, rendering the results invalid for further use. However, our AnimeInbet method can still preserve the general shape in the correct positions, even with a partial loss of details, which can be easily fixed with minor manual effort.

Table 3: Ablation study on vertex encoding.
<table><tr><td> $\mathcal { E } _ { I }$ </td><td> $\mathcal { E } _ { P }$ </td><td> ${ \mathcal { E } } _ { T }$ </td><td>Acc. (%)</td><td>Valid Acc. (%)</td><td>CD (↓)</td></tr><tr><td>√</td><td>X</td><td>X</td><td>51.66</td><td>31.01</td><td>12.30</td></tr><tr><td>√</td><td>√</td><td>X</td><td>61.87</td><td>55.62</td><td>11.55</td></tr><tr><td>√</td><td>x</td><td>√</td><td>59.28</td><td>45.45</td><td>11.86</td></tr><tr><td>√</td><td>√</td><td>√</td><td>65.51</td><td>61.28</td><td>11.12</td></tr></table>

Table 4: Ablation study on repositioning and visibility mask.
<table><tr><td>Method</td><td>CD (↓)</td></tr><tr><td>w/o. repositioning propagation</td><td>23.62</td></tr><tr><td>w/o. visibility mask</td><td>12.81</td></tr><tr><td>full model</td><td>11.12</td></tr></table>

User Study. To further evaluate the visual performance of our methods, we conduct a user study among 36 participants. For each participant, we randomly show 60 pairs, each composed of a result of AnimeInbet and that of a compared method, and ask the participant to select the better. To allow participants to take temporal consistency into the decision, we display these results in GIF formats formed by triplets of input frames and the inbetweened one. The winning rates of our method are shown in Figure 9, where AnimeInbet wins over 92% versus the compared methods. Notably, for “gap $= 5 ^ { \prime \prime } { \mathrm { ~ a n d ~ } } ^ {  } { \mathrm { g a p } } = 9 ^ { \prime \prime }$ slots, the winning rates of our methods are close to 100% with smaller deviations than $\mathrm { \mathop { g a p } } ^ { * } = 1 \mathrm { \mathop { s } } ,$ suggesting the advantages of our method on cases within large motions.

## 5.2. Ablation Study

Embedding Features. To investigate the effectiveness of the three types of embeddings mentioned in Section 4.1, we trained several variants by removing the corresponding modules. As shown in Table 3, for each variant, we list the matching accuracy for all vertices (“Acc.”), the accuracy for non-occluded vertices (“Valid Acc.”) and the final CD values of inbetweening on the validation set $( \mathrm { g a p } = 5 )$ . If removing the positional embedding $\mathcal { E } _ { P }$ , the “Valid $\operatorname { A c c } ? $ and the CD value drop 15.83% and 0.74, respectively; while the lacking of topological embedding ${ \mathcal { E } } _ { T }$ lowers “Valid Acc.” by 5.66% and worsens CD by 0.43, which reveals the importance of these two components.

Repositioning Propagation and Visibility Mask. We demonstrate the contribution of repositioning propagation (prepos. prop.) and visibility mask (vis. mask) both quantitatively and qualitatively. As shown in Table $^ { 4 , }$ without repositioning propagation, the CD value will be sharply wors ened by 12.50 (112%), while the lacking of visibility mask will also make a drop of 1.69 (15%). An example is shown in Figure 10, where “w/o. repos. prop.” appears within many messy lines due to undefined positions for those unmatched vertices, while “w/o. vis. mask” shows some redundant segments (red box) after repositioning; the complete AnimeInbet can resolve these issues and produce a clean yet complete result.

![](images/7c0b7ea0ee74c9c1cede489d836eb7d9b610ab08b85b2e1e2ec76b6783a1fea1.jpg)  
Figure 10: Visualization of ablation study. In predicted correspondence, matched vertices are marked in the same colors, while unmatched are black (please zoom in).

Table 5: Ablation study on data influence.
<table><tr><td>Validation data (gap = 5)</td><td>Occ. (%)</td><td>Shift</td><td>CD (↓)</td></tr><tr><td>Unseen characters (2 × 10)</td><td>34.30</td><td>44.59</td><td>14.70</td></tr><tr><td>Unseen actions (10 × 2)</td><td>37.71</td><td>31.53</td><td>8.98</td></tr><tr><td>Unseen both (2 × 2)</td><td>34.10</td><td>29.62</td><td>9.98</td></tr></table>

Geometrizor. As shown in Table 2, the quantitative metrics of AnimeInbet-VS are generally worse by around 0.6 com pared to AnimeInbet. This is because VirtualSketcher [15] does not vectorize the line arts as precisely as the ground truth labels (average vertex number 587 vs 1,351). As shown in Figure 10, the curves in “AnimeInbet-VS” become sharper and lose some details, which decreases the quality of the inbetweened frame. Using a more accurate geometrizer would lead to higher quality inbetweening results for raster image inputs.

Data Influence. As mentioned in Section 3, we created a validation set composed of 20 sequences of unseen characters but seen actions, 20 of unseen actions but seen characters and 4 of unseen both to explore the influence on data. Our experiment finds that whether the characters or the actions are seen does not fundamentally influence the inbetweening quality, while the motion magnitude is the key factor. As shown in Table 5, the CD value of unseen characters is 14.70, which is over 47% worse than that of unseen both due to larger vertex shifts (44.59 vs 29.62), while the difference between the CD values of unseen actions and unseen both is around 10% under similar occlusion rates and shifts.

## 6. Conclusion

In this study, we address the practical problem of cartoon line inbetweening and propose a novel approach that treats line arts as geometrized vector graphs. Unlike previous frame interpolation tasks on raster images, our approach formulates the inbetweening task as a graph fusion problem with vertex repositioning. We present a deep learning-based framework called AnimeInbet, which shows significant gains over existing methods in terms of both quantitative and qual itative evaluation. To facilitate training and evaluation on cartoon line inbetweening, we also provide a large-scale ge ometrized line art dataset, MixamoLine240. Our proposed framework and dataset facilitate a wide range of applications, such as anime production and multimedia design, and have significant practical implications.

Acknowledgement. This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG-PhD/2021-01- 031[T]). It is also supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). This study is partially supported by NTU NAP, MOE AcRF Tier 1 (2021-T1-001-088).

## References

[1] Mixamo. https://www.mixamo.com/. 3

[2] Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps for dimensionality reduction and data representation. Neural computation, 15(6):1373–1396, 2003. 5

[3] Leonardo Carvalho, Ricardo Marroquim, and Emilio Vital Brazil. Dilight: Digital light table–inbetweening for 2d animations using guidelines. Computers & Graphics, 2017. 3

[4] Evan Casey, V´ıctor Perez, and Zhuoru Li. The animation´ transformer: Visual correspondence via segment matching. In ICCV, 2021. 1, 3

[5] Shuhong Chen and Matthias Zwicker. Improving the perceptual quality of 2d animation interpolation. In ECCV, 2022. 1, 2, 6, 7

[6] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In ECCV, 2022. 1, 2, 6, 7

[7] Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik Learned-Miller, and Jan Kautz. Super SloMo: High quality estimation of multiple intermediate frames for video interpolation. In CVPR, 2018. 2

[8] Kangyeol Kim, Sunghyun Park, Jaeseong Lee, Sunghyo Chung, Junsoo Lee, and Jaegul Choo. Animeceleb: Largescale animation celebheads dataset for head reenactment. In ECCV, 2022. 3

[9] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2014. 7

[10] Johannes Kopf and Dani Lischinski. Digital reconstruction of halftoned color comics. ACM TOG, 31(6), 2012. 1, 3

[11] Xiaoyu Li, Bo Zhang, Jing Liao, and Pedro V. Sander. Deep sketch-guided cartoon video inbetweening. TVCG, 2020. 2

[12] Songtao Liu, Jin Huang, and Hao Zhang. End-to-end line drawing vectorization. In AAAI, 2022. 3, 4

[13] Ziwei Liu, Raymond A Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In CVPR, 2017. 2

[14] Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, and Jiaya Jia. Video frame interpolation with transformer. In CVPR, 2022. 1, 2, 6, 7

[15] Haoran Mo, Edgar Simo-Serra, Chengying Gao, Changqing Zou, and Ruomei Wang. General virtual sketching framework for vector line art. In SIGGRAPH, 2021. 3, 4, 7, 9

[16] Rei Narita, Keigo Hirakawa, and Kiyoharu Aizawa. Optical flow based line drawing frame interpolation using distance transform to support inbetweenings. In ICIP, 2019. 7

[17] Simon Niklaus and Feng Liu. Context-aware synthesis for video frame interpolation. In CVPR, 2018. 2

[18] Simon Niklaus and Feng Liu. Softmax splatting for video frame interpolation. In CVPR, 2020. 2

[19] Simon Niklaus, Long Mai, and Feng Liu. Video frame interpolation via adaptive convolution. In CVPR, 2017. 2

[20] Simon Niklaus, Long Mai, and Feng Liu. Video frame interpolation via adaptive separable convolution. In ICCV, 2017. 2

[21] Junheum Park, Keunsoo Ko, Chul Lee, and Chang-Su Kim. Bmbc: Bilateral motion estimation with bilateral cost volume for video interpolation. In ECCV, 2020. 2

[22] Yingge Qu, Tien-Tsin Wong, and Pheng-Ann Heng. Manga colorization. ACM TOG, 25(3), 2006. 1, 3

[23] Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. Film: Frame interpolation for large motion. In ECCV, 2022. 1, 2, 6, 7

[24] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020. 5

[25] Maria Shugrina, Ziheng Liang, Amlan Kar, Jiaman Li, Angad Singh, Karan Singh, and Sanja Fidler. Creative flow+ dataset. In CVPR, 2019. 3

[26] Hyeonjun Sim, Jihyong Oh, and Munchurl Kim. Xvfi: extreme video frame interpolation. In ICCV, 2021. 2

[27] Edgar Simo-Serra, Satoshi Iizuka, and Hiroshi Ishikawa. Mastering sketching: Adversarial augmentation for structured prediction. ACM TOG, 37(1), 2018. 2

[28] Edgar Simo-Serra, Satoshi Iizuka, Kazuma Sasaki, and Hiroshi Ishikawa. Learning to simplify: Fully convolutional networks for rough sketch cleanup. ACM TOG, 35(4), 2016. 2

[29] Li Siyao, Yuhang Li, Bo Li, Chao Dong, Ziwei Liu, and Chen Change Loy. Animerun: 2d animation visual correspondence from open source 3d movies. In NeurIPS, 2022. 3

[30] Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, and Ziwei Liu. Deep animation video interpolation in the wild. In CVPR, 2021. 2

[31] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, 2021. 5

[32] D. Sykora, J. Buri \` anek, and J. ´ Z<sup>ˇ</sup> ara. Unsupervised coloriza- ´ tion of black-and-white cartoons. In Int. Symp. NPAR, 2004. 1, 3

[33] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In CVPR, 2022. 5

[34] Xiangyu Xu, Li Siyao, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang. Quadratic video interpolation. In NeurIPS, 2019. 2

[35] Wenwu Yang. Context-aware computer aided inbetweening. IEEE TVCG, 24(2):1049–1062, 2017. 3

[36] Chih-Yuan Yao, Shih-Hsuan Hung, Guo-Wei Li, I-Yu Chen, Reza Adhitya, and Yu-Chi Lai. Manga vectorization and manipulation with procedural simple screentone. IEEE TVCG, 23(2), 2016. 3, 4

[37] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 3

[38] Lvmin Zhang, Jinyue Jiang, Yi Ji, and Chunping Liu. Smartshadow: Artistic shadow drawing tool for line drawings. In ICCV, 2021. 1, 3

[39] Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. Two-stage sketch colorization. In SIGGRAPH, 2018. 1, 3

[40] Song-Hai Zhang, Tao Chen, Yi-Fei Zhang, Shi-Min Hu, and Ralph R. Martin. Vectorizing cartoon animations. IEEE TVCG, 15(4), 2009. 3, 4