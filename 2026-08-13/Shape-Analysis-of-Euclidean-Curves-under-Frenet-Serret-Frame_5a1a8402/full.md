# Shape Analysis of Euclidean Curves under Frenet-Serret Framework

Perrine Chassat<sup>1</sup> <sup>1</sup>LaMME, University of Paris-Saclay perrine.chassat@univ-evry.fr

## Abstract

Geometric frameworks for analyzing curves are common in applications as theyfocus on invariantfeatures and provide visually satisfying solutions to standard problems such as computing invariant distances, averaging curves, or registering curves. We show that for any smooth curve in R<sup>d</sup>, d > 1, the generalized curvatures associated with the Frenet-Serret equation can be used to define a Riemannian geometry that takes into account all the geometricfeatures ofthe shape. This geometry is based on a Square Root Curvature Transform that extends the square root-velocity transform for Euclidean curves (in any dimensions) and provides likely geodesics that avoid artefacts encountered by representations using only first-order geometric information. Our analysis is supported by simulated data and is especially relevant for analyzing human motions. We consider trajectories acquired from sign language, and show the interest of considering curvature and also torsion in their analysis, both being physically meaningful.

## 1. Introduction

Identifying and comparing different types of visual objects is a fundamental task in machine learning and computer vision problems [13, 5, 8]. The shape is one of the essential features of objects that allow us to understand and characterize them. Nowadays, it is much easier to obtain data in the form of shapes, typically as dense point clouds or landmarks. The main task in shape analysis is to define a proper framework to compare and quantify the variation of the shapes. However, the shape space is generally nonlinear, and extracting meaningful information or features is complex. One of the successful approaches to shape analysis utilizes a Riemannian framework of differential geometry, where a metric can be defined between the shapes, which is invariant with respect to shape-preserving transformations such as translation and rotation. For instance, this gives rise to geodesic distances that are naturally invariant to smooth and optimal deformations through geodesic paths between the shapes [11]. This approach is very versatile as it can

Juhyun Park<sup>1,2</sup> <sup>2</sup>ENSIIE, Evry juhyun.park@ensiie.fr

Nicolas Brunel<sup>1,2,3</sup> <sup>3</sup>Quantmetry, Paris nicolas.brunel@ensiie.fr

be adapted to various kinds of manifold-value data and can be designed to emphasize important geometric information to be preserved. As a consequence, several choices of metrics are possible, such as the class of invariant Sobolev metrics, often called elastic, for the analysis of curves [1]. In this work, we are concerned with curves that often arise in the application as trajectories (function of time) or motions (animation, activity recognition) and with the definition of a framework to compare their shapes. The differential geometry of Euclidean curves is among the simplest (with respect to higher dimensional manifolds), and relatively simple Riemannian metrics are available with different mathematical representations of curves [28]. Quite remarkably, the introduction of the Square Root Velocity (SRV) transform [24] that consists of a particular representation of the shape of a curve enables to define a so-called elastic Riemannian distance, which has proven to be useful for the statistical shape analysis of 2D and 3D curves in applications. The SRV possesses interesting properties such as a principled theoretical framework, efficient computation, and generalization to higher dimensions [2]. Nevertheless, a limitation of the SRV transform and the corresponding elastic distance is the restrictive use of the first-order derivative, while the geometry of 3-D (or d-D) curves depends on the derivatives until order d. Indeed, it is well-known that a 3D curve is characterized by its curvature and its torsion: this is particularly critical when we consider trajectories or human movements, where the curvature and torsion can have a physical meaning.

## 2. Related works and contributions

As we will recall in section 3, the full geometry of a curve can be given either by the Frenet curvatures (standard curvature and torsion in 3D) or by the path of Frenet frames. There have been few attempts to directly deal with the Frenet curvatures: most of the works have been produced in 2D curves as an alternative representation [23]. Nevertheless, the potential for applications has not been investigated. In [17], the elastic shape analysis framework has been considered for 3D curves based on the Frenet frames, but the link to the physical parameters has been overlooked.

We can also mention the shape analysis of curves on Lie groups [4] with application in computer animation. Outside the Riemannian framework, an attempt has been made to use a direct curvature-based interpolation of curves [20, 25].

In this work, we introduce two representations of Euclidean curves for their shape analysis that use their complete geometry through Frenet curvatures. We provide the full development of the Riemannian frameworks associated with these two representations. As a consequence and in comparison with existing methods, our approaches also give explicit formulas for geodesics and geodesic distances. The first representation considered is based directly on unparametrized Frenet curvatures. We show through experiments that it defines a shape analysis framework that lacks elasticity. As the main contribution, we propose the definition of a second representation, called the Square-Root Curvature (SRC) Transform, which takes into account reparameterization and defines a metric on the space of shapes through the quotient space with the group of diffeomorphisms. One can imagine that the classical method associated with the SRVF, defining a Sobolev elastic metric [2], already implicitly uses all the geometric information necessary for a relevant curve analysis. We show here with simple examples that this is not the case. Through experiments on synthetic data, we compare the methods, and illustrate the limitations of the SRVF one, due to its lack of use of geometric information. To be able to judge and compare the quality of these metrics, we compare consistent sets of curves characterized by specific features. The SRC method shows a special strength in defining a framework that remains consistent with these sets. The straightforward example of geodesics between helices with different numbers of spins (Figure 1 in 2D and Figure 2 in 3D) shows that, in contrast, this is not the case for the SRVF method. In addition, we highlight the interest of these Frenet curvaturesbased representations in the real application case of human motion trajectory analysis.

## 3. Riemannian Geometry on Shape Space

We introduce useful notations and we review the main approach for constructing tractable representations of the shape of a curve and deriving a Riemannian geometry.

## 3.1. Shape Analysis of Euclidean Curves

We consider absolutely continuous curves that are smooth, open, and with values in some Euclidean space $\mathbb { R } ^ { d }$ we denote this set as $A C \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ . These curves are typically parametrized by a variable t that can usually be interpreted as time. Nevertheless, from a (statistical) shape analysis point of view, we focus on the geometric shape of curves that do not depend on a specific parametrization or standard transformations such as translations, rotations, scaling, or reparametrizations. To distinguish between parametrized curves that differ only by translation, we consider the set of absolutely continuous curves where $x ( 0 ) = 0$ , denoted by $A C _ { 0 } \left( [ 0 , \dot { 1 } ] , \mathbb { R } ^ { d } \right)$ . The natural and intrinsic parametrization that uniquely defines the shape of a curve x is the arc-length parametrization, defined with the arc length function $\begin{array} { r } { s ( t ) = \int _ { 0 } ^ { t } \| \dot { x } ( u ) \| d u , } \end{array}$ for $t \in [ 0 , 1 ]$ . In order to remove the scaling variability, the total length of the curve $s ( 1 )$ is set to 1. Under this parametrization, the shape $X : [ 0 , 1 ] \mapsto \mathbb { R } ^ { d }$ of the curve is the image of the function x such that $x ( t ) = X ( s ( t ) )$ . As we want to study shapes independently of their parameterizations, we introduce the reparametrization group $\mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ , of smooth orientation preserving diffeomorphisms of the interval [0, 1] onto itself. This group acts on the space of absolutely continuous curves by right composition, and this action only alters the parametrization of the curve, not the inherent shape $X .$ . The space of such shapes (or unparametrized curves) is often mathematically defined as the quotient space

$$
\begin{array} { r } { S ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) = A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right) / \mathrm { D i f f } _ { + } ( [ 0 , 1 ] ) . } \end{array}\tag{1}
$$

The purpose of shape analysis of curves is to define a distance function $d _ { S }$ on S and a framework to perform a complete statistical analysis on a set of curves in S (e.g. mean, classification, or Principal Component Analysis etc.). One of the main challenges in defining this distance is to choose an appropriate mathematical representation of the curves that can be made invariant to all shape-preserving transformations - translation, rotation, scaling, and reparametrization. Moreover, one of the stakes of such representation is to offer an (infinite-dimensional) Riemannian manifold structure that brings powerful and flexible tools for studying the geometry of shapes or statistical properties notably thanks to the tangent space of the manifold [11, 22]. In [23], a list of the few possible representations is given - coordinate functions, curvatures, angle function, and squareroot velocity function (SRVF) - and a framework for curve analysis is derived for the last two ones. While the angle representation is unparameterized, the SRVF representation depends on the parametrization, which is shown to be very useful as a tool for the registration of points across curves. As a consequence, the parameterization group Diff<sub>+</sub>([0, 1]) must be eliminated by using a quotient space. The classical approach is to define the Riemannian metric on the shape space through a metric on the space of parametrized representations that is invariant to reparametrization: $\forall h \in$ Diff<sub>+</sub>([0, 1])

$$
d _ { A C _ { 0 } } ( x _ { 0 } , x _ { 1 } ) = d _ { A C _ { 0 } } ( x _ { 0 } \circ h , x _ { 1 } \circ h ) .\tag{2}
$$

In that case, the distance on $s$ is defined as the infimum over all possible reparametrization. For $X _ { 0 } , X _ { 1 } \in { \mathcal { S } } _ { : }$

$$
d _ { S } ( X _ { 0 } , X _ { 1 } ) : = \operatorname* { i n f } _ { h \in \operatorname { D i f f } ( [ 0 , 1 ] ) } d _ { A C _ { 0 } } ( x _ { 0 } , x _ { 1 } \circ h ) .\tag{3}
$$

![](images/96e8c8246e15fa2065396877b7bfb640978b85615486db832798ddeb0bac5c6b.jpg)  
Figure 1: Geodesic paths between two 2D scaled spirals with different number of spins: SRVF $( 1 ^ { s t }$ row) and $\mathrm { S R C } \left( 2 ^ { n d } \mathrm { r o w } \right)$ .

In the following, we will denote with a dot the derivation with respect to the time variable, and with a prime the one with respect to the arc-length parameter.

## 3.2. Square Root Velocity Framework

The square-root velocity function framework is the most commonly used representation for curve shape analysis in $\mathbb { R } ^ { d } [ 2 4 , 2 3 ]$ . The square-root velocity function (SRVF) of $x \in A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ , denoted by $\mathcal { R } _ { \mathrm { S R V F } } ( x ) = q ,$ , is defined as

$$
q ( t ) = \frac { \dot { x } ( t ) } { \sqrt { \| \dot { x } ( t ) \| } } = \sqrt { \dot { s } ( t ) } T ( s ( t ) )\tag{4}
$$

where $\begin{array} { r } { T ( s ( t ) ) \ = \ \frac { \dot { x } ( t ) } { \| \dot { x } ( t ) \| } \ = \ X ^ { \prime } ( s ( t ) ) } \end{array}$ is the unit tangent vector of the curve. This transformation is a bijection with $A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ and its explicit inverse is $\begin{array} { r } { x ( t ) = \int _ { 0 } ^ { t } q ( u ) | q ( u ) | d u . } \end{array}$ . As we consider length-normalized curves, the SRVFs have a unit $\mathbb { L } ^ { 2 }$ norm, and their set is the convenient unit Hilbert sphere, a Riemannian submanifold of $\mathbb { L } ^ { 2 } ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ (with the $\mathbb { L } ^ { 2 }$ inner product). Then, the $\mathbb { L } ^ { 2 }$ metric on SRVF induces a Riemannian metric on $A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ where geodesics are given by the shorter arcs on great circles between SRV functions. The action of Diff ([0, 1]) on $A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ is reflected on q by the group action denoted by ∗ and defined as

$$
( q * h ) ( t ) = \sqrt { \dot { h } ( t ) } q ( h ( t ) )\tag{5}
$$

and if the curve is rotated by a matrix $O \in S O ( d )$ , its SRVF gets rotated by the same matrix. The key property of this representation is the invariance of its associated distance under the action of Diff<sub>+</sub>([0, 1]) and $S O ( d )$

$$
\| O ( q _ { 0 } * h ) - O ( q _ { 1 } * h ) \| _ { \mathbb { L } ^ { 2 } } = \| q _ { 0 } - q _ { 1 } \| _ { \mathbb { L } ^ { 2 } } .\tag{6}
$$

The metric can then be used to define a proper distance on the shape space $S ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$

$$
d _ { S } ^ { ( \mathrm { S R V F } ) } ( X _ { 0 } , X _ { 1 } ) : = \operatorname* { i n f } _ { \stackrel { O \in S O ( d ) } { h \in \mathrm { D i f f } ( [ 0 , 1 ] ) } } \ \cos ^ { - 1 } \langle q _ { 0 } , O ( q _ { 1 } * h ) \rangle\tag{7}
$$

and the geodesic path on the shape space is taken between $q _ { 0 }$ and $q _ { 1 } * h$

The definition of this distance on the shape space under the SRVF representation can be interpreted as the following registration problem

$$
h ^ { * } , O ^ { * } = \ \underset { { O \in S O ( d ) } } { \arg \operatorname* { m i n } } \ \int _ { 0 } ^ { 1 } \| q _ { 0 } ( t ) - O q _ { 1 } ( h ( t ) ) \sqrt { \dot { h } ( t ) } \| _ { 2 } ^ { 2 } d t .\tag{8}
$$

In [3], this registration problem has been reformulated with the unit tangent vector and the arc length functions. By defining $\gamma = s _ { 1 } \circ h \circ s _ { 0 } ^ { - 1 } \in \operatorname { D i f f } _ { + } ( [ 0 , 1 ] )$ the optimization problem 8 amounts to finding the optimal diffeomorphism of $\mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ that acts on the arc-length parameter s and solves the minimization problem:

$$
\gamma ^ { * } , O ^ { * } = \ \underset { O \in S O ( d ) } { \arg \operatorname* { m i n } } \ \int _ { 0 } ^ { 1 } \| T _ { 0 } ( s ) - O T _ { 1 } ( \gamma ( s ) ) \sqrt { \gamma ^ { \prime } ( s ) } \| _ { 2 } ^ { 2 } d s .\tag{9}
$$

It should be noted, in this reformulation, that the object $O T _ { 1 } ( \gamma ( s ) ) \sqrt { \gamma ^ { \prime } ( s ) }$ does not represent the same shape as $X _ { 1 } ( s )$ in the shape space. Here the element $\gamma$ of Diff<sub>+</sub>([0, 1]) is not used as a reparametrization of the curve but to deform the element of $\bar { \mathcal { S } ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) }$ . Under this point of view, the set of unit tangent vectors that can be reached by deforming the vector $T ( s )$ , with the group action $T * \gamma$ defines an equivalence class of shapes associated with that one, as in the setting of deformable templates of Grenander’s theory [29, 30].

Finally, the choice of a parametrized curve representation for shape analysis, discussed in [23], can be seen as the problem of choosing a good geometric representative of the shape as a template and defining an associated registration problem. Hence, an appropriate choice may be seen as a matter of modeling and should be done in interaction with the type of data analyzed and the dimension of the space. In the next sections, we use h to refer to functions of Diff<sub>+</sub>([0, 1]) that act on the time variable t and $\gamma$ for ones that act on the arc-length variable s.

![](images/592bb52999e74eb6cd8d49a3239ed955cf1b069b97b09cbc74e1cab2a04712e6.jpg)  
Figure 2: Geodesic paths between two scaled 3D circular helices with different number of spins: SRVF $( { 1 } ^ { s t } \ \mathrm { r o w } )$ and SRC $( 2 ^ { n d } \mathrm { r o w } )$ .

## 4. Exhaustive geometric information with Frenet representation

Based on these previous observations and with the intention of developing a more suitable framework for the analysis of three-dimensional curves, Brunel and Park [3] proposed a direct extension of the SRVF method, which considers not only the tangent vector as the geometric representation of the curve but the whole Frenet-Serret frame in three dimensions. Their idea is to use an exhaustive description of the geometry of curves by incorporating higher-order information about the geometry in the representation. To exploit this idea, we propose to study suitable representations based on this Frenet-Serret framework. Proofs of theoretical results are given in the supplementary material.

## 4.1. The Frenet-Serret framework

We introduce the Frenet-Serret framework for curves of any dimension d. Let $F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ be the set of curves $x \in A C _ { 0 } \left( [ 0 , 1 ] , \mathbb { R } ^ { d } \right)$ d-times continuously differentiable, and with the first d derivatives linearly independent. $F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ is called the set of Frenet curves. In the following, we will restrict the shape space to be the set

$$
\begin{array} { r } { S ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) = F ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) / \mathrm { D i f f } _ { + } ( [ 0 , 1 ] ) . } \end{array}\tag{10}
$$

The Frenet frame $e _ { 1 } , e _ { 2 } , \ldots , e _ { d }$ associated with $X \in$ $S ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ is uniquely defined by applying the Gram-Schmidt process to the first d derivatives of X. In dimension 3, the three vectors of the Frenet frame are known as the tangent, normal and bi-normal vector. We define the function $Q$ that maps to $s \in [ 0 , 1 ]$ along the curve the corresponding Frenet frame

$$
Q ( s ) = [ e _ { 1 } ( s ) \mid e _ { 2 } ( s ) \mid \ldots \mid e _ { d } ( s ) ] .\tag{11}
$$

The function $Q$ is a measurable curve from $[ 0 , 1 ]$ to the Lie group of rotation matrices $S O ( d )$ called the Frenet path.

Theorem 1 (Frenet-Serret equation [9]). Let $X \_ { \in }$ $S ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ and $Q ( s )$ its associated Frenet path. Then there are functions $\theta _ { 1 } , \ldots , \theta _ { d - 1 }$ defined on that curve with $\theta _ { 1 } , \ldots , \theta _ { d - 2 } > 0 ,$ , so that every $\theta _ { i } i s ( d - 1 - i )$ -times continuously differentiable and

$$
Q ^ { \prime } ( s ) = Q ( s ) A _ { \pmb { \theta } } ( s )\tag{12}
$$

where $\pmb { \theta } ( s ) = ( \theta _ { 1 } ( s ) , \ldots , \theta _ { d - 1 } ( s ) ) ^ { T }$ and

$$
A _ { \theta } ( s ) = \left[ \begin{array} { c c c c c c } { { 0 } } & { { - \theta _ { 1 } ( s ) } } & { { 0 } } & { { \dots } } & { { 0 } } \\ { { } } & { { } } & { { } } & { { } } & { { \vdots } } \\ { { \theta _ { 1 } ( s ) } } & { { 0 } } & { { - \theta _ { 2 } ( s ) } } & { { \ddots } } & { { } } & { { \vdots } } \\ { { } } & { { } } & { { } } & { { \ddots } } & { { } } & { { } } \\ { { 0 } } & { { \theta _ { 2 } ( s ) } } & { { } } & { { \ddots } } & { { } } & { { 0 } } \\ { { \vdots } } & { { \ddots } } & { { \ddots } } & { { } } & { { 0 } } & { { - \theta _ { d - 1 } ( s ) } } \\ { { 0 } } & { { \dots } } & { { } } & { { 0 } } & { { \theta _ { d - 1 } ( s ) } } & { { 0 } } \end{array} \right]
$$

and $\theta _ { i }$ is called the i-th Frenet curvature, and the equation is called the Frenet-Serret equation.

In dimension 3, the two Frenet curvatures are known as $s \mapsto \kappa ( s )$ the curvature function and $s \mapsto \tau ( s )$ the torsion function. They have an interpretable physical meaning. The curvature function measures how sharply the curve changes direction at a given point, and the torsion function measures the degree to which the curve twists and turns as it moves along its path. The Frenet-Serret equation with an initial condition $Q ( 0 ) = Q _ { 0 }$ defines an ordinary differential equation on the Lie group $S O ( d )$ where the function $s \mapsto A _ { \pmb \theta } ( s )$ has values in the Lie algebra of skew-symmetric matrices. This equation can also be expressed in function of the time variable t as

$$
\frac { d Q ( s ( t ) ) } { d t } = \dot { s } ( t ) Q ( s ( t ) ) A _ { \theta } ( s ( t ) ) .\tag{13}
$$

Lemma 1. The Frenet curvatures and the Frenet path are invariant under all Euclidean motions.

This means that for $x \in F ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) , O \in S O ( d ) , a \in$ $\mathbb { R } ^ { d }$ and $h \in \mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ , the curve defined by $\tilde { x } ( t ) = a +$ $O x ( h ( t ) )$ has the same Frenet curvatures as x and ${ \tilde { Q } } ( s ) =$ $O Q ( s )$

Theorem 2 (Fundamental theorem of the local theory of curves [9]). Let $\theta _ { 1 } , . . . , \theta _ { d - 1 } \ \in \ C ^ { \infty } ( [ 0 , 1 ] , \mathbb { R } )$ such that $\theta _ { 1 } , \ldots , \theta _ { d - 2 } > 0 .$ . For a given $X _ { 0 } \in \mathbb { R } ^ { d }$ and $Q _ { 0 } \in S O ( d )$ there is a unique $X \in \mathcal { S } ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ parametrized by arc length and satisfying the following three conditions:

• X(0) = X<sub>0</sub>,

$Q _ { 0 }$ is the Frenet frame of X at point $s = 0 ,$

$\theta _ { 1 } , . . . , \theta _ { d - 1 }$ are the Frenet curvatures ofX.

Theorems 1 and 2 state that there is a bijection, up to a translation and a rotation, between the shape space $S ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ , the set of admissible Frenet curvatures H and the set of corresponding Frenet paths $\mathcal { F } _ { 0 }$ , where

$$
\mathcal { H } = \left. \pmb { \theta } \in C ^ { \infty } ( [ 0 , 1 ] , \mathbb { R } ^ { d - 1 } ) | \theta _ { 1 } , \dots , \theta _ { d - 2 } > 0 \right. ,\tag{14}
$$

$$
\mathcal { F } _ { 0 } = \left\{ \begin{array} { c } { Q \in \mathbb { L } ^ { 2 } ( [ 0 , 1 ] , S O ( d ) ) \mathrm { ~ s u c h ~ t h a t ~ } } \\ { Q ^ { \prime } ( s ) = Q ( s ) A _ { \theta } ( s ) , Q ( 0 ) = I _ { d } , \pmb { \theta } \in \mathcal { H } } \end{array} \right\}\tag{15}
$$

From the detailed Frenet-Serret framework, one can think of the direct extension of the square-root velocity function 4 that simply consists in replacing the tangent vector with the entire Frenet frame. The representation of a parametrized curve $x \in F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ will be then

$$
\mathcal { R } _ { Q } ( x ) ( t ) = \sqrt { \dot { s } ( t ) } Q ( s ( t ) ) .\tag{16}
$$

This representation is used in [3] to define a new alignment method on S. They extend the SRVF registration problem 9 by using the Frobenius distance between the Frenet frames instead of only the $\mathbb { L } ^ { 2 }$ distance of the unit tangent vectors. They show to obtain more precise results with their method than the SRVF one. From the previous theorems, it is clear that this representation uniquely defines a parametrized curve $x \in F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ . However, as a result of the Frobenius theorem [12], the set of such representations $\mathcal { R } _ { Q }$ appears not to be a manifold. We leave the demonstration in the supplementary material.

## 4.2. Unparametrized Frenet curvatures

A possible representation of a parametrized curve, already suggested in [23, 25, 20], which keeps the idea of encoding more geometric information, is the unparametrized Frenet curvatures and the arc-length function pair

$$
\mathcal { R } _ { \pmb { \theta } } ( \boldsymbol { x } ) ( t ) = \left( \sqrt { \dot { s } ( t ) } , \pmb { \theta } ( s ( t ) ) \right) .\tag{17}
$$

We denote $\Psi ( [ 0 , 1 ] )$ , the set of square root velocity functions of length-normalized arc-length functions. This set is well-studied in the literature [15, 27]. It is the unit sphere of the Hilbert space $\mathbb { L } ^ { 2 } ( [ 0 , 1 ] , \mathbb { R } )$ and therefore a Riemannian manifold equipped with the $\mathbb { L } ^ { 2 }$ metric. Then, the geodesic distance between two elements in $\Psi ( [ 0 , 1 ] )$ is

$$
d _ { \Psi } \left( \sqrt { \dot { s } _ { 0 } } , \sqrt { \dot { s } _ { 1 } } \right) = \cos ^ { - 1 } \left( \left. \sqrt { \dot { s } _ { 0 } } , \sqrt { \dot { s } _ { 1 } } \right. \right)\tag{18}
$$

and the geodesic path connecting them is given by

$$
\alpha _ { \Psi } ( \tau ) = { \frac { \sin ( ( 1 - \tau ) \vartheta ) } { \sin ( \vartheta ) } } \sqrt { \dot { s } _ { 0 } } + { \frac { \sin ( \tau \vartheta ) } { \sin ( \vartheta ) } } \sqrt { \dot { s } _ { 1 } }\tag{19}
$$

where $\begin{array} { r } { \vartheta \ = \ d _ { \Psi } ( \sqrt { \dot { s } _ { 0 } } , \sqrt { \dot { s } _ { 1 } } ) } \end{array}$ . Moreover, any element of Diff<sub>+</sub>([0, 1]) is uniquely represented by an element of $\Psi ( [ 0 , 1 ] )$

Proposition 1. The set of Frenet curvatures H is a Riemannian submanifold of $\mathbb { L } ^ { 2 } ( [ 0 , 1 ] , \mathbb { R } ^ { d - 1 } )$ .

Proof. The set $M = \{ x \in \mathbb { R } ^ { d - 1 } | x _ { 1 } , \dots , x _ { d - 2 } > 0 \}$ is a open subset of the Riemannian manifold $\mathbb { R } ^ { d - 1 }$ . Then it is itself a differentiable Riemannian manifold with the standard inner product of $\mathbb { R } ^ { d - 1 }$ , and for any point $p \in M$ the tangent space $T _ { p } ( M )$ is $\mathbb { R } ^ { d - 1 }$ . The set of Frenet curvatures $\mathcal { H }$ is the set of measurable curves from [0, 1] to the Riemannian manifold M and thus also a manifold ([26]). Its tangent space is $\mathbb { L } ^ { 2 } ( [ 0 , 1 ] , \mathbb { R } ^ { d - 1 } )$ and it can be equipped with the $\bar { \mathbb { L } ^ { 2 } }$ Riemannian metric. □

Consequently, the geodesic distance on H is simply the $\mathbb { L } ^ { 2 }$ norm

$$
d _ { \mathcal { H } } ( \pmb { \theta } _ { 0 } , \pmb { \theta } _ { 1 } ) = \lVert \pmb { \theta } _ { 0 } - \pmb { \theta } _ { 1 } \rVert _ { \mathbb { L } ^ { 2 } }\tag{20}
$$

and the geodesic path is the straight line connecting them

$$
\begin{array} { r } { \alpha _ { \mathcal { H } } ( \tau ) = ( 1 - \tau ) \pmb { \theta } _ { 0 } + \tau \pmb { \theta } _ { 1 } . } \end{array}\tag{21}
$$

Proposition 2. The map $\mathcal { R } _ { \pmb { \theta } } : F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )  \Psi ( [ 0 , 1 ] ) \times$ H, defined above, is a bijection.

Proof. The element of $\Psi ( [ 0 , 1 ] )$ uniquely defines the arclength function by $\begin{array} { r } { s ( t ) = \int _ { 0 } ^ { t } ( \sqrt { \dot { s } ( u ) } ) ^ { 2 } d u } \end{array}$ . As mentioned before, we have a bijection between the unparametrized Frenet curvatures in H and the unparametrized curve in the shape space $\textit { S } ( 1 , 2 )$ . Then, from $X \in S$ , the initial parametrized curve is simply $x ( t ) = X ( s ( t ) )$ □

The set of such $\scriptstyle { \mathcal { R } } _ { \theta }$ is the Cartesian product of $\Psi ( [ 0 , 1 ] )$ and H and, therefore, is also a Riemannian manifold equipped with the product metric d ⊕ $d _ { \mathcal { H } }$ [16]. The induced metric on $\bar { F ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) }$ under the representation $\scriptstyle { \mathcal { R } } _ { \theta }$ is

$$
d _ { \pmb \theta } ( x _ { 0 } , x _ { 1 } ) = d _ { \Psi } \left( \sqrt { \dot { s } _ { 0 } } , \sqrt { \dot { s } _ { 1 } } \right) + d _ { \mathcal { H } } ( \pmb \theta _ { 0 } , \pmb \theta _ { 1 } ) .\tag{22}
$$

In order to define a distance on the shape space S from that one, we must quotient out the space $\mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ . The action of $\mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ on $\Psi ( [ 0 , 1 ] )$ is the same as $5 ,$ , and the Frenet curvatures are invariant under reparametrization of the corresponding parametrized curve (Lemma 1). Moreover, by simply taking $h ^ { \ast } \ = \ s _ { 1 } ^ { - 1 } \circ \ s _ { 0 } \ \in \ \mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ we have $\begin{array} { r c l } { { \sqrt { \dot { s } _ { 1 } } \ast h ^ { \ast } } } & { { = } } & { { \sqrt { \dot { s } _ { 0 } } . } } \end{array}$ , and thus the distance on $\Psi ( [ 0 , 1 ] ) / \mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ between $s _ { 0 }$ and $s _ { 1 }$ is zero. Hence, the induced distance on the shape space, between $X _ { 0 } , X _ { 1 } \in$ $s ,$ under the representation $\scriptstyle { \mathcal { R } } _ { \theta }$ is defined as

$$
d _ { S } ^ { ( \pmb \theta ) } ( X _ { 0 } , X _ { 1 } ) : = d _ { \mathcal { H } } ( \pmb \theta _ { 0 } , \pmb \theta _ { 1 } ) = \lVert \pmb \theta _ { 0 } - \pmb \theta _ { 1 } \rVert _ { \mathbb { L } ^ { 2 } }\tag{23}
$$

and the geodesic path connecting them is

$$
\alpha _ { S } ^ { ( \pmb { \theta } ) } ( \tau ) = \left( \sqrt { \dot { s } _ { 0 } } , \alpha _ { \mathcal { H } } ( \tau ) \right) .\tag{24}
$$

This immediate representation by the Frenet curvatures appears in the experiments not to be sufficiently elastic $( { \mathrm { F i g } } -$ ure 5). It has somewhat the same weakness as the angle representation proposed in [23], the Frenet curvatures being already independent of the parametrization.

## 4.3. Square Root Curvatures Transform

To overcome the “non-elasticity” issue of the representation defined above, we propose a second framework for shape analysis based on Frenet curvatures which uses, like the square root velocity function, the parametrization as a tool to register the curves and define a more “elastic” method. The latter is inspired by the square-root velocity transform of $S O ( d )$ -valued curves.

Definition 1 (SRV Transform for curves on $S O ( d ) )$ . Let $P \in C ^ { \infty } ( [ 0 , 1 ] , S O ( d ) )$ . The Square Root Velocity transform ofP is the map

$$
q ( P ) ( t ) = \frac { L _ { P ( t ) ^ { - 1 } } \dot { P } ( t ) } { \sqrt { \| \dot { P } ( t ) \| _ { F } } } = \frac { P ( t ) ^ { T } \dot { P } ( t ) } { \sqrt { \| \dot { P } ( t ) \| _ { F } } } ,\tag{25}
$$

where $\left\| . \right\| _ { F }$ is the Frobenius norm associated with the scalar product on the Lie Algebra of skew-symmetric matrices $\langle A , B \rangle = { \textstyle \frac { 1 } { 2 } } t r ( A ^ { T } B ) = - { \textstyle \frac { 1 } { 2 } } t r ( A B )$

Let $x \in F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ and $Q ( t ) \in { \cal C } ^ { \infty } ( [ 0 , 1 ] , S O ( d ) )$ be its associated Frenet path. Using the Frenet-Serret differential equation, the SRV Transform of the Frenet path is

$$
q ( Q ) ( t ) = \sqrt { \dot { s } ( t ) } \frac { A _ { \theta } ( s ( t ) ) } { \sqrt { \| A _ { \theta } ( s ( t ) ) \| _ { F } } } .\tag{26}
$$

Proposition 3. Let $\pmb \theta \in \mathcal H ,$ , we have

$$
\| A _ { \pmb \theta } ( s ( t ) ) \| _ { F } = \| \pmb \theta ( s ( t ) ) \| _ { 2 } .
$$

Based on the SRV Transform of a Frenet path and Proposition 3, we propose a new transformation of a parametrized curve, which we have called the Square-Root Curvatures (SRC) transform.

Definition 2 (Square-Root Curvatures Transform). Let $x \in$ $F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ . We consider its associated arc-length function $s ( t )$ and Frenet curvatures $\pmb \theta ( s ( t ) )$ defined as in Theorem 1. Then we define its square-root curvatures transform to be the map

$$
c ( t ) = \sqrt { \dot { s } ( t ) } \frac { \pmb { \theta } ( s ( t ) ) } { \sqrt { \| \pmb { \theta } ( s ( t ) ) \| } } .\tag{27}
$$

The set of such square-root curvatures transforms is

$$
\mathcal { C } = \left\{ c \in \mathbb { L } ( [ 0 , 1 ] , \mathbb { R } ^ { d - 1 } ) | c _ { 1 } , \dots , c _ { d - 2 } > 0 \right\} ,\tag{28}
$$

which is the same as the set of admissible Frenet curvatures H. We have already shown in the previous section that this set is a Riemannian manifold equipped with the $\mathbb { L } ^ { 2 }$ metric. Therefore, the geodesic distance between $c _ { 0 } , c _ { 1 } \in { \mathcal { C } }$ is the $\mathbb { L } ^ { 2 }$ distance between them, and the geodesic path is a straight line. We define the following representation of a parametrized curve $x \in F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ , from its Square-Root Curvatures transform, by

$$
\mathcal { R } _ { \mathrm { S R C } } ( x ) ( t ) = \left( \sqrt { \dot { s } ( t ) } , c ( t ) \right) .\tag{29}
$$

Proposition 4. The map $\begin{array} { r l r } { \mathcal { R } _ { S R C } } & { { } : } & { F ( [ 0 , 1 ] , \mathbb { R } ^ { d } ) \quad  } \end{array}$ $\Psi ( [ 0 , 1 ] ) \times \mathcal { C } ,$ , defined above, is a bijection.

Proof. This is again a result of theorems 1 and 2. To get x from $\mathcal { R } _ { \mathrm { S R C } } ( x )$ , it should be noted firstly that $c ( t ) \| c ( t ) \| =$ $\dot { s } ( t ) \pmb \theta ( s ( t ) )$ . From that, the skew-symmetric matrix function of the Frenet-Serret ODE can be reconstructed. By solving the corresponding Frenet-Serret ODE one gets the associated time parametrized Frenet path $Q ( t )$ . Then, using the first component of $\mathcal { R } _ { \mathrm { S R C } } ( x )$ , we get $x ( t ) = X ( s ( t ) ) =$ $\begin{array} { r } { \int _ { 0 } ^ { t } \dot { s } ( u ) T ( s ( u ) ) d u } \end{array}$ □

The set of such square root curvature representations $\mathcal { R } _ { \mathrm { S R C } }$ is the Cartesian product $\Psi ( [ 0 , 1 ] ) \times \mathcal { C }$ and therefore a Riemannian manifold with the product metric $d _ { \Psi } \oplus d _ { C }$ . This representation is, by definition, invariant under the action of $S O ( d )$ . Then, the corresponding shape space is the quotient space $\Psi ( [ 0 , 1 ] ) \times \mathcal { C } / \mathrm { D i f f } ( [ 0 , 1 ] )$ . Let’s $x \in F ( [ 0 , 1 ] , \hat { \mathbb { R } } ^ { d } )$ and $h \in \operatorname { D i f f } ( [ 0 , 1 ] )$ . The SRC representation of $\tilde { x } = x \circ h$ is

$$
\mathcal { R } _ { \mathrm { S R C } } ( \tilde { x } ) = \Big ( \sqrt { \dot { s } } * h , c * h \Big ) = \mathcal { R } _ { \mathrm { S R C } } ( x ) * h\tag{30}
$$

where ∗ is the group action defined in 5.

Proposition 5. The metric on $F ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ induced by the Riemannian metric on $\Psi ( [ 0 , 1 ] ) \times \stackrel { \triangledown } { \boldsymbol { C } }$ defined by $d _ { S R C } : = d _ { \Psi }$ ⊕ $d _ { C }$ is invariant under the action of ${ D i f f _ { + } ( [ 0 , 1 ] ) }$

The distance on the shape space $s$ under the representation $\mathcal { R } _ { \mathrm { S R C } }$ , between two elements $X _ { 0 } , X _ { 1 } \in { \mathcal { S } }$ , is defined as

$$
d _ { S } ^ { ( \mathrm { S R C } ) } ( X _ { 0 } , X _ { 1 } ) : = \operatorname* { i n f } _ { h \in \mathrm { D i f f } _ { + } ( [ 0 , 1 ] ) } d _ { \mathrm { S R C } } ( x _ { 0 } , x _ { 1 } \circ h ) .\tag{31}
$$

From the optimal wrapping function $h ^ { * }$ the geodesic path on $s$ between them is

$$
\begin{array} { r l } { \alpha _ { S } ^ { ( \mathrm { S R C } ) } ( \tau ) = } & { \Big ( \frac { \sin ( ( 1 - \tau ) \vartheta ) } { \sin ( \vartheta ) } \sqrt { \dot { s } _ { 0 } } + \frac { \sin ( \tau \vartheta ) } { \sin ( \vartheta ) } \big ( \sqrt { \dot { s } _ { 1 } } \ast h ^ { \ast } \big ) , } \\ & { \qquad ( 1 - \tau ) c _ { 0 } + \tau ( c _ { 1 } \ast h ^ { \ast } ) \big ) } \end{array}\tag{32}
$$

where $\vartheta = d _ { \Psi } ( \sqrt { \dot { s } _ { 0 } } , \sqrt { \dot { s } _ { 1 } } * h ^ { * } )$ .The registration problem consider here is to find the minimizer $h ^ { * }$ over $\mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ of

$$
\int _ { 0 } ^ { 1 } \| c _ { 0 } ( t ) - ( c _ { 1 } * h ) ( t ) \| ^ { 2 } + \| \sqrt { \dot { s } _ { 0 } ( t ) } - ( \sqrt { \dot { s } _ { 1 } } * h ) ( t ) \| ^ { 2 } d t\tag{33}
$$

Using the reformulation principle of [3], that is $\gamma = s _ { 1 } \circ h \circ$ $s _ { 0 } ^ { - 1 } \in \operatorname { D i f f } _ { + } ( [ 0 , 1 ] )$ ), this registration problem is shown to be equivalent to finding $\gamma ^ { * } \in \mathrm { D i f f } _ { + } ( [ 0 , 1 ] )$ that minimizes

$$
\int _ { 0 } ^ { 1 } \left\| \frac { \theta _ { 0 } ( s ) } { \sqrt { \| \theta _ { 0 } ( s ) \| } } - \sqrt { \gamma ^ { \prime } ( s ) } \frac { \theta _ { 1 } ( \gamma ( s ) ) } { \sqrt { \| \theta _ { 1 } ( \gamma ( s ) ) \| } } \right\| ^ { 2 }\tag{34}
$$

Note that this reformulation has the form of a penalized registration problem. The second term represents a penalty term on γ and ensures a certain smoothness of the warping function. In this framework, the deformable templates are the square-root normalized curvatures which encode more geometric information than the unit tangent vector.

## 5. Experiments

In this section, we report the experimental results of the proposed methods, comparing them with the SRVF method. We use both synthetic and real data. Additional results and figures are available in the supplementary material.

## 5.1. Statistical estimation of the Frenet curvatures

The main limitation of shape analysis methods based on the Frenet curvatures is the need of additional estimates of curvatures. Being dependent on higher-order derivatives (up to order d), they are quite sensitive to the observation noise of the Euclidean curve. We detail here a simple method that can be used for their smooth estimation, and we refer to [21, 18] for a more complex and detailed statistical estimation algorithm. First, it is possible to use a local polynomial smoothing algorithm to estimate the d first derivatives of the Euclidean curve [6]. From these derivatives, the raw estimates of the Frenet curvatures can be computed by using their extrinsic formulas. We propose here a second method based on the Frenet-Serret ODE approximation, to obtain the raw estimates, that appear to be more stable. A simple middle point approximation of the ODE solution gives

$$
Q ( s _ { j } ) \approx Q ( s _ { i } ) \exp \left( ( s _ { i } - s _ { j } ) A _ { \pmb \theta } \left( \frac { s _ { i } + s _ { j } } { 2 } \right) \right) .\tag{35}
$$

where exp(.) is the exponential map of the Lie group $S O ( d )$ . Then, using the inverse logarithm map log(.), this gives an approximation of the matrix $A _ { \pmb \theta } \left( ( s _ { i } + s _ { j } ) / 2 \right)$ ≈ $\begin{array} { r } { \frac { 1 } { s _ { i } - s _ { j } } \log \big ( Q ( s _ { i } ) ^ { T } Q ( s _ { j } ) \big ) } \end{array}$ and by identification raw estimates of $\pmb { \theta } \left( ( s _ { i } + s _ { j } ) / 2 \right)$ . As we consider here a problem of estimating a functional parameter, we formulate the final θ estimation problem as a penalized weighted functional regression with the obtained raw estimates, that we solve by using a B-spline approximation of θ.

## 5.2. Experiments with synthetic curves

We use synthetic data to highlight the differences between the methods discussed above (SRVF, SRC, and Frenet curvatures). The computations related to the SRVF method are made with the package fdasrsf. The SRC and Frenet curvatures methods are implemented with the code provided as supplementary material, including a dynamic programming algorithm for solving the registration problems.

We consider the simple case of a set of 20 curves in $\mathbb { R } ^ { 2 }$ with a single large peak of curvature. This one is created by generating curvature functions on [0, 1] with one peak of maximum value 60.5, width 0.15, and location chosen randomly between 0.1 and 0.9. These curves have the shape of a loop made with a wire, where the loop is more or less close to the right or left wire end, depending on the location of the curvature peak.

![](images/df91e9c83ea043bca44716d2dd9ab2b9b803881f366f9c2313023b1cb2f18117.jpg)  
Figure 3: Matrices of pairwise SRVF distance (left), SRC distance (middle), and unparametrized Frenet curvatures distance (right) by sorted location of the peak curvature.

We compare the three methods through the pairwise distance matrices in Figure $^ { 3 , }$ and the geodesic paths computed between two of these curves in Figure 4, with peaks located at 0.27 (red curve) and at 0.78 (blue curve). The corresponding deformations through the variations of the curvature along the different geodesic paths on Euclidean curves are shown in Figure 5, which highlights the strengths and weaknesses of each method. First, it emphasizes the ”nonelasticity” of the unparametrized Frenet curvatures method, as in the middle of the geodesic path, we have two peaks of curvature and, therefore, a completely different shape without any loop. This explains the inconsistency of the heatmap under this method. Conversely, there is an elastic deformation of the curvature with the SRC transform, and shapes along the geodesic are consistent with the set of curves considered, which is well summarised on the corresponding heatmap where all distances are rather close to zero. For the SRVF method, the chosen example with peaks of curvature that are quite far apart shows that artifacts appear along the geodesic; the middle curve has two small loops at the edges. This phenomenon gives unreliable distances, as shown in the heatmap, where the distances are not monotone as a function of the spacing between the curvature peaks.

![](images/0cb434bb964469ff857c155df255763915e8bc2534907d4201f2c3c476db61b2.jpg)

Figure 4: SRVF $( 1 ^ { s t }$ row), SRC $( 2 ^ { n d }$ row), and Frenet curvatures $( 3 ^ { r d }$ row) geodesic paths between curves with curvature peaks located at 0.27 (left) and 0.78 (right).  
![](images/db15c3fa16d5615797977448358500022830d7e1404ea5063e70b251105ab50a.jpg)  
Figure 5: Curvatures of the Euclidean curves along the geodesic paths plotted in Figure 4: SRVF $( 1 ^ { s t }$ row), SRC $( 2 ^ { n d }$ row), and Frenet curvatures $( 3 ^ { r d }$ row).

By considering a set of curves characterized by specific features, we observe a clear difference in consistency of the shapes along the geodesics with respect to the different methods. This phenomenon is quite visible on the geodesic between helices in 2D, Figure 1 or 3D, Figure 2. In that case, within both geodesic paths under SRVF method, the curves lose the characteristic geometry of the helix. A threedimensional circular helix is characterized by having a constant curvature and torsion, which is not the case along the SRVF geodesic, but preserved with the SRC method. In that case the geodesic under the Frenet curvature representation is very similar to SRC.

## 5.3. Application to sign language motion data

It appears that curvilinear velocity and Frenet curvatures are particularly relevant parameters for the analysis of human motion. Several laws involving these parameters can be found in the literature [10, 14, 19, 7]; among others, the power laws state a special relationship between the curvature, the torsion, and the velocity of a point trajectory representing human motion. Using a method that conserves the shape of these parameters is, therefore, of particular interest in this application. We demonstrate here with the case of wrist trajectories in sign language, acquired with a motion capture system by the company MocapLab (https://www.mocaplab.com/fr/). We compute the geodesic paths, under each of the frameworks, between the arbitrarily chosen red and blue curves within the set of several repetitions of the sign ”Femme”, shown in Figure 6 with the corresponding time-parameterized Frenet curvatures.

![](images/44e98ee2cb8659e3d540d4a69f5be64080b0723001db41861985ff1bfbe111df.jpg)  
Figure $6 { : }$ Trajectories of the right wrist while signing ”Femme” in sign language: 3D curves (left), timeparametrized curvatures (top right), torsions (bottom right). The blue and red ones are used to compute the geodesic in Figure 7.

Figure 7 emphasizes the advantage of considering a representation depending on the parameterization, allowing a registration before computing the geodesic. However, it also shows that considering only the tangent vector as a representative object (SRVF) is not sufficient to find the optimal reparametrization that correctly aligns the torsions, and could affect subsequent analyses made by the SRVF method, such as the mean. This results in the appearance of new minimums, maximums and zeros in the torsion functions along the SRVF geodesic. However, such characteristic points of the curvature, torsion and velocity functions are crucial in the observation of the laws of motion. It is therefore preferable to use a method that preserves these characteristics by directly optimizing the optimal alignment from these parameters, such as the proposed SRC method.

![](images/4111823d9b8ee036560e5226d744a9c2ca1c62e37d952c439ac8d6eddf4e6437.jpg)

![](images/b6fd76f8b2587031bde0835740f1e1110b3fd373ff0d415c58016a02b613e852.jpg)  
(a) SRVF

![](images/bd5ba946ff18278ba5d14b10b5dd22d674ce00484ecf99c516344998f5622c86.jpg)

![](images/64ee83f677e8794ecb64b9daf5618fde7926408aa9e92f4635a27fa05ec24ace.jpg)

![](images/878d9e5b5d3cf80766cc04c74abca30af4146f422c250ea64e929a9b51583932.jpg)  
(b) SRC

![](images/c776ed8f6b573a9a19aa1d10e8eaf7cc21543e5bae21ecf8fd4f50fabb5bfbd8.jpg)  
(c) Frenet curvatures

Figure 7: Comparison between time-parametrized curvature and torsion along the geodesic path under SRVF (left), SRC (middle), and Frenet curvatures (right).  
![](images/e1c814f73f21370ff425e9c329e37fdcf1d79f47b05020b32c1091e40ce47f5b.jpg)

![](images/c940a4774ed12dac061559f1c996ffce8808bc8cd7a550ec25e0fbceff36b8a1.jpg)  
(a) Warping functions h  
(b) Warping functions γ  
Figure 8: Comparison of estimated warping functions h (left) and $\gamma$ (right) to compute the geodesic in Figure 7.

## 6. Conclusion

The square-root curvature transform of a Euclidean curve in $\mathbb { R } ^ { d }$ is a representation that encodes more geometric information of the curves, and thus the results are easier to interpret than existing methods. The main limitation lies in the estimation of the Frenet curvatures from real and noisy data, nevertheless recent smooth statistical estimators can be used for computing the SRC [21, 18]. We believe our method is particularly interesting for motion trajectory analysis and could be developed further in the future as a tool for generation, segmentation and classification of complex trajectories.

## References

[1] Martin Bauer, Nicolas Charon, Eric Klassen, and Alice Le Brigant. Intrinsic riemannian metrics on spaces of curves: Theory and computation. Handbook ofMathematical Models and Algorithms in Computer Vision and Imaging, pages 1–35, 2021. 1

[2] Martin Bauer, Nicolas Charon, Eric Klassen, Sebastian Kurtek, Tom Needham, and Thomas Pierron. Elastic Metrics on Spaces of Euclidean Curves: Theory and Algorithms, 2022. arXiv:2209.09862 [math]. 1, 2

[3] Nicolas J.-B. Brunel and Juhyun Park. The frenet-serret framework for aligning geometric curves. In Frank Nielsen and Fred´ eric Barbaresco, editors,´ Geometric Science of Information, pages 608–617, Cham, 2019. Springer International Publishing. 3, 4, 5, 7

[4] Elena Celledoni, Markus Eslitzbichler, and Alexander Schmeding. Shape analysis on lie groups with applications in computer animation. Journal of Geometric Mechanics, 8(3):273–304, 2016. 2

[5] Marvin Eisenberger and Daniel Cremers. Hamiltonian dynamics for real-world shape interpolation. In European Conference on Computer Vision (ECCV), 2020. 1

[6] Jianqing Fan and Irene Gijbels. Local polynomial modelling and its applications. CRC monographs on statistics and applied probability 66. Chapman & Hall, 1996. 7

[7] Tamar Flash and Alain Berthoz. Space-Time Geometriesfor Motion and Perception in the Brain and the Arts. Lecture Notes in Morphogenesis. Springer, 2021. 8

[8] Lukas Koestler, Daniel Grittner, Michael Moeller, Daniel Cremers, and Zorah Lahner. Intrinsic neural fields: Learning¨ functions on manifolds. In European Conference on Computer Vision (ECCV), 2022. 1

[9] Wolfgang Kuhnel.¨ Differential Geometry Curves – Surfaces, volume 77. 4, 5

[10] Francesco Lacquaniti, Carlo Terzuolo, and Paolo Viviani. The law relating the kinematic and figural aspects of drawing movements. Acta Psychologica (Amst.), 54:115–130, 1983. 8

[11] Serge Lang. Differential and riemannian manifolds. Springer, 1999:1–6, 2006. 1, 2

[12] Robert Lyons. Frobenius theorem two ways. Lecture note, 2016. 5

[13] Zorah Lahner, Emanuele Rodol¨ a, Frank R. Schmidt,\` Michael M. Bronstein, and Daniel Cremers. Efficient globally optimal 2d-to-3d deformable shape matching. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), May 2016. 1

[14] Uri Maoz, Alain Berthoz, and Tamar Flash. Complex unconstrained three-dimensional hand movement and constant equi-affine speed. Journal of Neurophysiology, 101:1002– 1015, 2009. 8

[15] J. S. Marron, James O. Ramsay, Laura M. Sangalli, and Anuj Srivastava. Functional data analysis of amplitude and phase variation. Statistical Science, 30:468–484, 2015. 5

[16] Turk J Math. Submanifolds of Riemannian Product Manifolds. 29:389–401, 2005. 5

[17] Tom Needham. Shape Analysis of Framed Space Curves. Journal of Mathematical Imaging and Vision, 61(8):1154– 1172, Oct. 2019. 1

[18] Juhyun Park, Nicolas Brunel, and Perrine Chassat. Curvature and torsion estimation of 3d functional data: A geometric approach to build the mean shape under the frenet serret framework, 2022. arXiv:2203.02398 [stat]. 7, 9

[19] Frank E. Pollick, Uri Maoz, Amir A. Handzel, Peter J. Giblin, Guillermo Sapiro, and Tamar Flash. Three-dimensional arm movements at constant equi-affine speed. Cortex, 45:325–339, 2009. 8

[20] Marianna Saba. On the usage of the curvature for the comparison ofplanar curves. PhD thesis, University of Cagliari, 2012. 2, 5

[21] Laura M Sangalli, Piercesare Secchi, Simone Vantini, and Alessandro Veneziani. Efficient estimation of threedimensional curves and their derivatives by free-knot regression splines, applied to the analysis of inner carotid artery centrelines. Journal of the Royal Statistical Society: Series C (Applied Statistics), 58(3):285–306, 2009. 7, 9

[22] Stefan Sommer, Tom Fletcher, and Xavier Pennec. Introduction to differential and Riemannian geometry. 2020. 2

[23] Anuj Srivastava and Eric Klassen. Functional and Shape Data Analysis. Springer Series in Statistics. Springer New York, 2016. 1, 2, 3, 5, 6

[24] Anuj Srivastava, Eric Klassen, Shantanu H. Joshi, and Ian H. Jermyn. Shape analysis of elastic curves in euclidean spaces. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(7):1415–1428, 2011. 1, 3

[25] Tatiana Surazhsky and Gershon Elber. Metamorphosis of planar parametric curves via curvature interpolation. International Journal of Shape Modeling, 8:201–216, 2002. 2, 5

[26] Wang Tixiang. Morse theory on banach manifolds. Acta Mathematica Sinica, 5:250–262, 1989. 5

[27] J. Derek Tucker, Wei Wu, and Anuj Srivastava. Generative models for functional data using phase and amplitude separation. Computational Statistics and Data Analysis, 61:50–66, 2013. 5

[28] Laurent Younes. Computable Elastic Distances between Shapes. SIAM Journal on Applied Mathematics, 58(2):565– 586, 1998. Publisher: Society for Industrial and Applied Mathematics. 1

[29] Laurent Younes. Shapes and Diffeomorphisms. Applied Mathematical Sciences. Springer, 2010. 3

[30] Laurent Younes. Elastic distance between curves under the metamorphosis viewpoint, 2018. arXiv:1804.10155. 3