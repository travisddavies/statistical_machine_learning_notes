# Training DNNs
_Techniques specific to non-convex objectives, largely based on gradient descent_

## How to Train Your Network?
- You know the drill: Define the loss function and find parameters that minimise the loss on training data
- In the following, we are going to use **stochastic gradient descent** with a **batch size** of one. That is, we will process training examples one by one

## Example: Univariate Regression
- Consider regression
- Moreover, we'll use identity output activation function
$$
z = h(s) = s = \sum^p_{j=0}u_jw_j
$$
- This will simplify description of backpropagation. In other settings, the training procedure is similar.

![[simple_ann.png]]

## Loss Function for NNet Training
- Need **loss** between training example $\{x,y\}$ & prediction $\hat{f}(x, \theta)=z$, where $\theta$ is parameter vector of $v_{ij}$ and $w_j$
- As regression, can use **squared error**
$$
L = \frac{1}{2}(\hat{f}(x,\theta)-y)^2=\frac{1}{2}(z-y)^2
$$
	(the constant is used for mathematical convenience, see later)
- **Decision-theoretic** training: minimise $L$ w.r.t. $\theta$
	- Fortunately $L(\theta)$ is differentiable
	- Unfortunately no analytic solution in general

## Stochastic Gradient Descent for NNet
Choose initial guess $\theta^{(0)}, k=0$ 
	Here $\theta$ is a set of all weights form all layers
For $i$ from 1 to $T$ (**epochs**)
	For $j$ from 1 to $N$ (training examples - could **shuffle**)
		Consider example $\{x_j,y_j\}$
		<u>Update</u>: $\theta^{(k+1)}=\theta^{(k)}-\eta\nabla L(\theta^{(k)}); k \leftarrow k+1$ 

$$
L = \frac{1}{2}(z_j - y_j)^2
$$
Need to compute partial derivatives $\frac{\delta L}{\delta v_{ij}}$ and $\frac{\delta L}{\delta w_j}$ 

## Recap: Gradient Descent vs SGD
1. Choose $\theta^{(0)}$ and some $T$
2. For $i$ from 0 to $T-1$
	1. $\theta^{(i+1)} = \theta^{(i)} - \eta\nabla L(\theta^{(i)})$ 
3. Return $\hat{\theta} \approx \theta^{(T)}$ 

![[gd.png]]

**Stochastic G.D.**
1. Choose $\theta^{(0)}$ and some $T$, $k=0$
2. For $i$ from 1 to $T$
	1. For $j$ from 1 to $N$ (in random order)
		1. For $j$ from 1 to $N$ (in random order)
		2. $k++$
3. Return $\hat{\theta} \approx \theta^{k}$ 

## Mini-Batch SGD
- SGD works on single instances
	- High variance in gradients
	- Many, quick, updates
- GD works on whole datasets
	- Stable update, but slow
	- Computationally expensive
- Compromise: **mini-batch** (_often just called "SGD"_)
	- Process batches of size $1 < b < N$, e.g., $b=100$
	- Balances computation and stability
	- Parallelise over cluster of GPUs (size batch for GPU)

![[mini-batch.png]]

## (non-)Convex Objective Functions
- Recall linear regression, convex '**Bowl shaped**' objective
	- Gradient descent finds a **global** optimum

![[convex_sgd.png]]

- In contrast, most DNN objectives are **not convex**
	- Gradient methods get trapped in **local optima** or **saddle points**

![[global_optima.png]]

## Importance of Learning Rate
- Choice of $\eta$ has big effect on quality of final parameters
- Each SGD step:
	- $\theta^{(i)} = \theta^{(i-1)}-\eta\nabla L (\theta^{(i-1)})$ 
- Choosing $\eta$:
	- Large $\eta$ fluctuate around optima, even diverge
	- Small $\eta$ barely moves, stuck at local optima

![[learning_steps.png]]

## Momentum as a Solution
- Consider a ball with some mass rolling down the objective surface
	- **Velocity** increases as it rolls downwards
	- Momentum can carry it past local optima
- Mathematically, SGD update becomes
	- $\theta^{(t+1)} = \theta^{(t)} - v^{(t)}$ 
	- $v^{(t)} = \alpha v^{(t-1)}+\eta\nabla L (\theta^{(t)})$ 
	- $\alpha$ decays the velocity, e.g., 0.9
- Less oscillation, more robust

![[ball_rolling.png]]

## Adagrad: Adaptive Learning Rates
- Why just one learning rate applied to _all_ params?
	- Some features (parameters) are used more frequently than others $\rightarrow$ smaller updates for common features vs. rare
- **Adagrad** tracks the sum of squared gradient per-parameter, i.e., for parameter $i$
	- $g_i^{(t)}=g_i^{(t-1)}+\nabla L (\theta^{(t)})^2_i$ 
	- $\theta_i^{(t+1)}=\theta_i^{(t)}-\frac{\eta}{g_i^{(t)}+\epsilon}\nabla L (\theta^{(t)})_i$ 
		- Typically $\epsilon=10^{-8}$ $n=0.01$ 
- No need to tune learning rate! But can be conservative

## Adam
- Combining elements of momentum and adaptive learning rates
	1. $m^{(t)}=\beta_1 m^{(t-1)}+(1-\beta_1)\nabla L (\theta^{(t)})$ 
	2. $v^{(t)}= \beta_2 v^{(t-1)}+(1-\beta_2)\nabla L (\theta^{(t)})^2$
	3. $\hat{s} = \frac{v^{(t)}}{1-\beta_2}$
	4. $\hat{m} = \frac{m}{1-\beta_1^t}$ 
	5. $\theta^{(t+1)}=\theta^{(t)}-\frac{\hat{m}\eta}{\sqrt{\hat{s}}+\epsilon}$  
		$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ 
		2 and 5 are element-wise operations
 - Good work-horse method, current technique of choice for deep learning

## Zoo of Optimisation Algorithms
- Suite of batch-style algorithms, e.g., BFGS, L-BFGS, Conjugate Gradient, ...
- And SGD style:
	- Nesterov acc. grad.
	- Adadelta
	- AdaMax
	- RMSprop
	- AMSprop
	- Nadam
	- Adam
	- ...
 
 ![[optimiser.png]]
	- Lots of choice, and rapidly changing as deep learning matures

# Regularising Deep Nets
_Best practices in preventing overfitting, a big problem for such high capacity and complex models._

## Some Further Notes on DNN Training
- DNNs are flexible (recall universal approximation theorem), but the flipside is over-parameterisation, hence tendency to **overfitting**
- Starting weights usually randomly distributed about zero
- Implicit regularisation: **early stopping**
	- With some activation functions, this **shrinks** the DNN towards a linear model
![[tanh_function.png]]

## Explicit Regularisation
- Alternatively, an **explicit regularisation** can be used, much like in ridge regression
- Instead of minimising the loss $L$, **minimise regularised function** $L + \lambda(\sum^m_{i=0}\sum^p_{j=1}v^2_{ij}+\sum^p_{j=0}w_j^2)$ 
- This will simply add $2\lambda v_{ij}$ and $2\lambda w_j$ terms to the partial derivatives (aka **weight decay**)
- With some activation functions (e.g. tanh / sigmoid) this also **shrinks** the DNN towards a linear function

## Dropout
- Randomly mask fraction of units during training
	- Different masking each presentation
	- Promotes **redundancy** in network hidden representation (a form of regularisation)
	- A form of **ensemble** of exponential space
	- No masking at testing (requires **weight adjustment**)
- Results in smaller weights, and less overfitting
- Used in most SOTA deep learning systems

# Autoencoders
_A DNN training setup that can be used for unsupervised learning, initialisation, or just efficient coding_

## Autoencoding Idea
- Supervised learning:
	- Univariate regression: predict $y$ from $x$
	- Multivariate regression: predict $y$ from $x$
- Unsupervised learning: explore data $x_1, ..., x_n$
	- No response variable
- For each $x_i$ set $y_i \equiv x_i$
- Train a NNet to predict $y_i$ from $x_i$ i.e., model $p(x|x)$
- Pointless?

## Autoencoder Topology
- Given data without labels $x_1, ..., x_n$, set $y_i \equiv x_i$ and train a DNN to predict $z(x_i) \approx x_i$ 
- Set **bottleneck** layer $u$ in middle "thinner" than input, and/or
	- Corrupt input $x$ with noise
	- Regularise s.t. $u$ is sparse
	- Regularise to contract inputs

![[autoencoder_topology.png]]

## Introducing the Bottleneck
- Suppose you managed to train a network that gives a good **restoration** of the original signal $z(x_i) \approx x_i$
- This means that the data structure can be effectively described (**encoded**) by a lower dimensional representational $u$

![[bottleneck.png]]

## Under-/Over-Completeness
- Manner of bottleneck gives rise to:
	- **Undercomplete**: model with thinner bottleneck than input forced to generalise
	- **Overcomplete**: wider bottleneck than input, can just "copy" directly to output
- Even undercomplete models can learn trivial codes, given complex non-linear encoder and decoder
- Various methods to ensure learning

## Dimensionality Reduction
- Autoencoders can be used for
	- **Compression**
	- **Dimensionality reduction**
	- **Unsupervised pre-training**
	- Finding **latent feature space**
	... via a non-linear transformation 
- Related to **principle component analysis** (PCA)

## Principal Component Analysis
- Principal component analysis (PCA) is a popular method for **dimensionality reduction** and data analysis in general
- Given a dataset $x_1, ..., x_n, x_i \in R^m$, PCA aims to find a new coordinate system that most of the **variance is concentrated** along the first coordinate, then most of the remaining variance along the second (**orthogonal**) coordinate, etc.
- Dimensionality reduction is then based on **discarding coordinates** except the first $l < m$. Coordinates = axes of data = **principal components**

![[pca.png]]

## PCA: Solving the Optimisation
- PCA aims to find **principal component** $p_1$ that maximises variance of data projected onto the PC, $p_1'\sum_X p_1$
	- Subject to $\lVert p_1 \rVert = p_1'p_1 = 1$ 
	- Have to first subtract the centre of the data from the data
	![[pca_unexamined.png]]

- $\sum_Xp_1 = \lambda_1 p_1$
- Precisely defines $p_1$ as an **eigenvector** of covariance $\sum_X$ with $\lambda_1$ being the corresponding **eigenvalue**

## PCA vs Autoencoding
- If you use linear activation functions and only one hidden layer, then the setup becomes almost that of **Principal Component Analysis** (PCA)
	- PCA finds orthogonal basis where axes are aligned to capture maximum data variation
	- NNet might find a fidderent solution, doesn't use eigenvalues (directly)

## Uses of Autoencoders
- Data visualisation & clustering
	- Unsupervised first step towards understanding properties of the data
- As a feature representation
	- Allowing the use of off-the-shelf ML methods, applied to much smaller and informative representations of input
- Pre-training of deep models
	- Warm-starting training by initialising model weights with encoder parameters
	- In some fields of vision, mostly replaced with supervised pre-training on very large datasets