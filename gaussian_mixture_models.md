# Unsupervised Learning
_A large branch of ML that concerns with learning structure of the data in the absence of labels_

## Main Learning Paradigms So Far
- Supervised learning: Overarching aim is making predictions from data
- We studied methods in the context of this aim: e.g. linear/logistic regression, DNN, SVM
- We had instances $x_i \in R^m, \ i = 1, ..., n$ and corresponding labels $y_i$ for model fitting , aiming to predict labels for new instances
- Can be viewed as a function approximation problem, but with a big caveat: ability to generalise is critical
- Bandits: a setting of partial supervision where subroutine in contextual bandits requires supervised learning

## Now: Unsupervised Learning
- In unsupervised learning, there is no dedicated variable called a "label"
- Instead, we just have a set of points $x_i \in R^m, \ 1, ..., n$
- Aim of unsupervised learning is to **explore the structure** (patterns, regularities) of data
- The aim of "exploring the structure" is vague

![][Images/unsupervised_learning.png]

## Unsupervised Learning Tasks
- Diversity of tasks fall into unsupervised learning category
	- Clustering (now)
	- Dimensionality reduction (autoencoders)
	- Learning parameters of probabilistic models (before/now)
- Applications and related tasks are numerous:
	- Market basket analysis. E.g., use supermarket transaction logs to find items that are frequently purchased together
	- Outlier detection. E.g., find potentially fraudulent credit card transactions
	- Often unsupervised tasks in (supervised) ML pipelines

## Refresher: $k$-Means Clustering
1. <u>Initialisation</u>: choose $k$ cluster **centroids** randomly
2. <u>Update</u>:
	1. **Assign points** to the nearest* centroid
	2. **Compute centroids** under the current assignment
3. <u>Termination</u>: if no change then stop
4. Go to **Step 2**
*Distance represented by choice of metric typically $L_2$

Still one of the most popular data mining algorithms

![][Images/refresher-k-means.png]

# Gaussian Mixture Models
_A probabilistic view of clustering. Simple example of a latent variable model._

## Modelling Uncertainty in Data Clustering
- $k$-means clustering assigns each point to exactly one cluster
	- Does this make sense for points that are between two clusters?
	- Clustering is often not well defined to begin with!
- Like $k$-means, a probabilistic mixture model requires the user to choose the number of clusters in advance
- Unlike $k$-means, the probabilistic model gives us a power to express **uncertainty about the origin** of each point
	- Each point originates from cluster $c$ with probability $w_c$, $c=1, ..., k$ 
- That is, each point still originates from one particular cluster (aka component), but we are not sure from which one
- Next
	- Clustering becomes model fitting in probabilistic sense. Philosophically satisfying.
	- Individual components modelled as Gaussians
	- Fitting illustrates general Expectation Maximization (EM) algorithm

## Clustering: Probabilistic Model
Data points $x_i$ are independent and identically distributed (i.i.d.) samples from a **mixture** of $K$ distributions (components)

![][Images/clustering-probabilistic-model.png]

In principle, we can adopt any probability distribution for the **components**, however, the normal distribution is a common modelling choice $\rightarrow$ Gaussian Mixture Model

## Normal (aka Gaussian) Distribution
- Recall that a 1D Gaussian is

$$
N(x|\mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp\bigg(-\frac{(x-\mu)^2}{2 \sigma^2}\bigg)
$$

- And a $d$-dimensional Gaussian is

$$
N(x|\mu, \Sigma) = (2\pi)^{-\frac{d}{2}}|\Sigma|^{-\frac{1}{2}} \exp\bigg( -\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu) \bigg)
$$

- $\Sigma$ is a PSD symmetric $d \times d$ matrix, the **covariance matrix**
- $|\Sigma|$ denotes determinant
- No need to memorise the full formula

![][Images/normal_distribution_adka_gaussian.png]

## Gaussian Mixture Model (GMM): One Point
- Cluster assignment of point
	- Multinomial distribution on $k$ outcomes
	- $P(Z = j)$ described by $P(C_j) = w_j \geq 0$ with $\sum^k_{j=1} w_j = 1$ 
- Location of point
	- Each cluster has its own Gaussian distribution
	- Location of point governed by its cluster assignment
	- $P(X|Z=j) = N(\mu_j, \Sigma_j)$ class conditional density
- Model's parameters: $w_j, \mu_j, \Sigma_j, j = 1, ..., k$ 

![][Images/gaussin_mixture_model_one_point.png]

## From Marginalisation to Mixture Distribution
- When fitting the model to observations, we'll be maximising likelihood of observed portions of the data (the $X$'s) not the latent parts (the $Z$'s)
- Marginalising out the $Z$'s derives the "familiar" mixture distribution
- Gaussian mixture distribution:

$$
P(x) = \sum^k_{j=1}w_j N(x|\mu_j, \Sigma_j)
$$

$$
= \sum^k_{j=1} P(C_j) P(x|C_j)
$$

- A convex combination of Gaussians
- Simply marginalisation at work

![][Images/from-marginalisation-to-mixture-distribution.png]

## Clustering as Model Estimation
- Given a set of data points, we assume that data points are generated by a GMM
	- Each point in our dataset originates from our mixture distribution
	- Shared parameters between points: _w00t_ independence assumption
- Clustering now amounts to finding parameters of the GMM that "best explains" observed data
- Call upon old friend **MLE** principle to find parameters values that maximise $p(x_1, x_n)$

![][Images/clustering_as_model_estimation.png]

# Briefing Expectation-Maximisation Algorithm
_We want to implement MLE but we have unobserved r.v.'s that prevent clean decomposition as happens in fully observed settings_

## Fitting the GMM
- Modelling the data points as independent, aim is to find $P(C_j), \mu_j, \Sigma_j, j=1, ..., k$ that maximise

$$
P(x_1, ..., x_n) = \prod_{i=1}^n \sum^k_{j=1} P(C_j) P(x_i | C_j)
$$

where $P(x|C_j) = N(x|\mu_j, \Sigma_j)$ 

Can be solved analytically?

- Taking the derivative of this expression is pretty awkward, **try the usual log trick**

$$
\log P(x_1, ..., x_n) = \sum^n_{i=1} \log \Bigg(\sum^k_{j=1} P(C_j) P(x_i | C_j) \Bigg)
$$

$\rightarrow$ **Expectation-Maximisation (EM)**

![][Images/fitting-the-gmm.png]
## Motivation of EM
- Consider a parametric probabilistic model $p(X| \theta)$, where $X$ denotes data and $\theta$ denotes a vector of parameters
- According to MLE, we need to maximise $p(X|\theta)$ as a function of $\theta$
	- Equivalently maximise $\log p (X|\theta)$ 
- There can be a couple of issues with this task

![][Images/crying.png]

1. Sometimes we **don't observe** some of the variables needed to compute the log likelihood
	- Example: GMM cluster membership $Z$ is not known in advance
2. Sometimes the form of the log-likelihood is **inconvenient*** to work with
	- Example: taking a derivative of GMM log likelihood results in a cumbersome equation

## Expectation-Maximisation (EM) Algorithm

![][Images/expectation-maximisation-algorithm.png]