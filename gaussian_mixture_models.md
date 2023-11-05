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

![](unsupervised_learning.png)

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

![](refresher-k-means.png)

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

![](clustering-probabilistic-model.png)

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

![](normal_distribution_adka_gaussian.png)

## Gaussian Mixture Model (GMM): One Point
- Cluster assignment of point
	- Multinomial distribution on $k$ outcomes
	- $P(Z = j)$ described by $P(C_j) = w_j \geq 0$ with $\sum^k_{j=1} w_j = 1$ 
- Location of point
	- Each cluster has its own Gaussian distribution
	- Location of point governed by its cluster assignment
	- $P(X|Z=j) = N(\mu_j, \Sigma_j)$ class conditional density
- Model's parameters: $w_j, \mu_j, \Sigma_j, j = 1, ..., k$ 

![](gaussin_mixture_model_one_point.png)

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

![](from-marginalisation-to-mixture-distribution.png)

### My Explanation
This equation is just to determine the observed probability of $x$. What this looks like as a plot is shown in the above plot.

## Clustering as Model Estimation
- Given a set of data points, we assume that data points are generated by a GMM
	- Each point in our dataset originates from our mixture distribution
	- Shared parameters between points: _w00t_ independence assumption
- Clustering now amounts to finding parameters of the GMM that "best explains" observed data
- Call upon old friend **MLE** principle to find parameters values that maximise $p(x_1, x_n)$

![](clustering_as_model_estimation.png)

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

![](fitting-the-gmm.png)

### My Explanation
What this is saying is that like MLE, we want to find the parameters (as shown above) that maximises the probability of the function. To do that, we marginalise out the $Z_i$ node from $x_i$ and find the parameters with the above function. This is why we have the inner sum, it is to marginalise out $Z_i$ since it is unobserved, this makes things very difficult for us otherwise if we wanted to do the conditional probability.

## Motivation of EM
- Consider a parametric probabilistic model $p(X| \theta)$, where $X$ denotes data and $\theta$ denotes a vector of parameters
- According to MLE, we need to maximise $p(X|\theta)$ as a function of $\theta$
	- Equivalently maximise $\log p (X|\theta)$ 
- There can be a couple of issues with this task

![](crying.png)

1. Sometimes we **don't observe** some of the variables needed to compute the log likelihood
	- Example: GMM cluster membership $Z$ is not known in advance
2. Sometimes the form of the log-likelihood is **inconvenient*** to work with
	- Example: taking a derivative of GMM log likelihood results in a cumbersome equation

## Expectation-Maximisation (EM) Algorithm

![](expectation-maximisation-algorithm.png)

## Exercises
### Exercise 1
Gradient descent is typically preferred over coordinate descent. Given this, why is coordinate ascent use in the Expectation Maximisation algorithm? 

Because we want to maintain uncertainties as a component in our algorithm. This uncertainty allows us to gauge which location a point belongs to with a certain probability.

## Exercise 2
Gradient descent is typically preferred over coordinate descent. Given this, why is coordinate descent used in the Expectation Maximisation algorithm? 

Because we want to maintain uncertainties as a component in our algorithm. This uncertainty allows us to gauge which location a point belongs to with a certain probability.

## Exercise 3
Describe a benefit of using the _Gaussian mixture model_ (GMM) over _k-means clustering_. 

It also models uncertainty of a point belonging to a certain cluster.

## Exercise 4
Given a dataset comprising four training points $x_1 = [1,1]$, $x_2 = [2,1]$, $x_3 = [3,3]$ and $x_4 = [4,2]$, with labels $y_1=-1$, $y_2 = +1$, $y_3 = -1$ and $y_4 = +1$. We decide to train a linear logistic regression binary classifier on the dataset.

(a) Draw a diagram illustrating the training data, with the decision boundary shown for two training methods: maximum likelihood estimate (MLE), and maximum a posteriori (MAP) estimate where a Gaussian $L_2$ regularisation term is applied to the weight vector. Ensure the graph is labelled clearly. 

![[my-drawing.jpg]]

(b) Draw a second diagram showing the values of $P (y = +1|x)$ along the horizontal line segment $x = [α, 1]$ where $0 ≤ α ≤ 4$, for both the MLE and MAP trained models. Your graph should have α as on the horizontal axis and $P (y = +1|x)$ on the vertical axis. Ensure you label the two methods clearly. 

![[my-drawing2.jpg]]

(c) Next we try our classifier with a non-linear basis function, namely a radial basis function (RBF). Explain how the Bayesian marginal likelihood of the training data can be used to select how many RBF centres to use, and justify why this method should lead to good generalisation accuracy. 

We would first initialise a set of RBF centres.

We could map something the equivalent of:

$P(x_1, ..., x_n) = \prod_{i=1}^n \sum^k_{j=1} P(Z_j) P(x_i | Z_j)$

Then iteratively optimise and update our weights and $Z$ RBF centres