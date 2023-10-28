## Basis
- States background and thoughts - fundamental parts -- lectures 1 and 2
- Linear regression -- lecture 3
	- MLE is important
- Regularising linear regression -- lecture 5
	- Ridge regression
	- The lasso
	- Connections to Bayesian MAP
- Regularising non-linear regression -- lecture 5
- Bias-variance -- lecture 5

## Lecture 7
- PAC learning bounds:
	- Countably infinite case works as we've done so far
	- General infinite case? Needs new ideas!
- Growth functions for general PAC case
	- Considering patterns of labels possible on a data set
	- Gives good PAC bounds provided possible patterns don't grow too fast in the set size
- **Vapnik-Chervonenkis (VC) dimension
	- Max number of points that can be labelled in all ways
	- Beyond this point, growth function is polynomial in data set size
	- Leads to famous, general **PAC bound from VC theory**

## Lecture 9
- **Kernelisation (solution to SVM)**
	- Basis expansion on dual formulation of SVMs
	- "Kernel trick"; Fast computation of feature space dot product
- Constructing kernels
	- Overview of popular kernels and their properties
	- Mercer's theorem
	- Learning on unconventional data types

## Lecture 11, 12, 13, 14
- Fundamentals -- lecture 11
	- Networks, layers, activation functions
	- Training by gradient backpropagation
- Training by gradient backpropagation
- Training & Autoencoders -- lecture 12
- Network architectures
	- **Convolutional networks (CNN) -- lecture**
	- Recurrent networks (RNN) -- lecture

## Lecture 18 and 19
- Not included but Bayes theorem is important!!

## Lecture 20, 21
- Direct PGM - Lecture 20
- Independence lowers computational/model complexity
	- Conditional independence
- PGMs: compact representation of factorised joints
- Undirected PGMs - Lecture 21
	- Undirected PGM formulation
	- Directed to undirected

## Lecture 22
- Probabilistic inference: computing (conditional) marginals from the joint distributions
	- Needed to learn (posterior update) in Bayesian ML
	- Exact inference: Elimination algorithm
	- Approximate inference: Sampling
- Statistical inference: Parameter estimation
	- Fully observed case: Factors decompose under MLE
	- Latent variables: Motivates the EM algorithm