# Lagrangian Duality for the SVM
_An equivalent formulation, with important consequences_

## Soft-Margin SVM Recap
- Soft-margin SVM objective:

$$
\underset{w,b,\xi}{\text{argmin}}\Big(\frac{1}{2} \lVert w \rVert^2 + C \sum^n_{i=1}\Big)
$$

$$
\text{s.t. } y_i(w'x_i+b) \geq 1 - \xi_i \text{ for } i=1,..., n 
$$

$$
 \xi_i \geq 0 \text{ for } i = 1, ..., n 
$$

- While we can optimise the above "**primal**", often instead work with the **dual**

## Constrained Optimisation
- Constrained optimisation: **canonical form**

$$
\text{minimise } f(x)
$$

$$
\text{s.t. } g(x) \leq 0, i=1,...,n
$$

$$
h_j(x) = 0, j=1,...,m
$$

- E.g., find deepest point in the lake, _south of the bridge_
- Gradient descent doesn't immediately apply
- Hard-margin SVM: $\underset{w,b}{\text{argmin}} \frac{1}{2} \lVert w \rVert^2 \text{ s.t. } 1 - y_i(w'x_i+b) \leq 0 \text{ for } i = 1,..., n$ 
- Method of **Lagrange multipliers**
	- Transform to unconstrained optimisation
	- Transform **primal program** to a related **dual program**, alternate to primal
	- Analyse necessary & sufficient conditions for solutions of both programs

### My Explanation
What this all means is that we want to find the optimal value for our SVM, but at the moment we only have our **primal**, as shown in the previous objective function. We have our **equality** and **inequality** constraints, which are $h(x)$ and $g(x)$, which just follow the same constraints as what we stated for the soft-margin SVM objective function. 

We can, however, find our optimal value for the objective function by turning this objective function into a **Lagrangian**.

## The Lagrangian and Duality
- Introduce auxiliary objective function via auxiliary variables

$$
L(x,\lambda, v) = f(x) + \sum^n_{i=1}\lambda_i g_i(x) + \sum^m_{j=1}v_jh_j(x)
$$

- Called the _**Lagrangian**_ function
- New $\lambda$ and $v$ are called the _**Lagrange multipliers** or **dual variables**_ 
- (Old) **primal program**: $\min_x\max_{\lambda\geq 0,v} L(x, \lambda, v)$ `` 
- (New) **dual program**: $\max_{\lambda \geq 0, v}\min_x L(x,\lambda, v)$
	- The new program may be easier to solve, advantageous
  - Duality theory relates primal/dual:
	- Weak duality: dual optimum $\leq$ primal optimum
	- For convex programs (inc. SVM!) **strong duality**: optima coincide!

### My Explanation
What this all means is that we have made a Lagrangian that can find us an optimal value, and we are now essentially penalising the values that are **not less than or equal to zero**. So we are essentially encoding these objectives and constraints into our **dual program** and **primal program**. 

This can be explained by having a deeper look at what happens if we find $\max_{\lambda \geq 0, v}L(x,\lambda, v)$. If we have a sample that breaks $g_i(x) \leq 0$, then we will have a value of $g_i(x)$ that is greater than zero. Now to get the max in this instance, we can simply make $\lambda_i = \infty$, this is the largest possible value we could make for $L(x,\lambda,v)$. Now if we have a value of $x$ so that $h_j(x) \leq 0$, then we can get the largest number by applying $v_j = -\infty$, which will make our $L(x,\lambda,v) = \infty$. If $h_j(x) \geq 0$, then we can apply $v_j = \infty$ to get the same outcome.

We can demonstrate this below:

$L(x,\lambda, v) = f(x) +  \infty \times 1 + \sum^m_{j=1}v_jh_j(x)$

$L(x,\lambda,v) = \infty$

Or like this:

$L(x,\lambda, v) = f(x) +  \infty \times 1 + \sum^m_{j=1}v_jh_j(x)$

$L(x,\lambda, v) = f(x) + \sum^n_{i=1}\lambda_i g_i(x) + -\infty \times -1$

$L(x,\lambda, v) = \infty$  

So now let's consider this: we have a value that meets the inequality constraint, where $g_i(x) \leq 0$, then we will get the max by multiplying this value with $\lambda_i = 0$, this will give us a value of $L(x,\lambda,v) = 0$, and $0 > -\infty$, which is the value we would get if we multiplied $g_i(x)$ by $\infty$. If $h_j(x) = 0$, then no matter what value $v_j$ is, the product will always be 0. Other values will simply send the Lagrangian function to $\infty$. 

A demonstration is shown below:

$L(x,\lambda, v) = f(x) + 0 \times -1 + \sum^m_{j=1}v_jh_j(x)$

Or like this:

$L(x,\lambda, v) = f(x) + \sum^n_{i=1}\lambda_i g_i(x) + -\infty \times 0$

Now that we know how to get the max outcome for this part of the equation, let's look at the next part of the equation. So if we have values of $x$ that don't meet the equality and inequality constraints of the Lagrangian function, we basically won't be able to find the minimum value for $L(x,\lambda,v)$. So as demonstrated, if the equality and inequality constraints are met, then the **max of the inner part of the primal program will be the min of the outer part**. Thus, the optimal value for the Lagrangian function will be found when the constraints are met.

The **dual program** is simply just the max and min swapped around. In general, the dual optimal is more optimal than the primal optimal, and in this case we call this a **weak duality**, otherwise it is a **strong duality**, which basically implies that we can solve the objective function using either program and get the optimal value.

## Karush-Kuhn-Tucker (KKT) Necessary Conditions
- Lagrangian: $L(x,\lambda,v) = f(x) + \sum^n_{i=1}\lambda_i g_i(x) + \sum^m_{i=1}v_j h_j(x)$ \
- Necessary conditions for optimality of a primal solution
	- Souped-up version of necessary condition "derivative of zero" in **unconstrained** optimisation.
- **Primal feasibility**:
	- $g_i(x^*) \leq 0, i = 1,..., n$ 
	- $h_j(x^*) = 0, j=1,...,m$ 
- **Dual feasibility**: $\lambda_i^* \geq 0 \text{ for } i=1,...,n$ 
- **Complementary slackness**: $\lambda_i^*g_i(x^*)=0,i=1,...,n$ 
	- Don't penalise if constraint satisfied
- **Stationary**: $\nabla_xL(x^*,\lambda^*,v^*) =0$ 

<iframe width="560" height="315" src="https://www.youtube.com/embed/uh1Dk68cfWs?si=Yx0hJStEw2dwZzhg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## KKT Conditions for Hard-Margin SVM
The Lagrangian

$$
L(w,b,\lambda) = \frac{1}{2}\lVert w \rVert^2 - \sum^n_{i=1} \lambda_i(y_i(w'x_i+b) - 1)
$$

KKT conditions:
- Feasibility: $y_i((w^*)'x_i + b) - 1 \geq 0 \text{ for } i = 1,...,n$ 
- Feasibility: $\lambda_i^* \geq 0 \text{ for } i = 1,...n$ 
- Complementary slackness: $\lambda^*_i(y_i((w^*)'x_i + b^*) - 1) = 0$  
- Stationary: $\nabla_{w,b}L(w^*,b^*, \lambda^*) = 0$ 

### My Explanation
What this all means is that for the Lagrangian of the hard-margin SVM, we only consider the **inequality constraint**, so we only have $\lambda_i$ as a Lagrangian multiplier. 

For the first feasibility point, this makes sense since the point can either be outside the margin or just on the margin. The complementary slackness rule is just following the same rule that we discussed for KKT necessary conditions. We shall demonstrate:

$$
g_i(w^*,b^*,x_i) = y_i((w^*)'x_i+b^*)-1
$$

This makes sense since $g_i(x)$ is simply the inequality constraint, now if we consider this:

$$
\lambda_i^* g_i(w^*,b^*,x_i) = 0
$$

This rule then matches with the previously stated KKT necessary conditions.

## Let's Minimise Lagrangian w.r.t Primal Variables
- Lagrangian:

$$
L(w,b,\lambda) = \frac{1}{2}\lVert w \rVert^2 - \sum^n_{i=1}\lambda_i(y_i(w'x_i+b)-1)
$$

- Stationary conditions give us more information:

$$
\frac{\delta L}{\delta b} = \sum^n_{i=1}\lambda_iy_i=0
$$

- $\uparrow$  This one becomes our new constraint

$$
\frac{\delta L}{\delta w_j} = w_j^* - \sum^n_{i=1}\lambda_iy_i(x_i)_j = 0
$$

- $\uparrow$ This one eliminates the primal variables
- The Lagrangian becomes (with additional constraint, above)

$$
L(\lambda) = \sum^n_{i=1}\lambda_i - \frac{1}{2}\sum^n_{i=1}\sum^n_{j=1}\lambda_i\lambda_j y_i y_j x'_i x_j
$$

### My Explanation
What this means is that when we solve for $b^*$ and $w^*$, we find get the above solutions, the one for $b^*$ doesn't provide us with a solution, however with some complicated notation that we don't need to cover, this can be used as a constraint in the Lagrangian. The $w^*$ can be used and substituted into the original Lagrangian function, which becomes the above formula. Why this substituted part becomes a negative value also doesn't need to be explained.

What we have essentially done is find the min with respect to x, a.k.a the first part of the **dual program**!

Also note the $j$ index, this means the _element_ within a vector. So  $(x_i)_j$ means the $j$th element in the $i$th instance of $x$. 

## Dual Program for Hard-Margin SVM
- Having minimised the Lagrangian with respect to primal variables, now maximising w.r.t dual variables yields the **dual program**

$$
\underset{\lambda}{\text{argmax}}\sum^n_{i=1}\lambda_i - \frac{1}{2}\sum^n_{i=1}\sum^n_{j=1}\lambda_i\lambda_jy_iy_jx'_ix_j
$$

$$
\text{s.t. } \lambda_i \geq 0 \text{ and } \sum^n_{i=1}\lambda_iy_i = 0
$$

- **Strong duality**: Solving dual, solves the primal!!
- Like primal: A so-called _quadratic program_ - off-the-shelf software can solve - more later
- Unlike primal:
	- Complexity of solution is $O(n^3)$ instead of $O(d^3)$ - more later
	- Program depends on dot products of data only - more later on kernels!

### My Explanation
So now that we have solved the min with respect to $x$ for the first part of the dual program, we now need to just find the max with respect to $\lambda_i$ and we have now got a solution to the dual program!

If you pay attention to the $\sum^n_{i=1}\sum^n_{j=1}y_iy_jx'_ix_j$ part of the solution, this part is actually constant if the data is fixed, therefore the only part that is subject to change in equation are the $\lambda$'s. 

Because the hard-margin SVM has a strong convex property, we can say that it has a strong duality, and therefore if we solve the dual then we also solve the primal.

Also note that we can say that primal and dual optima are always equal because they are both derived from the Lagrangian of the primal problem (as shown above)

## Making Predictions with Dual Solution
Recovering Primal Variables
- Recall from stationarity: $\textcolor{red}{w_j^*} - \sum^n_{i=1}\lambda_iy_i(x_i)_j=0$ 
- Complementary slackness: $\textcolor{red}{b^*}$ can be recovered from dual solution, noting for any example $j$ with $\lambda_i^* > 0$, we have $y_i(b^*+\sum^n_{i=1}\lambda_i^*y_ix'_ix_j) = 1$ (these are the **support vectors**) 
- <u>Testing</u>: classify new instance $x$ based on sign of

$$
s = b^*+\sum^n_{i=1}\lambda^*_iy_ix'_ix
$$

### My Explanation
So since we have solved for $w^*$ and we can recover $b^*$ from the dual solution, we can now find $y=w'x+b$, which is our hyperplane for the SVM, which is shown in the bottom equation.

## Soft-Margin SVM's Dual
- <u>Training</u>: find the $\lambda$ that solves

$$
\underset{\lambda}{\text{argmax}}\sum^n_{i=1}\lambda_i-\frac{1}{2}\sum^n_{i=1}\sum^n_{j=1}\lambda_i\lambda_jy_iy_jx_i'x_j
$$

$$
\text{s.t. } C \geq \lambda_i \geq 0 \text{ and } \sum^n_{i=1}\lambda_iy_i = 0
$$

- <u>Making predictions</u>: same pattern as in hard-margin case

### My Explanation
The dual program for the soft-margin is basically the same as the hard-margin, except we also use the new constraint for $\lambda_i$ 

## Finally... Training the SVM
- The SVM dual problems are quadratic programs, solved in $O(n^3)$, or $O(d^3)$ for the primal.
- This can be inefficient; specialised solutions exist
	- Chunking: original SVM training algorithm exploits fact that many $\lambda$s will be zero (sparsity)
	- Sequential minimal optimisation (SMO), an extreme case of chunking. An iterative procedure that analytically optimises randomly chosen pairs of $\lambda$s per iteration.

# Kernelising the SVM
_Feature transformation by basis expansion; sped up by direct evaluation of kernels - the 'kernel trick'_ 

## Handling Non-Linear Data with the SVM
- Method 1: Soft-margin SVM
- Method 2: **Feature space** transformation
	- Map data into a new feature space
	- Run hard-margin or soft-margin SVM in new space
	- Decision boundary is non-linear in original space

![](data_transformation.png)

## Feature Transformation (Basis Expansion)
- Consider a binary classification problem
- Each example has features $[x_1, x_2]$ 
- Not linearly separable

![](not_linearly_separable.png)

- Now 'add' a feature $x_3 = x_1^2 + x_2^2$ 
- Each point is now $[x_1,x_2,x_1^2+x_2^2]$ 
- Linearly separable

![](linearly_separable_1.png)

## Naive Workflow
- Choose/design a linear model
- Choose/design a high-dimensional transformation $\varphi(x)$ 
	- Hoping that after adding <u>a lot</u> of various features some of them will make the data linearly separable
- **For each** training example, and **for each** new instance, compute $\varphi(x)$ 
- Train classifier/Do predictions
- <u>Problem</u>: **impractical/impossible to compute $\varphi(x)$** for high/infinite-dimensional $\varphi(x)$

## Hard-Margin SVM's Dual Transformation

![](dual_transformation.png)

## Hard-Margin SVM in <u>Feature Space</u> 

![](feature_space.png)

## Observation: Kernel Representation
- Both parameter estimation and computing predictions depend on data <u>only in a form of a dot product </u>
	- In original space $u'v = \sum^m_{m=1}u_iv_i$ 
	- In transformed space $\varphi(u)'\varphi(v) = \sum^l_{i=1}\varphi(u)_i\varphi(v)_i$ 

- **Kernel** is a function that can be expressed as a dot product in some feature space $K(u,v) = \varphi(u)'\varphi(v)$ 

## Kernel as a Shortcut: Example
- For some $\varphi(x)$'s, **kernel is faster to compute** directly than first mapping to feature space then taking dot product.
- For example, consider two vectors $u = [u_1]$ and $v = [v_1]$ and transformation $\varphi(x) = [x_1^2, \sqrt{2c}v_1,c]$, some $c$
	- So $\varphi(u) = \overset{\textcolor{red}{\text{2 operations}}}{[u_1^2, \sqrt{2c}u_1,c]'}$ and $\varphi(v) = \overset{\textcolor{red}{\text{+2 operations}}}{[v_1^2, \sqrt{2c}v_1,c]'}$  
	- Then $\varphi(u)'\varphi(v) = (u_1^2v_1^2+2cu_1v_1+c^2)$ $\textcolor{red}{\text{+4 operations = 8 ops}}$.
- This can be <u>alternatively computed directly as</u>
$$
\varphi(u)'\varphi(v) = (u_1v_1+c)^2 \textcolor{aqua}{\text{ 3 operations}}
$$
- Here $K(u,v) = (u_1v_1+c)^2$ is the corresponding kernel

## More Generally: The "Kernel Trick"
- Consider two training points $x_i$ and $x_j$ and their dot product in the transformed space.
- $k_{ij} = \varphi(x_i)'\varphi(x_j)$ **kernel matrix** can be computed as:
	1. Compute $\varphi(x_i)'$
	2. Compute $\varphi(x_j)$
	3. Compute $k_{ij} = \varphi(x_i)'\varphi(x_j)$
- However, for some transformations $\varphi$, there's a "shortcut" function that gives exactly the same answer $K(x_i,x_j)=k_{ij}$ 
	- Doesn't involve steps 1 - 3 and no computation of $\varphi(x_i)$ and $\varphi(x_j)$
	- Usually $k_{ij}$ computable in $O(m)$, but computing $\varphi(x)$ requires $O(l)$, where $l \gg m$ (**impractical**) and even $l = \infty$ (**infeasible**)

### My Explanation
Basically if we do the feature mapping of $\varphi(x)$, it can take infinite time if we have a dataset with infinite features which is obviously infeasible. However, if we know the kernel function first, we don't need to do these two feature mappings and can instead have a time complexity based off **the number of instances $m$** rather than **the number of features $l$**. This is a solution that is much more efficient for our models.

## Kernel Hard-Margin SVM
- <u>Training</u>: finding $\lambda$ that solve

$$
\underset{\lambda}{\text{argmax}}\sum^n_{i=1}\sum^n_{j=1}\lambda_i\lambda_jy_iy_jK(x_i,x_j)
$$

$$
\text{s.t. } \lambda_i \geq 0 \text{ and } \sum^n_{i=1}\lambda_iy_i
$$

- <u>Making predictions</u>: classify new instance $x$ based on sign of

$$
s = b^* + \sum^n_{i=1}\lambda^*_iy_i\textcolor{red}{K(x_i,x_j)}
$$

- the $K(x_i,x)$ is the **kernel feature mapping**
- Here $b^*$ can be found by noting that for support vector $j$ we have $y_i(b^*+\sum^n_{i=1}\lambda_i^*y_i\textcolor{red}{K(x_i,x_j)})=1$ 

## Approaches to Non-Linearity
### Neural Nets
- Elements of $u = \varphi(x)$ are transformed input $x$
- This $\varphi$ has weights learned from data

![](nnet.png)

### SVMs
- Choice of kernel $K$ determines features $\varphi$ 
- Don't learn $\varphi$ weights
- But, don't even need to compute $\varphi$ so can support very high dim. $\varphi$
- Also support arbitrary data types

# Modular Learning
_Kernelisation beyond SVMs; separating the "learning module" from feature space transformation_

## Modular Learning
- All information about feature mapping is concentrated within the kernel
- In order to use a different feature mapping, simply change the kernel function
- Algorithm design decouples into choosing a "learning method" (e.g., SVM vs logistic regression) and choosing feature space mapping, i.e., kernel
- But how to know if an algorithm is a kernel method?

## Representer Theorem
**Theorem**:
For any training set $\{x_i,y_i\}^n_{i=1}$, any empirical risk function $E$, monotonic increasing function $g$, then any solution

$$
f^* \in \text{argmin}_fE(x_1,y_1,f(x_1),...,x_n,y_n,f(x_n)) + g(\lVert f \rVert)
$$

has representation for some coefficients

$$
f^*(x) = \sum^n_{i=1}\alpha_ik(x,x_i)
$$

- Tells us when a (decision-theoretic) learner is kernelisable
- The dual tells us the form this linear kernel representation takes
- SVM not the only case:
	- Ridge regression
	- Logistic regression
	- Principal component analysis (PCA)
	- Canonical correlation analysis (CCA)
	- Linear Discriminant Analysis (LDA)
	- and many more ...
- Kernel method solutions always in "span" of the data.

### My Explanation
So if we have an optimal model that can be represented as a linear combination of a kernel and some vector of values $\alpha$, then we can say that this function is kernelisable.

# Constructing Kernels
_An overview of popular kernels, kernel properties for building and recognising new kernels_

## Polynomial Kernel
- Function $K(u,v) = (u'v+c)^d$ is called _**polynomial kernel**_ 
	- Here $u$ and $v$ are vectors with $m$ components
	- $d \geq 0$ is an integer and $c \geq 0$ is a constant
- Without loss generality, assume $c=0$ 
	- If it's not, add $\sqrt{c}$ as a dummy feature to $u$ and $v$ 
$(u'v)^d = (u_1v_1+...+u_mv_m)(u_1v_1+...+u_mv_m)...(u_1v_1+...+u_mv_m)$
We can then basically break it down into the below expression, as it is basically just the sum of $u_iv_i$ to some power.

$= \sum^l_{i=1}(u_1v_1)^{a_{i1}}...(u_mv_m)^{a_{im}}$ 

Here $0 \leq a_{ij} \leq d$ and $l$ are integers

This can then be essentially broken down into the below expression, as we can just separate the $u$ values and the $v$ values and give them a power.

$=\sum^l_{i=1}(u_1^{a_{i1}}...u_m^{a_{im}})'(v_1^{a_{i1}}...v_m^{a_{im}})$ 

This therefore is basically a polynomial kernel, which is expressed below.

$=\sum^l_{i=1}\varphi(u)_i\varphi(v)_i$ 

- Feature map: $\varphi: \mathbb{R}^m \rightarrow \mathbb{R}^l$, where $\varphi_i(x) = (x^{a_{i1}}...x_m^{a_{im}})$ 

## Identifying New Kernels
- <u>Method 1</u>: Let $K_1(u,v)$, $K_2(u,v)$ be kernels, $c \geq 0$ be a constant, and $f(x)$ be a real-valued function. Then each of the following is also a kernel:
	- $K(u,v) = K_1(u,v)+K(u,v)$ 
	- $K(u,v) = cK_1(u,v)$ 
	- $K(u,v) = f(u)K_1(u,v)f(v)$ 
- <u>Method 2</u>: Using Mercer's theorem

## Radial Basis Function Kernel
- Function $K(u,v) = \exp(-\gamma \lVert u - v \rVert^2)$ is the **_radial basis function kernel_** (aka Gaussian kernel)
	- Here $\gamma > 0$ is the spread parameter
- $\exp(-\gamma \lVert u - v \rVert^2) = \exp(-\gamma(u - v)'(u-v))$ 

$= \exp(-\gamma(u'u-2u'v+v'v))$

$= \exp(-\gamma u'u)\exp(2\gamma u'v)\exp(-\gamma v'v)$ 

$=f(u)\exp(2\gamma u'v)f(v)$ 

This below part requires Power series expansion

$=f(u)(\sum^{\infty}_{d=0}r_d(u'v)^d)f(v)$ 

- Here, each $(u'v)^d$ is a polynomial kernel. Using kernel identities, we conclude that the middle term is a kernel, and hence the whole expression is a kernel.

## Mercer's Theorem
- Question: given $\varphi(u)$, is there a good kernel to use?
- Inverse question: given some function $K(u,v)$, **is this a valid kernel**? In other words, is there a mapping $\varphi(u)$ implied by the kernel?

**Mercer's theorem**:
- Consider a finite sequence of objects $x_1,...,x_n$
- Construct $n \times n$ matrix of pairwise values $K(x_i,x_j)$ 
- $K$ is a valid kernel is this matrix is positive-semidefinite, for all possible sequences $x_1, ..., x_n$ 

## Handling Arbitrary Data Structures
- Kernels are powerful approach to deal with many data types
- Could define similarity function on variable length strings $K(\text{"science is organised knowledge"}, \text{"wisdom is organised life"})$ 
- However, not every function on two objects is a valid kernel
- Remember that we need that function $K(u,v)$ to imply a dot product in some feature space

# Exercises
## Exercise 1
List two general strategies for proving that a _learning algorithm can be kernelised_. 
1. The mercer's theorem, where we map features into a higher dimensional space and then prove that in that space it can be represented by a positive semi-definite kernel function.
2. Representers theorem, which says that the addition of a valid kernel with another valid kernel regardless of factor, is also a kernel

## Exercise 2
Suppose both $k(u, v)$ and $k^′(u, v)$ are valid kernels on some input space $X$ , with inputs $u, v \in X$. You may assume for simplicity, that these kernels have corresponding feature mappings $φ$, $φ′$ that map inputs $d d′$ from $X$ to $R$ and $R$ respectively. In other words, $k(u, v)$ equals $φ(u)$ dot product with $φ(v)$ (and similarly for the primed kernel and feature map); and the feature maps are finite dimensional . Prove that new function $g(u, v) = k(u, v) × k ^′ (u, v)$ is a valid kernel. 

$g(u,v) = k(u,v)  k'(u,v)$
$g(u, v) = \phi(u) \cdot \phi(v) \times \phi '(u) \cdot \phi '(v)$
$g(u,v) = \sum_{i,j} \phi(u_i) \phi '(u_j) \phi(v_i) \phi '(v_j)$ 
let $\sum_{i,j} \phi(u_i) \phi '(u_j) = \phi_k(u)$, and then apply the same for the other
$g(u,v) = \phi_k(u)\phi_k(v)$ 

Therefore is a kernel

## Exercise 3
The soft-margin support vector machine imposes box constraints on the dual parameters in training, namely $0 ≤ λ_i ≤ C$. Explain how reducing the value of $C$ has the effect of increasing the margin of separation of the training data. You should assume a linear kernel, and recall the margin can be expressed as $\frac{1}{\lVert w \rVert}$ where $w$ are the primal weights. [8 marks]

Given the following:

$\xi_i \geq y_i(w \times x_i + b)$

And:

$$
\underset{w,b,\xi}{\text{argmin}}\Big(\frac{1}{2}\lVert w \rVert^2+C\sum^n_{i=1}\xi_i \Big)
$$

The dual equivalent is:

$$
\underset{\lambda}{\text{argmax}}\sum^n_{i=1}\lambda_i-\frac{1}{2}\sum^n_{i=1}\sum^n_{j=1}\lambda_i\lambda_jy_iy_jx_i'x_j
$$

$$
\text{s.t. } C \geq \lambda_i \geq 0 \text{ and } \sum^n_{i=1}\lambda_iy_i = 0
$$

In this dual for $\lambda_i$, it essentially works as a regulariser just like for in the primal, it must follow the KTT conditions of being greater than one, but is bounded by $C$ just like the primal. 

Therefore the $C$ value essentially applies a certain penalty to the misclassifications. If the $C$ value is low, the penalty is also small and the SVM will naturally have a larger margin at the expense of lots of misclassifications. If the $C$ value is large, the penalty is large and naturally the SVM will aim to make the margin as small as possible to avoid penalties.

(b) We wish to use the function $k(u, v) = u Bv$ as a kernel, where $B$ is a $d × d$ matrix and $u$, $v$ are both d dimensional vectors. Is this a valid kernel? Do any conditions need to be placed on $B$ for this to be the case? Present a mathematical argument to support your answer. [8 marks]

If $u'Bv \geq 0$, and $B$ has no negative eigenvalues. Also, if $u'Bv = (u'Bv)^T$, therefore is symmetrical and we can say that it is positive semi-definite

## Exercise 4
Weak duality guarantees that a primal optimum always upper bounds a dual optimum. How can we tell that the support vector machine’s primal and dual optima are always equal? [5 marks]

Because both are derived from the Lagrangian of the primal problem

## Exercise 5
Suppose you have trained a soft-margin support vector machine (SVM) with a RBF kernel , and the performance is very good on the training set while very poor on the validation set. How will you change the hyperparameters of the SVM to improve the performance of the validation set? List two strategies. [5 marks]

We can adjust the gamma value in the RBF kernel, or we could decrease the $C$ value to increase the margin width

## Exercise 6
For a support vector machine with a quadratic kernel , in what situation, if any, would it be better to use the primal program instead of the dual program for training? [4 marks]

Dual SVM requires $n^3$ time complexity, primal requires $d^3$, therefore a dataset with low dimensionality and high quantities of instances would be better to use primal.

## Exercise 7
If $k_1 (a, a_0 )$ and $k_2 (b, b_0 )$ are both valid kernels over vector valued inputs of size $d_1$ and $d_2$ respectively, prove that

![[q15-20210s1.png]]

is also a valid kernel over vectors of size $d_1$ + $d_2$.

Given that both $k_1$ and $k_2$ are valid kernels, they are both positive semi-definite. Therefore the addition of both kernels will also result in a positive semi-definite kernel and is also a valid kernel.

## Exercise 8
Consider a polynomial kernel of degree $p = 12$ (you may refer to the formula sheet). Explain a benefit of using the kernel trick when using this kernel in the SVM vs explicit basis expansion. 

Using the kernel trick can allow you to save on time complexity and computation by using a factorised version for matrix multiplication, rather than performing a matrix multiplication 12 times. This is an advantage over something like explicit basis expansion

## Exercise 9
Suppose you have trained a soft-margin SVM with a RBF kernel , but the performance is poor on both training and validation sets. How will you change the hyperparameters of the SVM to improve the performance? List two strategies. 

Change gamma and the C-value to have a smaller margin and make tighter bends to fit the data.