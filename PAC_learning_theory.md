# PAC Learning Theory
## Generalisation and Model Complexity
- Theory we've seen so far (mostly statistics)
	- Asymptotic notions (consistency, efficiency)
	- Model complexity undefined
- Want: finite sample theory; convergence _rates_, trade-offs
- Want: define model complexity and relate it to test error
	- Test error can't be measured in real life, but it can be provably bounded!
	- Growth function, VC dimension
- Want: distribution-independent, learner independent theory
	- A fundamental theory applicable _throughout ML_
	- Unlike bias-variance: distribution dependent, no model complexity

# Probably Approximately Correct Learning
_The bedrock of machine learning theory in computer science_

## Standard Set Up
**Problem we consider here**: Supervised binary classification of
- data in $X$ into label set $y = \{-1,1\}$
**What we have**:
- i.i.d data $D^{train} = \{(x_i, y_i)\}^m_{i=1} \thicksim D$ some fixed unknown distribution. The $D^{train}$ is called training data.
- **Training error** of a function $f$ on $D^{train}$ can be expressed by $\hat{R}[f] = \frac{1}{m}\sum^m_{i=1} l(y_i,f(x_i))$ 
**What we will do** in supervised binary classification
- Learn a function $f_m$ from a class of functions $F$ mapping (classifying) $X$ into $y$ such that $f_m = \text{argmin}_{f \in F} \hat{R}[f]$ 

Now we have
$$
f_m = \underset{f \in F}{\text{argmin}}\hat{R}[f] = \text{argmin}_{f \in F} \frac{1}{m}\sum_{i=1}^m l(y_i, f(x_i))
$$
and **want to** analyse the performance of $f_m$ on **new data** from the fixed distribution $D$

Can you write down the test error based on $f_m$ and $D$ ?
$$
R[f_m] = E_{(X,Y)} = E_{(X,Y) \thicksim D}[l(Y,f_m(X))] 
$$
This above function represents the risk (or test error) of $f_m$ on $D$ 

- What parts depend on the sample of data?
	- Empirical risk $\hat{R}[f]$ that averages loss over the sample
	- $f_m \in F$ the learned model (it could be same or different; theory is actually fully general here)

## The Bayes Risk: One Thing We Cannot Ignore
- We usually cannot even hope for perfection!
	- $R^* \in \inf_f R[f]$ called the **Bayes Risk**;
	- **Cannot** expect zero $R[f]$ and a clear decision boundary
- Thus, we care about the following risk more:
$$
R[f_m] - R^*
$$
This is what we call **excess risk**

## Decomposed Risk: The Good, Bad and Ugly
$$
R[f_m] - R^* = (R[f_m] - R[f^*])+(R[f^*]-R^*)
$$
![[formula.png]]

- **Good**: what we'd aim for in our class, with infinite data
	- $R[f^*]$ true risk of **best in class** $f^* \in \text{argmin}_{f \in F}R[f]$ 
- **Bad**: we get what we get and don't get upset
	- $R[f_m]$ true risk of **learned** $f_m \in \text{argmin}_{f \in F} \hat{R}[f] + C \lVert f \rVert^2$ (e.g.)
- **Ugly**: we usually cannot even hope for perfection!
	- $R^* \in \inf_f R[f]$ called the **Bayes risk**;
	- **Cannot** expect zero $R[f]$ and a clear decision boundary

- **Estimation error** is the difference between the test error of your current model and the test error of the best model (i.e. the one with the lowest test error). Generally, the **lower** this error gets, the **more complex** the model gets and risks **overfitting**, and the **higher** the value means that the model is **simpler**.
- **Approximation error** is the different between the test error of your best model and the best test error possible with an infinite choice of models.
## A Familiar Trade-Off: More Intuition
- Simple family $\rightarrow$ may underfit due to approximation error
- Complex family $\rightarrow$ may overfit due to estimation error
![[approx_estimation_error.png]]

## About Bayes Risk
![[bayes_risk.png]]

- **Bayes risk** $R^* \in \inf_f R[f]$
	- Best risk possible, ever; but can be large
	- Depends on distribution and loss function
- **Bayes classifier** achieves Bayes risk
	- $f_{Bayes}(x) = \text{sgn} E(Y|X=x)$ 

## Let's Focus on $R[f_m]$ 
- Since we don't know data distribution, we need to bound generalisation to be small
	- Bound by test error $\hat{R}[f_m] = \frac{1}{m} \sum_{i=1}^m f(X_i, Y_i)$ 
	- Abusing notation: 
$$f(X_i, Y_i) = l(Y_i, f(X_i))$$
$$R[f_m] \leq \hat{R}[f_m] + \epsilon(m, F)$$
- Unlucky training sets, no always guarantees possible!
- With $\text{probability} \geq 1 - \delta: R[f_m] \leq \hat{R}[f_m] + \epsilon(m,F,\delta)$  
- Called Probably Approximately Correct (PAC) learning
	- $F$ called **PAC learnable** if $m = O(poly(1/\epsilon, 1/\delta))$ to learn $f_m$ for any $\epsilon$, $\delta$
	- This means that we don't require exponential growth in training size $m$ 

What this all means is that we can conclude that the **true risk** will be **no more** than the **measured risk plus some confidence value of $\epsilon$**. However, this is not always the case depending on the situation, so what we can say is, as the epsilon of the dataset reduces, the probability of this claim increases. What's more, we can say that **the more data $m$ that we have**, the **lower the $\epsilon$** will be, and therefore the more confident we will be in our claims.     

# Bounding True Risk of One Function
_One step at a time_

## We Need a Concentration Inequality
![[concentration_inequality.png]]

- $\hat{R}[f]$ is an unbiased estimate of $R[f]$ for any fixed $f$ (why?)
	- Because given enough data, the $\hat{R}[f]$ will converge with $R[f]$ 
- That means on average $\hat{R}[f]$ lands on $R[f]$ 
- What's the likelihood $1-\delta$ that $\hat{R}[f]$ lands within $\epsilon$ of $R[f]$? Or more precisely, what $1 - \delta(m,\epsilon)$ achieves a given $\epsilon > 0$?
- Intuition: Just bounding CDF of $\hat{R}[f]$, independent of distribution!!

## Hoeffding's Inequality
- Many such concentration inequalities; a simplest one...
- **Theorem**: Let $Z_1, ... Z_m, Z$ be i.i.d random variables and $h(z) \in [a,b]$ be a bounded function. For all $\epsilon > 0$ 
$$
	Pr(|E[h(Z)] - \frac{1}{m}\sum^m_{i=1}h(Z_i)| \geq \epsilon) \leq 2\exp(-\frac{2m\epsilon^2}{(b-a)^2})
$$
$$
	Pr(E[h(Z)] - \frac{1}{m}\sum^m_{i=1}h(Z_i) \geq \epsilon) \leq \exp(-\frac{2m\epsilon^2}{(b-a)^2})
$$
- Two-sided case in words: The probability that the empirical average is far from the expectation is **small**.
![[bell_curve.png]]

## Et Voila: A Bound on True Risk!
![[bound_on_risk.png]]

### Proof
- Take the $Z_i$ as labelled examples $(X_i, Y_i)$ 
- Take $h(X,Y) = l(Y,f(X))$ zero-one loss for some fixed $f \in F$ then $h(X,Y) \in [0,1]$
- Apply one-sided Hoeffding: $Pr(R[f] - \hat{R}[f] \geq \epsilon) \leq \exp(-2m\epsilon^2)$ 
- Then, substitute $\epsilon = \sqrt{\frac{\log(1/ \delta)}{2m}}$ into the above inequality, we have
- $Pr(R[f] - \hat{R}[f] \geq \sqrt{\frac{1/ \delta}{2m}}) \geq \delta$, i.e., $Pr(R[f]-\hat{R}[f] \leq \sqrt{\frac{\log(1/\delta)}{2m}}) \geq 1 - \delta$ 

## Common Probability 'Tricks'
- Inversion:
	- For any event $A$, $Pr(\bar{A}) = 1 - Pr(A)$ 
	- Application: $Pr(X > \epsilon) \leq \delta$ implies $Pr(X \leq \epsilon) \geq 1 - \delta$ 

![[sets.png]]

- Solving for, in high-probability bounds:
	- For given $\epsilon$ with $\delta(\epsilon)$ function $\epsilon: Pr(X > \epsilon) \leq \delta(\epsilon)$ 
	- Given $\delta'$ can write $\epsilon = \delta^{-1}(\delta'): Pr(X > \delta^{-1}(\delta')) \leq \delta'$ 
	- Let's you specify either parameter
	- Sometimes sample size $m$ a variable we can solve for too

# Uniform Deviation Bounds
_Why we need our bound to **simultaneously** (or uniformly) hold over a family of functions_ 

## Our Bound Doesn't Hold for $f = f_m$ 

![[bounds.png]]

- Result says there's set $S$ of good samples for which $R[f] \leq \hat{R}[f] + \sqrt{\frac{\log(1/\delta)}{2m}}$ and $Pr(Z \in S) \leq 1 - \delta$ 
- But for different functions $f_1, f_2, ...$ we might get very different sets $S_1, S_2, ...$ 
- $S$ observed may be bad for $f_m$. Learning minimises $\hat{R}[f_m]$, exacerbating this

What this means is that although we say that there is some bounds for $f$, in reality each function may have very different results for different samples (i.e. some data will be really bad, and will break our bounding rules), and therefore this bound that we have created won't really hold in reality. Therefore we need to change our perspective on how we should bound the risk.

## Uniform Deviation Bounds

![[deviation_bounds.png]]

- We could analyse risks of $f_m$ from specific learner
	- But repeating for new learners? How to compare learners?
	- Note there are ways to do this, and data-dependently
- Bound uniformly deviations across whole class $F$ 
$$
R[f_m] - \hat{R}[f_m] \leq \sup_{f \in F}(R[f]- \hat{R}[f]) \leq ?
$$

![[pdf_deviation.png]]

- Worst deviation over an entire class bounds learned risk!
- Convenient, but could be much worse than the actual gap for $f_m$ 

What this means is that rather than the previous method for bounding, which could be broken depending on the sample of data used, we will bound based on the largest possible deviation between the mean risk of a function and some risk of the same function but different sample.
## Relation to Estimation Error?
- Recall **estimation error**?  _Learning_ part of excess risk!
$$
R[f_m] - R^* = (R[f_m] - R[f^*])+(R[f^*]-R^*)
$$
- The estimation error part is:
$$
R[f_m] - R[f^*]
$$
**Theorem**: ERM's estimation error is at most twice the uniform divergence
- Proof:

$R[f_m] \leq (\hat{R}[f^*] - \hat{R}[f_m])+R[f_m] - R[f^*]+R[f^*]$ 
$\ \ \ \ \ \ \ \ \ \ = \hat{R}[f^*]-R[f^*]+R[f_m] - \hat{R}[f_m]+R[f^*]$ 
$\ \ \ \ \ \ \ \ \ \ \leq |R[f^*]-\hat{R}[f^*] | + |R[f_m] - \hat{R}[f_m] | + R[f^*]$ 
$\ \ \ \ \ \ \ \  \ \ \leq 2 \sup_{f \in F}|R[f]- \hat{R}[f]| + R[f^*]$     

# Error Bound for Finite Function Classes
_Our first uniform deviation bound_

## The Union Bound
- If each model $f$ having large risk deviation is a "bad event", we need a tool to bound the probability that any bad event happens. I.e. the union of bad events!
- Union bound: for a sequence of events $A_1, A_2, ...$ 
$$
Pr(\cup_i A_i) \leq \sum_i Pr(A_i)
$$
Proof:
Define $B_i = A_i \backslash \cup^{i-1}_{j=1}A_j$ with $B_1 = A_1$ 
1. We know: $\cup_i B_i = \cup_i A_i$ (could prove by induction)
2. The $B_i$ are disjoint (empty intersections)
3. We know: $B_i \subseteq A_i$ so $Pr(B_i) \leq Pr(A_i)$ by monotonically
4. $Pr(\cup_i A_i) = Pr(\cup_i B_i) = \sum_i Pr(B_i) \leq Pr(A_i)$ 

## Bound for Finite Classes $F$ 
- A uniform deviation bound over _any_ finite class or distribution

![[theorem.png]]

Proof:
- If each model $f$ having large risk deviation is a "bad event", we bound the probability that any event happens
- $Pr(\exists f \in F, R[f] - \hat{R}[f] \geq \epsilon) \leq \sum_{f \in F} Pr(R[f] - \hat{R}[f] \geq \epsilon)$ 
- $\leq |F| \exp(-2m \epsilon^2)$ by the union bound
- Followed by inversion, setting $\delta = |F| \exp(-2m\epsilon^2)$ 

## Discussion
- Hoeffding's inequality only uses boundedness of the loss, not the variance of the loss random variables
	- Fancier concentration inequalities leverage variance
- Uniform deviation is worst-case, ERM on a very large over-parameterised $F$ may approach the worst-case, but learners generally may not
	- Custom analysis, data-dependent bounds, PAC-Bayes, etc.
- Dependent data?
	- Martingale theory
- Union bounds is in general loose, as bad is if all the bad events were independent (not necessarily the case even though underlying data modelled as independent); and **finite $F$** 
	- VC theory coming up next!
