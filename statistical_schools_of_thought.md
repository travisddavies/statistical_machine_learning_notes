# Frequentist Statistics
Wherein unknown model parameters are treated as having fixed but unknown values.

## About Frequentist Statistics
- Abstract problem
	- Given: $X_1, X_2, ..., X_n$ drawn i.i.d* from some distribution
	- Want to: identify unknown distribution, or a property of it

- Parametric approach ("**parameter estimator**")
	- Class of models $\{p_{\theta}(x): \theta \in \varTheta\}$ indexed by **parameters** $\varTheta$ (could be a real number, or vector, or ...)
	 - **Point estimate** $\hat{\theta}^*$ $(X_1, ..., X_n)$ a function (or **statistic**) of data

  - Examples:
	  - Given $n$ coin flips, determine probability of landing heads
	  - Learning a classifier

Notes \*:
- i.i.d means **independent and identically distributed**
- The hat in $\hat{\theta}$ means that it is an estimate or estimator

### My Explanation
What this basically means is that frequentist statistics will not consider prior knowledge to form its prediction, it will will rather form a model and do an arbitrary number of experiments to measure its model against the reality.

Take for example a coin being flipped, the frequentist will say that there is no point saying that there's X% probability that it is heads or tails since at this point in time it doesn't have the results to come to a conclusion of the probability. A frequentist will perform $n$ flips and then conclude that if heads occurred 50% of the time, then the probability of heads or tails occurring is 50%.

Frequentists have some ways of measuring the performance of their estimators, as shown with the following:

## Estimator Bias
Frequentists seek good behaviour, in ideal conditions
- **Bias**: $B_{\theta}(\hat{\theta}) = E_{\theta}[\hat{\theta}(X_1, ..., X_n)] - \theta$
	- What this means is that the bias is simply the difference between the actual value and the estimated value given a certain estimator and its parameters.

**Example**: $\text{for } i=1 ... 40$
- $X_{i, 1}, X_{i, 20} \thicksim p_{\theta} = Normal(\theta = 0, \sigma ^ 2)$
- $\hat{\theta}_i = \frac{1}{20} \sum^{20}_{j=1} X_{i,j}$  the sample mean, plot as a blue dot (as shown below)

![](bias_estimator.png)

### My Explanation
What this example means is that a frequentist has essentially tried to predict the true $\theta$ value in the above probability distribution by sampling 20 instances, 40 times and taken the mean of the instances each time and recorded it with a blue dot on the above plot. The black triangle shows the true $\theta$ and the red triangle shows the mean from the estimator. The gap between the two triangle is the **bias** of the estimator.

As you can see by the gap, the bias of this mode is quiet low. If the model's probability distribution was more skewed away from the true mean, as shown below, then we would say that the bias is much higher. This would be considered a **bias** estimator.

![](high_bias_estimator.png)

## Estimator Variance
Frequentists seek good behaviour, in ideal conditions
- **Variance**: $Var(\hat{\theta}) = E_{\theta}[(\hat{\theta} - E_{\theta} [\hat{\theta}])^2]$

**Example** continued:
- Plot each $(\hat{\theta}_{i} - E_{\theta}[\hat{\theta}_i])^2 = \hat{\theta}_i^2$ as a blue dot as shown below

![](variance_estimator.png)

### My Explanation
What the above is essentially saying is how much does the **estimator's** predictions vary from the mean predicted value, which basically means how stable is our estimator at predicting. This should not be mistaken with the variance of the data itself, this variance is focused on the estimator.

The example above follows the same method as the previous example, except this time it is measuring the average variance of the estimator's predictions and plotting it, then comparing the average variance of all the measurements with the variance of the data itself. As can be seen in the figure above, the variance of the estimator is similar to the true variance.

## Asymptotically Well Behaved
For our example estimator (sample mean), we could calculate its exact bias (zero) and variance ($\sigma^2$). Usually can't guarantee low bias/variance exactly :(

Asymptotic properties often hold! :)
- **Consistency**: $\hat{\theta}(X_1, ..., X_n) \rightarrow \theta$ in probability*
- **Asymptotic efficiency**: $Var_{\theta}(\hat{\theta}(X_1, ..., X_n))$ converges to the smallest possible variance of any estimator of $\theta$

**Note** *:
- The $\rightarrow \theta$ in this case means that bias closer and closer to zero 

### My Explanation
What the above is basically saying is that despite some estimators performing poorly on given data, there are some asymptotic guarantees. These guarantees are that the **bias** and **variance** will converge to their minimum are the data approaches **infinity**. 

## Maximum-Likelihood Estimation
- A **general principle** for designing estimators
- Involves **optimisation**
- $\hat{\theta}(x_1, ..., x_n) \in \underset{\theta \in \varTheta}{\text{argmax}} \prod^n_{i=1} p_{\theta}(x_i)$ 
- "_The best estimator is one under which observed data is most likely_" - Fischer

### My Explanation
What this is basically saying is that for the frequentist approach of designing estimators, they use the **maximum-likelihood estimation** approach. This basically is a general principle where the estimator which maximises the probability of predicting observed data. The rule above shows that the product of the probabilities for each parameter is calculated for each model, and the one with the highest probability is chosen as the estimator.

### Example I: Bernoulli
- Know data comes from Bernoulli distribution with unknown parameter (e.g., biased coin); find mean
MLE for mean:

$$p_{\theta}(x) = \begin{cases}
   \theta, &\text{if } x=1 \\
   1-\theta, &\text{if } x = 0
\end{cases}
$$

(note: $p_{\theta}(x) = 0$ for all other $x$)

$$p_{\theta}(x) = \theta^x(1-\theta)^{1-x}$$

Maximising likelihood yields 

$$\hat{\theta} = \frac{1}{n}\sum^n_{i=1}X_i$$

The derivation of this is as follows:
1. Write as joint distribution over all $X$

$$\prod^n_{i=1}p_{\theta}(x_i) = \prod^n_{i=1} \theta^{x_i}(1 - \theta^{1 - x_i})$$   

2. Take the logarithms. **Note**: the log brought the powers down, hence why the $x_i$ and $1-x_i$ are brought down.  The multiplication is split up because of the rule $\log(a \times b) = \log(a) + \log(b)$ . This is also the reason why the product went away and became a sum.

$$
L(\theta) = \log \theta \sum^n_{i=1}x_i + \log(1-\theta) \sum^n_{i=1} (1-x_i) 
$$

3. We can now find the derivative with respect to $\theta$. The derivative of a log is just the inverse, which is why the fractions formed. We get a negative for the second fraction because of the chain rule and the derivative of $-x_i$ is -1.

$$
\frac{dL}{d\theta} = \frac{\sum^n_{i=1} x_i}{\theta} - \frac{\sum^n_{i=1}(1-x_i)}{1- \theta}
$$

4. We can now simplify this slightly so it's easier to solve, let's change the sums to just $\bar{X}$, and simplify the sum of 1's to just $n$. 

$$
\frac{dL}{d \theta} = \frac{\bar{X}}{\theta} - \frac{n - \bar{X}}{1 - \theta}
$$

5. Set the derivative to zero and then solve for $\theta$

$$
0 = \frac{\bar{X}}{\theta} - \frac{n-\bar{X}}{1 -\theta}
$$

$$
\theta = \frac{1}{n}\sum^n_{i=1}x_i
$$

### Example II: Normal
- Know data comes from Normal distribution with variance 1 but unknown mean; find mean
MLE for mean:

$$
p_{\theta}(x) = \frac{1}{\sqrt{2 \pi}}\exp(-\frac{1}{2}(x-\theta)^2)
$$

Maximising likelihood yields 

$$
\hat{\theta} = \frac{1}{n}\sum^n_{i=1}X_i
$$

Exercise: derive MLE for variance $\sigma^2$ based on

$$
p_{\theta}(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2) \text{ with } \theta = (\mu, \sigma^2) 
$$

$$
\prod^n_{i=1}p_{\theta}(x) = \prod^n_{i=1}\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)
$$

$$
\log(L) = \sum^n_{i=1}[-\frac{1}{2}\log(2\pi) -\frac{1}{2}\log(\sigma^2)-\frac{(x_i-\mu)^2}{2\sigma^2}]
$$

$$
	\frac{dL}{d\sigma^2} = \sum^n_{i=1}[-\frac{1}{2\sigma^2}+\frac{(x_i-\mu)^2}{2(\sigma^2)^2}]=0 
$$

$$
-\frac{n}{2\sigma^2}+\frac{\sum^n_{i=1}(x_i-\mu)^2}{2(\sigma^2)^2}=0
$$

$$
\frac{\sum^n_{i=1}(x_i-\mu)^2}{\sigma^2}=n
$$

$$
\sigma^2=\frac{1}{n}\sum^n_{i=1}(x_i-\mu)^2
$$

$$
\frac{dL}{d\mu} = \sum^n_{i=1}[\frac{x_i - \mu}{\sigma^2}] = 0
$$

$$
\sum^n_{i=1}{x_i} - n\mu = 0
$$

$$
\mu = \frac{1}{n}\sum^n_{i=1}x_i
$$
## MLE 'Algorithm'
1. Given data $X_1, ..., X_2$ **define** probability distribution, $p_{\theta}$, assumed to have **generated the data**
2. Express likelihood of data, $\prod^n_{i=1}p_{\theta}(X_i)$ 
	- (usually its **logarithm**... why?) Because logarithms are monotonic, so this prevents stack underflow.
3. Optimise to find _best_ (most likely) parameters $\hat{\theta}$ 
	1. Take partial derivatives of log likelihood wrt $\theta$ 
	2. Set to 0 and solve
	    (failing that, use **gradient descent**)

# Statistical Decision Theory
## Decision Theory
- Act to maximise utility - connected to economics and operations research
- **Decision rule** $\delta(x) \in A$ an action space
	- E.g. point estimate $\hat{\theta}(x_1, ..., x_2)$
	- E.g. Out-of-sample prediction $\hat{Y}_{n+1}|X_1, Y_1, ..., X_n, Y_n, X_{n+1}$ 
- **Loss function** $l(a,\theta)$: economic cost, error metric
	- E.g. square loss of estimate $(\hat{\theta}-\theta)^2$ 
	- E.g. 0-1 loss of classifier predictions $1[y \not = \hat{y}]$ 

### My Explanation
What this basically means is that **decision theory** is about choosing **actions** that are within an **action space** given the provided **data**. The goal is to maximise the **utility** of the decision, which is basically like a **score** for the outcome of the decision.

Take for example the table below, this is an example of utilities from decision theory. If the doctor performs surgery on a patient and the estimation is **correct** that the patient has cancer, then the score is 100%. However, if the doctor is **incorrect** in its prediction and the cancer is absent, then the utility is only 40%. Much like for if the surgery is not performed, the misclassified case has a utility of 0%, and the correct estimation is 85%. The goal of the estimator would be to **minimise** the loss of its estimations as much as possible.  

|                | Surgery Performed | Surgery Not Performed |
|----------------|-------------------|-----------------------|
| Cancer Present | 100%              | 0%                    |
| Cancer Absent  | 40%               | 80%                   |

## Risk & Empirical Risk Minimisation (ERM)
- In decision theory, we really care about _expected_ loss
- **Risk** $R_{\theta}[\delta] = E_{X \thicksim \theta}  [l(\delta (X), \theta)]$
	- E.g. true test error
	- aka generalisation error
- Want: Choose $\delta$ to minimise $R_{\theta}[\delta]$ 
- Can't directly! Why?
	- **Need to come back to this**
 - **ERM**: Use training set $X$ to approximate $R_{\theta}$ 
	 - Minimise **empirical risk**:  $\hat{R}_{\theta} [\delta] = \frac{1}{n}\sum^n_{i=1}l(\delta(X_i), \theta)$ 

### My Explanation
What this basically means is that we really want a very low loss if we were to pick out a random sample and make a prediction with it, so what we do is use the **expected** loss, a.k.a the **risk**, as a measurement for the loss if were to predict a random sample. This is much like how we would measure the average loss from a test dataset and use that as our expectation of the model's loss out in the wild. Therefore, the aim is to minimise the **risk** of our model. 

Much like for other methods of calculating the expected value, the **risk** in this case is measured by calculating the average loss of the model over a finite number of samples

## Decision Theory vs. Bias-Variance
We've already seen

- Bias: $B_{\theta}(\hat{\theta}) = E_{\theta}[\hat{\theta}(X_n, ..., X_n)] - \theta$ 
- Variance: $Var_{\theta}(\hat{\theta}) = E_{\theta}[(\hat{\theta} - E_{\theta}[\hat{\theta}])^2]$ 

But are they equally important? How related?
- **Bias-variance decomposition** of square-loss risk

$$
E_{\theta}[(\theta - \hat{\theta})^2] = [B(\hat{\theta})]^2 + Var_{\theta}(\hat{\theta})
$$ 

### My Explanation
What this basically means is that there is this tension between the **bias** and **variance** for the **loss** of a model. If the bias is high in a model, then the variance will likely be low, and vice-versa. So to **minimise** the loss of the function, both the bias and the variance need to be **reduced**.

It also means that Decision theory's concept of **risk** and the frequentist's concept of **bias** and **variance** are much of a muchness, just a different way of looking at the problem

# Extremum Estimators
Very general framework that covers elements of major statistical learning frameworks; enjoy good asymptotic behaviour in general!!

## About Extremum Estimators
- $\hat{\theta}_n (X) \in \underset{\theta \in \varTheta} {\text{argmin}} Q_n(X, \theta)$ for any objective $Q_n()$ 
- Generalises bits of all statistical frameworks.
	- MLE and ERM seen earlier this lecture; and
	- MAP seen later in this lecture.
	- These are all $M$-estimators, with $Q$ as a sum over data (i.e. of log-likelihood, loss, log-likelihood plus log prior)
- And it generalises other frameworks too!

## Consistency of Extremum Estimators
- Recall consistency: stochastic convergence to 0 bias
- Theorem for extremum estimators: $\hat{\theta}_n \rightarrow \theta$ in probability

## A Game Changer
- Frequentists: estimators that aren't even correct with infinite data (inconsistent), aren't adequate in practice
- Proving consistency for every new estimator? Ouch!
- So many estimators are extremum estimators - general guarantees **make it much easier** (but not easy!) to prove
- **Asymptotic Normality** 
	- Extremum estimators converge to Gaussian in distribution
	- Asymptotic efficiency: the variance of that limiting Gaussian
- Practical: **Confidence intervals** - think error bars!!
$\rightarrow$ Frequentists like to have this asymptotic theory for their algorithms

# Bayesian Statistics
Wherein unknown model parameters have associated distributions reflecting prior belief

## About Bayesian Statistics
- Probabilities correspond to beliefs
- Parameters
	- Modelled as random variables having distributions
	- Prior belief in $\theta$ encoded by **prior distribution** $P(\theta)$ 
		- Parameters are modelled like random variables (even if not really random)
		- Thus: data likelihood $P_{\theta}(X)$ written as conditional $P(X|\theta)$
	- Rather than point estimate $\hat{\theta}$,  Bayesians update belief $P(\theta)$ with observed data to $P(\theta|X)$ the **posterior distribution**)

### My Explanation
 What this basically means is that as opposed to the frequentist approach, Bayesian statistics uses prior knowledge and historic data to make its estimations, and will iteratively update its model as it gets information that goes against its model's estimations.

The model will like a little something like this:

$$
P(X) = P(X|\theta_1, \theta_2, ... \theta_3)
$$

This means that will make its estimations about a sample of data based on prior knowledge of different factors.

## Tools of Probabilistic Inference
- Bayesian probabilistic inference
	- Start with prior $P(\theta)$ and likelihood $P(X|\theta)$ 
	- Observe data $X=x$ 
	- Update prior to posterior $P(\theta|X=x)$

### Primary Tools to Obtain the Posterior
- **Bayes Rule**: reverses order of conditioning (more info [here](math_review.md))

$$
P(\theta|X=x) = \frac{P(X=x| \theta)P(\theta)}{P(x=x)}
$$

- **Marginalisation**: eliminates unwanted variables (more info [here](math_review.md))

$$
P(X=x) = \sum_t P(X=x, \theta = t)
$$

### Example 
- We model $X|\theta$ as $N(\theta, 1)$ with prior $N(0, 1)$
- Suppose we observe $X=1$, then update posterior
 
$P(\theta|X=1) = \frac{P(X=1|\theta)P(\theta)}{P(X=1)}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \propto P(X=1| \theta) P(\theta)$ 

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = [\frac{1}{\sqrt{2\pi}}\exp (-\frac{(1-\theta)^2}{2})][\frac{1}{\sqrt{2\pi}}\exp(-\frac{\theta^2}{2})]$ 

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \propto \exp(-\frac{(1-\theta)^2+\theta^2}{2}) = \exp(-\frac{2\theta^2-2\theta+1}{2})$ 

Now we make the leading numerator coefficient 1: $\times \frac{1}{2}$ on top and bottom

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \propto \exp(-\frac{\theta^2-\theta+\frac{1}{2}}{2 \times \frac{1}{2}})$   

Now we need to set it up as the normal distribution to find where $\sigma$ and $\mu$ are in this new gaussian distribution. In the numerator we need to form the $(x - \mu^2)$ and in the denominator we need to form the $2\sigma^2$. We already can find our $\sigma$ since we already have the 2 out the front, but we still need to find $\mu$. 

To find $\mu$, we need to apply the square trick, which is the following rule:

$\theta^2 - \theta + \frac{1}{2} \rightarrow (\theta-?)^2$ 

$ax^2 +bx + c = a(x + \frac{b}{2a})^2 + c - \frac{b^2}{4a}$ 

$\theta^2 - \theta + \frac{1}{2} = (\theta - \frac{1}{2})^2 + \frac{1}{4}$ 

Now let's separate out the constants from the equation to form the gaussian equation

$= \exp(-\frac{(\theta-\frac{1}{2})^2}{2 \times \frac{1}{2}}) \times \exp(-\frac{\frac{1}{4}}{2 \times \frac{1}{2}})$ 

$\propto \exp(-\frac{(\theta-\frac{1}{2})^2}{2 \times \frac{1}{2}})$ 

$P(\theta|X=1) \propto N(0.5,0.5)$ 

## How Bayesians Make Point Estimates
- They don't, unless forced at gunpoint!
	- The posterior carries full unformation, why discard it?
- But, there are common approaches
	- Posterior mean $E_{\theta|X}[\theta] = \int\theta P(\theta|X)\delta \theta$ 
	- Posterior mode $\underset{\theta}{\text{argmax}}P(\theta|X)$ (max a posteriori or MAP)
	- There're Bayesian decision-theoretic interpretations of these
 
![](MAP.png)

## MLE in Bayesian Context
- MLE formulation: find parameters that best fit data

$$\hat{\theta} \in \underset{\theta}{\text{argmax}}P(X=x| \theta)$$

- Consider the MAP under a Bayesian formulation
$\hat{\theta} \in \underset{\theta}{\text{argmax}} P(\theta|X=x)$  

$\ \ = \underset{\theta}{\text{argmax}}\frac{P(X=x|\theta)P(\theta)}{P(X=x)}$ 

$\ \ = \underset{\theta}{\text{argmax}} P(X=x|\theta)P(\theta)$ 

- Prior $P(\theta)$ weights; MLE like uniform $P(\theta) \propto 1$ 
- What the above dot point means is that the prior acts as a weight for the posterior probability, and if the value is uniform across all parameters, then it is basically a constant
- Extremum estimator: Max $\log P(X=x|\theta) + \log P(\theta)$ 

## Frequentists vs Bayesians - Oh My!
- Two key schools  of statistical thinking
	- Decision theory complements both
- Past: controversy; animosity; almost a 'religious' choice
- Nowadays: deeply connected

![](bayesians_v_frequentists.png)

# Exercises
## Exercise 1
Why must the _evidence_ be computed when evaluating a _Bayesian posterior_, but when maximising the same posterior to find the _max a posteriori (MAP)_ estimate, the _evidence_ can be ignored/cancelled? 

- When finding the posterior, we write the following formula:
$$
P(\theta|X,y) = \frac{p(y|X, \theta)p(\theta)}{p(y|X)}
$$
For MAP, we are interested in the parameters given the data that gives the highest probability. In this case we can write it as:
$$
MAP = \arg \max_{\theta} \frac{p(y|X, \theta)p(\theta)}{p(y|X)}
$$

Where the denominator is the evidence. In this case, the evidence is actually a constant with respect to $\theta$, so we can say that:

$$
MAP \propto \arg \max_{\theta} {p(y|X, \theta)p(\theta)}
$$

## Exercise 2
Why are both maximum-likelihood estimators and maximum a posteriori estimators both asymptotically efficient? 

Because if given infinite data, their variance would converge to 0 of estimator $\theta$

## Exercise 3
Let $E$ be the set of all extremum estimators, $L$ be the set of all maximum-likelihood estimators, and $M$ be the set of all $M$ -estimators. Fill in the blanks in your answers with $E$, $L$, $M$ to make the following expression correct:  $\_\_\_⊂\_\_\_⊂\_\_\_$. 

$\text{MLE} \subset M\text{-estimators} \subset \text{extremum estimators}$

## Exercise 4
(a) Consider an i.i.d. sequence of random variables $X_1 , X_2 , . . .$ coming from some distribution with mean $θ$. Consider a simple estimator $\hat{θ̂}_n = \hat{\theta}(X_1 , . . . , X_n ) = X_1$ of the mean. That is, use the first observation $X_1$ as the estimate and ignore the rest.

$E(\hat{\theta}_n) = E(X_1) = \theta$

(b) Is the estimator $\hat{\theta}_n$ _consistent_? 

No

(c) Explain why your answer to the previous part is correct 

Because variance is not zero, for variance to approach zero the sample size has to approach infinity

## Exercise 5
Consider a setting with high uncertainty over model parameters. Describe what effect, if any, this will have for the maximum likelihood estimate. 

High variance in the parameters of the model

## Exercise 6
Explain why using the training likelihood, $p(y|X, θ)$, for model selection can be problematic when choosing between models from different families. 

Choosing based on training likelihood may lead to choosing a model that is overfitting the training data, not accurately predicting the data to unseen data.

## Exercise 7
In words or a mathematical expression, what quantity is minimised by _linear regression_? 

The squared error between the true label and the predicted label

## Exercise 8
In words or a mathematical expression, what is the _marginal likelihood_ for a _Bayesian probabilistic model_? 

$P(x|\theta) = \int P(\theta|x)P(\theta)\delta \theta$

Where $\theta$ is the parameters, $x$ is the data

## Exercise 9
Explain why maximum likelihood estimation, max a posteriori and empirical risk minimisation are all instances of extremum estimators. 

Because they all involve finding the maximum or minimum of certain objective functions to estimate parameters in statistical and machine learning models.

## Exercise 10
Describe how the frequentist and Bayesian approaches differ in their modelling of unknown parameters. 

Frequentist statistics will not consider prior knowledge to form its prediction, it will will form a model and do an arbitrary number of experiments to measure its model against the reality.

Bayesian statistics uses prior knowledge and historic data to make its estimations, and will iteratively update its model as it gets information that goes against its model's estimations.

## Exercise 11
Why doesn’t max a posteriori estimation require computation of the evidence through costly marginalisation, while computing the posterior distributions does? 

Because it the evidence is a constant with respect to $\theta$, therefore can be can be ignored.
