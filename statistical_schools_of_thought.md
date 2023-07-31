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

What this example means is that a frequentist has essentially tried to predict the true $\theta$ value in the above probability distribution by sampling 20 instances, 40 times and taken the mean of the instances each time and recorded it with a blue dot on the above plot. The black triangle shows the true $\theta$ and the red triangle shows the mean from the estimator. The gap between the two triangle is the **bias** of the estimator.

As you can see by the gap, the bias of this mode is quiet low. If the model's probability distribution was more skewed away from the true mean, as shown below, then we would say that the bias is much higher. This would be considered a **bias** estimator.

![](Images/high_bias_estimator.png)

## Estimator Variance
Frequentists seek good behaviour, in ideal conditions
- **Variance**: $Var(\hat{\theta}) = E_{\theta}[(\hat{\theta} - E_{\theta} [\hat{\theta}])^2]$

**Example** continued:
- Plot each $(\hat{\theta}_{i} - E_{\theta}[\hat{\theta}_i])^2 = \hat{\theta}_i^2$ as a blue dot as shown below

![](Images/variance_estimator.png)

What the above is essentially saying is how much does the **estimator's** predictions vary from the mean predicted value, which basically means how stable is our estimator at predicting . This should not be mistaken with the variance of the data itself, this variance is focused on the estimator.

The example above follows the same method as the previous example, except this time it is measuring the average variance of the estimator's predictions and plotting it, then comparing the average variance of all the measurements with the variance of the data itself. As can be seen in the figure above, the variance of the estimator is similar to the true variance.

## Asymptotically Well Behaved
For our example estimator (sample mean), we could calculate its exact bias (zero) and variance ($\sigma^2$). Usually can't guarantee low bias/variance exactly :(

Asymptotic properties often hold! :)
- **Consistency**: $\hat{\theta}(X_1, ..., X_n) \rightarrow \theta$ in probability*
- **Asymptotic efficiency**: $Var_{\theta}(\hat{\theta}(X_1, ..., X_n))$ converges to the smallest possible variance of any estimator of $\theta$

**Note** *:
- The $\rightarrow \theta$ in this case means that bias closer and closer to zero 

What the above is basically saying is that despite some estimators performing poorly on given data there are some asymptotic guarantees. These are that the **bias** and **variance** will converge to their minimum are the data approaches **infinity**. 

## Maximum-Likelihood Estimation
- A **general principle** for designing estimators
- Involves **optimisation**
- $\hat{\theta}(x_1, ..., x_n) \in \underset{\theta \in \varTheta}{\text{argmax}} \prod^n_{i=1} p_{\theta}(x_i)$ 
- "_The best estimator is one under which observed data is most likely_" - Fischer

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
Maximising likelihood yields $\hat{\theta} = \frac{1}{n}\sum^n_{i=1}X_i$

The derivation of this is as follows:
1. Write as joint distribution over all $X$

$$\prod^n_{i=1}p_{\theta}(x_i) = \prod^n_{i=1} \theta^{x_i}(1 - \theta^{1 - x_i})$$   

2. Take the logarithms. **Note**: the log brought the powers down, hence why the $x_i$ and $1-x_i$ are brought down.  The multiplication is split up because of the rule $\log(a \times b) = \log(a) + \log(b)$ . This is also the reason why the product went away and became a sum.

$$
L(\theta) = \log \theta \sum^n_{i=1}x_i + \log(1-\theta) \sum^n_{i=1} (1-x_i) 
$$

3. We can now find the derivative with respect to $\theta$. The derivative of a log is just the inverse, which is why the fractions formed.

$$
\frac{dL}{d\theta} = \frac{\sum^n_{i=1} x_i}{\theta} + \frac{\sum^n_{i=1}(1-x_i)}{1- \theta}
$$

4. We can now simplify this slightly so it's easier to solve, let's change the sums to just $\bar{X}$, and simplify the sum of 1's to just $n$. 

$$
\frac{dL}{d \theta} = \frac{\bar{X}}{\theta} + \frac{n - \bar{X}}{1 - \theta}
$$

5. Set the derivative to zero and then solve for $\theta$

$$
0 = \frac{\bar{X}}{\theta} + \frac{n-\bar{X}}{1 -\theta}
$$

$$
\theta = \frac{1}{n}\sum^n_{i=1}x_i
$$

### Example II: Normal
- Know data comes from Normal distribution with variance 1 but unknown mean; find mean
- MLE for mean