# Probabilistic Inference on PGMs
_Computing marginal and conditional distributions from the joint of a PGM using Bayes rule and marginalisation_

_This deck: how to do it effectively_

## Two Familiar Examples
- Naive Bayes **(frequentist/Bayesian)**
	- Chooses most likely class given data
	- $Pr(Y|X_1, ..., X_d) = \frac{Pr(Y, X_1, ..., X_d)}{Pr(X_1, ..., X_d)} = \frac{Pr(Y, X_1, ..., X_d)}{\sum_y Pr(Y=y, X_1, ..., X_d)}$ 

![[two_familar_examples1.png]]

- Data $X|\theta \thicksim N(\theta, 1)$ with prior $\theta \thicksim N(0, 1)$ **(Bayesian)**
	- Given observation $X = x$ update posterior
	- $Pr(\theta|X) = \frac{Pr(\theta, X)}{Pr(X)} = \frac{Pr(\theta, X)}{\sum_{\theta}Pr(\theta, X)}$  

![[two_familiar_examples2.png]]

- **Joint + Bayes rule + marginalisation $\rightarrow$ anything**

## Nuclear Power Plant
- Alarm sounds; meltdown?!
- $Pr(HT|AS = t) = \frac{Pr(HT, AS=t)}{Pr(AS=t)}$ 
- $\frac{\sum_{FG, HG, FA} Pr(AS=t, FA, HG, FG, HT)}{\sum_{FG, HG, FA, HT'}Pr(AS=t, FA, HR, FG, FG, HT')}$ 

![[nuclear_power_plant_22_example.png]]

- Numerator (denominator similar) 
	expanding out sums, joint _summing once over 2^5 table_
	$= \sum_{FG}\sum_{HG} \sum_{FA} Pr(HT) Pr(HG|HT, FG)Pr(FG)Pr(AS=t|FA, HG)Pr(FA)$ 
	distributing the sums as far down as possible _summing over several smaller tables_	
	
	$=Pr(HT)\sum_{FG}Pr(FG)\sum_{HG} Pr(HG|HT, FG)\sum_{FA}Pr(FA)Pr(AS=t|FA = t | FA, HG)$

## Nuclear Power Plant (Cont.)

![[nuclear_power_plant_cont.png]]

## Elimination Algorithm

![[elimination_algorithm.png]]

## Runtime of Elimination Algorithm

![[runtime_of_elimination_algorithm.png]]

- Each step of elimination
	- Removes a node
	- Connects node's remaining neighbours
		- **forms a clique** in the "reconstructed" graph (_cliques are exactly r.v.'s involved in each sum_)
- Time complexity **exponential in largest clique**
- Different complexity **exponential in largest clique**
	- **Treewidth**: minimum over orderings of the largest clique
	- Best possible time complexity is exponential in the treewidth e.g. $O(2^{tw})$ 

## Probabilistic Inference by Simulation
- Exact probabilistic inference can be expensive/impossible
	- Integration may not have analytical solution!
- Can we approximate numerically?
- Idea: **sampling methods**
	- Approximate **distribution** by **histogram of a sample**
	- We can't trivially sample: (1) only know desired distribution up to a (normalising) constant (2) naive sampling approaches are inefficient in high dimensions

![[probabilistic_inference_by_solution.png]]

## Gibbs Sampling

![[gibbs_sampling.png]]

1. Given: D-PGM on $d$ random variables
2. Given: evidence values $x_E$ over variables $E \subset \{1, ..., d\}$ 
3. Goal: many approximately independent samples from joint conditioned on $x_E$ 

1. Initialise with a string starting $X^{(0)} = \big(X_1^{(0)}, ..., X_d^{(0)} \big)$ with $X_E^{(0)} = x_E$ 
2. Repeat many times
	1. Pick non-evidence node $X_j$ uniformly at random
	2. Sample single node $X_j' \thicksim p(X_j | X_1^{(i-1)}, ..., X_{j-1}^{(i-1)}, X_{j+1}^{(i-1)}, ..., X_d^{(i-1)})$ 
	3. Save entire joint sample $X^{(i)} = \big( X_1^{(i-1)}, ..., X_{j-1}, \textcolor{red}{X_j'}, X_{j+1}^{(i-1)}, ..., X_d^{(i-1)} \big)$ 
- Exercise: Why always $X_E^{(i)} = x_E$?
- Need not update nodes in random order, e.g. **parents first order**. But do need to be able to **sample from conditionals** (e.g. conjugacy)

## Markov Blanket
- Intuition: all nodes that you directly depend on. _Not just your parents/children_!
- Consider node $X_i$ in D-PGM on nodes in $N = \{1, ..., d\}$ 
- Markov blanket MB(i) of $X_i$: 
	- Nodes $B \subseteq N \backslash \{i\}$ such that...
	- $X_i$ independent of $X_{\bar{B} \backslash \{i\}}$ given $X_B$
	- $p(X_i | X_1, ..., X_{i-1}, X_{i+1}, ..., X_d) = p(X_i | MB(X_i))$ 
- In D-PGM Markov blanket is:
	- Parents of $i$, children of $i$, parents of children of $i$
	- $p(X_i | MB(X_i)) \propto p(X_i | X_{\pi_i}) \prod_{k:i \in \pi_k} p(X_k | X_{\pi_k})$ 

![[markov_blanket.png]]

## Now What??
- With our $X^{(1)}, ..., X^{(T)}$ in hand after running Gibbs for a while with burn-in and thinning...
- These form "i.i.d." sample of $p(X_{\bar{E}}|X_E = x_E)$ 
- We can do heaps!
	1. Can approximate the distribution via a histogram of these samples (make bins, form counts)
	2. Marginalising out variables == Dropping components from samples
	3. Expectations: Estimating by sample mean of samples
- Posterior $p(w|X_{tr}, y_{tr})$ combine (a) and (b). Mean posterior point estimate, combine with (c)

# Statistical Inference on PGMs
_Learning from data - fitting probability tables to observations (eg as a frequentist; a **Bayesian would just use probabilistic inference** to update prior to posterior)_

## Have PGM, Some Observations, No Tables

![[have-ogm-some-observation-no-tables.png]]

## Fully-Observed Case in "Easy"
- Max-Likelihood Estimator (MLE) says
	- If we observe _all_ r.v.'s $X$ in a PGM independently $n$ times $x_i$
	- Then maximise the _full_ joint
		$\underset{\theta \in \Theta}{\text{argmax}} \prod^n_{i=1}\prod_j p(X^j = x_i^j | X^{parents(j)} = x_i^{parents(j)})$  

![[fully-observed-case-in-easy.png]]

- Decomposes easily, leads to counts-based estimates
	- Maximise log-likelihood instead; becomes sum of logs
		$\underset{\theta \in \Theta}{\text{argmax}} \sum^n_{i=1} \sum_j \log p (X^j = x_i^j | X^{parents(j)} = x_i^{parents(j)})$ 
		Big maximisation of all parameters together, **decouples into small independent problems**
- Example is training a naive Bayes classifier

## Example: Fully-Observed Case

![[example-fully-observed-case.png]]

## Presence of Unobserved Variables Trickier
- But most PGMs you'll encounter will have latent, or unobserved, variables
- What happens to the MLE?
	- Maximise likelihood of observed data only
	- Marginalise full joint to get to desired "partial" joint
	- $\underset{\theta \in \Theta}{\text{argmax}} \prod^n_{i=1} \sum_{\text{latent} j} \prod_j p(X^j = X_i^j | X^{parents(j)} = x_i^{parents(j)})$
	- This won't decouple - oh no's!!
- Use **EM algorithm**!

