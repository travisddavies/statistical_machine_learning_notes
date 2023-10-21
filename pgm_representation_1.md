# Probabilistic Graphical Models
_Marriage of graph theory and probabilistic theory. Tool of choice for Bayesian statistical learning._

_We'll stick with easier discrete case, ideas generalise to continuous_

## Motivation by Practical Importance

![[motivation_by_practical_importance.png]]

## Motivation by Way of Comparison

![[motivation_by_way_of_comparison.png]]

## Everything Starts at the Joint Distribution
- All joint distributions on discrete r.v.'s can be represented as tables
- #rows grow exponentially with #rv's
- Example: Truth Tables
	- $M$ Boolean r.v.'s require $2^M-1$ rows
	- Table assigns probability per row
 
![[Statistical Machine Learning/Images/everything_starts_at_the_joint_distribution.png|inlL]]

### My Explanation
We have $2^3$ combinations for this joint probability since our values are boolean. We do $2^3-1$ because we omit the last row, this is because if we know all the other combinations, then we will definitely know the last row. This is also so the sum of the column is equal to 1.

# The Good: What We Can Do with the Joint
- **Probabilistic inference** from joint on r.v.'s
	- Computing any other distributions involving our r.v.'s
- Pattern: want a distribution, have joint; use:
	$\textcolor{blue}{\text{Bayes rule}} + \textcolor{red}{\text{marginalisation}}$
- Example: **naive Bayes classifier**
	- Predict class $y$ of instance $x$ by maximising

$$
Pr(Y=y|X=x) = \textcolor{blue}{\frac{Pr(Y=y, X=x)}{Pr(X=x)}} = \frac{Pr(Y=y, X=x)}{\textcolor{red}{\sum_y Pr(X=x, Y=y)}}
$$

Recall: _integration (over parameters)_ continuous equivalent of sum (both referred to as marginalisation)

## The Bad & Ugly: Tables Waaaaaay Too Large!!
- **The Bad:** Computational complexity
	- Tables have exponential number of rows in number of r.v.'s
	- Therefore $\rightarrow$ poor space & time to marginalise
- **The Ugly**: Model complexity
	- Way too flexible
	- Way too many parameters to fit $\rightarrow$ needs lots of data OR will overfit
- Antidote: assume independence!

![[tables_are_way_to_large.png]]

## Example: You're Late!
- Modelling a tardy lecturer. Boolean r.v.'s
	- $T$: Ben teaches the class
	- $S$: It is sunny (o.w. bad weather)
	- $L$: The lecturer arrives late (o.w. on time)

![[umbrella_chair.png]]

- Assume: Ben sometimes delayed by bad weather, Ben more likely late than other lecturers
	- $Pr(S|T) = Pr(S), \ Pr(S) = 0.3, \ Pr(T) = 0.6$
- Lateness not independent on weather, lecturer
	- Need $Pr(L|T=t, S=s)$ for all combinations
- Need just 6 parameters

![[pov_youre_late.png]]

### My Explanation
We require 6 parameters since we know that we need a parameter each for $Pr(T)$ and $Pr(S)$, and as shown in the table above, we have $2^2$ combinations for $T$ and $S$ since each can be either true or false. Therefore when we sum this up we get $1 + 1 + 2^2 = 6$. We need to know the first two probabilities so we can choose True or False in the table above. 

Therefore, because we assume that $P(T)$ and $P(S)$ are independent, we saved on the number of parameters required for the model. If we didn't assume this independence, we would have $2^3 - 1$ parameters to consider, therefore this is much more efficient.

## Independence: Not a Dirty Word

![[independence_not_a_dirty_word.png]]

- Independence assumptions:
	- Can be reasonable in light of domain expertise
	- Allow us to factor $\rightarrow$ Key to tractable models

## Factoring Joint Distributions
- **Chain Rule**: for <u>any ordering</u> of r.v.'s can always factor:

$$
Pr(X_1, X_2, ..., X_k) = \prod^k_{i=1} Pr(X_i|X_{i+1}, ..., X_k)
$$

- Model's independence assumptions correspond to
	- Dropping conditioning r.v.'s in the factors!
	- Example **unconditional indep.**: $Pr(X_1|X_2) = Pr(X_1)$ 
	- Example **conditional indep.**: $Pr(X_1|X_2, X_3) = Pr(X_1|X_2)$
- Example: independent r.v.'s $Pr(X_1, ..., X_k) = \prod^k_{i=1}Pr(X_i)$ 
- Simpler factors: **speed up inference** and **avoid overfitting**

## Directed PGM
- **Nodes**
- **Edges** (acyclic)
- **Random variables**
- Conditional dependence
	- **Node table**: $Pr(child|parents)$
	- Child directly depends on parents
- **Joint factorisation**

$$
Pr(X_1, X_2, ..., X_k) = \prod^k_{i=1}Pr(X_i|X_j \in parents(X_i))
$$

_Tardy Lecturer Example_

![[tardy_lecture_example.png]]

## Example: Nuclear Power Plant
- Core temperature
	- Temperature gauge
	- Alarm
- Model uncertainty in monitoring failure
	- GRL: gauge reads low
	- CTL: core temperature low
	- FG: faulty gauge
	- FA: faulty alarm
	- AS: alarm sounds
- PGMs to the rescue!

![[nuclear_power_plant_example.png]]

## Naive Bayes

![[naive_bayes_diagram.png]]

## Short-Hand for Repeats: Plate Notation

![[short-hand-for-repeats-plate-notation.png]]

### My Explanation
The above diagram is just showing us that we can simply the Naive Bayes graph into a simple edge and two vertices plot as shown to the right.

## (from here) PGM's: Frequentist OR Bayesian
- PGMs represent joints, which are central to Bayes
- Catch is that Bayesians add: **node per parameters**, with table being the parameter's priors

![[pgms_bayesian.png]]
