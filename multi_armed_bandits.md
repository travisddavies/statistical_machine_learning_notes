## Multi-Armed Bandits
_Where we learn to take actions; we receive only indirect supervision in the form of rewards; and we only observe rewards for actions taken - the simplest setting with an explore-exploit trade-off_

## Exploration vs. Exploitation
- "Multi-armed bandit" (MAB)
	- Simplest setting for balancing exploration, exploitation
	- Same family of ML tasks as reinforcement learning
- Numerous applications
	- Online advertising 
	- Caching in databases
	- Stochastic search in games (e.g. AlphaGo!)
	- Adaptive A/B testing

![](pokies.png)

## Stochastic MAB Setting
- Possible actions $\{1,...,k\}$ called "**arms**"
	- Arm $i$ has distribution $P_i$ on bounded **rewards** with mean $\mu_i$
- In round $t=1 ... T$ 
	- Play action $i_t \in \{1,...,k\}$
	- Receive reward $R_{i_t}(t) \thicksim P_{i_t}$ 
- Goal: minimise cumulative **regret**
	- $\mu^* T - \sum^T_{t=1} E[R_{i_t}(t)]$ 
		- $\mu^* = \max_i \mu_i$  
		- $\mu^*T$ is the **best expected cumulative reward with hindsight**
		- $\sum^T_{t=1}E[R_{i_t}(t)]$ is the **expected cumulative reward of bandit**
- Intuition: Do as well as a rule that is simple but has knowledge of the future

## $\epsilon$-Greedy
- At round $t$
	- **Estimate value** of each arm $i$ as average reward observed
 
$$
Q_{t-1}(i) = \begin{cases}
   \frac{\sum^{t-1}_{s=1}R_i(s)1[i_s = i]}{\sum^{t-1}_{s=1}1[i_s = i]}, &\text{if } \sum^{t-1}_{s=1}1[i_s = i] > 0 \\
   Q_0, &\text{otherwise }
\end{cases}
$$

... some init constant $Q_0(i) = Q_0$ used until arm $i$ has been pulled
- This formula basically means that the $Q$ value for arm $i$ is the total sum of rewards for the states that it was pulled on, divided by the number of times the arm was pulled.
- If it is not the initial point, then we just set an initial $Q$ value $Q_0$ 

- **Exploit**, baby exploit... probably; or possibly **explore**
$$
i_t \thicksim \begin{cases}
    \text{argmax}_{1 \leq i \leq k} Q_{t-1}(i) & w.p. 1 - \epsilon \\
   Unif(\{1,...,k\}) & w.p. \epsilon
\end{cases}
$$
- Tie breaking randomly
- This formula means that if we roll a random number and it less than or equal to our value $\epsilon$, then we will pull the arm with the best $Q$ value, otherwise we will randomly pull an arm for exploratory purposes.
- Hyperparam. $\epsilon$ controls exploration vs. exploitation

## Kicking the Tyres

![](kicking_the_tyres.png)

- 10-armed bandit
- Rewards $P_i = Normal(\mu_i,1)$ with $\mu_i \thicksim Normal(0,1)$ 
- Play game for 300 rounds
- Repeat 1,000 games, plot average per-round rewards

## Kicking the Tyres: More Rounds

![](kicking_the_tyres_more_rounds.png)

- Greedy increases fast, but levels off at low rewards
- $\epsilon$-Greedy does **better long-term by exploring**
- 0.01-Greedy initially slow (little explore) but eventually superior to 0.1-Greedy (**exploits after enough exploration**)

## Optimistic Initialisation Improves Greedy

![](optimistic_initialisation_improves_greedy.png)

- **Pessimism**: init $Q$'s below observable rewards $\rightarrow$ Only try one arm
- **Optimism**: Init $Q$'s above observable rewards $\rightarrow$ Explore arms once
- Middle-ground init $Q$ $\rightarrow$ Explores arms at most once

But pure greedy never explores an arm more than once

## Limitations of $\epsilon$-Greedy
- While we can improve on basic Greedy with optimistic initialisation and decreasing $\epsilon$...
- Exploration and exploitation are too "distinct"
	- Exploration actions completely blind to **promising arms**
	- **Initialisation trick** only helps with "cold start"
- Exploitation is blind to **confidence** of estimates
- These limitations are serious in practice

# Upper-Confidence Bound (UCB)
_Optimism in the face of uncertainty: A smarter way to balance exploration-exploitation_

## (Upper) Confidence Interval for Q Estimates
- Theorem: **Hoeffding's inequality**
	- Let $R_1, ..., R_n$ be i.i.d. random variables in $[0,1]$ mean $\mu$, denote by $R_n$ their sample mean
	- For any $\epsilon \in (0,1)$ with probability at least $1 - \epsilon$
 
$$
\mu \leq \bar{R}_n + \sqrt{\frac{\log(1/\epsilon)}{2n}}
$$

- Application to $Q_{t-1}(i)$ estimate - also i.i.d mean!!
	- Take $n = N_{t-1}(i) = \sum^{t-1}_{s=1}1[i_s = i]$ number of $i$ plays
	- Then $\bar{R}_n = Q_{t-1}(i)$ 
	- Critical level $\epsilon = 1/t$, take $\epsilon = 1/t^4$ 

## Upper Confidence Bound (UCB) Algorithm
- At round $t$
	- **Estimate value** of each arm $i$ as average reward observed
$$
Q_{t-1}(i) = \begin{cases}
   \hat{u}_{t-1}(i) + \sqrt{\frac{2\log(t)}{N_{t-1}(t)}}, &\text{if } \sum^{t-1}_{s=1}1[i_s = i] > 0 \\
   Q_0, &\text{otherwise }
\end{cases}
$$
...some constant $Q_0(i) = Q_0$ used until arm $i$ has been pulled; where:
$$
N_{t-1}(i) = \sum^{t-1}_{s=1}1[i_s = i] \ \ \ \hat{u}_{t-1}(i) = \frac{\sum^{t-1}_{s=1}R_i(s)1[i_s=i]}{\sum^{t-1}_{s=1}1[i_s=i]}
$$
- "**Optimism in the face of uncertainty**"

$$
i_t \thicksim \text{argmax}_{i \leq i \leq k} Q_{t-1}(i)
$$

...tie breaking randomly
- This basically means the arm pulled for a given action will be the one with the highest $Q$ value.
- Addresses several limitations of $\epsilon$-greedy
- Can "pause" in a bad arm for a while, but eventually find best

## Kicking the Tyres: How Does UCB Compare?

![](kicking_the_tyre_how_does_ucb_compare.png)

- UCB quickly overtakes the $\epsilon$-greedy approaches

![](kicking_the_tyre_how_does_ucb_compare2.png)

- Continues to overtake on per round rewards for some time

![](kickingthe_tyre_down_the_road_3.png)

- More striking when viewed as mean cumulative rewards

## Notes on UCB
- Theoretical **regret bounds**, optimal up to multiplicative constant
	- Grows like $O(\log(t))$ i.e. averaged regret goes to zero!
- Tunable $\rho > 0$ exploration hyperparam. replaces "2"

$$
Q_{t-1}(i) = \begin{cases}
   \hat{u}_{t-1}(i) + \sqrt{\frac{\textcolor{red}{\rho}\log(t)}{N_{t-1}(t)}}, &\text{if } \sum^{t-1}_{s=1}1[i_s = i] > 0 \\
   Q_0, &\text{otherwise }
\end{cases}
$$

Captures different $\epsilon$ rates & bounded rewards outside $[0,1]$ 
- Many variations e.g. different confidence bounds
- Basis for Monte Carlo Tree Search used in AlphaGo!

# Beyond Basic Bandits
_Adding state with contextual bandits;_
_State transitions/dynamics with reinforcement learning._

## But Wait, There's More!! Contextual Bandits
- Adds concept of "**state**" of the world
	- Arms' rewards now depend on state
	- E.g. best ad depends on user and webpage
- Each round, observe arbitrary context (feature) **vector** representing state $X_i(t)$ per arm
	- Profile of web page visitor (state)
	- Web page content (state)
	- Features of a potential ad (arm)
- Reward estimation
	- Was unconditional: $E[R_i(t)]$
	- **Now conditional**: $E[R_i(t)|X_i(t)]$
- A **regression problem**!!

![](but_wait_theres_more.png)

## MABs vs. Reinforcement Learning
- Contextual bandits introduce state
	- But don't model actions as causing state transitions
	- New state arrives "somehow"
- RL has rounds of states, actions, rewards too
- But (state, action) determines the next state
	- E.g. playing Go, moving a robot, planning logistics
- Thus, RL still learns value functions $w$ regression, but has to "roll out" predicted rewards into the future
