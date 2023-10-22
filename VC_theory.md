## Countably Infinite $F$?

$$
Pr(R[f] - \hat{R}[f] \geq \sqrt{\frac{\log(\frac{1}{\delta(f)})}{2m}}) \leq \delta(f)
$$

... where we're free to choose (varying) $\delta(f)$ in $[0,1]$ 
- Union bound "works" (sort of) for this case

$$
Pr(\exists f \in F, R[f] - \hat{R}[f] \geq \sqrt{\frac{\log(\frac{1}{\delta(f)})}{2m}}) \leq \sum_{f \in F} \delta(f)
$$

- Choose confidences to sum to constant $\delta$, then this works
	- E.g. $\delta(f) = \delta \times p(f) \text{ where } 1 = \sum_{f \in F} p(f)$ 
- By inversion: w.h.p $1 - \delta$, for all $f$, $R[f] \leq \hat{R}[f] + \sqrt{\frac{\log(\frac{1}{p(f)}) + \log(\frac{1}{\delta})}{2m}}$

## Ok Fine, but General Case?
- Much of ML has continuous parameters
	- Countably infinite covers only discrete parameters
- Our argument fails!
	- $p(f)$ becomes a density
	- It's zero for all $f$. No divide by zero!
	- Need a new argument!
 
 ![[divide_zero.png]]
 
- Idea introduced by **VC theory**: intuition
	- Don't focus on whole class $F$ as if each $f$ is different
	- Focus on differences over sample $Z_1, ..., Z_m$ 

What this means is that if we think of the probabilities of different functions $f$ occurring, $p(f)$, then we can think of it much like a Gaussian distribution. Since the probability of part of a Gaussian curve relies on the area under the curve, since just the probability of one function has 0 width, the probability of a function occurring becomes zero. This doesn't really make sense and is therefore an ill-posed problem.
# Growth Function
_Focusing on the size of model families on data samples_

## Bad Events: Unreasonably Worst Case?
- Bad event $B_i$ for model $f_i$ 

$$
R[f_i] - \hat{R}[f_i] \geq \epsilon \text{ with probability } \leq 2 \exp(-2m\epsilon^2)
$$

- Union bound: bad events don't overlap!?

$$
Pr(B_1 \text{ or } B_{|F|}) \leq Pr(B_1)+... + Pr(B_{|F|}) \leq 2|F|\exp(-2m\epsilon^2)
$$

![](Images/overlaps.png)

## How Do Overlaps Arise?

![](Images/overlaps2.png)

Significantly **overlapping** events $B_1$ and $B_2$ 

VC theory focuses on the pattern of labels any $f \in F$ could make

![](Images/overlaps3.png)

## Dichotomies and Growth Function

![](Images/dichotomies.png)

- **Unique dichotomies** $F(x) = \{(f(x_1), ..., f(x_m)): f \in F\}$, patterns of labels with the family
- Even when $F$ infinite, $|F(x)| \leq 2^m$ (why?)
	- Because it is binary
- And also (relevant for $F$ finite, tiny), $|F(x) | \leq |F|$ (why?)
	- Because $|F|$ is infinity
- Intuition: $|F(x)|$ might replace $|F|$ in union bound? How remove $x$? 

![](growth_function.png)

## $S_F(3)$ for $F$ linear classifiers in 2D

![](Images/sf_3.png)
sf_3.png
$|F(x)| = 6$
but still have
$S_F(3) = 8$ 
because as proven in the previous example, we can have an example with 3 samples that has a **dichotomy** of 8.
![[f6-sf-8.png]]

## $S_F(4)$ for $F$ Linear Classifiers in 2D
- What about $m = 4$ points?
- Can never produce the criss-cross (XOR) dichotomy

![](Images/sf_4.png)

- In fact $S_F(4) = 14 < 2^4$ 
	- This is because if we exclude these two _impossible classifications with a linear model_, we will have $2^8 - 2$ dichotomies for 4 samples, which results in $S_F(4) = 14$  
- Guess/exercise: What about general $m$ and dimensions
	- dimensions + 1

## PAC Bound with Growth Function

![](Images/pac_bound_growth_function.png)

- Compare to PAC bounds so far
	- A few negligible extra constants (the 2s, the 4)
	- **$|F|$ has become $S_F(2_m)$** 
	- $S_F(m) \leq |F|$, not "worse" than union bound for finite $F$ 
	- $S_F(m) \leq 2^m$, **very bad for big family with exponential growth** function gets $R[f] \leq \hat{R}[f] + \text{ Big Constant}$. Even $R[f] \leq \hat{R}[f] +1$ meaningless!!

# The VC Dimension
_Computable, bounds growth function_

## Vapnik-Chervonenkis Dimension

![](Images/vc_dimension.png)

- Points $x = (x_1, x_m)$ are **shattered** by $F$ if $|F(x)| = 2^m$ 
- So $VC(F)$ is the size of the largest set shattered by $F$ 
- Example: linear classifiers in $\mathbb{R}^2$, $VC(F)=3$

![](Images/shattered.png)

## Example: $VC(F)$ From $F(x)$ on Whole Domain
- Columns are _all_ points in domain
- Each row is a dichotomy on entire input domain
- Obtain dichotomies on a subset of points $x' \subseteq \{x_1, ... x_2\}$ by columns, drop dupe rows
- $F$ shatters $x'$ if number of rows is $2^{|x'|}$ 

![](Images/shatter_example.png)

This example:
- Dropping column 3 leaves 8 rows behind: $F$ shatters $\{x_1,x_2,x_4\}$ 
- Original table has $< 2^4$ rows: $F$ doesn't shatter more than 3
- $VC(F) = 3$

![](Images/shatter_example2.png)

What this means is that this function $F$ for 4 samples only has a dichotomy of 10, which is less than $2^4$ and thus cannot shatter $m=4$, so what we do instead is we then check if it can shatter $m=3$, which turns out has $2^8$ dichotomies once the duplicate rows were removed. We can therefore say that for this function, $VC(F)=3$, which means that this function can shatter up to $m=3$. 
## Sauer-Shelah Lemma

![](Images/sauer_shela_lemma.png)

- From basic facts of Binomial coefficients
	- Bound is $O(m^k)$: finite VC $\implies$ eventually polynomial growth!
	- For $m \geq k$, it is bounded by $(\frac{em}{m})^k$ 

![](Images/vc_bound.png)

What the above basically means is that we can consider growth function as being bounded a _combination_ in the mathematical term of $k$ shatterable samples of function $F$ and $m$ total samples. However, we should remember that a combination is bounded by $m^k$, simply because it will always be bigger. 

So according to some derivations that we can overlook, we can claim that given that number of samples is greater than or equal to the number of samples that a function can shatter, we can claim the bottom formula as the bound between the true risk of the function and the measured risk. 
## VC Bound Big Picture

![](Images/big_picture.png)

This above graph shows that even if $m$ continues to increase past the $VC(F)$, the number of dichotomies that the function can have becomes lower and lower (as you can imagine with the above examples).

- (Uniform) difference between $R[f], \hat{R}[f]$ is $O(\sqrt{\frac{k\log m}{m}})$ down from $\infty$, where $k$ is the VC dimension. Where $k$ is the VC dimension.
- Limiting complexity of $F$ leads to better generalisation
- VC dim, growth function measure "effective" size of $F$ 
- VC dim doesn't count functions, but uses geometry of family: projections of family members onto possible samples
- Example: linear "gap-tolerant" classifiers (like SVMs) with "margin" $\Delta$ have $VC = O(1/\Delta^2)$. Maximising "margin" reduces VC-dimension.
