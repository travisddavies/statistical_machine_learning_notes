# Undirected PGMs
_Undirected variant of PGM, parameterised by arbitrary positive values functions of the variables, and global normalisation_

_A.K.A Markov Random Field_

## Undirected vs Directed
**Undirected PGM**
- Graph
	- Edges undirected
- Probability
	- Each note a r.v.
	- Each clique $C$ has "factor" $\psi_c(X_j:j \in C) \geq 0$ 
	- Joint $\propto$ product of factors
 
**Directed PGM**
- Graph
	- Edged directed
- Probability
	- Each node a r.v.
	- Joint = product of cond'ls
	- Each node has conditional $p(X_i|X_j \in parents(X_i))$ 

**Key difference = normalisation**

## Undirected PGM Formulation
- Based on notion of 
	- **Clique**: a set of _fully connected nodes_ (e.g., A-D, C-D, C-D-F)
	- **Maximal clique**: _largest cliques in graph (not C-D, due to C-D-F_)

![[undirected_pgm_formulation.png]]

- Joint probability defined as

$$
P(a,b,c,d,e,f) = \frac{1}{Z} \psi_1(a,b) \psi_2(b,c) \psi_3(a,d) \psi_4 (d,c,f) \psi_5(d,e)
$$

where each $\psi$ is a positive function and $Z$ is the normalising **'partition' function**

$$
Z = \sum_{a,b,c,d,e,f} \psi_1(a,b) \psi_2(b,c) \psi_3(a,d) \psi_4(d,c,f) \psi_5(d,e)
$$

## Directed to Undirected
- Directed PGM formulated as

$$
P(X_1, X_2, ..., X_k) = \prod^k_{i=1}Pr(X_i|X_{\pi_i})
$$
where $\pi$ indexes parents

- Equivalent to U-PGM with
	- Each conditional probability term is included in one factor function $\psi_c$ 
	- Clique structure links **_groups of variables_**, i.e., $\{\{X_i\} \cup X_{\pi_i}, \forall i\}$ 
	- Normalisation term trivial, $Z=1$

![[undirected_to_directed.png]]

## Why U-PGM
- Pros
	- Generalisation of D-PGM
	- Simpler means of modelling without the need for per-factor normalisation
	- General inference algorithms use U-PGM representation (supporting both types of PGM)
- Cons
	- (slightly) weaker independence
	- Calculating global normalisation term ($Z$) intractable in general 

# Example PGMs
_The hidden Markov model (HMM);_ 
_lattice Markov random field (MRF);_
_Conditional random field (CRF);_

## The HMM (and Kalman Filter)
- Sequential observed **outputs** from hidden state

![[the_hmm.png]]

- The **Kalman filter** same with continuous Gaussian r.v.'s
- A **CRF** is the undirected analogue

![[kalman_filter.png]]

## HMM Applications
- NLP - **part of speech tagging**: given words in sentence, infer hidden parts of speech

"I love Machine Learning $\rightarrow$ noun, verb, noun, noun"

- **Speech recognition**: given waveform, determine phonemes

![[hmm_applications.png]]

- Biological sequences: classification, search, **alignment**
- Computer vision: identify who's walking in video, **tracking**

## Fundamental HMM Tasks

![[fundamental_hmm_tasks.png]]

## Pixel Labelling Tasks in Computer Vision

![[pixel_labelling_in_compute_vision.png]]

## What These Tasks Have in Common
- Hidden state representation semantics of image
	- Semantic labelling: Cow vs. tree vs. grass vs. sky vs. house
	- Fore-back segment: Figure vs. ground
	- Denoising: Clean pixels
- Pixels of image
	- What we observe of hidden state
- Remind you of HMMs?

![[what-these-tasks-have-in-common.png]]

## A Hidden Square-Lattice Markov Random Field

![[a-hidden-square-lattice.png]]

## Application to Sequences: CRFs
- **Conditional Random Field: Same model applied to sequences**
	- Observed outputs are words, speech, amino acids etc
	- States are tags: part-of-speech, phone, alignment
- CRFs are discriminative, model $P(Q|O)$ 
	- Versus HMM's which are generative, $P(Q|O)$
	- Undirected PGM more general and expressive

![[application-to-sequences.png]]