a Multilayer Perceptron
_Modelling non-linearity via function composition_

## Recap: Perceptron Model

![](perceptron_recap.png)

- A linear classifier
- $x_1, x_2$ - inputs
- $w_1, w_2$ - synaptic weights
- $w_0$ - bias weights
- $f$ - activation function

## Limitations of Linear Models
Some problems are linearly separable, but many are not
- In/out value 1
- In/out value 0

![](problem_with_perceptron.png)

Possible solution: **composition**
$x_1 \text{ XOR } x_2 = (x_1 \text{ OR } x_2) \text{ AND } \text{ not} (x_1 \text{ AND } x_2)$ 

We are going to compose perceptrons...

## Perceptron is _Sort of_ a Building Block for ANN
- ANNs are not restricted to binary classification
- Nodes in ANN can have various **activation functions**

![](ann_build_block.png)

_...many others, many variations_

## Feed-Forward Artificial Neural Network

![](feed-forward.png)

## ANN as Function Composition

![](ann_function_composition.png)

## ANN in Supervised Learning
- ANNs can be naturally adapted to various supervised learning setups. Requires setting: output layer dimension, output layer activations, appropriate loss
- Univariate regression $y = f(x)$ 
	- e.g., linear regression earlier in the course
- Multivariate regression $y = f(x)$
	- Predicting values for multiple continuous outcomes
- Binary classification
	- e.g., predict whether a patient has type II diabetes
- Multiclass classification
	- e.g., handwritten digits recognition with labels "1", "2", etc.

## The Power of ANN as a Non-Linear Model
- ANNs are capable of approximating plethora non-linear functions, e.g., $z(x) = x^2$ and $z(x) = \sin(x)$ 
- For example, consider the following network. In this example, hidden unit activation functions are $\tanh$ 

![](ann_example.png)

![](tanh.png)

Blue points are the function values evaluated at different $x$. Red lines are the predictions from the ANN. Dashed lines are outputs of the hidden units

![](fitting_in.png)


- **Universal approximation theorem** (_Cybenko 1989_): An ANN with a hidden layer with a finite numbers of units, and mild assumptions on the activation function, can approximate continuous functions on compact subsets of $R^n$ arbitrarily well

# Deep Learning and Representation Learning
_Hidden layers viewed as feature space transformation_

## Representational Capacity
- ANNs with a single hidden layer are **universal approximators**
- For example, such as ANNs can represent any Boolean function

![](representational_capacity.png)

- Any Boolean function over $m$ variables can be implemented using a hidden layer with up to $2^m$ elements
- More **efficient to stack** several hidden layers

## Deep Networks
"Depth" refers to number of hidden layers

![](dnn.png)

## Deep ANNs as Representational Learning
- Consecutive layers form **representations** of the input of increasing complexity
- An ANN can be have a simple _linear_ output layer, but using complex _non-linear_ representation

$$
z = \tanh(D'(\tanh(C'(\tanh(B')(\tanh(A'x))))))
$$

- Equivalently, a hidden layer can be thought of as the transformed feature space, e.g. $u = \varphi(x)$ compare to **basis** / **kernel learning** 
- Parameters of such a  transformation are learned from data

## ANN Layers as Data Transformation

![](data_transformation_1.png)

![](preprocessing_data_transformation.png)

![](preprocessed_data_transformation2.png)

![](preprocessed_data_transformation3.png)

## Depth vs. Width
- A single arbitrarily wide layer in theory gives a universal approximator
- However (empirically) depth yields more accurate models
	Biological inspiration from the 
	- First detect small edges and colour patches
	- Compose these into smaller shapes
	- Building to more complex detectors, of e.g. texture, faces, etc.
- Seek to mimic layered complexity in a network
- However **vanishing gradient problem** affects learning with very deep models

## Vs Manual Feature Representation
- Standard pipeline
	- Input $\rightarrow$ feature pipeline $\rightarrow$ classification algorithm
- Deep learning automates feature engineering
	- No need for expert analysis
	
![](bird_data.png)

![](bird_type_images.png)

## Backpropagation
##### = "backward propagation of errors"
_Calculating the gradient of loss of a composition_

## Backpropagation: Start with the Chain Rule
- Recall that the output $z$ of an ANN is a function composition, and hence $L(z)$ is also a composition
	- $L = 0.5(z-y)^2 = 0.5(h(s) - y)^2 = 0.5(s-y)^2$ 
	- $= 0.5(\sum^p_{j=0}u_jw_j) = 0.5(\sum^p_{j=0}g(r_j)w_j-y)^2 = ...$ 
- Backpropagation makes use of this fact by applying the **chain rule** for derivatives
- $\frac{\delta L}{\delta w_j}=\frac{\delta L}{\delta z}\frac{\delta z}{\delta s}\frac{\delta s}{\delta w_j}$ 
- $\frac{\delta L}{\delta v_{ij}}=\frac{\delta L}{\delta z}\frac{\delta z}{\delta s}\frac{\delta s}{\delta u_j}\frac{\delta u_j}{\delta r_j}\frac{\delta r_j}{\delta v_{ij}}$ 

![](backprop_ann.png)

## Backpropagation Equations

![](backpropagation_equations.png)

## Forward Propagation

![](foward_propagation.png)

## Backward Propagation of Errors

![](backprop_of_errors.png)

# Exercises
## Exercise 1
Explain how artificial neural networks can be considered to be a form of non-linear basis function when learning a linear model. 

Because it goes through several layers of linear functions, applied with a non-linear activation function through each layer.

## Exercise 2
Consider networks with hundreds of layers of perceptrons - neurons with linear activation functions that output a constant times the weighted sum of their inputs. Are these deep networks? Explain why. 

No. Because the activation functions are linear, this model essentially acts as a linear model and is unable to model complex relationships like a deep neural network.