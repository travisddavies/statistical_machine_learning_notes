## Motivating Example
- Image classification
	- Instance is matrix of pixels

![[x_and_tick.png]]

- How can we apply a neural net?
	- Flatten into vector, then use fully connected network
 
![[flatten_image.png]]

## Fully-Connected Net, No Spatial Invariane
- Disadvantage: must learn same concept again & again!

![[fc_v_cnn.png]]

- **Translation invariance**: architecture that activates on the same pattern even if "translated" spatially

## Use More Depth?
- **Inefficent**, requires huge number of parameters with more hidden layers. Could **overfit**

![[overfitting_fcann.png]]

- In computer vision, **filters** are small square patterns such as line segments or textures, used as features
- Need ways to: match filters against image (**next**); learn filters
- Key idea: learn **translation invariant** filters - parameter sharing

![[lecun_cnn.png]]

# Convolution Operator
_Allows us to match a small filter across multiple patches of a 2D image or range of a 1D input_

## Convolution
- Concept from signal processing, with wide-spread application
	- Defined as
 
$$
(f * g)(t) = \int^{\infty}_{-\infty}f(\tau)g(t-\tau)d\tau
$$

- Measures how the shape of one function matches the other as it **slides** along
- **ConvNets** use this idea applied to **discrete** inputs

![[conv_graph.png]]

## Convolution in 1D

![[cnn_1d.png]]

![[cnn_1d_1.png]]

## Convolution on 2D Images

![[cnn_on_images.png]]

## Convolution in 2D
- Use filter/kernel to perform element-wise multiplication and sum for every local patch

![[cnn_2d.png]]

## Image Decomposes into Local Patches
- Different local patches include different patterns
	- We can first extract local featurs (local patterns) and then combine features for classification

![[cnn_slide_window.png]]

## Convolution Filters (aka Kernels)
- Filters/kernels can identify different patterns

![[kernels.png]]

- When input and kernel have the same pattern high activation response

## Different Kernels Identify Different Patterns

![[kernel_patterns.png]]

## Convolution in 2D Example (MNIST)
- Response (Feature map) for single kernel

![[mnist_cnn.png]]

- Different kernels identify different patterns: use several filters in each layer of network

## Convolution Parameters
- **Filters are parameters** themselves to be learned (next)
- Key **hyperparameters** in convolution
	- Kernel size: size of the patches
	- Number of filters: depth (channel) of the output
	- Stride: how far to "slide" patch across input
	- Padding of input boundaries with zeros (black here)

## Convolution on Multiple-Channel Input

![[colour-channels.png]]

# Convolutional Neural Networks (CNN)
_Deep networks combining convolutional filters, pooling and other techniques_

## CNN for Computer Vision
- LeNet-5 sparked modern deep models of vision
	- "C" = convolutional, "S" = down-sampling,
	- "F" = fully connected
 
![[lenet.png]]

## Components of a CNN
- **Convolution** layers
	- Complex input representations based on convolution operation
	- Filter **weights are learned** from training data
- Downsampling, usually via **Max Pooling**
	- Re-scales to a smaller resolution, limits parameter explosion
- **Fully connected** parts and output layer
	- Merges representations together

## Downsampling via Max Pooling
- Special types of processing layer. For an $m\times m$ patch 
$$
v = \max(u_{11}, u_{12}, ..., u_{mm})
$$
- Strictly speaking, not everywhere differentiable. Instead, gradient is defined according to "sub-gradient"
	- Tiny changes in values of $u_{ij}$ that is not max do not change $v$
	- If $u_{ij}$ is max value, tiny changes in that value change $v$ linearly
	- Use $\frac{\delta v}{\delta u_{ij}}=1$ if $u_{ij} = v$, and $\frac{\delta v}{\delta u_{ij}}=0$ otherwise
- Forward pass records maximising element, which is then used in the backward pass during back-propagation

## Convolution + Max Pooling $\rightarrow$ Translation Invariance
- Consider shift input image
	- Exact same kernels will activate, with same response
	- Max-pooling over the kernel outputs gives same output
	- Size of max-pooling patch limits the extent of invariance
- Can include padding around input boundaries

![[sliding_window.png]]

## Convolution as a Regulariser

![[convolution_reg.png]]

## Conv Nets Learn Hierarchical Patterns
- Stacking several layers of convolution: larger size receptive field (more of input is seen)

![[conv_net_hierarchy.png]]

## Inspecting Learned Kernels

![[kernel_layers.png]]

## ConvNets in Computer Vision
- ResNet represents modern state-of-the-art
	- Up to 151 layers (!)
	- Mixture of convolutions, pooling, fully connected layers
- Critical innovation is the "**residual connection**"
	- Linear copy of input to output
	- Easier to optimise despite depth, solving gradient vanishing problem
- Standard practise to _**pretrain**_ big model on large dataset, then _**fine-tune**_ (continue training) on small target task

![[resnet.png]]

## ConvNets for Language
- Application of 1d kernels to word sequences
	- Capture patterns of nearby words

![[conv_nets_words.png]]