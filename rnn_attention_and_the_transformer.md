# Recurrent Networks
_A DNN tailored to variable length sequential inputs_

## Sequential Input
- Until now, we have assumed fixed-sized input
	- Vectors of features $x$ in $d$ dimensions
	- Matrices of pixels in an image
- What if our input is a **sequence**?
	- Frames in a video clip
	- Time steps in an audio clip
	- Words in a sentence
	- A protein sequence
	- Stock prices over time
- How can we model this in a DNN?

## FCNNs are Poor for Sequences
- Consider classifying sentences
	- "This is the worst movie of all time, a real stinker" $\rightarrow$ :(
	- "The movie is a real stinker" $\rightarrow$ :(
- Issue: inputs are _**different lengths**_
	- **Pad** them with empty "words" to be a fixed size
- Issue: how do we _**represent words**_ as vectors?
	- Learn an "**embedding**" vector for each word
- Issue: phrases have _**similar meaning**_ even when at _different locations_
	- "a real stinker" is a key predictive feature
	- If we naively apply FCNN needs to learn this concept repeatedly

## ConvNets for Sequences
- Sequences are just rectangular shaped images (e.g., embedding dim. times length): apply CNNs
	- With **1D filters**
	- The filter parameters are shared across time, and can find patterns in the input
- This is called the _**time delay neural network_**
- Downside:
	- Receptive field of filters are limited to finite size, i.e., the width of the convolutional filters, which can be expanded with deeper networks

## Recurrent Neural Nets (RNNs)
- RNNs create networks dynamically, based on input sequence
	- Given sequence of inputs $x^{(1)}, x^{(2)}, ..., x^{(t)}$ 
	- Process each symbol from left to right, to form a sequence of hidden states $h^{(t)}$
	- Each $h^{(t)}$ encodes all inputs up to $t$

![[rnn_architecture.png]]

## RNN Applications: Seq. Classification
- Sequence classification: labelling sequence
	- Use last hidden state as input to linear model (classifier etc.)

![[rnn_classification.png]]

## Sequence Tagging RNN
- Assign each item/token a label in sequence
	- Given targets per item, can measure loss per item

![[sequence_tagging_rnn.png]]

## Encoder-Decoder for Sequence Translation

![[encoder-decoder.png]]

## RNN Parameterisation

![[rnn_parameterisation.png]]

- Parameters are $b, W, U, c, V$
	- Not specific to timestep $t$, but shared across all positions
	- This "template" can get unrolled arbitrarily

## Training RNNs: Backprop. Thru. Time
- Backpropagation algorithms can be applied to network
	- Called **backpropagation through time (BPTT)**
	- Gradients from the loss at every position must be propagated back to the very start of the network
- Suffer from **gradient vanishing** problem
	- Consider linear RNN, gradients of $\frac{\delta g^{(T)}}{\delta h^{(1)}} = W^{T-1}$, thus can explode or vanish with large $T$, depending on largest eigenvalue of $W$ (i.e., greater than / less than one).
	- Can't _learn_ long distance phenomena (over 10+ steps)

## Long Short-Term Memory (LSTM)
- In RNN, previous state is provided as an input
	- Multiplied by weight matrix, and non-linearity applied
- LSTM introduces state self-loop, based on copying
	- Takes **copy** of previous state, scaled by sigmoid **forget gate**
- Gradient magnitude now maintained
	- Can handled 100+ distance phenomena (vs 5-10 for RNN)

![[lstm.png]]

# Transformers
_A method for processing sequence inputs in highly parallelisable manner, using **attention**_

## Attention
- **RNNs** over long sequences not too good at representing properties of the full sequence
	- **Biased** towards the end (or ends) of the sequence
	- Last hidden layer / context: A **bottleneck**!
- **Attention** averages over hidden sequence
	- $x = \sum_j \alpha_j h^{(j)}$ summary weighted average
	- $\alpha_j = \exp(e_j)/(\sum_{j'}\exp(e_{j'}))$ softmax
	- $e_j = f(h^{(j)})$ 
- E.g., key phrase in review

![[attention.png]]

## Repeated Attention in Seq2Seq Models
- Consider multiple sequential outputs
	- $s_i = f(s^{(i-1)}, y^{(i-1)}, c_i)$ 
	- $c_i = \sum_j \alpha_{ij}h^{(j)}$ 
	- $\alpha_{ij} = \exp(e_{ij})/\sum_{j'}\exp(e_{ij'})$ 
	- $e_{ij} = a(s^{i-1}, h^{(j)})$ 
- Avoids bottleneck, and uncovers meaningful structure

![[se2seq.png]]

## Attention in Vision
- Can attend to other representations, e.g., images
	- Attention over matrix input
	- Roves during generation of caption

![[cnn_transformer.png]]

![[bird_kernels.png]]

## Self-Attention
- **Transformers** use attention as means of representing sequences directly, instead of RNN
	- Representation of item $i$ is based on attention to the rest of the sequence
	- Use item $i$ as the query in attention against all items $j \not = i$ 
- Compared to RNNs
	- No explicit position information (add to each symbol position index)
	- Cheap: easily done in parallel

![[self-attention.png]]

## Transformer

![[transformer1.png]]