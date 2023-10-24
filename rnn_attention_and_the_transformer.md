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

![](Images/rnn_architecture.png)

## RNN Applications: Seq. Classification
- Sequence classification: labelling sequence
	- Use last hidden state as input to linear model (classifier etc.)

![](Images/rnn_classification.png)

## Sequence Tagging RNN
- Assign each item/token a label in sequence
	- Given targets per item, can measure loss per item

![](Images/sequence_tagging_rnn.png)

## Encoder-Decoder for Sequence Translation

![](Images/encoder-decoder.png)

## RNN Parameterisation

![](Images/rnn_parameterisation.png)

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

![](Images/lstm.png)

### My Explanation
Due to the transformation that the data goes through when traversing an RNN, some information is lost at each time step. After a while, the RNN's state contains virtually no trace of the first inputs. This can be a showstopper. Imagine Dory the fish trying to translate a long sentence; by the time she's finished reading it, she has no clue how it started. To tackle this problem, various types of cells with long-term memory have been introduced. They have been proven to be so effective that the basic cells are not used much anymore. Let's first look at the most popular of these long-term memory cells: the LSTM cell.

Essentially the utilises a system where a state traverses through the network from left to right and passes through a _forget gate_. This forget gate drops some information from the state. This output is then added with the input state to add new memories via the addition operator. This output then undergoes no more transformations and is added to the output from the output gate. This is how we add long term memory to the model.

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

![](Images/attention.png)

### My Explanation
Attention mechanisms allow a decoder to focus on the appropriate words (as encoded by the encoder) at each time step. For example, we want our decoder to focus more on the word "le football" at each time step if we output the word "soccer". It does this by taking the output of each sequence and performing a weighted sum over the outputs, allowing us to put more focus on certain words in the sequence. For the word "I like soccer", it is appropriate to expect that the verb "like" is very fitting with the noun "soccer". This solves the bottleneck formed when just looking at the output for the last part of the sequence.

## Repeated Attention in Seq2Seq Models
- Consider multiple sequential outputs
	- $s_i = f(s^{(i-1)}, y^{(i-1)}, c_i)$ 
	- $c_i = \sum_j \alpha_{ij}h^{(j)}$ 
	- $\alpha_{ij} = \exp(e_{ij})/\sum_{j'}\exp(e_{ij'})$ 
	- $e_{ij} = a(s^{i-1}, h^{(j)})$ 
- Avoids bottleneck, and uncovers meaningful structure

![](Images/se2seq.png)

## Attention in Vision
- Can attend to other representations, e.g., images
	- Attention over matrix input
	- Roves during generation of caption

![](Images/cnn_transformer.png)

![](Images/bird_kernels.png)

## Self-Attention
- **Transformers** use attention as means of representing sequences directly, instead of RNN
	- Representation of item $i$ is based on attention to the rest of the sequence
	- Use item $i$ as the query in attention against all items $j \not = i$ 
- Compared to RNNs
	- No explicit position information (add to each symbol position index)
	- Cheap: easily done in parallel

![](Images/self-attention.png)

## Transformer

![](Images/transformer1.png)