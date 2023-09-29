CNNs, as you've seen are
1. Flexible
2. Computationally efficient
3. Expressive
4. Give great results

![[cnn_chess.png]]

## The Dark Side of CNNS
But what is assumed when you use a CNN?
The data should be:
1. Regular
2. Dense (or non-sparse)
3. Have some local relational properties
4. Consistent
5. Suitable for taking layers of filtering

## Breaking These Assumptions?
**What happens if these assumptions don't hold?**
You can always try to map data into a more regular structure, or downsample your data - Uber does this for demand estimation

Imputation is another option too, where we try and fill in missing data to make a regular structure

![[cnn_data_organisation.png]]

**And even if the assumptions hold, it doesn't mean that there's not a better way**

## Motivating Question

![[unstructured_cnn.png]]

What happens if the data isn't nicely structured and suitable for a CNN? How do we manage real world data sets?

## Real World Data
Real world data doesn't often exist on nice grids!
1. Traffic Graphs
2. Sensor network data
3. Taxi demand
4. Social network

We need a general language to describe the relationships between entities

One potential solution is to use _graphs_

![[graph_cnn.png]]

This above diagram shows how graph neural networks are able to learn from complex learning spaces such as elements interacting with one another in drugs.

![[graph_network.png]]

It is possible to represent this as a matrix - specifically an adjacency matrix corresponding to all the links. But the matrix would inherently be incredibly sparse, which would make it inappropriate for CNNs.

## Graphs

![[unstructured_cnn.png]]

- Graphs are a collection of Vertices (or Nodes) $V$ and Edges $E$. For deep learning, we presume that the Graph Nodes have attributes $X \in R^{n \times d}$.
- It is also possible that the edges have attributes $X^e \in R^{m \times c}$. These properties can be constant or vary with time.
- With a Graph Convolutional Network, we want to aggregate information from the neighbours of a node.
- A neighbourhood can be more than just the nodes immediate neighbours.

**My explanation**:
What the above is trying to say is that we can think of CNN convolution multiplication kind of like a graph, where we seek out information $n$ neighbours away from the target node. This is basically the logic of a graph neural network, it seeks out information $n$ neighbours away from the target node. However, in this case, both the edge and the node can have attributes.

## Predictions on Graphs
The big question is: How do we take advantage of this relational structure to make better predictions?

Rather than abstracting them to make the data fit modelling frameworks (like CNNs), we instead want to explicitly model these relationships to improve predictive performance.

## Graph Neural Networks
With a graph neural network, we want to learn how to aggregate and propagate information across the graph, in a way that helps us extra **local** (node specific) of **global** (graph specific) features.

![[graph_neural_networks.png]]

## What Exactly is Convolutional Again?
$$
(f * g)(x) \int_{R^d}f(y)g(x-y)dy = \int_{R^d}g(y)dy
$$
- In general, a convolution is the distortion of one function by another, so one takes the properties of the other.
- In a CNN, we project the data onto the convolution kernel, and extract properties about the local neighbourhood within the matrix representation.

## CNN Networks

![[cnn_graph.png]]

$h_i \in R^F$ are the hidden layer activations of each pixel. 
Update the hidden layer by 
$$
h^{(l+1)} = \sigma(\sum_{\forall i}W_i^{(l)}h_i^{(l)})
$$

## Graph Convolutional Networks

![[unstructured_cnn.png]]

To update:
$$
h_0^{(l+1)} = \sigma(\textcolor{red}{W_0^{(l)}} + \textcolor{blue}{W_1^{(l)}h_1^{(l)}+ W_2^{(l)}h_2^{(l)}+W_3^{(l)}h_3^{(l)}+W_4^{(l)}h_4^{(l)}})
$$
Note how **each node** has a **single weight $W_i$**, rather than a unique weight for each link.

But how do we account for the fact that some nodes have fewer connections?
Weight the update, so that:
$$
\textcolor{red}{h_i^{(l+1)}}=\sigma(\textcolor{red}{W_ih_0^{(l)}}+\textcolor{blue}{\sum_{\forall j \in N_i} f(i, |N_j|)h_j^{(l)}W_j^{(l)}})
$$
One possible weighting is $f(i,|N_j|)=\frac{1}{|N_i|}$ 

**My explanation**:
To keep things consistent across the graph where some nodes have fewer connections and some have more, we average the weights based on the number of connections. So we basically sum up all the connections to a node, then divide by the total number of connections to the node.

## Graph Network Extensions
$$
\textcolor{red}{H^{(l+1)}} = \sigma(\textcolor{red}{H^{(l)}W_0^{(l)}}+\textcolor{blue}{\tilde{A}H^{(l)W_1^{(l)}}})
$$
Can be generalised as
$$
H^{(l+1)}=\sigma(H^{(l)}W_0^{(l)}+Agg(\{h_j^{(l)},\forall j \in N_i\}))
$$
Where the $Agg$ function can be nearly anything. Examples include max-pooling
$$
Agg = \gamma(\{Qh\})
$$
where $\gamma$ is a mean, max, or min function. We can also apply LSTMs to the output (or even, in some cases, on the hidden layers too)
$$
Agg = LSTM(h)
$$

**My explanation**:
The above formula is just saying that since graphs are unstructured and we need something like a matrix for out matrix multiplications, what we can do is form an adjacency matrix with the connections like what we would do for a traditional graph (represented with the $\tilde{A}$).

We can also do our traditional aggregation techniques like what we would traditionally do for CNNs such as max-pooling.

## Efficient Graph Updates
Just summing over all the connecting nodes is neither efficient, nor tensor-like. The update procedure can instead by framed as
$$
\textcolor{red}{H^{(l+1)}}=\sigma(\textcolor{red}{H^{(l)}W_0^{(l)}}+\textcolor{blue
}{\tilde{A}H^{(l)}W_1^{(l)}})
$$
where $\tilde{A} = D^{-1/2}AD^{1/2}$ is the Laplacian operator.
$H^{(l)}=[h_1^{(l)}, h_1^{(l)},...,h_N^{(l)}]$. In this $A$ is the graph adjacency matrix, for which $A_{i,j}=1$ if there's a link from node $i$ to $j$, $D$ is the diagonal degree matrix of $A$, where
$$
D_{ii} = \sum_{\forall i}A_{ij}
$$
Instead of using $\tilde{A} = D^{-1/2}AD^{1/2}$, can use the modified Laplacian $L = I_N +D^{-1/2}AD^{1/2}$, so that
$$
H^{(l+1)}= \sigma(LH^{(l)}W^{(l)}+b^{(l)})
$$

**My explanation**:
The above is saying that we can basically apply a Laplacian step to the adjacency matrix to make it essentially like a typical machine learning problem, as shown in the final formula.

## Graph Edge Networks
But real data of interest might exist on the nodes and the edges.

![[graph_cnn_embedding.png]]

Edge hidden weights can be assigned as
$$
h_{(i,j)} = f_e^{(l)}(h_i^{(l)},h_j^{(l)};x_{(i,j)})
$$
Vertex hidden weights are then
$$
h_j^{(l+1)}=f_v^{(l)}(h_i^{(l)},h_j^{(l)};x_{(i,j)})
$$
Vertex hidden weights are then
$$
h_j^{(l+1)}=f_v^{(l)}\Bigg(\sum_{\forall i \forall N_j}h^{(l)}_{(i,j)};x_i\Bigg)
$$

**My explanation**:
The above is basically saying that we can encode edges and learn information about the edges between nodes by applying some function to the information of two connecting nodes and information about the edge itself. So therefore, to update the weights of our edges, we would apply a function to the sum of our neighbouring nodes for the vertex and the information about the edge.

## The Final Layer

![[graph_cnn_final_layer.png]]

The final layer can be processed in a number of ways. $\text{softmax}(z_i)$ can be used for node classification, and $\text{softmax}(\sum z_i)$ for graph classification. The importance of links can be predicted by $\sigma(x_j^T z_j)$. Other activation functions can be used for regression.

## Training
- With a CNN, all information is loaded into memory. If our data is an image, then we load the entire image at once.
- In a GCN, we can do the same thing. But a Graph $G = (E,V)$ has a lot more information to load in, and there's no implicit structure.
- But with a GCN, we don't need global information to train or run the model. So can randomly select a node, expand over its neighbours, and train over a subset of the graph. Changing this subset over training allows global behaviour to be learned.

## Case Study: Google Maps

![[google_map_graph.png]]

Create the graph network, and then at each node enter a sequence of time series data.

![[lstm_google_map_graph.png]]

Behaviour across the whole network can be described by a Graph Convolution Network, embedded within a LSTM-like structure.

![[google_maps_performances.png]]

![[random_google_maps_diagram.png]]

## Case Study: Point Clouds

![[point_clouds.png]]

These can be represented as a density matrix, but this approach may fail in the case of complex geometries, noisy data, or areas with holes. Would require significant preprocessing, as point clouds are irregular and unordered.

![[point_cloud_process.png]]

![[point_cloud_performances.png]]

While GNNs are less efficient than CNNs, as a relatively new and emerging architecture there's still plenty of scope for improvements.

## Overview
The biggest feature of Graph Neural Networks is their flexibility. A GNN can perform almost any operation that you could see with a traditional CNN, but with extra flexibility in how you design the network.
This flexibility in turn allows ML practitioners to both approach a greater range of problems, and to tackle traditional problems with the potential for greater accuracy.
With careful design, even with the additional overhead of managing the graph, they also had the potential to be more scalable to large data sets than other NN approaches.