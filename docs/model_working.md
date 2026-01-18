# GNN Model for Track 1
# Mathematical working


The GNN works to reconstruct a 2-qubit system, in which case our density matrix rho will be a 4x4 complex matrix.
A 2-qubit state lives in a Hilbert space formed by the tensor product of two individual qubits: H1xH2(kronecker product)

which comprise of the Computational Basis States. They are the quantum equivalent of the four possible states of two classical bits(00, 01, 10, 11), which are orthonormal basis vectors in a 4-dimensional complex Hilbert space.

Every entry in our 4x4 density matrix rho corresponds to a combination of these basis states. In which the diagonal elements represent the probabilities of finding the system in that specific basis state.
Non-diagonal elements represent the quantum correlations between different basis states.
Our measurement data will be the input for the GNN, which in turn will understand which of the measurements and guess which of the 4 states our measurements are pointing towards.

The GNN works by treating the quantum system as a graph G = (V, E), where V are our qubits and E represents the connections or interactions between them.
# GNN architecture
**INPUT LAYER**
Node initialisation:

in the first step each qubit i is assigned an initial feature vector. This happens in the first layer(input layer). Each of the 2 nodes receives its measurement data (e.g., node 1 was measured in Z and resulted in $+1$).

**HIDDEN LAYERS**
**Message passing:**

Nodes exchange information to learn how local measurement outcomes on Qubit A relate to Qubit B.

AGGREGATION:

Node I looks at all the other nodes it is connected to, which are called its neighbors. Each neighbor has a set of features or a mathematical "hidden state" from the previous layer.

* Node I collects all of these neighbor states



* It then combines them into a single summary vector using a mathematical function.

* Common functions for this step include taking the average (Mean), adding them together (Sum), or picking the largest value (Max).

* This step is important because it doesn't matter what order the neighbors are in; the summary will be the same.
UPDATE:


Now that Node I has a summary of what its neighbors "know," it needs to update its own state.

* The model takes the neighbor summary and combines it with Node I's own current state.

* This combined information is multiplied by a weight matrix, which contains the parameters the model learns during training.

* Finally, the result is passed through an activation function like ReLU.

* This activation function helps the model learn complex, non-linear patterns in the quantum measurement data.
**Data Pooling:**

At this stage, each of our 2 qubits (nodes) have a final hidden state that contain information about itself and its neighbor. However, the density matrix describes the whole system, not just one qubit.

* The model takes the final hidden states from all nodes in the graph.

* It combines them into one single vector, often called a "graph-level embedding".

* A common method is Global Mean Pooling, where the model calculates the average of all node states.

* This step ensures that the final prediction is based on the combined "knowledge" of the entire qubit network.
**OUTPUT LAYER**

**Prediction stage:**

The single vector from the pooling step is now passed through a standard neural network (often called a Multi-Layer Perceptron or MLP).

* This network maps the combined graph data into a specific number of output values.

* For 2 qubits, you need to predict the values that will fill your 4x4 lower triangular matrix, known as L.

* The model outputs a set of real numbers that represent the real and imaginary parts of the L matrix entries.

* These raw numbers are the final "guess" made by the neural network before the physical constraints are applied.
# Final mathematical constraints

The final mathematical step happens outside the GNN layers to ensure the result is physically valid for a quantum state.

The model uses the predicted L values to build the density matrix using the formula: rho = (L multiplied by its conjugate transpose) divided by the Trace.

This forces the output to be a valid density matrix that is Hermitian, Positive Semi-Definite, and has a Unit Trace.

This final rho is what we compare against our "true" quantum state to calculate our success metrics like Fidelity and Trace Distance.
To ensure our model produces a physically valid quantum state, we do not predict the density matrix $\rho$ directly. Instead, our Graph Neural Network outputs a set of parameters that we use to construct a lower triangular matrix $L$. We then derive the density matrix through a Cholesky-like decomposition.

1. Matrix ConstructionFor our 2-qubit system, the density matrix must be $4 \times 4$. We construct the lower triangular matrix $L$ as follows:$$L = \begin{pmatrix}
l_1 & 0 & 0 & 0 \\
l_5 + il_6 & l_2 & 0 & 0 \\
l_9 + il_{10} & l_{11} + il_{12} & l_3 & 0 \\
l_{13} + il_{14} & l_{15} + il_{16} & l_{17} + il_{18} & l_4
\end{pmatrix}$$
* Diagonal Elements: We enforce that $l_1, l_2, l_3, l_4$ are real and non-negative by applying a softplus or exponential activation function to the model's raw output.
* Off-Diagonal Elements: We treat the remaining model outputs as pairs of real and imaginary components to form the complex entries below the diagonal.

2. Physical Requirement Checks

Once $L$ is constructed, we calculate the unnormalized density matrix $\tilde{\rho} = LL^\dagger$, where $L^\dagger$ is the conjugate transpose. This approach allows our model to satisfy the following requirements automatically:

* Hermitian: Because $\tilde{\rho} = (LL^\dagger)$, it is guaranteed that $\tilde{\rho} = \tilde{\rho}^\dagger$.
* Positive Semi-Definite (PSD): The $LL^\dagger$ form ensures that all eigenvalues of our reconstructed matrix are non-negative, representing valid probability distributions.
* Unit Trace: To satisfy the final constraint, we normalize our matrix by its own trace:$$\rho = \frac{LL^\dagger}{\text{Tr}(LL^\dagger)}$$

3. Performance Metrics

We evaluate the accuracy of our reconstruction by comparing our predicted matrix $\rho$ against the true state $\sigma$ using two primary metrics:
* Quantum Fidelity: We calculate $F(\rho, \sigma) = (\text{Tr} \sqrt{\sqrt{\sigma}\rho\sqrt{\sigma}})^2$ to measure how closely our predicted state overlaps with the target state.
* Trace Distance: We use $D(\rho, \sigma) = \frac{1}{2} \text{Tr} \sqrt{(\rho - \sigma)^\dagger (\rho - \sigma)}$ to quantify the statistical distinguishability between the two states.


our results:
 FINAL EVALUATION
Mean Fidelity:         0.7437
Mean Trace Distance:   0.3965
Avg Inference Latency: 3.40 ms