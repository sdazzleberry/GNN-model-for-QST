## Replication guide

This guide details the steps necessary to set up the environment, generate the quantum measurement dataset, and execute the training loop for our GNN-based reconstruction model.
1. Environment Setup
Our project requires Python 3.8+ and the following core libraries for deep learning and quantum simulation:

      * PyTorch: For building the neural network and handling backpropagation.

      * PyTorch Geometric: To implement the GCNConv layers and graph-level pooling.

      * Qiskit: Used for generating the ground-truth density matrices and simulating classical shadows.
2. Dataset Generation:

      * Before training, we must generate a dataset of measurement outcomes (classical shadows) from random 2-qubit states. Our generation script creates a series of random Pauli measurements (X, Y, Z) and records the resulting bitstrings.


3. Training Execution:

      * Once the dataset is ready, we initiate the training process. The model will learn to map the graph-represented measurement data to the parameters of the L matrix.
      * Monitoring: During training, the script will log the Mean Fidelity and Trace Distance to ensure the model is converging toward valid physical states.

      * Saving: The final model weights will be saved to the /outputs directory as a .pt file.