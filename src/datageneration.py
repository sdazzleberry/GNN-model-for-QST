import torch
import numpy as np
from qiskit.quantum_info import random_density_matrix
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def generate_quantum_data(num_samples=1000):
    dataset = []
    simulator = AerSimulator()

    for _ in range(num_samples):
        # Generate a random 2-qubit density matrix (4x4 matrix)
        true_rho = random_density_matrix(4)
        
        # Pick random bases for measurement (0=X, 1=Y, 2=Z)
        bases = np.random.randint(0, 3, size=2)
        
        # Simulate the measurement
        qc = QuantumCircuit(2)
        qc.set_density_matrix(true_rho)
        
        # Apply basis change for measurement
        for i, b in enumerate(bases):
            if b == 0: 
                qc.h(i)      # Measure in X basis
            elif b == 1: 
                qc.sdg(i)    # Measure in Y basis
                qc.h(i)
            # Z basis (b==2) is the default, no gate needed
        
        qc.measure_all()
        
        # Run simulation
        result = simulator.run(transpile(qc, simulator), shots=1).result()
        outcome_str = list(result.get_counts().keys())[0]
        outcomes = [int(bit) for bit in outcome_str] 

        # Store for GNN
        node_features = []
        for i in range(2):
            node_features.append([float(bases[i]), float(outcomes[i])])
            
        dataset.append({
            'x': torch.tensor(node_features, dtype=torch.float),
            'y': torch.tensor(true_rho.data, dtype=torch.complex64)
        })

    return dataset

if __name__ == "__main__":
    print("Generating 1000 quantum samples...")
    data = generate_quantum_data(1000)
    torch.save(data, 'data/dataset.pt')
    print("Dataset saved to data/dataset.pt")