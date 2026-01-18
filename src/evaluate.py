import torch
import numpy as np
import time
from scipy.linalg import sqrtm
from model import QuantumGNN

def evaluate():
    # 1. Load data and model
    try:
        dataset = torch.load('data/dataset.pt')
        model = QuantumGNN()
        model.load_state_dict(torch.load('outputs/model_weights.pt'))
        model.eval()
    except FileNotFoundError:
        print("Error: Ensure 'data/dataset.pt' and 'outputs/model_weights.pt' exist.")
        return

    fidelities = []
    distances = []
    latencies = []
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    print("Evaluating model performance on 2-qubit reconstruction...")
    
    with torch.no_grad():
        for item in dataset:
            # Prepare single sample for model
            data = Data(x=item['x'], edge_index=edge_index)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            
            # Measure Inference Latency 
            start_time = time.time()
            pred_rho = model(data).squeeze(0).numpy()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) 
            
            true_rho = item['y'].numpy()

            # Manual Fidelity Calculation 
            # Formula: F = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
            try:
                sqrt_rho = sqrtm(pred_rho)
                product = sqrt_rho @ true_rho @ sqrt_rho
                fidelity = np.real(np.trace(sqrtm(product)))**2
                fidelities.append(np.clip(fidelity, 0, 1)) # Keep between 0 and 1
            except:
                continue

            # Manual Trace Distance Calculation 
            # Formula: 0.5 * Tr|rho - sigma|
            diff = pred_rho - true_rho
            evals = np.linalg.eigvals(diff)
            td = 0.5 * np.sum(np.abs(evals))
            distances.append(td)

    # 2. Output Results

    print(f" FINAL EVALUATION")
   
    print(f"Mean Fidelity:         {np.mean(fidelities):.4f}")
    print(f"Mean Trace Distance:   {np.mean(distances):.4f}")
    print(f"Avg Inference Latency: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    from torch_geometric.data import Data # Ensure Data is imported
    evaluate()