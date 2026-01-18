import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import QuantumGNN  # Imports your GNN class

def train():
    # 1. Load the generated dataset
    raw_data = torch.load('data/dataset.pt')
    
    # 2. Convert to PyTorch Geometric Data objects
    formatted_data = []
    # Every qubit is connected to the other (fully connected 2-node graph)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    for item in raw_data:
        formatted_data.append(Data(
            x=item['x'], 
            edge_index=edge_index, 
            y=item['y'].unsqueeze(0) # Target density matrix
        ))
    
    loader = DataLoader(formatted_data, batch_size=32, shuffle=True)

    # 3. Initialize Model, Optimizer, and Loss
    model = QuantumGNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # We use Mean Squared Error between the matrices
    criterion = torch.nn.MSELoss()

    # 4. Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(1, 101):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Prediction
            pred_rho = model(batch)
            
            # Instead of a single criterion, split the real and imaginary parts
            # This avoids the ComplexFloat error on CPU
            pred_real = pred_rho.view(-1, 16).real
            pred_imag = pred_rho.view(-1, 16).imag
            target_real = batch.y.view(-1, 16).real
            target_imag = batch.y.view(-1, 16).imag

            loss = criterion(pred_real, target_real) + criterion(pred_imag, target_imag)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.6f}")

    # 5. Save the trained weights
    torch.save(model.state_dict(), 'outputs/model_weights.pt')
    print("Training complete. Weights saved to outputs/model_weights.pt")

if __name__ == "__main__":
    train()