import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class QuantumGNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(QuantumGNN, self).__init__()
        # 1. Message Passing Layers (Aggregation & Update)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 2. Output Head (MLP)
        # We need 16 real numbers for a 4x4 L matrix:
        # 4 real diagonals + 6 complex off-diagonals (6 real + 6 imaginary) = 16
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16) 
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Step 1: Aggregation & Update (Hidden Layers)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Step 2: Global Pooling
        # Collapses 2 node vectors into 1 system-level vector
        z = global_mean_pool(x, batch) 
        
        # Step 3: Prediction
        params = self.fc(z)
        
        # Step 4: Physical Constraint Reconstruction
        rho = self.reconstruct_density_matrix(params)
        return rho

    def reconstruct_density_matrix(self, params):
        # params: batch_size x 16
        batch_size = params.shape[0]
        
        # Initialize L matrix as complex zeros
        L = torch.zeros((batch_size, 4, 4), dtype=torch.complex64, device=params.device)
        
        # Diagonal elements (l1, l2, l3, l4) must be positive real
        # We use softplus to ensure they stay > 0
        diag_indices = [0, 1, 2, 3]
        diagonals = F.softplus(params[:, 0:4])
        for i in range(4):
            L[:, i, i] = torch.complex(diagonals[:, i], torch.zeros_like(diagonals[:, i]))
            
        # Off-diagonal elements (lower triangular)
        # We take pairs of values for real and imaginary parts
        idx = 4
        for i in range(4):
            for j in range(i):
                real = params[:, idx]
                imag = params[:, idx + 1]
                L[:, i, j] = torch.complex(real, imag)
                idx += 2
                
        # Calculate unnormalized rho = L * L_dagger
        L_dagger = L.resolve_conj().transpose(1, 2)
        rho_unnorm = torch.bmm(L, L_dagger)
        
        # Normalize by Trace to ensure Unit Trace
        trace = torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1).real
        rho = rho_unnorm / trace.view(-1, 1, 1)
        
        return rho