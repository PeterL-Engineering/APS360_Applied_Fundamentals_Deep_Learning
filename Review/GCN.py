import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

class DenseGCN(nn.module):
    def __init__(self):
        super(DenseGCN, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, A, x):
        A = A + torch.eye(A.shape[0])
        D = torch.diag(torch.sum(A, dim=0))
        D = torch.pow(D, -0.5)
        A = torch.mm(torch.mm(D, A), D)
        x = torch.mm(A, x)
        x = F.relu(self.fc1(x))
        x = torch.mm(A, x)
        x = F.relu(self.fc2(x))         # Node embeddings
        g = torch.sum(x, dim=0)         # Graph embedding
        y = torch.Sigmoid(self.fc3(g))  # Graph-level prediction
        return y

edge_index = torch.tensor([[0, 1, 1, 2, 3, 3, 3, 4],
                           [3, 2, 3, 1, 0, 1, 4, 3]],
                           dtype=torch.long)

x = torch.tensor([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [1, 0, 10]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=[1.0])

class SparseGCN(nn.Module):
    def __init__(self):
        super(SparseGCN, self).__init__()
        self.gcn1 = nn.GCNConv(3, 32)
        self.gcn2 = nn.GCNConv(32, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        g = torch.sum(x, dim=0)
        y = torch.Sigmoid(self.fc(g))
        return y
    
# DataLoader for GNN/GCN

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)
    break

# Training for GNN/GCN

model = SparseGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

model.train()
for epoch in range(100):
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()