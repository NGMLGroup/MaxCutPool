import os
import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean
from source.data import TUDataModule
from source.layers.maxcutpool.modules import MaxCutPool

# Constants
max_nodes = 150

# Setup paths
path = osp.join('data', 'MUTAG_dense')
os.makedirs(path, exist_ok=True)

# Configuration
args = type('Args', (), {
    'dataset': 'PROTEINS',
    'seed': 42,
    'n_folds': 10,
    'fold_id': 0,
    'batch_size': 20
})

# Initialize data module
data_module = TUDataModule(args)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                     out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, edge_index):
        x1 = self.bn1(self.conv1(x, edge_index).relu())
        x2 = self.bn2(self.conv2(x1, edge_index).relu())
        x3 = self.bn3(self.conv3(x2, edge_index).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * max_nodes)
        
        self.gnn1_embed = GNN(data_module.dataset.num_features, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)

        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, data_module.dataset.num_classes)

        # Initialize MaxCutPool layers
        pool_kwargs = {'ratio': 0.5, 'beta': 1.0}
        score_net_kwargs = {'mp_units': [64], 'mp_act': 'ReLU', 'mlp_units': [32], 'mlp_act': 'ReLU'}
        self.pool1 = MaxCutPool(3*64, **pool_kwargs, **score_net_kwargs)
        self.pool2 = MaxCutPool(3*64, **pool_kwargs, **score_net_kwargs)

    def forward(self, x, edge_index, batch=None):
        # If batch is not provided, assume all nodes belong to one graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.gnn1_embed(x, edge_index)
        x, edge_index, edge_weight, batch, _, _, mc_loss1, _, _ = self.pool1(x, edge_index, edge_weight=None, batch=batch)

        x = self.gnn2_embed(x, edge_index)
        x, edge_index, edge_weight, batch, _, _, mc_loss2, _, _ = self.pool2(x, edge_index, edge_weight=edge_weight, batch=batch)

        x = self.gnn3_embed(x, edge_index)

        x = x.mean(dim=0) if batch is None else scatter_mean(x, batch, dim=0)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), mc_loss1 + mc_loss2

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    loss_all = 0

    for data in data_module.train_dataloader():
        data = data.to(device)
        optimizer.zero_grad()
        output, mc_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y.view(-1)) + mc_loss
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(data_module.train_dataloader().dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)

# Training loop
best_val_acc = test_acc = 0
times = []
for epoch in range(1, 151):
    start = time.time()
    train_loss = train(epoch)
    val_acc = test(data_module.val_dataloader())
    if val_acc > best_val_acc:
        test_acc = test(data_module.test_dataloader())
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")