import os
import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
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
    'dataset': 'MUTAG',
    'seed': 42,
    'n_folds': 10,
    'fold_id': 0,
    'batch_size': 20
})

# Initialize data module
data_module = TUDataModule(args)

# Define the GNN model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_features = data_module.dataset.num_features
        num_classes = data_module.dataset.num_classes
        hidden_channels = 64  

        # First GINConv layer
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        # MaxCutPool layer
        pool_kwargs = {'ratio': 0.5, 'beta': 1.0}
        score_net_kwargs = {
            'mp_units': [hidden_channels],
            'mp_act': 'ReLU',
            'mlp_units': [32, 32],
            'mlp_act': 'ReLU'
        }
        self.pool = MaxCutPool(hidden_channels, **pool_kwargs, **score_net_kwargs)

        # Second GINConv layer
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        # Readout layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # First MP layer
        x = self.conv1(x, edge_index)

        # MaxCutPool layer
        x, edge_index, edge_weight, batch, _, _, mc_loss, _, _ = self.pool(
            x, edge_index, edge_weight=None, batch=batch
        )

        # Second MP layer
        x = self.conv2(x, edge_index)

        # Max pooling layer
        x = x.mean(dim=0) if batch is None else scatter_mean(x, batch, dim=0)

        # Readout layer
        x = self.lin(x)

        return F.log_softmax(x, dim=-1), mc_loss

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