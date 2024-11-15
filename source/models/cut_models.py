import torch
from torch_geometric.nn import GINConv, MLP
from torch_geometric.nn.resolver import activation_resolver

# Local imports
from source.layers.maxcutpool.modules import MaxCutPool


class CutModel(torch.nn.Module):

    def __init__(self, 
                 in_channels,                           # Size of node features
                 hidden_channels=64,                    # Dimensionality of node embeddings
                 num_layers_pre=1,                      # Number of GIN layers before pooling
                 activation='ELU',                      # Activation of the MLP in GIN 
                 pool_kwargs={},                        # Pooling method kwargs
                 aux_net_kwargs={}                      # Auxiliary network kwargs
                 ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        if activation == "Identity":
            self.act = lambda x: x
        else:
            self.act = activation_resolver(activation)

        # Pre-pooling block            
        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
            self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        if num_layers_pre == 0:
            hidden_channels = in_channels 
        # Pooling block
        score_net_kwargs = aux_net_kwargs
        self.pool = MaxCutPool(hidden_channels, **pool_kwargs, **score_net_kwargs)

    def forward(self, data):
        x = data.x    
        adj = data.edge_index
        batch = data.batch

        ### pre-pooling block
        for layer in self.conv_layers_pre:  
            x = self.act(layer(x, adj))

        ### pooling block
        (x, _, _, _, kept_nodes,
        score, aux_loss, _, _) = self.pool(x, adj, edge_weight=None, batch=batch)

        return x, kept_nodes, score, aux_loss