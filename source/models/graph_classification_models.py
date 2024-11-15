import torch
from torch.nn import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import (GINConv, GINEConv, MLP, DenseGINConv,
                                dense_mincut_pool, dense_diff_pool, 
                                global_add_pool, DMoNPooling, TopKPooling,
                                ASAPooling, graclus)

# Local imports
from source.layers.maxcutpool.modules import MaxCutPool
from source.layers.edgepool.edge_pool import EdgePooling
from source.layers.kmis.kmis_pool import KMISPooling
from source.layers.sum_pool import sum_pool


class GraphClassificationModel(torch.nn.Module):

    def __init__(self, 
                 in_channels,                           # Size of node features
                 out_channels,                          # Number of classes
                 pooler,                                # Pooling method
                 edge_channels=None,                    # Size of edge features
                 num_layers_pre=1,                      # Number of GIN layers before pooling
                 num_layers_post=1,                     # Number of GIN layers after pooling
                 hidden_channels=64,                    # Dimensionality of node embeddings
                 activation='ELU',                      # Activation of the MLP in GIN 
                 pool_kwargs={},                        # Pooling method kwargs
                 pooled_nodes=None,                     # Number of nodes after pooling
                 aux_net_kwargs={},                     # Auxiliary network kwargs
                 use_gine=False,                        # Use GINE instead of GIN
                 ):
        super().__init__()
        
        self.num_classes = out_channels
        self.act = activation_resolver(activation)
        self.pooler = pooler
        if edge_channels is not None and use_gine:
            self.using_gine = True
        else:
            self.using_gine = False

        ### Pre-pooling block            
        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
            if self.using_gine:
                self.conv_layers_pre.append(GINEConv(nn=mlp, train_eps=False, edge_dim=edge_channels))
            else:
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
                        
        ### Pooling block
        if pooler in ['diffpool','mincut']:
            self.pool = Linear(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
        elif pooler=='dmon':
            self.pool = DMoNPooling(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
            self.l3_weight = pool_kwargs['l3_weight']
        elif pooler=='maxcutpool':
            score_net_kwargs = aux_net_kwargs
            self.pool = MaxCutPool(hidden_channels, **pool_kwargs, **score_net_kwargs)
        elif pooler=='topk':
            self.pool = TopKPooling(hidden_channels, **pool_kwargs)
        elif pooler=='asapool':
            self.pool = ASAPooling(hidden_channels, **pool_kwargs)  
        elif pooler=='edgepool':
            self.pool = EdgePooling(hidden_channels)
        elif pooler in ['graclus','nopool']:
            pass
        elif pooler=='kmis':
            self.pool = KMISPooling(hidden_channels, **pool_kwargs)

        ### Post-pooling block
        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            if pooler in ['diffpool','mincut','dmon']:
                self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))
            elif pooler in []: # 'diffndp', 'kmis'
                self.conv_layers_post.append(GINEConv(nn=mlp, train_eps=False, edge_dim=1))
            else:
                self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

        ### Readout
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=0.5)

    def forward(self, data):
        """
        ⏩ 
        """
        x = data.x    
        adj = data.edge_index
        ea = getattr(data, 'edge_attr', None)
        batch = data.batch
        edge_weight = None # torch.ones(adj.size(1)).to(adj.device)

        ### Pre-pooling block
        if self.using_gine:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj, edge_attr=ea))
        else:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj))

        ### Pooling block
        if self.pooler in ['diffpool','mincut','dmon']:
            x, mask = to_dense_batch(x, batch)
            adj = to_dense_adj(adj, batch)
            if self.pooler=='diffpool':
                s = self.pool(x)
                x, adj, l1, l2 = dense_diff_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2
            elif self.pooler=='mincut':
                s = self.pool(x)
                x, adj, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l2_weight*l2
            elif self.pooler=='dmon':  
                s, x, adj, l1, l2, l3 = self.pool(x, adj, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2 + self.l1_weight*l3
        elif self.pooler=='topk':
            x, adj, _, batch, _, _ = self.pool(x, adj, edge_attr=None, batch=batch)
        elif self.pooler=='asapool':
            x, adj, _, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooler=='edgepool':
            x, adj, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooler=='kmis':
            x, adj, _, batch, _, _ = self.pool(x, adj, edge_attr=edge_weight, batch=batch)
        elif self.pooler=='graclus':
            data.x = x
            cluster = graclus(adj, num_nodes=data.x.size(0))
            data = sum_pool(cluster, data)
            x = data.x    
            adj = data.edge_index
            batch = data.batch
        elif self.pooler=='maxcutpool':
            (x, adj, edge_weight, batch, kept_nodes,
             score, aux_loss, s, maps_list) = self.pool(x, adj, edge_weight=edge_weight, batch=batch)
        elif self.pooler=='nopool':
            pass
        else:
            raise KeyError("unrecognized pooling method")
        
        ### Post-pooling block
        for layer in self.conv_layers_post:  
            if self.pooler in []: # 'maxcutpool', 'kmis'
                x = self.act(layer(x, adj, edge_weight.unsqueeze(-1)))
            else:
                x = self.act(layer(x, adj))

        ### Readout
        if self.pooler in ['diffpool','mincut','dmon']:
            x = torch.sum(x, dim=1)
        else:
            x = global_add_pool(x, batch)
        x = self.mlp(x)

        if 'edge_weight' not in locals():
            edge_weight=None
        if 'kept_nodes' not in locals():
            kept_nodes=None
        if 'score' not in locals():
            score=None
        if 'map_list' not in locals():
            maps_list=None
        if 'aux_loss' not in locals():
            aux_loss=0
        if 's' not in locals():
            s=None

        return x, adj, edge_weight, batch, kept_nodes, score, aux_loss, s, maps_list