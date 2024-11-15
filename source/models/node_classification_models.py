import einops
import torch
from torch.nn import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import (GINConv, GINEConv, MLP, DenseGINConv,
                                dense_mincut_pool, dense_diff_pool,
                                DMoNPooling, TopKPooling, ASAPooling, graclus)
from torch_geometric.utils import (
    to_dense_adj, to_dense_batch
)

# Local imports
from source.layers.maxcutpool.modules import MaxCutPool
from source.layers.kmis.kmis_pool import KMISPooling
from source.layers.sum_pool import sum_pool
from source.layers.ndp import ndp


class NodeClassificationModel(torch.nn.Module):
    """
    Autoencoder model with the following architecture:
    [MP encoder] ➡️ Pooling ⬇️ [MP bottleneck] ➡️ Unpooling ⬆️ [MP decoder] ➡️ Readout
    """

    def __init__(self, 
                 in_channels,                           # Size of node features
                 out_channels,                          # Number of classes
                 pooler,                                # Pooling method
                 edge_channels=None,                    # Size of edge features
                 num_mp_layers=1,                       # Number of MP layers in each block
                 hidden_channels=64,                    # Dimensionality of node embeddings
                 activation='ELU',                      # Activation of the MLP in GIN 
                 pool_kwargs={},                        # Pooling method kwargs
                 pooled_nodes=None,                     # Number of nodes after pooling
                 aux_net_kwargs={},                     # Auxiliary network kwargs
                 use_gine_enc=False,                    # Use GINE instead of GIN in the Encoder
                 use_gine_bottleneck=False,             # Use GINE instead of GIN in the bottleneck
                 lift_zero_pad=False,                   # Lift zero padded nodes
                 res_connect=None,                      # Residual connections (None, 'sum', 'cat')
                 dropout=0.1,                           # Dropout rate
                 ):
        super().__init__()
        
        self.num_classes = out_channels
        self.act = activation_resolver(activation)
        self.pooler = pooler
        if edge_channels is not None and use_gine_enc:
            self.using_gine_enc = True
        else:
            self.using_gine_enc = False
        self.using_gine_bottleneck = use_gine_bottleneck
        self.lift_zero_pad = lift_zero_pad
        self.res_connect = res_connect
        self.dropout = dropout

        ### Encoder MP block
        self.encoder_mp_layers = torch.nn.ModuleList()
        for _ in range(num_mp_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation, dropout=dropout)
            if self.using_gine_enc:
                self.encoder_mp_layers.append(GINEConv(nn=mlp, train_eps=False, edge_dim=edge_channels))
            else:
                self.encoder_mp_layers.append(GINConv(nn=mlp, train_eps=False))
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
            raise NotImplementedError("Edge pooling not implemented yet.")
        elif pooler == 'graclus':
            pass
        elif pooler == 'ndp':
            self.adj_pool = None
            self.edge_weight_pool = None
            self.pool_indices = None
            self.sparsify = pool_kwargs['sparsify']
            self.sparse_thresh = pool_kwargs['sparse_thresh']
        elif pooler=='kmis':
            self.pool = KMISPooling(hidden_channels, **pool_kwargs)

        ### Bottleneck MP block
        self.bottleneck_mp_layers = torch.nn.ModuleList()
        for _ in range(num_mp_layers):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None, dropout=dropout)
            if pooler in ['diffpool','mincut','dmon']:
                self.bottleneck_mp_layers.append(DenseGINConv(nn=mlp, train_eps=False))
            elif pooler in ['maxcutpool'] and self.using_gine_bottleneck:
                self.bottleneck_mp_layers.append(GINEConv(nn=mlp, train_eps=False, edge_dim=1))
            else:
                self.bottleneck_mp_layers.append(GINConv(nn=mlp, train_eps=False))

        ### Decoder MP block
        self.decoder_mp_layers = torch.nn.ModuleList()
        in_channels = 2*hidden_channels if res_connect=='cat' else hidden_channels
        for _ in range(num_mp_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation, norm=None, dropout=dropout)
            if pooler in ['diffpool','mincut','dmon']:
                self.decoder_mp_layers.append(DenseGINConv(nn=mlp, train_eps=False))
            else:
                self.decoder_mp_layers.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        ### Readout
        self.mlp = MLP([hidden_channels, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=dropout)

    def forward(self, data):
        """
        ⏩ 
        """
        x = data.x    
        adj = data.edge_index
        ea = getattr(data, 'edge_attr', None)
        batch = data.batch

        ### Encoder MP block ➡️
        if self.using_gine_enc:
            for layer in self.encoder_mp_layers:  
                x = self.act(layer(x, adj, edge_attr=ea))
        else:
            for layer in self.encoder_mp_layers:  
                x = self.act(layer(x, adj))

        ### Pooling block ⬇️
        if self.pooler in ['diffpool','mincut','dmon']:
            x, mask = to_dense_batch(x, batch)
            adj = to_dense_adj(adj, batch)
            if self.pooler=='diffpool':
                s = self.pool(x)
                x_pool, adj_pool, l1, l2 = dense_diff_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2
            elif self.pooler=='mincut':
                s = self.pool(x)
                x_pool, adj_pool, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l2_weight*l2
            elif self.pooler=='dmon':  
                s, x_pool, adj_pool, l1, l2, l3 = self.pool(x, adj, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2 + self.l1_weight*l3
        elif self.pooler=='topk':
            x_pool, adj_pool, _, batch, perm, _ = self.pool(x, adj, edge_attr=None, batch=batch)
        elif self.pooler=='asapool':
            x_pool, adj_pool, _, batch, perm = self.pool(x, adj, batch=batch)
        elif self.pooler=='edgepool':
            x_pool, adj_pool, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooler=='kmis':
            x_pool, adj_pool, edge_weight_pool, batch, _, cluster = self.pool(x, adj, edge_attr=torch.ones(adj.size(1)).to(adj.device), batch=batch)
        elif self.pooler=='graclus':
            data = data.clone()
            data.x = x    
            cluster = graclus(adj, num_nodes=data.x.size(0))
            data = sum_pool(cluster, data)
            x_pool = data.x    
            adj_pool = data.edge_index
            batch = data.batch
        elif self.pooler=='ndp':
            if self.adj_pool is None and self.edge_weight_pool is None and self.pool_indices is None:
                self.adj_pool, self.edge_weight_pool, self.pool_indices = ndp(
                    data, sparsify=self.sparsify, sparse_thresh=self.sparse_thresh, return_all=False)
            kept_nodes = self.pool_indices
            x_pool = x[kept_nodes]
            adj_pool = self.adj_pool
            edge_weight_pool = self.edge_weight_pool.float()
        elif self.pooler=='maxcutpool':
            (x_pool, adj_pool, edge_weight_pool, batch, kept_nodes,
             score, aux_loss, s, maps_list) = self.pool(x, adj, edge_weight=torch.ones(adj.size(1)).to(adj.device), batch=batch)
        else:
            raise KeyError(f"unrecognized pooling method: {self.pooler}")
        
        ### Bottleneck MP block ➡️
        for layer in self.bottleneck_mp_layers:  
            if self.pooler in ['maxcutpool'] and self.using_gine_bottleneck:
                x_pool = self.act(layer(x_pool, adj_pool, edge_weight_pool.unsqueeze(-1)))
            else:
                x_pool = self.act(layer(x_pool, adj_pool))

        ### Lifting (unpooling) ⬆️
        if self.pooler in ['diffpool','mincut','dmon']:
            x_lift = einops.einsum(s, x_pool, 'b n k, b k d -> b n d')
        elif self.pooler=='maxcutpool':
            if self.lift_zero_pad:
                x_lift = torch.zeros_like(x)
                x_lift[kept_nodes] = x_pool[s[1][kept_nodes]]
            else:
                x_lift = x_pool[s[1]]
        elif self.pooler in ['topk','asapool']:
            x_lift = torch.zeros_like(x)
            x_lift[perm] = x_pool
        elif self.pooler=='graclus':
            unique_clusters = sorted(set(cluster.tolist()))
            cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
            remapped_clusters = [cluster_map[old_id] for old_id in cluster.tolist()]
            x_lift = x_pool[remapped_clusters]
        elif self.pooler=='ndp':
            x_lift = torch.zeros_like(x)
            x_lift[kept_nodes] = x_pool
        elif self.pooler=='kmis':
            x_lift = x_pool[cluster]
        else:
            raise NotImplementedError(f"Lifting not implemented yet for method: {self.pooler}")

        ### Residual connection ⏭️
        if self.res_connect=='sum':
            x_lift = x_lift + x
        elif self.res_connect=='cat':
            x_lift = torch.cat([x_lift, x], dim=-1)

        ### Decoder MP block ➡️
        x_dec = x_lift
        for layer in self.decoder_mp_layers:  
            x_dec = self.act(layer(x_dec, adj))

        ### Dropout
        # x_dec = torch.nn.functional.dropout(x_dec, p=self.dropout, training=self.training)

        ### Readout ▶️
        out = self.mlp(x_dec)

        if len(out.size())==3:
            out = out[0]

        if 'edge_weight_pool' not in locals():
            edge_weight_pool=None
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

        return out, adj_pool, edge_weight_pool, batch, kept_nodes, score, aux_loss, s, maps_list