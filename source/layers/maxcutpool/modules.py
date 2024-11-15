from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn.resolver import activation_resolver

# Local imports
from .losses import max_cut_loss
from .utils import topk, get_propagation_matrix, get_new_adjacency_matrix


class ScoreNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 mp_units=[32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8],
                 mp_act="TanH",
                 mlp_units=[16,16],
                 mlp_act="ReLU",
                 delta=2.0):
        super().__init__()
    
        # MP layers
        if mp_act == "Identity":
            self.mp_act = lambda x: x
        else:
            self.mp_act = activation_resolver(mp_act)
        self.mp_convs = torch.nn.ModuleList()
        in_units = in_channels
        for out_units in mp_units:
            self.mp_convs.append(GCNConv(in_units, out_units, normalize=False, cached=False))
            in_units = out_units
        
        # MLP layers
        if mlp_act == "Identity":
            self.mlp_act = lambda x: x
        else:
            self.mlp_act = activation_resolver(mlp_act)
        self.mlp = torch.nn.ModuleList()
        for out_units in mlp_units:
            self.mlp.append(Linear(in_units, out_units))
            in_units = out_units

        self.final_layer = Linear(in_units, 1)
        
        self.delta = delta

    def forward(self, x, edge_index, edge_weight):
        """Compute scores for each node.
        
        Args:
            x (Tensor): Node features
            edge_index (Tensor): Graph connectivity in COO format
            edge_weight (Tensor): Edge weights
            
        Returns:
            Tensor: Scores for each node, normalized to [-1,1] via tanh
        """
        # Get real propagation matrix
        edge_index, edge_weight = get_propagation_matrix(edge_index, edge_weight, delta=self.delta)

        # Propagate node feats
        for mp in self.mp_convs:
            x = mp(x, edge_index, edge_weight)
            x = self.mp_act(x)
        
        # Cluster assignments (logits)
        for layer in self.mlp:
            x = layer(x)
            x = self.mlp_act(x)

        score = self.final_layer(x)

        return torch.tanh(score)
    

class MaxCutPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        beta: float = 1., # scale aux_loss
        score_net: Optional[Callable] = None,
        expressive: Optional[bool] = False,
        expressive_reduce: Optional[str] = 'sum',
        flip: Optional[bool] = False,
        max_iter: Optional[int] = 10,
        just_cut: Optional[bool] = False,
        initial_embedding: Optional[bool] = True,
        **score_net_kwargs
    ):
        super().__init__()
        
        # Parameters
        self.in_channels = in_channels
        self.ratio = ratio
        self.beta = beta
        self.expressive = expressive
        self.expressive_reduce = expressive_reduce
        self.flip = flip
        self.flip_state = 1
        self.max_iter = max_iter
        self.just_cut = just_cut
        self.initial_embedding = initial_embedding
        
        # Layers
        if initial_embedding:
            self.embedding = Linear(in_channels, in_channels)
        if score_net is None:
            try:
                self.score_net = ScoreNet(in_channels=in_channels, **score_net_kwargs)
            except:
                self.score_net = ScoreNet(in_channels=in_channels)
                print("Warning: ScoreNet not initialized with provided kwargs, using default.")
                # raise ValueError("Incorrect score_net_kwargs")
        else:
            self.score_net = score_net


    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None, # edge_attr is the edge weight
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        """Forward pass of MaxCutPool layer.
        
        Args:
            x (Tensor): Node features
            edge_index (Tensor): Graph connectivity in COO format
            edge_weight (Tensor, optional): Edge weights. Defaults to None.
            batch (Tensor, optional): Batch assignments. Defaults to None.
            
        Returns:
            tuple:
                - x (Tensor): Pooled node features
                - edge_index (Tensor): New graph connectivity
                - edge_weight (Tensor): New edge weights
                - batch (Tensor): Updated batch assignments
                - kept_nodes (Tensor): Indices of kept nodes
                - score (Tensor): Score vector
                - aux_loss (Tensor): Auxiliary loss value
                - s (Tensor): Assignment matrix
                - maps_list (list): Step-by-step mappings (for logging purpose)
        """

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # get score
        if self.initial_embedding:
            x = self.embedding(x)
        score = self.score_net(x, edge_index, edge_weight)
        
        # get loss
        aux_loss = max_cut_loss(z=score, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        aux_loss = self.beta * aux_loss
        
        # get kept nodes
        kept_nodes = topk(self.flip_state * score, self.ratio, batch)
        
        # update flip_state
        if self.flip:
            self.flip_state = -self.flip_state
        
        if self.just_cut:
            return x, None, None, None, kept_nodes, score.squeeze(), aux_loss, None, None

        # get new adjacency matrix
        edge_index, edge_weight, assignments, maps_list = get_new_adjacency_matrix(
            edge_index, edge_weight, 
            kept_nodes, max_iter=self.max_iter, 
            batch=batch)
        batch = batch[kept_nodes]
        
        # get new node features
        if self.expressive:
            x_pool = scatter(x, assignments[1], dim=0, reduce=self.expressive_reduce) * score[kept_nodes].view(-1, 1)
        else:
            x_pool = torch.zeros_like(x[kept_nodes])
            x_pool[assignments[1][kept_nodes]] = x[kept_nodes] * score[kept_nodes].view(-1, 1)

        return x_pool, edge_index, edge_weight, batch, kept_nodes, score.squeeze(), aux_loss, assignments, maps_list