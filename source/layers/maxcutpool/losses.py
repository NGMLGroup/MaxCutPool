# z^TAz losses

import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter


def max_cut_loss(z, edge_index, edge_weight, batch):
    """Compute the MaxCut loss for graph partitioning.
    
    Args:
        z (Tensor): Node scores/assignments
        edge_index (Tensor): Graph connectivity in COO format
        edge_weight (Tensor, optional): Edge weights
        batch (Tensor): Batch assignments for each node
        
    Returns:
        Tensor: Mean normalized cut loss across all graphs in batch
    """
    
    num_nodes = batch.size(0)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    
    az = adj.matmul(z).squeeze()
    z = z.squeeze()
    cut_losses = scatter(z*az, batch, dim=0, reduce='sum')
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=batch.device)
    edge_batch = batch[edge_index[0]]
    
    volumes = scatter(edge_weight, edge_batch, dim=0, reduce='sum')
    
    return torch.mean(cut_losses / volumes)


def max_cut_loss_tanh(z, edge_index, edge_weight, batch):
    """MaxCut loss with tanh normalization of scores.
    
    Similar to max_cut_loss but applies tanh to normalize scores to [-1,1].
    
    Args:
        z (Tensor): Node scores/assignments
        edge_index (Tensor): Graph connectivity in COO format
        edge_weight (Tensor, optional): Edge weights
        batch (Tensor): Batch assignments for each node
        
    Returns:
        Tensor: Mean normalized cut loss across all graphs in batch
    """
    z = torch.tanh(z) # normalize z
    
    num_nodes = batch.size(0)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    
    az = adj.matmul(z).squeeze()
    z = z.squeeze()
    cut_losses = scatter(z*az, batch, dim=0, reduce='sum')
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=batch.device)
    edge_batch = batch[edge_index[0]]
    
    volumes = scatter(edge_weight, edge_batch, dim=0, reduce='sum')
    
    return torch.mean(cut_losses / volumes)


def max_cut_loss_norm(z, edge_index, edge_weight, batch):
    """MaxCut loss with L2 normalization of scores.
    
    Similar to max_cut_loss but applies L2 normalization to the scores.
    
    Args:
        z (Tensor): Node scores/assignments
        edge_index (Tensor): Graph connectivity in COO format
        edge_weight (Tensor, optional): Edge weights
        batch (Tensor): Batch assignments for each node
        
    Returns:
        Tensor: Mean normalized cut loss across all graphs in batch
    """
    z_squared = z**2
    norms_squared = scatter(z_squared, batch, dim=0, reduce='sum')
    batched_norms = norms_squared[batch]
    z = z / torch.sqrt(batched_norms) # normalize z
    
    num_nodes = batch.size(0)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    
    az = adj.matmul(z).squeeze()
    z = z.squeeze()
    cut_losses = scatter(z*az, batch, dim=0, reduce='sum')
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=batch.device)
    edge_batch = batch[edge_index[0]]
    
    volumes = scatter(edge_weight, edge_batch, dim=0, reduce='sum')
    
    return torch.mean(cut_losses / volumes)


def max_cut_loss_gw(z, edge_index, edge_weight, batch):
    """MaxCut loss following Goemans-Williamson relaxation.
    
    Implements the semidefinite programming relaxation of MaxCut.
    
    Args:
        z (Tensor): Node embeddings
        edge_index (Tensor): Graph connectivity in COO format
        edge_weight (Tensor, optional): Edge weights
        batch (Tensor): Batch assignments for each node
        
    Returns:
        Tensor: Mean normalized cut loss across all graphs in batch
    """
    
    zzt = torch.matmul(z, z.t())
    zzt_edge = zzt[edge_index[0], edge_index[1]]
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=batch.device)
    edge_batch = batch[edge_index[0]]
    
    cut_loss_elementwise = edge_weight*zzt_edge
    cut_loss_graphwise = scatter(cut_loss_elementwise, edge_batch, dim=0, reduce='sum')

    volumes = scatter(edge_weight, edge_batch, dim=0, reduce='sum')
    
    return torch.mean(cut_loss_graphwise / volumes)