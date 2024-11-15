from typing import Optional, Tuple, Union
from torch_sparse import SparseTensor, eye as torch_sparse_eye
import torch
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter, remove_isolated_nodes, coalesce, get_laplacian, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7
) -> Tensor:
    """Select top-k nodes based on scores, maintaining batch structure.
    
    Args:
        x (Tensor): Node scores
        ratio (float or int): If float, ratio of nodes to keep. If int, exact number of nodes.
        batch (Tensor): Batch assignments for each node
        min_score (float, optional): Minimum score threshold. Defaults to None.
        tol (float, optional): Tolerance for numerical stability. Defaults to 1e-7.
        
    Returns:
        Tensor: Indices of selected nodes
        
    Note:
        Either ratio or min_score must be specified. If min_score is used,
        ensures at least one node per graph is kept.
    """
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = num_nodes.new_zeros(num_nodes.numel() + 1)
        torch.cumsum(num_nodes, 0, out=ptr[1:])
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")
    

def get_propagation_matrix(edge_index, edge_weight, delta, num_nodes=None):
    """Compute the propagation matrix for message passing.
    
    Constructs a modified Laplacian matrix with self-loops for message propagation.
    
    Args:
        edge_index (Tensor): Graph connectivity in COO format
        edge_weight (Tensor): Edge weights
        delta (float): Scale factor for Laplacian
        num_nodes (int, optional): Number of nodes. Defaults to None.
        
    Returns:
        tuple:
            - Tensor: Updated edge indices
            - Tensor: Updated edge weights
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization='sym')
    edge_weight = -delta*edge_weight
    
    eye_index, eye_weight = torch_sparse_eye(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    
    A = SparseTensor(row=torch.cat([edge_index[0], eye_index[0]]),
                     col=torch.cat([edge_index[1], eye_index[1]]),
                     value=torch.cat([edge_weight, eye_weight]),
                     sparse_sizes=(num_nodes, num_nodes)).coalesce("sum") # sommo i pesi degli archi con gli archi della diagonale
    
    row, col, edge_weight = A.coo()
    edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_weight


def reset_node_numbers(edge_index, edge_attr=None):
    """Reset node indices after removing isolated nodes.
    
    Args:
        edge_index (Tensor): Graph connectivity in COO format
        edge_attr (Tensor, optional): Edge attributes. Defaults to None.
        
    Returns:
        tuple:
            - Tensor: Updated edge indices
            - Tensor: Updated edge attributes
    """
    edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr=edge_attr)
    return edge_index, edge_attr


def create_one_hot_tensor(num_nodes, kept_node_tensor, device):
    """Create one-hot encoding tensor for node assignments.
    
    Args:
        num_nodes (int): Total number of nodes
        kept_node_tensor (Tensor): Indices of nodes to keep
        device (torch.device): Device to create tensor on
        
    Returns:
        Tensor: One-hot encoding matrix [num_nodes, num_kept_nodes + 1]
    """
    tensor = torch.zeros(num_nodes, len(kept_node_tensor)+1, device=device)
    tensor[kept_node_tensor, 1:] = torch.eye(len(kept_node_tensor), device=device)
    return tensor

def get_sparse_map_mask(x, edge_index, kept_node_tensor, mask):
    """Compute sparse assignment mapping using message passing.
    
    Args:
        x (Tensor): Node features/assignments
        edge_index (Tensor): Graph connectivity
        kept_node_tensor (Tensor): Indices of kept nodes
        mask (Tensor): Boolean mask of already assigned nodes
        
    Returns:
        tuple:
            - Tensor: Propagated features
            - Tensor: Node assignment mapping
            - Tensor: Updated assignment mask
    """
    sparse_ei = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0)))
    y = sparse_ei.matmul(x) # propagation step
    first_internal_mask = torch.logical_not(mask) # get the mask of the nodes that have not been assigned yet
    am = torch.argmax(y, dim=1) # get the visited nodes
    nonzero = torch.nonzero(am, as_tuple=True)[0] # get the visited nodes that are not zero (since the zero-th node is a fake node)
    second_internal_mask = torch.zeros_like(first_internal_mask, dtype=torch.bool) # initialize the second mask
    second_internal_mask[nonzero] = True # set the mask to True for the visited nodes
    final_mask = torch.logical_and(first_internal_mask, second_internal_mask) # newly visited nodes that have not been assigned yet
    indices = torch.arange(x.size(0), device=x.device) # inizialize the indices
    out = kept_node_tensor[am-1] # get the supernode indices of the visited nodes (am-1 because the zero-th node is a fake node)
    
    indices = indices[final_mask] # get the indices of the newly visited nodes that have not been assigned yet

    mappa = torch.stack([indices, out[indices]]) # create the map
    mask[indices] = True # set the mask to True for the newly visited nodes that have not been assigned yet

    return y, mappa, mask

def get_random_map_mask(kept_nodes, mask, batch=None):
    """Randomly assign remaining unassigned nodes.
    
    Args:
        kept_nodes (Tensor): Indices of kept nodes
        mask (Tensor): Boolean mask of already assigned nodes
        batch (Tensor, optional): Batch assignments. Defaults to None.
        
    Returns:
        Tensor: Random assignment mapping for unassigned nodes
    """
    neg_mask = torch.logical_not(mask)
    zero = torch.arange(mask.size(0), device=kept_nodes.device)[neg_mask]
    one = torch.randint(0, kept_nodes.size(0), (zero.size(0),), device=kept_nodes.device)
    
    if batch is not None:
        s_batch = batch[kept_nodes]
        s_counts = torch.bincount(s_batch)
        
        cumsum = torch.zeros(s_counts.size(0), device=batch.device).to(torch.long)
        cumsum[1:] = s_counts.cumsum(dim=0)[:-1]

        count_tensor = s_counts[batch].to(torch.long)
        sum_tensor = cumsum[batch].to(torch.long)
        
        count_tensor = count_tensor[neg_mask]
        sum_tensor = sum_tensor[neg_mask]

        one = one % count_tensor + sum_tensor
        
        one = kept_nodes[one]
    
    mappa = torch.stack([zero, one])
    return mappa


def get_assignments(maps_list):
    """Compute final node assignments from hierarchical mappings.
    
    Args:
        maps_list (list): List of assignment mappings at each level
        
    Returns:
        Tensor: Final node assignment matrix
    """
    """
    Compute the assignments from the original to the pooled graph
    """
    assignments = torch.cat(maps_list, dim=1)
    assignments = assignments[:, assignments[0].argsort()]
    _, unique_one = torch.unique(assignments[1], return_inverse=True) # get idx of pooled graph
    assignments[1] = unique_one

    return assignments


def get_new_adjacency_matrix(edge_index, edge_weight, kept_node_indices, max_iter=5, batch=None):
    """Compute new adjacency matrix after pooling.
    
    Performs hierarchical node assignment and constructs the pooled graph structure.
    
    Args:
        edge_index (Tensor): Original graph connectivity
        edge_weight (Tensor): Original edge weights
        kept_node_indices (Tensor): Indices of nodes to keep
        max_iter (int, optional): Maximum propagation iterations. Defaults to 5.
        batch (Tensor, optional): Batch assignments. Defaults to None.
        
    Returns:
        tuple:
            - Tensor: New edge indices
            - Tensor: New edge weights
            - Tensor: Node assignment matrix
            - list: Hierarchical assignment mappings
    """
    if isinstance(kept_node_indices, torch.Tensor):
        kept_node_tensor = torch.squeeze(kept_node_indices).to(torch.long)
    else:
        kept_node_tensor = torch.tensor(kept_node_indices, dtype=torch.long)

    if batch is None:
        num_nodes = edge_index.max().item() + 1
    else:
        num_nodes = batch.size(0)
    device = edge_index.device
    
    edge_index = edge_index.clone()
    if edge_weight is not None:
        edge_weight = edge_weight.clone()
    
    x = create_one_hot_tensor(num_nodes, kept_node_tensor, device) # initialize the one-hot tensor for x
    
    maps_list = []
    mask = torch.zeros(num_nodes, device=device, dtype=torch.bool) # initialize the mask
    mask[kept_node_indices] = True # set the mask to True for the supernodes

    _map = torch.stack([kept_node_tensor, kept_node_tensor]) # create the first map
    maps_list.append(_map) # append the first map to the list of maps
    
    for _ in range(max_iter): # iterate over the max path length
        if mask.all(): # if the mask is all True
            break # break the loop
        x, _map, mask = get_sparse_map_mask(x, edge_index, kept_node_tensor, mask) # propagate the kept nodes for one step
        maps_list.append(_map)

    if not mask.all():
        _map = get_random_map_mask(kept_node_tensor, mask, batch) # randomly assign unassigned nodes
        maps_list.append(_map)
    
    # Get the assignments from the old to the pooled graph
    assignments = get_assignments(maps_list)

    new_ei, new_ea = coalesce(assignments[1][edge_index], edge_attr=edge_weight)
    new_ei, new_ea = remove_self_loops(new_ei, new_ea)
    if new_ea is not None:
        new_ea = new_ea/new_ea.max()
    return new_ei, new_ea, assignments, maps_list