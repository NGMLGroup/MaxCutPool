import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.io import fs, read_txt_array
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(
    folder: str,
    prefix: str,
) -> Tuple[Data, Dict[str, Tensor], Dict[str, int]]:
    """Reads a TU dataset from disk.
    
    Args:
        folder (str): Path to the dataset folder
        prefix (str): Prefix of the dataset files
        
    Returns:
        tuple containing:
            - Data: PyG Data object with the graph data
            - Dict[str, Tensor]: Mapping of data splits
            - Dict[str, int]: Dataset statistics including number of node/edge attributes and labels
    """
    files = fs.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [osp.basename(f)[len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attribute = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attribute = read_file(folder, prefix, 'node_attributes')
        if node_attribute.dim() == 1:
            node_attribute = node_attribute.unsqueeze(-1)

    node_label = torch.empty((batch.size(0), 0))
    if 'node_labels' in names:
        node_label = read_file(folder, prefix, 'node_labels', torch.long)
        if node_label.dim() == 1:
            node_label = node_label.unsqueeze(-1)
        node_label = node_label - node_label.min(dim=0)[0]
        node_labels = node_label.unbind(dim=-1)
        node_labels = [one_hot(x) for x in node_labels]
        if len(node_labels) == 1:
            node_label = node_labels[0]
        else:
            node_label = torch.cat(node_labels, dim=-1)

    edge_attribute = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attribute = read_file(folder, prefix, 'edge_attributes')
        if edge_attribute.dim() == 1:
            edge_attribute = edge_attribute.unsqueeze(-1)

    edge_label = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_label = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(-1)
        edge_label = edge_label - edge_label.min(dim=0)[0]
        edge_labels = edge_label.unbind(dim=-1)
        edge_labels = [one_hot(e) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_label = edge_labels[0]
        else:
            edge_label = torch.cat(edge_labels, dim=-1)

    x = cat([node_attribute, node_label])
    edge_attr = cat([edge_attribute, edge_label])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        try:
            y = read_file(folder, prefix, 'graph_labels', torch.long)
        except:
            y = read_file(folder, prefix, 'graph_labels', torch.float)
            y = y.to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = int(edge_index.max()) + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attribute.size(-1),
        'num_node_labels': node_label.size(-1),
        'num_edge_attributes': edge_attribute.size(-1),
        'num_edge_labels': edge_label.size(-1),
    }

    return data, slices, sizes


def read_file(
    folder: str,
    prefix: str,
    name: str,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Reads a specific file from a TU dataset.
    
    Args:
        folder (str): Path to the dataset folder
        prefix (str): Dataset prefix
        name (str): Name of the file to read (e.g., 'A' for adjacency, 'graph_indicator' etc.)
        dtype (torch.dtype, optional): Data type to cast the contents to
        
    Returns:
        Tensor: Contents of the file as a tensor
    """
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    """Concatenates a sequence of tensors along the last dimension.
    
    Args:
        seq (List[Optional[Tensor]]): Sequence of tensors to concatenate
        
    Returns:
        Optional[Tensor]: Concatenated tensor, or None if no valid tensors
    """
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None


def split(data: Data, batch: Tensor) -> Tuple[Data, Dict[str, Tensor]]:
    """Splits a large data object into batches.
    
    This function takes a PyG Data object and splits it into batches based on the
    batch assignments. It handles both node and edge attributes appropriately.
    
    Args:
        data (Data): PyG Data object to split
        batch (Tensor): Batch assignments for each node
        
    Returns:
        tuple containing:
            - Data: The processed data object with batch information
            - Dict[str, Tensor]: Slice information for each attribute
    """
    node_slice = cumsum(torch.from_numpy(np.bincount(batch)))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(torch.from_numpy(np.bincount(batch[row])))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, Tensor)
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, int(batch[-1]) + 2, dtype=torch.long)

    return data, slices
