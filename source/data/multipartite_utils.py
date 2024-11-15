import numpy as np
import networkx as nx
import torch
import torch_geometric
import math


def rotate_list(lst, i):
    """Rotates a list by i positions.
    
    Args:
        lst (list): List to rotate
        i (int): Number of positions to rotate by
        
    Returns:
        list: Rotated list
    """
    return lst[-i:] + lst[:-i]


def get_graph(centers, max_N):
    """Creates a multipartite graph with nodes clustered around centers.
    
    Args:
        centers (list): List of tuples, each tuple is a center of a cluster
        max_N (int): Maximum number of nodes per cluster
        
    Returns:
        tuple: (networkx.Graph, numpy.ndarray) containing:
            - The multipartite graph
            - Node positions and cluster assignments
    """
    sizes = [np.random.randint(1, max_N) for _ in range(len(centers))]
    cluster_pos = [np.random.normal(center, 1, (size, 2)) for center, size in zip(centers, sizes)]
    cluster_pos_and_index = [np.concatenate((pos, np.full((pos.shape[0], 1), i)), axis=1) for i, pos in enumerate(cluster_pos)]
    X = np.concatenate(cluster_pos_and_index, axis=0)

    G = nx.complete_multipartite_graph(*sizes)
    return G, X


def get_graph_data(centers, max_N):
    """Generates a multipartite graph torch_geometric data object.

    Args:
        centers (list): List of center coordinates for each partition
        max_N (int): Maximum number of nodes per partition

    Returns:
        torch_geometric.data.Data: Graph data object containing:
            - x: Node features
            - pos: Node positions
            - node_labels: Partition assignments
            - edge_index: Graph connectivity
    """
    G, X = get_graph(centers, max_N)
    pos = X[:, :2]
    color = X[:, 2]
    data = torch_geometric.utils.from_networkx(G)
    data.x = torch.tensor(X, dtype=torch.float32)
    data.pos = torch.tensor(pos, dtype=torch.float32)
    data.node_labels = torch.tensor(color)
    
    return data


def get_polygon_vertices(N):
    """Generates vertices of a regular polygon.
    
    Args:
        N (int): Number of vertices
        
    Returns:
        list: List of (x,y) coordinates for each vertex
    """
    vertices = []
    radius = 20*N

    for i in range(N):
        angle = 2 * math.pi * i / N
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y))

    return vertices


def get_centers(centers):
    """Gets center coordinates for multipartite graph partitions.
    
    Args:
        centers (int or list): Either number of centers to generate in polygon formation,
                             or list of (x,y) coordinates for centers
        
    Returns:
        list: List of (x,y) coordinates for partition centers
    """
    if isinstance(centers, int):
        centers = get_polygon_vertices(centers)
    else:
        assert isinstance(centers, list) and len(centers) > 1

    return centers