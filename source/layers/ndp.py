from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import numpy as np
import torch


def eval_cut(total_volume, L, z):
    """
    Computes the normalized size of a cut in [0,1]
    """
    cut = z.T.dot(L.dot(z))
    cut /= 2 * total_volume
    return cut


def ndp(data, sparsify=True, sparse_thresh=1e-2, return_all=False):
    """Node Decimation Pooling (NDP) implementation.
    http://dx.doi.org/10.1109/TNNLS.2020.3044146
    
    Performs graph pooling based on node degree information.
    
    Args:
        data (Data): Input graph data
        sparsify (bool, optional): Whether to sparsify the output by removing small edges. Defaults to True.
        sparse_thresh (float, optional): Threshold for sparsification. Defaults to 1e-2.
        return_all (bool, optional): Whether to return additional info. Defaults to False.
        
    Returns:
        tuple: 
            - edge_index (Tensor): New graph connectivity
            - edge_weight (Tensor): New edge weights
            - idx_pos (Tensor): Indices of kept nodes
            - cut_size (float, optional): Size of the cut
            - V (Tensor, optional): Additional node information
    """
    num_nodes = data.num_nodes
    device = data.x.device
    
    # compute Laplacian and symmetric Laplacian
    edge_index_L, edge_weight_L = get_laplacian(data.edge_index, data.edge_weight, normalization=None)
    L = to_scipy_sparse_matrix(edge_index_L, edge_weight_L, num_nodes).tocsr()
    edge_index_Ls, edge_weight_Ls = get_laplacian(data.edge_index, data.edge_weight, normalization='sym')
    Ls = to_scipy_sparse_matrix(edge_index_Ls, edge_weight_Ls, num_nodes)

    # Compute spectral cut
    if num_nodes==1:
        # No need for pooling
        idx_pos = np.array([0])
    else:
        try:
            V = eigsh(Ls, k=1, which='LM', v0=np.ones(num_nodes))[1][:, 0]
        except Exception:  # ArpackNoConvergence:
            # Random split if eigendecomposition is not possible
            print('Level %d --Eigendecomposition is not possible, splitting randomly instead')
            V = np.random.choice([-1, 1], size=(num_nodes,))
            
        idx_pos = np.nonzero(V >= 0)[0]
        idx_neg = np.nonzero(V < 0)[0]

        # Evaluate the size of the cut 
        z = np.ones((num_nodes, 1))  # partition vector
        z[idx_neg] = -1
        if data.edge_weight is None:
            total_volume = data.num_edges
        else:
            total_volume = torch.sum(data.edge_weight)
        cut_size = eval_cut(total_volume, L, z)
        
        # If the cut is smaller than 0.5, return a random cut
        if cut_size < 0.5:
            print("Spectral cut lower than 0.5 (%.2f): returning random cut" % (cut_size))
            V = np.random.choice([-1, 1], size=(Ls.shape[0],))
            idx_pos = np.nonzero(V >= 0)[0]
            idx_neg = np.nonzero(V < 0)[0]

    if len(idx_pos) == 0:
        idx_pos = np.array([0])

    # Link reconstruction with Kron reduction
    if len(idx_pos) <= 1:
        # No need to compute Kron reduction with 0 or 1 node
        Lnew = sp.csc_matrix(-np.ones((1, 1)))  # L = -1
        idx_pos = np.array([0])  # make sure there is at least 1 index
    else:
        # Kron reduction
        L_red = L[np.ix_(idx_pos, idx_pos)]
        L_in_out = L[np.ix_(idx_pos, idx_neg)]
        L_out_in = L[np.ix_(idx_neg, idx_pos)].tocsc()
        L_comp = L[np.ix_(idx_neg, idx_neg)].tocsc()
        
        try:
            Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(L_comp, L_out_in))
        except RuntimeError:
            # If L_comp is exactly singular, damp the inversion with
            # Marquardt-Levenberg coefficient ml_c
            ml_c = sp.csc_matrix(sp.eye(L_comp.shape[0]) * 1e-6)
            Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(ml_c + L_comp, L_out_in))

        # Make the laplacian symmetric if it is almost symmetric
        if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
            Lnew = (Lnew + Lnew.T) / 2.
            
    # Get the new adjacency matrix
    A_new = -Lnew
    if sparsify:
        A_new = A_new.multiply(np.abs(A_new) > sparse_thresh)
    A_new.setdiag(0)
    A_new.eliminate_zeros()
    
    edge_index, edge_weight = from_scipy_sparse_matrix(A_new)
    idx_pos = torch.tensor(idx_pos)
    
    if return_all:
         return edge_index.to(device), edge_weight.to(device), idx_pos.to(device), cut_size, V
    else:
        return edge_index.to(device), edge_weight.to(device), idx_pos.to(device)