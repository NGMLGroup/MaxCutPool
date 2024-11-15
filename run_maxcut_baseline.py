from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_networkx
from source.data import PyGSPDataset, GsetDataset
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf
import pandas as pd


from source.utils import register_resolvers
from source.layers.ndp import ndp
from source.layers.ndp import eval_cut
from source.utils.GW import goemans_williamson

try:
    register_resolvers()
except Exception as e:
    print(f"An error occurred: {e}")

results = []

@hydra.main(version_base=None, config_path="config", config_name="run_maxcut_baseline")
def run(cfg: DictConfig) -> float:
    """Run MaxCut baseline experiments for comparison.
    
    Implements both Goemans-Williamson and NDP baselines for MaxCut computation.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing experiment parameters
        
    Returns:
        tuple:
            - float: Cut size ratio
            - int: Number of cut edges
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 📊 Load data 
    if cfg.dataset.name=='PyGSPDataset':
        torch_dataset = PyGSPDataset(root='data/PyGSP', name=cfg.dataset.hparams.dataset, 
                                     kwargs=cfg.dataset.params, force_reload=cfg.dataset.hparams.reload)
    elif cfg.dataset.name=='GsetDataset':
        torch_dataset = GsetDataset(root='data/Gset', name=cfg.dataset.hparams.dataset, 
                                    directed=cfg.dataset.hparams.directed, force_reload=cfg.dataset.hparams.reload)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not recognized")
      
    data_loader = DataLoader(torch_dataset, batch_size=cfg.batch_size, shuffle=False)

    g = next(iter(data_loader))

    if cfg.method == 'GW': 
        colors, _, _ = goemans_williamson(to_networkx(g))
        edge_index_L, edge_weight_L = get_laplacian(g.edge_index, g.edge_weight, normalization=None)
        L = to_scipy_sparse_matrix(edge_index_L, edge_weight_L, g.num_nodes).tocsr()

        cut_size = eval_cut(g.num_edges, L, colors)
        cut_edges = int(cut_size * g.num_edges)

    elif cfg.method == 'NDP':
        edge_index_pool, edge_weight_pool, idx_pos, cut_size, Vmax = ndp(g, return_all=True)
        cut_size = cut_size[0,0].item()
        cut_edges = int(cut_size * g.num_edges)

    else:
        raise ValueError(f"Method {cfg.method} not recognized")
    
    print(f"Cut size: {cut_size:.3f} ({cut_edges} edges)")
    
    results.append([cfg.method, cfg.dataset.hparams.dataset, cut_size, cut_edges])

    return cut_size, cut_edges

if __name__ == "__main__":
    run()
    df = pd.DataFrame(results, columns=["Method", "Dataset", "Cut_Size", "Cut_Edges"])
    print(df)
    df.to_csv("results.csv", index=False)