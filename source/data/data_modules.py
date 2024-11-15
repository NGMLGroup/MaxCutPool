import pytorch_lightning as pl

from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.transforms import Constant
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily

# Local imports
from source.utils import DataToFloat, get_train_val_test_datasets
from .torch_datasets import GraphClassificationBench, PyGSPDataset, EXPWL1Dataset, MultipartiteGraphDataset, TUDataset

trans = Constant()

class EXPWL1DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the EXPWL1 dataset.
    
    This module handles loading and preprocessing of the EXPWL1 dataset, which is used
    to evaluate the expressive power of graph neural networks. The dataset is from
    `"The expressive power of pooling in Graph Neural Networks" 
    <https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs>`_ paper.

    Args:
        args: Configuration object containing:
            - seed (int): Random seed for data splitting
            - n_folds (int): Number of cross-validation folds
            - fold_id (int): Which fold to use (0 to n_folds-1)
            - batch_size (int): Batch size for dataloaders
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        path = "data/EXPWL1/"
        self.dataset = EXPWL1Dataset(path, transform=DataToFloat())
        self.dataset = self.dataset.shuffle()
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(self.dataset, args.seed, args.n_folds, args.fold_id)
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        if self.dataset[0].edge_attr is not None:
            self.num_edge_features = self.dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.avg_nodes = int(self.dataset._data.num_nodes / len(self.dataset))
        self.max_nodes = max([d.num_nodes for d in self.dataset])
        self.seed = args.seed
        self.n_folds = args.n_folds
        if args.fold_id is not None:
            assert args.fold_id < args.n_folds
            self.fold_id = args.fold_id
        else:
            self.fold_id = 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)


class BenchHardDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the Bench-hard graph classification dataset from 
    `"Pyramidal Reservoir Graph Neural Network" <https://arxiv.org/abs/2104.04710>`_ paper.
    
    Args:
        args: Configuration object containing:
            - seed (int): Random seed for data splitting
            - n_folds (int): Number of cross-validation folds
            - fold_id (int): Which fold to use (0 to n_folds-1)
            - batch_size (int): Batch size for dataloaders
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        path = "data/Bench-hard/"
        orig_train_dataset = GraphClassificationBench(path, split='train', easy=False, small=False)
        orig_val_dataset = GraphClassificationBench(path, split='val', easy=False, small=False)
        orig_test_dataset = GraphClassificationBench(path, split='test', easy=False, small=False)
        
        train_data_list = [data for data in orig_train_dataset]
        val_data_list = [data for data in orig_val_dataset]
        test_data_list = [data for data in orig_test_dataset]
        data_list = train_data_list + val_data_list + test_data_list
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(data_list, args.seed, args.n_folds, args.fold_id)
        if self.train_dataset[0].edge_attr is not None:
            self.num_edge_features = self.train_dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.num_features = orig_train_dataset.num_features
        self.num_classes = orig_train_dataset.num_classes
        self.avg_nodes = int(orig_train_dataset.data.num_nodes / len(orig_train_dataset))
        self.max_nodes = max([d.num_nodes for d in orig_train_dataset])

        self.seed = args.seed
        self.n_folds = args.n_folds
        if args.fold_id is not None:
            assert args.fold_id < args.n_folds
            self.fold_id = args.fold_id
        else:
            self.fold_id = 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)
    

class MultipartiteDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for multipartite graph datasets.
    
    This module handles loading and preprocessing of multipartite graph datasets, which
    contain graphs with multiple partitions.
    
    Args:
        hparams: Hydra config containing:
            - seed (int): Random seed
            - n_folds (int): Number of cross-validation folds
            - fold_id (int): Which fold to use
            - batch_size (int): Batch size for dataloaders
            - reload (bool): Whether to force reload the dataset
            - use_downloaded (bool): Whether to use downloaded data
        params: Additional parameters containing:
            - centers (int): Number of centers
            - max_N (int): Maximum number of nodes
            - graphs_per_class (int): Number of graphs per class
            - seed (int): Random seed for graph generation
    """
    def __init__(self, hparams, params):
        super().__init__()
        self.hargs = hparams
        self.args = params
        path = "data/Multipartite/"
        self.dataset = MultipartiteGraphDataset(path, centers=params.centers, max_N=params.max_N, graphs_per_class=params.graphs_per_class, 
                                                seed=params.seed, force_reload=hparams.reload, use_downloaded=hparams.use_downloaded)
        self.dataset = self.dataset.shuffle()
        self.num_features = 3
        self.num_classes = self.dataset.num_classes
        self.avg_nodes = params.max_N
        if self.dataset[0].edge_attr is not None:
            self.num_edge_features = self.dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.max_nodes = params.max_N*self.num_classes
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(self.dataset, hparams.seed, hparams.n_folds, hparams.fold_id)
        self.seed = hparams.seed
        self.n_folds = hparams.n_folds
        if hparams.fold_id is not None:
            assert hparams.fold_id < hparams.n_folds
            self.fold_id = hparams.fold_id
        else:
            self.fold_id = 0
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hargs.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hargs.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hargs.batch_size)


class TUDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for TU datasets.
    
    This module handles loading and preprocessing of TU graph classification datasets
    from the TU Dortmund benchmark collection (`TU Dataset Collection <https://chrsmrrs.github.io/datasets/>`_).
    
    Args:
        args: Configuration object containing:
            - dataset (str): Name of TU dataset to load
            - seed (int): Random seed for data splitting
            - n_folds (int): Number of cross-validation folds
            - fold_id (int): Which fold to use
            - batch_size (int): Batch size for dataloaders
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.dataset in ['COLLAB']:
            self.dataset = TUDataset(root="data/TUDataset", name=args.dataset, cleaned=True, transform=trans)
        elif args.dataset in ['REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-MULTI-5K']:
            self.dataset = TUDataset(root="data/TUDataset", name=args.dataset, cleaned=True, transform=trans)
        else:
            self.dataset = TUDataset(root="data/TUDataset", name=args.dataset, cleaned=True)
        self.dataset = self.dataset.shuffle()
        if self.dataset[0].edge_attr is not None:
            self.num_edge_features = self.dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.avg_nodes = int(self.dataset.data.num_nodes / len(self.dataset)) # perché così e non con sum()/len()?
        self.max_nodes = max([d.num_nodes for d in self.dataset])

        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(self.dataset, args.seed, args.n_folds, args.fold_id)
        
        self.seed = args.seed
        self.n_folds = args.n_folds
        if args.fold_id is not None:
            assert args.fold_id < args.n_folds
            self.fold_id = args.fold_id
        else:
            self.fold_id = 0
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)
    

class PyGSPDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for PyGSP graph datasets.
    
    This module handles loading and preprocessing of PyGSP graph datasets, which provide
    various synthetic and real-world graph structures.
    
    Args:
        args: Configuration object containing:
            - pygsp_graph (str): Name of PyGSP graph to load
            - reload (bool): Whether to force reload the dataset
            - batch_size (int): Batch size for dataloaders
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        path = "data/PyGSP"

        self.dataset = PyGSPDataset(root=path, name=args.pygsp_graph, force_reload=args.reload)
        self.num_features = self.dataset.num_features

    def train_dataloader(self):
        return DataLoader(self.dataset, self.args.batch_size)
    
class NodeClassDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for node classification datasets.
    
    This module handles loading and preprocessing of various node classification datasets.
    
    Args:
        name (str): Name of the dataset to load
        transform (callable, optional): Transform to apply to the data
        pre_transform (callable, optional): Transform to apply before saving the data
        force_reload (bool): Whether to force reload the dataset
        **args: Additional arguments including:
            - fold (int, optional): Which data fold to use
            - split (str): Split type for Planetoid datasets
            - geom_gcn_preprocess (bool): Whether to use geometric GCN preprocessing
            
    Methods:
        dataloader(): Returns a DataLoader for the dataset
        compute_homophily(method='edge_insensitive'): Computes homophily score for the dataset
    """
    def __init__(self, name, transform=None, pre_transform=None, force_reload=False, **args): 
        super().__init__()
        root = "data/NodeClass/"
        self.fold = args.get('fold', None)
        heterophilous__datasets = ['Roman-empire','Amazon-ratings', 'Minesweeper',
                                   'Tolokers', 'Questions']
        
        available_datasets = heterophilous__datasets
        assert name in available_datasets, f"Available datasets are {available_datasets}"

        if name in heterophilous__datasets:
            self.torch_dataset = HeterophilousGraphDataset(
            root=root, name=name, 
            transform=transform,
            pre_transform=pre_transform,
            force_reload=force_reload)
        
    
    def dataloader(self):
        return DataLoader(self.torch_dataset, batch_size=1)
    
    def compute_homophily(self, method='edge_insensitive'):
        return homophily(self.torch_dataset[0].edge_index, 
                         self.torch_dataset[0].y, 
                         method=method)