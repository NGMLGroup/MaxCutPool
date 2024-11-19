import os
from os import path
import numpy as np
import pickle
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx
from pygsp2 import graphs
import os.path as osp
import shutil
from typing import Callable, List, Optional
import networkx as nx
from io import StringIO

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from .tu import read_tu_data
from .multipartite_utils import rotate_list, get_graph_data, get_centers


class EXPWL1Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPWL1Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXPWL1.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url('https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs/raw/main/data/EXPWL1/raw/EXPWL1.pkl', 
                     self.raw_dir)  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXPWL1.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class GraphClassificationBench(InMemoryDataset):
    """The synthetic dataset from `"Pyramidal Reservoir Graph Neural Network"
    <https://arxiv.org/abs/2104.04710>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If `"train"`, loads the training dataset.
            If `"val"`, loads the validation dataset.
            If `"test"`, loads the test dataset. Defaults to `"train"`.
        easy (bool, optional): If `True`, use the easy version of the dataset.
            Defaults to `True`.
        small (bool, optional): If `True`, use the small version of the
            dataset. Defaults to `True`.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to `None`.
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. Defaults to `None`.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. Defaults to `None`.
    """
    base_url = ('http://github.com/FilippoMB/'
                'Benchmark_dataset_for_graph_classification/'
                'raw/master/datasets/')
    
    def __init__(self, root, split='train', easy=True, small=True, transform=None, pre_transform=None, pre_filter=None):
        self.split = split.lower()
        assert self.split in {'train', 'val', 'test'}
        if self.split != 'val':
            self.split = self.split[:2]
        
        self.file_name = ('easy' if easy else 'hard') + ('_small' if small else '')
        
        super(GraphClassificationBench, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.file_name)
    
    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.file_name+ '_' + self.split)

    def download(self):
        download_url('{}{}.npz'.format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        npz = np.load(path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        raw_data = (npz['{}_{}'.format(self.split, key)] for key in ['feat', 'adj', 'class']) 
        data_list = [Data(x=torch.FloatTensor(x), 
                          edge_index=torch.LongTensor(np.stack(adj.nonzero())), 
                          y=torch.LongTensor(y.nonzero()[0])) for x, adj, y in zip(*raw_data)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


class PyGSPDataset(InMemoryDataset):
    def __init__(self, root, 
                 name='community',
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 force_reload=False,
                 kwargs=None):
        
        self._GRAPHS = [
            'Graph',
            'Airfoil',
            'BarabasiAlbert',
            'Comet',
            'Community',
            'DavidSensorNet',
            'ErdosRenyi',
            'FullConnected',
            'Grid2d',
            'Logo',
            'LowStretchTree',
            'Minnesota',
            'Path',
            'RandomRegular',
            'RandomRing',
            'Ring',
            'Sensor',
            'StochasticBlockModel',
            'SwissRoll',
            'Torus'
        ]

        self._NNGRAPHS = [
            'NNGraph',
            'Bunny',
            'Cube',
            'ImgPatches',
            'Grid2dImgPatches',
            'Sphere',
            'TwoMoons'
        ]

        # check if the graph is in the list of available graphs.
        if name not in self._GRAPHS and name not in self._NNGRAPHS:
            raise ValueError(f"Graph {name} not available in PyGSP. Available graphs are:\n{self._GRAPHS}\nand\n{self._NNGRAPHS}")

        if name in self._GRAPHS:
            graph = getattr(graphs, name)
        else:
            graph = getattr(graphs.nngraphs, name)
        self.G = graph(**kwargs) if kwargs is not None else graph()

        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, force_reload=force_reload)
        
        if torch_geometric.__version__ > '2.4':
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        edge_index, edge_weights = from_scipy_sparse_matrix(self.G.W)

        # Set coords if the graph does not have them
        if not hasattr(self.G, 'coords'):
            self.G.set_coordinates(kind='spring', seed=42)

        # self.num_features = self.G.coords.shape[1]

        data_list = [Data(x=torch.tensor(self.G.coords.astype('float32')), 
                          edge_index=edge_index,
                          edge_attr=edge_weights,
                          y=torch.zeros(self.G.N, dtype=torch.long))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if torch_geometric.__version__ > '2.4':
            self.save(data_list, self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])


class MultipartiteGraphDataset(InMemoryDataset):
    def __init__(self, root, 
                 centers=10,
                 max_N=20,
                 graphs_per_class=500,
                 seed=777,
                 use_downloaded=True,
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 force_reload=False):
        
        self.centers = get_centers(centers)
        self.max_N = max_N
        self.graphs_per_class = graphs_per_class
        self.use_downloaded = use_downloaded
        self.seed = seed

        super(MultipartiteGraphDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return len(torch.unique(self.data.y))

    @property
    def raw_file_names(self):
        return ['Multipartite.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url('https://zenodo.org/records/11617423/files/Multipartite.pkl?download=1', 
                     self.raw_dir)

    def process(self):

        if not self.use_downloaded:
            np.random.seed(self.seed)
            data_list = []
            for i in range(len(self.centers)):
                rotated_centers = rotate_list(self.centers, i)
                graphs = [get_graph_data(rotated_centers, self.max_N) for _ in range(self.graphs_per_class)]
                for graph in graphs:
                    graph.y = torch.tensor([i], dtype=torch.long)
                    data_list.append(graph)
                print(f"Class {i} done.")

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            
            torch.save(data_list, self.root + '/raw/Multipartite_local.pkl')
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        
        else:
            print(self.root)
            data_list = torch.load(self.root + '/raw/Multipartite.pkl')
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])



class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.*,
    :obj:`"IMDB-BINARY"`, :obj:`"REDDIT-BINARY"` or :obj:`"PROTEINS"`,
    collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - MUTAG
          - 188
          - ~17.9
          - ~39.6
          - 7
          - 2
        * - ENZYMES
          - 600
          - ~32.6
          - ~124.3
          - 3
          - 6
        * - PROTEINS
          - 1,113
          - ~39.1
          - ~145.6
          - 3
          - 2
        * - COLLAB
          - 5,000
          - ~74.5
          - ~4914.4
          - 0
          - 3
        * - IMDB-BINARY
          - 1,000
          - ~19.8
          - ~193.1
          - 0
          - 2
        * - REDDIT-BINARY
          - 2,000
          - ~429.6
          - ~995.5
          - 0
          - 2
        * - ...
          -
          -
          -
          -
          -
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        self.data, self.slices, self.sizes = out

        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

class GsetDataset(InMemoryDataset):
    """
    Dataset class for the Gset dataset http://web.stanford.edu/~yyye/yyye/Gset
    The dataset is mainly used to evaluate the performance of MAXCUT algorithms.
    """

    base_url = "http://web.stanford.edu/~yyye/yyye/Gset/"

    def parse_graph(self, data):
        """ Parse the graph data and create a NetworkX graph. """
        # Create a buffer from the data string
        buf = StringIO(data)
        
        # Read the first line separately which contains the number of nodes and edges
        num_nodes, num_edges = map(int, buf.readline().strip().split())
        
        # Create a graph with the given number of nodes
        if self._DIRECTED:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_nodes_from(range(1, num_nodes+1))

        # Read the rest of the lines, which contain the edges
        for line in buf:
            parts = line.strip().split()
            if len(parts) == 3:
                u, v, weight = map(int, parts)
                # Add an edge to the graph
                G.add_edge(u, v, weight=weight)

        return G

    def __init__(self, root, 
                 name="G1",
                 directed=False,
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 **kwargs):
        
        self._DIRECTED = directed
        
        self._GRAPHS = ["G1", "G2", "G3", "G4", "G5", 
                        "G6", "G7", "G8", "G9", "G10", 
                        "G11", "G12", "G13", "G14", "G15", 
                        "G16", "G17", "G18", "G19", "G20", 
                        "G21", "G22", "G23", "G24", "G25", 
                        "G26", "G27", "G28", "G29", "G30", 
                        "G31", "G32", "G33", "G34", "G35", 
                        "G36", "G37", "G38", "G39", "G40", 
                        "G41", "G42", "G43", "G44", "G45", 
                        "G46", "G47", "G48", "G49", "G50", 
                        "G51", "G52", "G53", "G54", "G55", 
                        "G56", "G57", "G58", "G59", "G60", 
                        "G61", "G62", "G63", "G64", "G65", 
                        "G66", "G67", "G70", "G72", "G77", 
                        "G81"]

        if name not in self._GRAPHS:
            raise ValueError(f"Invalid graph name: {name}")
        self.file_name = name

        super(GsetDataset, self).__init__(root, transform, pre_transform, pre_filter, **kwargs)
        
        if torch_geometric.__version__ > '2.4':
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.file_name
    
    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.file_name)

    def download(self):
        download_url("{}{}".format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        with open(path.join(self.raw_dir, self.raw_file_names), 'rb') as f:

            data = f.read().decode('utf-8')
            graph = self.parse_graph(data)
            pyg_graph = from_networkx(graph)

            rnd_feats = torch.randn(pyg_graph.num_nodes, 32)

            data_list = [Data(
                # x=pyg_graph.x,
                x=rnd_feats,
                edge_index=pyg_graph.edge_index, 
                edge_attr=pyg_graph.weight, 
                num_nodes=pyg_graph.num_nodes,)
                ]

            if self.pre_filter is not None:
                data = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data = [self.pre_transform(data) for data in data_list]

            if torch_geometric.__version__ > '2.4':
                self.save(data_list, self.processed_paths[0])
            else:
                torch.save(self.collate(data_list), self.processed_paths[0])