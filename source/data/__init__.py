"""Data handling modules for graph datasets and data loading.

This package provides data modules and datasets for various graph learning tasks:

Data Modules:
    - EXPWL1DataModule: For evaluating GNN expressiveness
    - BenchHardDataModule: For challenging graph classification tasks
    - MultipartiteDataModule: For multipartite graph datasets
    - TUDataModule: For TU graph classification benchmarks
    - PyGSPDataModule: For PyGSP graph datasets
    - NodeClassDataModule: For node classification tasks

Datasets:
    - EXPWL1Dataset: Dataset for GNN expressiveness evaluation
    - GraphClassificationBench: Benchmark datasets for graph classification
    - PyGSPDataset: PyGSP graph datasets
    - GsetDataset: Gset graph datasets for MAXCUT problems
    - MultipartiteGraphDataset: Multipartite graph datasets
    - TUDataset: TU graph classification benchmarks
"""

from .data_modules import (EXPWL1DataModule, 
                           BenchHardDataModule, 
                           MultipartiteDataModule, 
                           TUDataModule,
                           PyGSPDataModule,
                           NodeClassDataModule)

from .torch_datasets import (EXPWL1Dataset, 
                             GraphClassificationBench, 
                             PyGSPDataset,
                             GsetDataset,
                             MultipartiteGraphDataset,
                             TUDataset
                             )

from .multipartite_utils import *