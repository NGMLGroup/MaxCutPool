# MaxCutPool

Official implementation of the paper ["MaxCutPool: differentiable feature-aware Maxcut for pooling in graph neural networks"]

## Overview

This repository contains the implementation of MaxCutPool, a novel approach to compute the MAXCUT in attributed graphs. The method is designed to work with graphs that have features associated with both nodes and edges. Key features:

- Fully differentiable architecture
- Robust to underlying graph topology
- Optimizes MAXCUT jointly with other objectives
- Implements a hierarchical graph pooling layer for GNNs
- Particularly effective for heterophilic graphs

## Installation

To install the required packages, create a conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate maxcutpool
```


The main dependencies include:
- Python 3.10
- PyTorch >= 2.0.0
- PyTorch Geometric (PyG)
- PyTorch Lightning
- Hydra
- Neptune (for logging)
- Various scientific computing libraries (numpy, scipy, etc.)

For a complete list of dependencies, see `environment.yml`.

## Usage

### Quick Start

For a basic example of how to use the MaxCutPool layer, check out `example.py`. This file demonstrates:
- Basic setup of the layer
- Integration with PyTorch Geometric
- Training and evaluation on a sample dataset

### Experiments

To replicate the experiments from the paper, you can use the following run scripts:

1. MaxCut experiments:

```bash
python run_maxcut.py
```

2. Graph classification experiments:

```bash
python run_graph_classification.py
```

3. Node classification experiments:

```bash
python run_node_classification.py
```

Each script uses [Hydra](https://hydra.cc/) for configuration management. The corresponding YAML config files can be found in the `config` directory. You can override any configuration parameter from the command line, for example:

```bash
python run_graph_classification.py dataset=expwl1 pooler=edgepool
```

## Project Structure

```
./
├── config/                     # Run configuration files
├── source/                     # Main package directory
│   ├── data/                   # Dataset handling
│   ├── layers/                 # Neural network layers
│   │   ├── maxcutpool/         # MaxCutPool implementation
│   │   ├── edgepool/           # EdgePool implementation
│   │   ├── kmis/               # KMIS implementation
│   │   └── ndp.py              # NDP implementation
│   ├── models/                 # Model architectures
│   ├── pl_modules/             # PyTorch Lightning modules
│   └── utils/                  # Utility functions
├── example.py                  # Quick start example
├── run_maxcut.py               # MaxCut experiment runner
├── run_maxcut_baseline.py      # MaxCut baseline comparisons
├── run_classification.py       # Classification experiment runner
├── run_node_classification.py  # Node classification runner
├── environment.yml             # Conda environment specification
├── README.md                   # This README file
├── requirements.txt            # PyG dependencies
└── LICENSE                     # MIT License
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
