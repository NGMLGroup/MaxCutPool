# {Paper Title} ({Venue} {Year})

[![ICLR](https://img.shields.io/badge/{Venue}-{Year}-blue.svg?style=flat-square)]({Link to paper page})
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)]({Link to paper pdf})
[![arXiv](https://img.shields.io/badge/arXiv-2409.05100-b31b1b.svg?style=flat-square)](http://arxiv.org/abs/2409.05100)

This repository contains the code for the reproducibility of the experiments presented in the paper "{Paper Title}" ({Venue} {Year}). {Paper TL;DR}.

**Authors**: [Author 1](mailto:{Author1 mail}), [Author 2](mailto:{Author2 mail})

---

## In a nutshell

{Paper description}.

<!-- p align=center>
	<img src="./overview.png" alt="{Image description}"/>
</p -->

---

## Directory structure

The directory is structured as follows:

```
.
├── config/
│   ├── exp1/
│   └── exp2/
├── datasets/
├── lib/
├── requirements.txt
├── conda_env.yaml
└── experiments/
    ├── exp1.py
    └── exp2.py

```


## Datasets

All datasets are automatically downloaded and stored in the folder `datasets`.

The datasets used in the experiment are provided by [pyg](). GCB-H and EXPWL1 datasets are downloaded from these links:
- [GCB-H]().
- [EXPWL1]().

### Multipartite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11616515.svg)](https://zenodo.org/doi/10.5281/zenodo.11616515)

Descrivere come è fatto il dataset

## Configuration files

The `config` directory stores all the configuration files used to run the experiment. They are divided into subdirectories according to the experiment they refer to.

## Requirements

We run all the experiments in `python 3.8`. To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yml
conda activate env_name
```

Alternatively, you can install all the requirements listed in `requirements.txt` with pip:

```bash
pip install -r requirements.txt
```

## Library

The support code, including the models and the datasets readers, are packed in a python library named `lib`. Should you have to change the paths to the datasets location, you have to edit the `__init__.py` file of the library.


## Experiments

The scripts used for the experiments in the paper are in the `experiments` folder.

* `exp1.py` is used to ... . An example of usage is

	```
	python experiments/exp1 --config exp1/config.yaml args
	```


## Bibtex reference

If you find this code useful please consider to cite our paper:

```
{Bibtex reference}
```
