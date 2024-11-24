# Towards a General Recipe for Combinatorial Optimization with Multi-Filter GNNs

### [Frederik Wenkel*](https://wenkelf.github.io/), [Semih Cantürk*](https://semihcanturk.github.io/), [Stefan Horoi](https://shoroi.github.io/), [Michael Perlmutter](https://sites.google.com/view/perlmutma/home), [Guy Wolf](https://guywolf.org/)

_Accepted as Spotlight at Learning on Graphs (LoG) 2024_

[![arXiv](https://img.shields.io/badge/arXiv-2405.20543-b31b1b.svg)](https://arxiv.org/abs/2405.20543)

![img](GCON.jpg)

## Installation

For `conda`, using the `environment.yml` file:
```bash
conda env create -f environment.yml
```

For `pip`, using `requirements.txt`:
```bash
pip install -r requirements.txt
```

Alternatively, you can install the packages manually via `conda`:
```bash
conda create -n copt python=3.10
conda activate copt

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pyg -c pyg
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg
# might need to install latest torch-sparse via pip instead
pip install git+https://github.com/rusty1s/pytorch_sparse.git
conda install lightning -c conda-forge
pip install yacs einops loguru dwave-networkx ogb performer-pytorch wandb
```

## Quick start

This codebase is built on top of [PyG GraphGym](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html), and uses configuration files to run experiments.
The default GCON configurations are found in `configs/benchmarks/{TASK_NAME}`. E.g., to run GCON for MCut on BA-small:
```bash
python main.py --cfg configs/benchmarks/maxclique/maxclique_rb_small.yaml
```

To use the non-decoupled architecture, override the `gcon` layer in the default config with the `hybridconv` convolution:
```bash
python main.py --cfg configs/benchmarks/maxclique/maxclique_rb_small.yaml gnn.layer_type=hybridconv
```

The Erdos' GNN configurations are denoted with an `-erdos` suffix:
```bash
python main.py --cfg configs/benchmarks/maxclique/maxclique_rb_small-erdos.yaml
```

To use Erdos' GNN with entropy annealing, use the `optim.entropy.enable` flag:
```bash
python main.py --cfg configs/benchmarks/maxclique/maxclique_rb_small-erdos.yaml optim.entropy.enable=True
```

You can use your WandB account for logging by setting `wandb.entity` to your own entity. You can also use local-only logging by setting `wandb.use=False`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wenkel2024generalgnnframeworkcombinatorial,
      title={Towards a General GNN Framework for Combinatorial Optimization}, 
      author={Frederik Wenkel and Semih Cantürk and Michael Perlmutter and Guy Wolf},
      year={2024},
      eprint={2405.20543},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.20543}, 
}
```