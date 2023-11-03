## Installation

```
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