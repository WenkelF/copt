import torch
import numpy
import random


def set_seed(seed):
    
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)