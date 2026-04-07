import os
import random
import numpy as np
import torch
import torch.nn as nn


def calc_model_params(model:nn.Module) -> int:
    params = model.parameters()
    total = 0
    for item in params:
        total += item.numel()
    return total


def init_rand_seed(seed:int=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    