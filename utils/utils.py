import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn 

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(task, shuffle=True):
    task_dataset = task if isinstance(task,Dataset) else task.dataset
    dataloader = DataLoader(task_dataset, batch_size=1, shuffle=shuffle,
                            num_workers=16, drop_last=True)
    return dataloader

def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True