import numpy as np
import random
import pickle
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset,random_split
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
    
def load_memory_data(cur_tid, memory_path, args):
    if cur_tid <= 0:
        return None
    else:
        mem_lists = []
        ave_mem = int(args.mem_size/cur_tid)
        for tid in range(cur_tid):
            pre_mem_fname = memory_path+'/task_{}.pkl'.format(tid)
            assert os.path.exists(pre_mem_fname)
            with open(pre_mem_fname, "rb") as file:
                pre_mem = pickle.load(file)
            split_sizes = [ave_mem, len(pre_mem) - ave_mem]
            mem_lists.append(random_split(pre_mem,split_sizes)[0])
        return ConcatDataset(mem_lists)

def save_memory_data(cur_id, train_task, memory_path, args):
    print("train_task_len:",len(train_task.dataset))
    assert len(train_task.dataset) >= args.mem_size
    assert os.path.exists(memory_path)
    split_sizes = [args.mem_size, len(train_task.dataset) - args.mem_size]
    memorized_data = random_split(train_task.dataset, split_sizes)
    memorized_data = memorized_data[0]
    mem_data_fname = memory_path + "/task_{}.pkl".format(cur_id)
    with open(mem_data_fname, "wb") as file:
        pickle.dump(memorized_data, file)