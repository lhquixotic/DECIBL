from argparse import Namespace
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader
import numpy as np

from DECIBL.metrics import *

def reset_frozen_gradients(network):
    mask_index = 0
    for module in network.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[network.freeze_masks[mask_index][0]] = 0  # type: ignore
            module.bias.grad[network.freeze_masks[mask_index][1]] = 0    # type: ignore
            mask_index = mask_index + 1
    return network

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def task_val(network,val_loader, args):
    network.eval()
    is_1st_loss = True
    loss_batch = 0
    
    for case_id, case in enumerate(val_loader):    
        # Get data from the loader
        case = [tensor.to(get_device()) for tensor in case]
        
        l = get_loss(network,case)
        if (case_id+1) % args.batch_size != 0:
            if is_1st_loss:
                loss = l
                is_1st_loss = False
            else:
                loss += l
        else:
            loss = loss / args.batch_size 
            is_1st_loss = True
            loss_batch += loss.item()

    return loss_batch / (case_id+1)

def task_train(network, optimizer, train_loader: DataLoader, args: Namespace):
    is_1st_loss = True
    loss_batch = 0
    network.train()
    
    for count, case in enumerate(train_loader):
        if count >= args.max_train_cases_num: # max train case
            break
        case = [tensor.to(get_device()) for tensor in case]
        optimizer.zero_grad()
        if (count+1) % args.batch_size != 0:
            l = get_loss(network,case)
            
            if is_1st_loss:
                graph_loss = l
                is_1st_loss = False
            else:
                graph_loss += l
        else:
            graph_loss = graph_loss / args.batch_size
            is_1st_loss = True
            graph_loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(network.parameters(), args.clip_grad)
            
            optimizer.step()
            loss_batch += graph_loss.item()

    train_loss = loss_batch / (count+1)

    return network, train_loss

def task_test(network, task_id, test_loader, args):
    test_times = args.test_times
    test_case_num = args.test_case_num
    eval_ade = np.zeros((test_times,test_case_num))
    eval_fde = np.zeros((test_times,test_case_num))
    for i in range(test_times):
        dataloader = test_loader
        for idx, case in enumerate(dataloader):
            if idx >= test_case_num:
                break
            else:
                case_ade,case_fde = eval_case(case,network,task_id)
                eval_ade[i,idx] = case_ade
                eval_fde[i,idx] = case_fde
        del dataloader
    return eval_ade,eval_fde