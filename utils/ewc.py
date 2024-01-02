from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from utils.utils import get_device
from utils.train_eval import get_graph_loss

def variable(t:torch.Tensor, **kwargs):
    t = t.to(get_device())
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, memory_dataset: Dataset) -> None:
        self.model = model
        self.dataset = memory_dataset
        
        self.params = self.get_parameters()
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        
        for n,p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
    
    def get_parameters(self):
        params = {}
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                params[n] = p
        return params
    
    def _diag_fisher(self):
        precision_matrices = {}
        for n,p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        
        self.model.eval()
        mem_loader = DataLoader(self.dataset)
        for cnt, case in enumerate(mem_loader):
            ''' For previous tasks '''
            self.model.zero_grad()
            case = [tensor.to(get_device()) for tensor in case]
            l = get_graph_loss(self.model,case)
            l.backward()
            
            for n,p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
            
        precision_matrices = {n:p for n, p in precision_matrices.items()}
        return precision_matrices
    
    def penalty(self, model:nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss  = self._precision_matrices[n] * (p-self._means[n]) ** 2
            loss += _loss.sum()
        return loss        

def ewc_task_train(model, optimizer, dataloader, ewc, args):
    is_1st_loss = True
    loss_batch = 0
    model.train()
    for cnt, case in enumerate(dataloader):
        case = [tensor.to(get_device()) for tensor in case]
        optimizer.zero_grad()
        
        l = get_graph_loss(model,case)
        if is_1st_loss:
            graph_loss = l
            is_1st_loss = False
        else:
            graph_loss += l
        
        if (cnt+1) % args.batch_size == 0:
            graph_loss = graph_loss / args.batch_size + args.ewc_importance * ewc.penalty(model)
            is_1st_loss = True
            graph_loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            loss_batch += graph_loss.item()
            

    train_loss = loss_batch / (int((cnt+1)/args.batch_size))

    return model, train_loss