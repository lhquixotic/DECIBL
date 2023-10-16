import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import copy

import torch.optim as optim

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    
class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
        
        # model construct
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        
        return v,a

class progressive_social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(progressive_social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.pred_seq_len = pred_seq_len

        # For Progressive Neural Network
        self.columns = nn.ModuleList()
        self.block_number = n_stgcnn + n_txpcnn

        # for expert selection
        self.expert_selector = dict()
        self.batch_norm_para = dict()
        
    def load_batch_norm_para(self, task_id):
        expert_id = self.select_expert(task_id)
        n = 0 # number of loaded para
        for i in range(expert_id+1):
            for k, module in enumerate(self.columns[i].modules()):
                if isinstance(module, nn.BatchNorm2d):      
                    module.running_mean = copy.deepcopy(self.batch_norm_para[task_id][0][n])
                    module.running_var = copy.deepcopy(self.batch_norm_para[task_id][1][n])
                    n+=1
    
    def save_batch_norm_para(self, task_id):
        # save the batch norm para of all columns respect to task_id
        bn_mean = []
        bn_var = []
        for i in range(len(self.columns)):    # for all column 
            for module in self.columns[i].modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_mean.append(module.running_mean)
                    bn_var.append(module.running_var)
        # print("[save] bn para:",self.batch_norm_para[0][0][0])
        self.batch_norm_para[task_id] = copy.deepcopy([bn_mean,bn_var])
        
    def new_task(self):
        expert_id = len(self.columns)

        # define the blocks contained in the baseline model
        defined_blocks = []
        for k in range(self.block_number):
            # stgcnns
            if k < self.n_stgcnn:
                defined_blocks.append(st_gcn(self.input_feat,self.output_feat,
                                             (self.kernel_size,self.seq_len)))
            # tpcnns
            elif k == self.n_stgcnn:
                defined_blocks.append(nn.Conv2d(self.seq_len,self.pred_seq_len,3,padding=1))
            elif k < self.block_number - 1:
                defined_blocks.append(nn.Conv2d(self.pred_seq_len,self.pred_seq_len,3,padding=1))
            else:
                defined_blocks.append(nn.Conv2d(self.pred_seq_len,self.pred_seq_len,3,padding=1))
        
        # for a new task, add a new list contains lateral blocks of all layers
        new_blocks = [] 

        for k in range(self.block_number):
            new_blocks.append(LateralBlock(col=expert_id,
                             depth=k,
                             block=defined_blocks[k]))
        
        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)
        
        # update expert selector
        self.expert_selector[expert_id] = []

    def forward(self,v,a,expert_id=-1):
        # check whether columns exist
        assert self.columns

        # calculate the first layer output as inputs
        inputs = []

        for column in self.columns:
            # print("col 0, layer 0:",column[0])
            v_,a_ = column[0]([v,a],is_1st_layer=True) # column[0] -> st_gcns
            
            # reshape 
            v_ = v_.view(v_.shape[0],v_.shape[2],v_.shape[1],v_.shape[3])
            inputs.append(v_)
        
        for l in range(1,self.block_number): # from the 2nd block to the last block
            out = []
            for col_id, col in enumerate(self.columns): 
                if l == 1: # 1st layer of tpcnn
                    layer_out = col[l](inputs[:col_id+1], activation=nn.PReLU().to(get_device()))
                    out.append(layer_out)
                elif l < self.block_number - 1: # tpcnn out before the output layer
                    layer_out = col[l](inputs[:col_id+1], activation=nn.PReLU().to(get_device())) + inputs[col_id]
                    out.append(layer_out)
                elif l == self.block_number - 1:
                    layer_out = col[l](inputs[:col_id+1])
                    layer_out = layer_out.view(layer_out.shape[0],layer_out.shape[2],
                                               layer_out.shape[1],layer_out.shape[3])
                    out.append(layer_out)
            inputs = out
        return out[expert_id]
    
    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for name, params in c.named_parameters():
                    params.requires_grad = False
    
    def select_expert(self, task_id):
        expert_id = len(self.columns) - 1
        # task id is recorded
        seen_tasks = []
        for key, value in self.expert_selector.items():
            for v in value:
                    seen_tasks.append(v)
            if task_id in value:
                expert_id = key
                return expert_id
        
        # task id is not recorded
        if task_id == len(seen_tasks):
            return len(self.columns) - 1
        
        return None

class LateralBlock(nn.Module):
    def __init__(self,col,depth,block):
        super(LateralBlock,self).__init__()
        self.col = col     # task id
        self.depth = depth # layer index of the current block
        self.block = block # block should contain nn.Conv2d
        
        if self.depth > 0:
            # determine the shape of lateral connection
            if isinstance(self.block,list) or isinstance(self.block,nn.ModuleList):
                for layer in self.block:
                    if isinstance(nn.Conv2d):
                        weight_shape = layer.weight.shape
                        break
            else:
                assert isinstance(self.block,nn.Conv2d)
                weight_shape = self.block.weight.shape
            out_channels = weight_shape[0]
            in_channels = weight_shape[1] * self.col
            
            if in_channels > 0:
                self.lateral_connection = nn.Conv2d(in_channels,out_channels,3,padding=1)
                # When sel.col > 0, we add lateral connection for the block
            else:
                self.lateral_connection = None
                # When sel.col == 0, No lateral connection

    def forward(self, inputs, activation=None,is_1st_layer=False):
        if is_1st_layer:
            v,a = inputs[0],inputs[1]
            return self.block(v,a)
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        cur_column_out = self.block(inputs[-1])
        # if lateral connection exists, calculate it
        if self.lateral_connection is not None:
            prev_columns_out = self.lateral_connection(torch.cat(inputs[:-1],dim=1))
        else:
            prev_columns_out = 0
        # assert cur_column_out.shape == prev_columns_out.shape
        
        res = cur_column_out + prev_columns_out
        if activation is not None:
            res = activation(res)
        return res

