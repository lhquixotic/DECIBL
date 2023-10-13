import os
import datetime

import torch
import torch.nn as nn
import numpy as np

import argparse
import copy

from DECIBL.metrics import *
from scenarios.utils import *
# from metrics import *
from DECIBL.psstgcnn import *
from DECIBL.train_eval import get_device

import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.distributions.multivariate_normal as torchdist


import matplotlib.pyplot as plt

import itertools

# Indicate a model and dataset to test in terms of ADE and FDE
# The first scenario dataset will be test after each addtional scenario is used to train

# region [Arguments] add arguments parser
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--model_name',default='social-stgcnn')
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model_path',default='./checkpoint/social-stgcnn-MA/val_best.pth')
parser.add_argument('--train_method',default='vanilla')

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)

# Test specific parameters

parser.add_argument('--dataset', default='1-MA',
                    help='MA, FT, SR, EP, ZS')   
parser.add_argument('--load_tag', default='MA',
                    help='load model name')
parser.add_argument('--cur_task',default=1)
parser.add_argument('--is_demo',type=int,default=0)
parser.add_argument('--test_case_num', type=int, default=20,
                    help='number of observed cases')
parser.add_argument('--test_times',type=int,default=10)
parser.add_argument('--obs_case_num',type=int,default=1000)

args = parser.parse_args()

task_dict = {1:'MA',2:'FT',3:'ZS',4:'EP',5:'SR'}
# endregion

# region [Model] load trained model
# Defining the model and load values
model = progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
print("cur task:",int(args.cur_task))
# for i in range(int(args.cur_task)):
#     print("new_task {}".format(i))
model.new_task()
if int(args.cur_task) == 1:
    model.new_task()
model = model.to(get_device())
# model_path = './checkpoints/' + args.train_method+ '/'+ args.model_name   + '/'+ args.load_tag+ '/' + 'val_best.pth'
model_path = './checkpoints/' + args.train_method + '/' + 'val_best_model_{}.pth'.format(args.cur_task)
assert os.path.exists(model_path)
print("[Model] load model path is ",model_path)
model.load_state_dict(torch.load(model_path))
torch.manual_seed(args.seed)
# endregion

# region [Data] load dataset used to test
# load datasets
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
dataset = './scenarios/datasets/' + args.dataset + '/'
test_dataset = dataset + 'test/'
print("[Data] load dataset is ",test_dataset)
# dataset and dataloader for test
test_data = TrajectoryDataset(
    test_dataset,
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1,
    norm_lap_matr=True
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    num_workers=16,
    drop_last=True
)
# endregion

# region [Test] test and restore the results
# variables for test results
test_times = args.test_times
eval_ade = np.zeros((test_times,args.obs_case_num))
eval_fde = np.zeros((test_times,args.obs_case_num))
# directory for evaluation
# eval_dir = './evaluations/' + args.train_method+ '/'+ args.model_name   + '/'+ args.load_tag + '/'
eval_dir = './evaluations/' + args.train_method+ '/' + args.load_tag + '/'

# eval_dir means which model is used for test
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# test for a certain times
for i in range(test_times):
    copy_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        drop_last=True
    )
    
    for idx, case in enumerate(copy_loader):
        if idx >= args.obs_case_num:
            break
        else:
            case_ade,case_fde = eval_case(case,model,task_id=int(args.load_tag[0])-1)
            eval_ade[i,idx] = case_ade
            eval_fde[i,idx] = case_fde
            # print("Test times {}, test case {}, ade {}, fde {}".format(i,idx,case_ade,case_fde))
            
    del copy_loader

# save the results
np.save(eval_dir+"ADE-"+args.dataset+'.npy',eval_ade)
np.save(eval_dir+"FDE-"+args.dataset+'.npy',eval_fde)

# display the results
# x = np.arange(args.obs_case_num)
# y_ade_means = np.mean(eval_ade,axis=0)
# y_ade_var = np.var(eval_ade,axis=0)
# y_fde_means = np.mean(eval_fde,axis=0)
# y_fde_var = np.var(eval_fde,axis=0)

# print("Model "+args.load_tag+" tested on Dataset "+args.dataset+", ADE: "+str(np.mean(y_ade_means))+"("+str(np.mean(y_ade_var))+"), FDE: "+str(np.mean(y_fde_means))+"("+str(np.mean(y_fde_var))+").")
# print(y_ade_means)
# print(y_ade_var)
# print(y_fde_means)
# print(y_fde_var)

# endregion