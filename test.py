import os
import datetime

import torch
import torch.nn as nn
import numpy as np

import argparse
import copy

from DECIBL.metrics import *
from scenarios.utils import *
from DECIBL.train_eval import task_test
# from metrics import *
from DECIBL.psstgcnn import *
from DECIBL.train_eval import get_device

import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from DECIBL.learner import Learner

# Indicate a model and dataset to test in terms of ADE and FDE
# The first scenario dataset will be test after each addtional scenario is used to train

# region [Arguments] add arguments parser
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--model_name',default='social-stgcnn')
parser.add_argument('--experiment_name', default="experiment-1-4")
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model_path',default='./checkpoint/social-stgcnn-MA/val_best.pth')
parser.add_argument('--train_method',default='vanilla')
parser.add_argument('--expert_id', type=int, default=-1)
parser.add_argument('--test_sequence', default="4")
parser.add_argument('--start_task', type=int, default=0)


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


def load_trained_network(network, task_index, args):
    if len(network.columns) == 0:
        network.new_task()
    else:
        print("network column number is {}".format(len(network.columns)))
        # Load checkpoint to load previous knowledge, including model paras, expert selector, and bn paras
        ckpt_path = "./logs/{}/checkpoint/val_best_model_{}.pth".format(args.experiment_name,task_index)
        print("Load existing model knowledge: {}".format(ckpt_path))
        assert os.path.exists(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        print("Load success!")    
        
        # trained_tasks = [seen_task for key, value in checkpoint["expert_selector"].items() for seen_task in value]     
        expand_times = task_index
        
        print("For task_{} the model need to expand {} times.".format(task_index, expand_times))
        
        # expand if load pretrained model
        for t in range(expand_times+1):
            
            if t == expand_times: # before last time for expansion, load the existing model
                    
                    if 'expert_selector' in checkpoint.keys():
                        pass
                    else:
                        last_ckpt_path = "./logs/{}/checkpoint/val_best_model_{}.pth".format(args.experiment_name,task_index-1)
                        last_ckpt = torch.load(last_ckpt_path)
                        checkpoint['expert_selector'] = last_ckpt['expert_selector']
                        print("last expert selector is {}".format(last_ckpt['expert_selector']))
                        existing_tasks = [tid for key,value in checkpoint['expert_selector'].items() for tid in value]
                        checkpoint['expert_selector'][len(checkpoint['expert_selector'])] = [max(existing_tasks) + 1]
                    
                    network.load_state_dict(checkpoint['model_state_dict'])
                    
                    network.expert_selector = copy.deepcopy(checkpoint["expert_selector"])
                    network.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
                    network.freeze_columns() # freeze the existing knowledge
                    network.load_batch_norm_para(task_index-1)
            else:
                network.new_task()
                print("net columns: {}".format(len(network.columns)))
    
    print("Current column number is {}, expert-task dict is {}, bn para number is {}".format(
        len(network.columns), network.expert_selector,len(network.batch_norm_para))) 
    
    return network

def test_given_task(network, test_task_id, expert_id, test_datasets, save_dir=None):
    assert expert_id < len(network.columns) # expert id should < column number
    network.load_batch_norm_para(test_task_id)

    for id,task_dataset in enumerate(test_datasets):
        if id + args.start_task == test_task_id:
            test_loader = DataLoader(task_dataset, batch_size=1, shuffle=False,num_workers=16,drop_last = True)
            ade, fde = task_test(network, expert_id, test_loader, args)
            
            # save if save_dir is provided
            if save_dir is not None:        
                np.save(save_dir+"ADE-task-{}-exp-{}.npy".format(test_task_id,expert_id),ade)
                np.save(save_dir+"FDE-task-{}-exp-{}.npy".format(test_task_id,expert_id),fde)
                
                print("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                    len(network.columns), test_task_id, expert_id, ade.mean(), fde.mean()))
            return ade, fde
    return None,None

task_dict = {1:'MA',2:'FT',3:'ZS',4:'EP',5:'SR'}
# endregion

# region [Model] load trained model
# Defining the model and load values
model = progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
model.new_task()

model = load_trained_network(model,3,args)
model = model.to(get_device())
torch.manual_seed(args.seed)

# endregion

# load datasets
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len

test_sequence = [int(i) for i in args.test_sequence.split("-")]
test_seq_name = [str(i)+"-"+task_dict[i] for i in test_sequence]
test_dirs = ['./scenarios/datasets/' + d_name + '/test/' for d_name in test_seq_name]
print("[Data] load dataset is ",test_dirs)

test_datasets = [TrajectoryDataset(
                    test_dir,
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,
                    norm_lap_matr=True
                ) for test_dir in test_dirs]

save_dir = "./logs/{}/evaluation/columns_{}/".format(args.experiment_name,len(model.columns))
if not os.path.exists(save_dir):
                os.makedirs(save_dir)

for id in range(len(test_datasets)):
    expert_id = model.select_expert(id)
    ade,fde = test_given_task(model,test_task_id=id, expert_id=expert_id, test_datasets=test_datasets, save_dir=save_dir)
    ade,fde = test_given_task(model,test_task_id=3,expert_id=id,test_datasets=test_datasets,save_dir=save_dir)