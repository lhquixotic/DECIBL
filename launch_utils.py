import argparse
import numpy as np
import random
from typing import Tuple
import os

import torch
from torch.backends import cudnn 


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DECIBL')

    # experiment information parameters
    parser.add_argument('--experiment_name', type=str, default = 'test')
    parser.add_argument('--is_demo',type=int,default=0)
    parser.add_argument('--debugging', type=int, default=0)
    parser.add_argument('--no_train',type=int, default=0)
    parser.add_argument('--no_test',type=int, default=0)
    
    # scenario parameters
    parser.add_argument('--train_start_task', type=int, default=0)
    parser.add_argument('--test_start_task', type=int, default=0)
    parser.add_argument('--task_seq', type=str, default="1-2-3")
    parser.add_argument('--datasets_split_num', type=int, default=1)
    parser.add_argument('--obs_seq_len', type=int, default=20)
    parser.add_argument('--pred_seq_len', type=int, default=40)
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)
    
    # Training method parameters
    parser.add_argument("--train_method", type=str, default="DEM")
    parser.add_argument('--seed', type=int,  default=1)
    parser.add_argument('--deterministic', type=int,  default=1)
    parser.add_argument('--batch_size', type=int, default=5,help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=5,help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,help='gadient clipping')    
    parser.add_argument('--early_stop_epochs', type=int, default=50, 
                        help="early stop if val loss didn't decrease for long")
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default = 'SGD')
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    parser.add_argument('--lr_sh_rate', type=int, default=100,help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', type=int, default=0,help='Use lr rate scheduler')
    
    # DEM parameters
    parser.add_argument('--init_new_expert',type=int, default=0)
    parser.add_argument('--expand_thres',type=float,default=0.3)    
    parser.add_argument('--task_free',type=int, default=0, 
                        help="Task free means no approach to task label")
    parser.add_argument('--gae_lr', type=float, default=0.01)
    parser.add_argument('--no_gae_train', type=int,default=1)
    
    # GSM parameters
    parser.add_argument('--mem_size', type=int, default=3500)
    parser.add_argument('--margin', type=float, default=0.5,help='for quadprog computing')
    parser.add_argument('--eps', type=float, default=0.001,help='for quadprog computing too')

    # Test parameters
    parser.add_argument('--test_times', type=int, default=3)
    parser.add_argument('--test_case_num', type=int, default=10)
    
    # loggings TODO
    parser.add_argument('--verbose_logging', type=int, default = '0', choices=[0, 1, 2, 3, 4, 5, 6])  
    parser.add_argument('--store_loss', type=int, default=1)

    return parser.parse_args()

def create_log_dirs(args: argparse.Namespace) -> None:
    dirpath = f"./logs/{args.train_method}/{args.experiment_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath+"/evaluation")
        os.makedirs(dirpath+"/checkpoint")
        os.makedirs(dirpath+"/loss")

def save_args(args:argparse.Namespace):
    dirpath = f"./logs/{args.train_method}/{args.experiment_name}"
    param_dict = vars(args)
    with open(dirpath+"/args.txt","w") as file:
        for param,value in param_dict.items():
            file.write(f"{param}:{value}\n")
