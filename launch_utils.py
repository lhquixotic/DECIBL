import argparse
import numpy as np
import torch
import random
from typing import Tuple
from torch.backends import cudnn 
import os


def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DECIBL')

    parser.add_argument('--experiment_name', type=str, default = 'test')

    parser.add_argument('--is_demo',type=int,default=0)
    parser.add_argument('--start_task', type=int, default=0)
    parser.add_argument('--task_seq', type=str, default="1-2-3")
    
    parser.add_argument('--expand_thres',type=float,default=0.3)    
    parser.add_argument('--dynamic_expand',type=int, default=1)
    
    # Learning params
    parser.add_argument('--seed', type=int,  default=1)
    parser.add_argument('--deterministic', type=int,  default=1)

    # Anything under torch.optim works. e.g., 'Adam' and 'SGD'
    parser.add_argument('--optimizer', type=str, default = 'SGD')
    parser.add_argument('--learning_rate', type=float, default = 0.01)

    
    # 0 = No log
    # 1 = Accuracies, #stable/#plastix and model checkpoing after learning all tasks
    # 2 = "1" and Accuracies on all earlier tasks and #stable/#plastic after each task
    # 3 = "2" and model checkpoint and average_activations after each task
    # 4 = "3" and #stable, #plastic, #candidate stable units after each phase
    # 5 = "4" and accuracies on current task for each phase
    # 6 = "5" and model checkpoints and average_activations
    parser.add_argument('--verbose_logging', type=int, default = '0', choices=[0, 1, 2, 3, 4, 5, 6])
    # TODO: add verbose logging mode

    # parameters of stgcnn and dataset

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    #Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=20)
    parser.add_argument('--pred_seq_len', type=int, default=40)
    parser.add_argument('--dataset', default='MA',help='MA, FT, SR, EP, ZS')    
    
    #Training specifc parameters
    parser.add_argument('--max_train_cases_num', type=int, default=7000,help='max train cases number')
    parser.add_argument('--max_test_cases_num', type=int, default=1000,help='max test cases number')
    
    parser.add_argument('--batch_size', type=int, default=5,help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=5,help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,help='gadient clipping')        

    parser.add_argument('--lr_sh_rate', type=int, default=150,help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', action="store_true", default=False,help='Use lr rate scheduler')
    parser.add_argument('--test_times', type=int, default=3)
    parser.add_argument('--test_case_num', type=int, default=10)

    return parser.parse_args()

def create_log_dirs(args: argparse.Namespace) -> None:
    dirpath = "./logs/"+args.experiment_name
    if not os.path.exists(dirpath):
        os.makedirs(dirpath+"/evaluation")
        os.makedirs(dirpath+"/checkpoint")
        os.makedirs(dirpath+"/loss")

