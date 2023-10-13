import argparse
import config
import numpy as np
import torch
import random
from typing import Tuple
from torch.backends import cudnn 
# from torchvision import transforms
import os
import pickle
import shutil


def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='psstgcnn')
    # Run tag
    parser.add_argument('--run_tag',type=str, default= 'test-1')
    parser.add_argument('--is_demo',type=bool,default=True)
    parser.add_argument('--start_task', type=int, default=0)
    parser.add_argument('--task_seq', type=str, default="1-2-3")
    
    parser.add_argument('--expand_thres',type=float,default=0.3)    
    parser.add_argument('--dynamic_expand',type=bool, default=True)
    
    # Logging paramss
    parser.add_argument('--experiment_name', type=str, default = 'Test')
    parser.add_argument('--experiment_note', type=str, default = '')

    # Architectural params
    parser.add_argument('--model', type=str, default = 'Mlp_Reg')
    parser.add_argument('--prune_perc', type=float, default=80)

    # Learning params
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--deterministic', type=int,  default=1)

    # Anything under torch.optim works. e.g., 'Adam' and 'SGD'
    parser.add_argument('--optimizer', type=str, default = 'SGD')
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    # parser.add_argument('--batch_size', type=int, default = 64)
    # Anything under torch.nn will work see https://pytorch.org/docs/stable/nn.html#loss-functions
    parser.add_argument('--loss_func', type=str, default = 'MSELoss')

    # Algortihm params
    parser.add_argument('--recovery_perc', type=float, default = 0.1)
    parser.add_argument('--phase_epochs', type=int, default = 10)
    parser.add_argument('--reinit',  type=int, default = 1)
    parser.add_argument('--tau_schedule', type=str, default = 'cosine_anneling', choices=['cosine_anneling', 'exp_decay','linear'])
    parser.add_argument('--tau_param',  type=float, default = 30)
    parser.add_argument('--grow_init', type=str, default = 'zero_init', choices=['zero_init'])
    parser.add_argument('--grow_method', type=str, default = 'random', choices=['random']) 

    # Output head params
    parser.add_argument('--multihead', type=int, default = 1)
    parser.add_argument('--output_mechanism', type=str, default = 'vanilla')

    # Log params
    parser.add_argument('--log_prefix', type=str, default = 'Mlp')
    parser.add_argument('--log_suffix', type=str, default = 'verbose6')
    # 0 = No log
    # 1 = Accuracies, #stable/#plastix and model checkpoing after learning all tasks
    # 2 = "1" and Accuracies on all earlier tasks and #stable/#plastic after each task
    # 3 = "2" and model checkpoint and average_activations after each task
    # 4 = "3" and #stable, #plastic, #candidate stable units after each phase
    # 5 = "4" and accuracies on current task for each phase
    # 6 = "5" and model checkpoints and average_activations
    parser.add_argument('--verbose_logging', type=int, default = '0', choices=[0, 1, 2, 3, 4, 5, 6])

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
    parser.add_argument('--dataset', default='MA',
                        help='MA, FT, SR, EP, ZS')    
    

    #Training specifc parameters
    parser.add_argument('--max_train_cases_num', type=int, default=7000,help='max train cases number')
    parser.add_argument('--max_test_cases_num', type=int, default=1000,help='max test cases number')
    
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='tag',
                        help='personal tag for the model ')
    parser.add_argument('--test_times', type=int, default=3)
    parser.add_argument('--test_case_num', type=int, default=10)

    return parser.parse_args()

def get_log_param_dict(args: argparse.Namespace) -> dict:
    return {
        "LogPath":  config.LOG_PATH,
        "DirName": args.log_prefix + "_" + args.experiment_name + "_" + args.log_suffix,
        "save_activations_task": args.verbose_logging in [3, 4, 5, 6],
        "save_activations_phases": args.verbose_logging in [6],
        "save_model_phase": args.verbose_logging in [6],
        "eval_model_phase": args.verbose_logging in [5, 6],
        "save_model_task": args.verbose_logging in [3, 4, 5, 6],
        "write_phase_log": args.verbose_logging in [4, 5, 6],
        "write_task_log": args.verbose_logging in [2, 3, 4, 5, 6],
        "write_sequence_log": args.verbose_logging in [1, 2, 3, 4, 5, 6],
        "no_log": args.verbose_logging == 0
    }

def create_log_dirs(args: argparse.Namespace) -> None:
    dirpath = "./logs/"+args.run_tag
    if not os.path.exists(dirpath):
        os.makedirs(dirpath+"/evaluation")
        os.makedirs(dirpath+"/checkpoint")
        os.makedirs(dirpath+"/loss")

# def create_log_dirs(args: argparse.Namespace, log_params: dict) -> None:
    
    
#     dirpath = os.path.join(log_params["LogPath"], log_params["DirName"])
#     # Remove existing files/dirs
#     if os.path.exists(dirpath) and os.path.isdir(dirpath):
#         shutil.rmtree(dirpath)
    
#     if log_params["no_log"]:
#         return
#     # Create log dirs and save experiment args
#     os.makedirs(dirpath)
#     with open(os.path.join(dirpath, 'args.pkl'), 'wb') as file:
#         pickle.dump(args, file)

#     if log_params["write_task_log"]:
#         number_of_tasks = len(args.task_seq.split("-"))
#         for task_id in range(1, number_of_tasks +1):
#             os.makedirs(os.path.join(dirpath, "Task_{}".format(task_id)))




