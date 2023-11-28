import pickle
import copy
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributions.multivariate_normal as torchdist

from utils.utils import get_device,get_dataloader
from models.task_detector.graph_autoencoder import GraphAutoEncoder
from utils.train_eval import task_test_with_given_expert
from utils.metrics import ade,fde,seq_to_nodes,nodes_rel_to_nodes_abs,closer_to_zero
from models.social_stgcnn import social_stgcnn
from models.social_stgcnn_dem import social_stgcnn_dem
from scenarios.benchmark import get_continual_scenario_benchmark
from launch_utils import set_seeds

# Arguments
parser = argparse.ArgumentParser(description="test autoencoder")

parser.add_argument('--experiment_name', type=str, default='test_GAE')
parser.add_argument('--is_demo', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--task_detect', type=int, default=0)
parser.add_argument('--task_predict', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--vis_detect_res', type=int, default=0)

# scenario parameters
parser.add_argument('--task_seq', type=str, default="1-2-3-4")
parser.add_argument('--datasets_split_num', type=int, default=1)
parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)
parser.add_argument('--train_start_task', type=int, default=0)
parser.add_argument('--test_start_task', type=int, default=0)

# Model parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# test parameters
parser.add_argument('--test_times', type=int, default=1)
parser.add_argument('--test_case_num', type=int, default=1000)
parser.add_argument('--task_free', type=int, default=0)

args = parser.parse_args()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if args.deterministic: set_seeds(args.seed)

# Data
scenarios = get_continual_scenario_benchmark(args)
task_num = len(args.task_seq.split('-')) * args.datasets_split_num

# Model
task_gaes = []
for tid in range(task_num):
    task_gaes.append(GraphAutoEncoder(seq_len=args.obs_seq_len))
    task_gaes[tid].load_state_dict(torch.load("./logs/FAE/layer-2-latent-32/checkpoint/GAE_task_{}.pth".format(tid)))
    task_gaes[tid].to(get_device())
loss_func = nn.MSELoss()

# Task detect
def task_detect_test(model, test_loader):
    all_loss = []
    model.eval()
    for step, case in enumerate(test_loader):
        case = [tensor.to(get_device()) for tensor in case]
        _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
        V_obs_tmp = V_obs.permute(0,3,1,2)
        ev,dv = model(V_obs_tmp,A_obs.squeeze())
        loss = loss_func(dv,V_obs_tmp) 
        all_loss.append(loss.item())
    all_loss = np.array(all_loss)
    return all_loss

task_detect_res = []

if args.task_detect:
    res = np.zeros((task_num,task_num))
    for tid, test_task in enumerate(scenarios.test_stream):
        tid += args.test_start_task
        test_loader = get_dataloader(test_task,False)
        # get loss for each gae
        for gae_id, gae in enumerate(task_gaes):
            task_loss = task_detect_test(gae, test_loader)
            if gae_id == 0:
                all_gae_loss = np.zeros_like(task_loss)[np.newaxis,:].repeat(task_num,0)
            all_gae_loss[gae_id] = task_loss
        # get gae indices with minimum loss
        max_indices = np.argmin(all_gae_loss,axis=0)
        for i in range(task_num):
            res[tid,i] = np.count_nonzero(max_indices==i)/len(max_indices)
            # x label: task true value, y label: task_detected value, z label: Percentage
        task_detect_res.append(max_indices)
        # display the detect result
        print("[Task {}] Total: {}; Detected as [0: {}, 1: {}, 2: {}, 3: {}, 4: {}]".format(tid,
            len(max_indices),np.count_nonzero(max_indices==0),np.count_nonzero(max_indices==1),
            np.count_nonzero(max_indices==2),np.count_nonzero(max_indices==3),np.count_nonzero(max_indices==4)))        
    # visualization
    if args.vis_detect_res:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        x,y = np.meshgrid(np.arange(1,task_num+1), np.arange(1,task_num+1))
        x = x.flatten()
        y = y.flatten()
        z = res.flatten()
        
        colors = ['lightblue' if i==j else 'lightcoral' for i,j in zip(x,y)]
        ax.bar3d(x,y,np.zeros(len(z)), dx=0.8, dy=0.8, dz=z, color=colors)
        
        ax.set_xlabel("Task label detect value")
        ax.set_ylabel("Task label ground truth")
        ax.set_zlabel("Percentage")
        
        plt.show()

# test predict models
def load_previous_knowledge(model, task_id):
    root_path = "./logs/DEM/experiment-1"
    ckpt_fname = root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id-1)
    print(f"Load model in {ckpt_fname}")
    assert os.path.exists(ckpt_fname)
    checkpoint = torch.load(ckpt_fname)
    
    # load expert selector
    assert "expert_selector" in checkpoint.keys()
    model.expert_selector = checkpoint['expert_selector']
    prev_expert_num = len(checkpoint['expert_selector'])
    
    # load model state dict only when expert number equals to column num        
    if prev_expert_num == len(model.columns):
        model.load_state_dict(checkpoint['model_state_dict'])
        assert "batch_norm_para" in checkpoint.keys()
        model.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
        model.freeze_columns()
        model.load_batch_norm_para(task_id-1)
    return model

def load_best_model(model, task_id):
    root_path = "./logs/DEM/experiment-1"
    ckpt_fname = root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id)
    assert os.path.exists(ckpt_fname), "No checkpoint named {}".format(ckpt_fname)
    checkpoint = torch.load(ckpt_fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    # load batch_norm parameters
    assert "batch_norm_para" in checkpoint.keys()
    model.batch_norm_para = checkpoint["batch_norm_para"]
    # load expert_select dict
    assert "expert_selector" in checkpoint.keys()
    model.expert_selector = checkpoint["expert_selector"]
    return model

if args.task_predict:
    model = social_stgcnn_dem(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
    model.set_task_detector(use_task_detector=True)
    # model = load_previous_knowledge(model,task_num-1)
    model = load_previous_knowledge(model,9)
    
    expand_times = len(model.expert_selector) - len(model.columns) + 1
    for t in range(expand_times):
        # if t == expand_times - 1:
        #     model = load_previous_knowledge(model, task_num-1)
        model.add_column()
    model = load_best_model(model, 9)
    
    print("columns num: {}, task num: {}, GAE num: {}, expert_selector: {}".format(
        len(model.columns), task_num, len(model.task_gaes), model.expert_selector))
    
    model.set_task_detector(use_task_detector=True)
    for tid in range(task_num):
        model.task_gaes[tid].load_state_dict(torch.load("./logs/FAE/layer-2-latent-32/checkpoint/GAE_task_{}.pth".format(tid)))
    model = model.to(get_device())
    
    if not args.task_free:
        model.set_task_detector(use_task_detector=False)
    for id, task in enumerate(scenarios.test_stream):
        id += args.test_start_task
        if id >= task_num:
            break
        eid = model.select_expert(id)
        ade, fde = task_test_with_given_expert(model,id,expert_id=eid,test_task=task,
                                               args=args, save_dir="./logs/FAE/layer-2-latent-32//evaluation")
        print("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(model.columns), id, eid, ade.mean(), fde.mean()))