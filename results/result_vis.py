import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch

# args
parser = argparse.ArgumentParser(description="result vis args")
parser.add_argument('--task_id', type=int, default=0)

args = parser.parse_args()

# colors setup
def rgb_to_hex(rgb):
    return '#'+"".join(f'{val:02X}'for val in rgb)

# RED:    231,098,084
# ORANGE: 239,138,071
# YELLOW: 255,208,111
# BLUE_l: 170,222,224
# BLUE_m: 082,143,173
# BLUE_d: 030,070,110

color_dict = {
              "Multiple": "#D3E2B7",
              # "Multiple": rgb_to_hex([55,103,149]),
              "Vanilla":rgb_to_hex([114,118,213]),
              "GSM":rgb_to_hex([170,220,224]),
              "PNN":rgb_to_hex([239,138,71]),
              "Joint":rgb_to_hex([231,98,84]),
              "DEM":rgb_to_hex([30,70,110]),
              "EWC":rgb_to_hex([82,143,173])}

marker_dict = {"Multiple": '.',
              "Vanilla":'.',
              "GSM":'.',
              "PNN":'.',
              "Joint":'.',
              "DEM":'^',
              "EWC":''}

# expert selector
def get_expert_dict(task_id, method_name, experiment_name):
    if method_name == "PNN":
        expert_dict = dict()
        for tid in range(task_id+1):
            expert_dict[tid] = [tid]
        return expert_dict
    # load and assert the expert_dict
    # if experiment_name == "experiment-2":
    #     experiment_name = "experiment-1"
    ckpt_fname = f"./logs/{method_name}/{experiment_name}/checkpoint/checkpoint_task_{task_id}.pth"
    assert os.path.exists(ckpt_fname), 'No checkpoint file: {}.'.format(ckpt_fname)
    ckpt = torch.load(ckpt_fname)
    assert "expert_selector" in ckpt.keys(), 'No expert_selector in the checkpoint file.'    
    expert_dict = ckpt["expert_selector"]
    return expert_dict

def select_expert(task_id, expert_dict):
    # select the expert according to the dict
    for key, val in expert_dict.items():
        if task_id in val:
            return key
    return None

# load result info
def load_result_data(start_tid, end_tid, method_name, experiment_name,
                     only_last_eval=False, given_tid=None):
    '''
    Evaluation of current performance (tested with the dataset of the scenario just trained) in the 
    continual learning settings.
    '''
    # method setup
    use_expert_selector = False
    if method in ["PNN","DEM"]:
        use_expert_selector = True
    
    # Loop for each task
    for tid in range(start_tid, end_tid+1):# load task_{start_tid} to task_{end_tid}
        # determine the expert id 
        if not use_expert_selector:
            eid = 0
        else:
            expert_dict = get_expert_dict(tid, method_name, experiment_name)
            eid = select_expert(tid, expert_dict)
        assert eid is not None, "No expert is indicated."

        # load evaluation result for given eid and tid
        if only_last_eval: # only load the result from the last task
            pass
        else:
            dir_name = f"./logs/{method_name}/{experiment_name}/evaluation/task_{tid}"
            if method == "Joint":
                dir_name = f"./logs/{method_name}/{experiment_name}/evaluation/task_0"
            
            # if indicating a task
            gtid = tid
            if given_tid is not None:
                gtid = given_tid
                eid = 0 if not use_expert_selector else select_expert(gtid,expert_dict)
                
            ade_fname = dir_name + "/ADE-task-{}-exp-{}.npy".format(gtid,eid)
            fde_fname = dir_name + "/FDE-task-{}-exp-{}.npy".format(gtid,eid)
            print(ade_fname)
            ade_single = np.load(ade_fname)
            fde_single = np.load(fde_fname)
            if tid==start_tid:
                ade_data = np.zeros_like(ade_single)[np.newaxis,:].repeat(end_tid-start_tid+1,0)
                fde_data = np.zeros_like(fde_single)[np.newaxis,:].repeat(end_tid-start_tid+1,0)
            ade_data[tid] = ade_single
            fde_data[tid] = fde_single
    return ade_data,fde_data   


def visualize_result(data_dict:dict, axes, save_fig=False):
    num = len(data_dict.keys()) # number of comparison models
    for key, res in data_dict.items():
        model_num, test_times, obs_case_num = res.shape
        res_mean = np.mean(res, axis=1)
        res_var  = np.var(res, axis=1)
        dataset_mean = np.mean(res_mean, axis=1)
        dataset_var  = np.mean(res_var, axis=1)
        
        # plot the result
        x = np.arange(1,1+model_num)
        axes.plot(x, dataset_mean, label=key, color=color_dict[key],linestyle="-",marker=marker_dict[key])
        axes.fill_between(x,dataset_mean-dataset_var,dataset_mean+dataset_var,
                          alpha=0.3,color=color_dict[key])
        print(key,dataset_mean)
    

methods_zoo = {"Vanilla": "experiment-3",
              "Multiple": "experiment",
              "Joint":    "experiment",
              "PNN":      "experiment",
              "DEM":      "experiment-3",
            #   "GSM":      "experiment-1",
              "EWC":      "experiment-2"}

scenario_list = ["MA","FT","ZS", "EP", "SR"]

plot_str_tid = 0
plot_end_tid = 9


# 1. Plot ADE (Evaluation on current dataset) 
res_ade_dict = dict()
res_fde_dict = dict()
for method, experiment_name in methods_zoo.items():
    ade_data, fde_data = load_result_data(plot_str_tid,plot_end_tid,method,experiment_name)  
    res_ade_dict[method] = ade_data
    res_fde_dict[method] = fde_data

plt.rcParams["axes.edgecolor"] = 'gray'
fig, axs = plt.subplots(figsize=(12,5))
axs.axvspan(0.5,5.5,alpha=0.5,color='lightblue')
axs.axvspan(5.5,10.5,alpha=0.5,color='lightyellow')
axs.axvline(5.5,color="white",linewidth=1)

visualize_result(res_ade_dict, axs) 

axs.set_xlabel("Trained dataset index")
axs.set_ylabel("Test Error (ADE)")
axs.set_xlim(0.5,10.5)
axs.set_xticks(range(1,11))
axs.set_xticklabels([s+ "-0" for s in scenario_list] + [s+"-1" for s in scenario_list])
axs.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5)   
axs.grid(color='gray',linewidth=0.3)
plt.tight_layout()
plt.savefig("./results/figs/current-ade-10-tasks.png",dpi=600)
plt.show()

# 2. Plot FDE (evaluation on current dataset)
fig, axs = plt.subplots(figsize=(12,5))
axs.axvspan(0.5,5.5,alpha=0.5,color='lightblue')
axs.axvspan(5.5,10.5,alpha=0.5,color='lightyellow')
axs.axvline(5.5,color="white",linewidth=1)

visualize_result(res_fde_dict, axs) 

axs.set_xlabel("Trained dataset index")
axs.set_ylabel("Test Error (FDE)")
axs.set_xlim(0.5,10.5)
axs.set_xticks(range(1,11))
axs.set_xticklabels([s+ "-0" for s in scenario_list] + [s+"-1" for s in scenario_list])
axs.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5)   
axs.grid(color='gray',linewidth=0.3)
plt.tight_layout()
plt.savefig("./results/figs/current-fde-10-tasks.png",dpi=600)
plt.show()

# 3. Plot ADE (evaluation on MA-0)
methods_zoo = {"Vanilla": "experiment-3",
              "Joint":    "experiment",
              "PNN":      "experiment",
              "DEM":      "experiment-3",
            #   "GSM":      "experiment-1",
              "EWC":      "experiment-2"}

res_ade_dict = dict()
res_fde_dict = dict()
for method, experiment_name in methods_zoo.items():
    ade_data, fde_data = load_result_data(plot_str_tid,plot_end_tid,method,experiment_name,given_tid=0)  
    res_ade_dict[method] = ade_data
    res_fde_dict[method] = fde_data

plt.rcParams["axes.edgecolor"] = 'gray'
fig, axs = plt.subplots(figsize=(12,5))
axs.axvspan(0.5,5.5,alpha=0.5,color='lightblue')
axs.axvspan(5.5,10.5,alpha=0.5,color='lightyellow')
axs.axvline(5.5,color="white",linewidth=1)

visualize_result(res_ade_dict, axs) 

axs.set_xlabel("Trained dataset index")
axs.set_ylabel("Test Error (ADE)")
axs.set_xlim(0.5,10.5)
axs.set_xticks(range(1,11))
axs.set_xticklabels([s+ "-0" for s in scenario_list] + [s+"-1" for s in scenario_list])
axs.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5)   
axs.grid(color='gray',linewidth=0.3)
plt.tight_layout()
plt.savefig("./results/figs/task1-ade-10-tasks.png",dpi=600)
plt.show()

# 4. Plot FDE (evaluation on MA-0)
fig, axs = plt.subplots(figsize=(12,5))
axs.axvspan(0.5,5.5,alpha=0.5,color='lightblue')
axs.axvspan(5.5,10.5,alpha=0.5,color='lightyellow')
axs.axvline(5.5,color="white",linewidth=1)

visualize_result(res_fde_dict, axs) 

axs.set_xlabel("Trained dataset index")
axs.set_ylabel("Test Error (FDE)")
axs.set_xlim(0.5,10.5)
axs.set_xticks(range(1,11))
axs.set_xticklabels([s+ "-0" for s in scenario_list] + [s+"-1" for s in scenario_list])
axs.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5)   
axs.grid(color='gray',linewidth=0.3)
plt.tight_layout()
plt.savefig("./results/figs/task1-fde-10-tasks.png",dpi=600)
plt.show()

# # 5. Compare the usage of storage
# methods_zoo = {"Vanilla": "experiment",
#               "Multiple": "experiment",
#               "PNN":      "experiment",
#               "DEM":      "experiment-3",
#               "GSM":      "experiment-1"}

# def get_file_size(filepath):
#     return os.path.getsize(filepath)

# def get_storage_results(start_tid, end_tid, method_name, experiment_name): 
#     storage_results = [] # list(storage_size): storage used after each task
#     for tid in range(start_tid,end_tid):
#         ckpt_file_name = f"./logs/{method_name}/{experiment_name}/checkpoint/checkpoint_task_{tid}.pth"
#         file_size = get_file_size(ckpt_file_name)
#         if method_name in ["GSM"]:
#             mem_file_name =  f"./logs/{method_name}/{experiment_name}/memory/mem_3500/task_{tid}.pkl"
#             mem_size = get_file_size(mem_file_name)
#             file_size += mem_size
#         storage_results.append(file_size/1000)
#     return storage_results

# storage_res = []
# storage_final = {}
# for method, exp_name in methods_zoo.items():
#     storage_used = get_storage_results(plot_str_tid,plot_end_tid,method,exp_name)
#     storage_res.append(storage_used)
#     storage_final[method] = storage_used[-1]
# #     plt.plot(range(plot_end_tid-plot_str_tid),storage_used)
# # plt.show()

# # print final storage sizes
# print(storage_final)