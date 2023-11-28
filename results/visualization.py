import os
import numpy as np
import matplotlib.pyplot as plt

def get_dir_name(train_method, experiment_name=None):
    if train_method == "single" or train_method == "GSM" or train_method == "vanilla":
        return "./logs/{}/evaluation".format(train_method)
    if train_method == "DECIBL":
        return "./logs/{}/evaluation".format(experiment_name)

# train_method = "single"
# data_dir = get_dir_name(train_method)
# test_dataset_id = "1"

# model_results = os.listdir(data_dir)
# model_num = len(model_results)
# print(model_num," results in toatal: ",model_results)
def rgb_to_hex(rgb):
    return '#'+"".join(f'{val:02X}'for val in rgb)

# def get_eval_result(train_method,start_task=0,end_task=5,exp_name=None):
#     data_dir = get_dir_name(train_method,exp_name)
#     col_names = os.listdir(data_dir)
#     if train_method == "DECIBL":
#         col_names.remove('expert_performance.txt')

def get_eval_data(train_method,test_dataset_id=0,start_model_id=0,end_model_id=5,experiment_name=None):
    
    data_dir = get_dir_name(train_method,experiment_name)
    print("Data dir:",data_dir)
    model_results = os.listdir(data_dir)
    print("Results:",model_results)
    
    if train_method == "DECIBL":
        model_results.remove('expert_performance.txt')
    max_model_num = len(model_results)
    assert end_model_id - start_model_id <= max_model_num
    assert end_model_id <= max_model_num
    print("Max model number: ", max_model_num," results in toatal: ",model_results,
          ", start with ", start_model_id, ", end with ", end_model_id)
    display_model_n = end_model_id - start_model_id

    is_1st_result = True
    for model_result in model_results:
        if train_method == "single":
            model_idx = int(model_result[0]) - 1
        elif train_method == "DECIBL":
            model_idx = int(model_result[-1]) - 1
        else:
            model_idx = model_result.count("-")
        if model_idx >= end_model_id:
            continue
        
        result_dir = data_dir + "/" + model_result
        np_files = os.listdir(result_dir)
        for np_file in np_files:
            np_dir = result_dir + "/" + np_file
            
            # select the results we will use
            flag = str(model_idx+1) if test_dataset_id==0 else str(test_dataset_id) 
            
            # DECIBL 
            
            if train_method == "DECIBL":
                f = np_file.split(".")[0]
                filename_split = f.split("-")
                if str(filename_split[-1])==str(int(flag)-1) and str(filename_split[2])==str(int(flag)-1):
                    if "ADE" in np_file:
                        print(np_file)
                        ade_single = np.load(np_dir)
                        print("method:{}, flag:{}, np_file:{}".format(train_method,flag,np_file))
                    else:
                        fde_single = np.load(np_dir)
                
            else:
                if  flag in np_file:
                    if "ADE" in np_file:
                        ade_single = np.load(np_dir)
                        print("method:{}, flag:{}, np_file:{}".format(train_method,flag,np_file))
                    elif "FDE" in np_file:
                        fde_single = np.load(np_dir)
                        
        if is_1st_result:
            ade_data = np.zeros_like(ade_single)[np.newaxis,:].repeat(display_model_n,0)
            fde_data = np.zeros_like(fde_single)[np.newaxis,:].repeat(display_model_n,0)
            is_1st_result = False
        
        ade_data[model_idx] = ade_single
        fde_data[model_idx] = fde_single
    
    return ade_data, fde_data

color_dict = {"single": rgb_to_hex([55,103,149]),
              "vanilla":rgb_to_hex([114,118,213]),
              "GSM":rgb_to_hex([170,220,224]),
              "DECIBL":rgb_to_hex([231,98,84])}

marker_dict = {"single":"o",
               "vanilla":"o",
               "GSM":"o",
               "DECIBL":"^"}

def expert_selector(task_id):
    es = {0: [0, 5], 1: [1, 4, 6], 2: [2, 7], 3: [3]}
    for key, val in es.items():
        if task_id in val:
            return key
    return None

def get_DECIBL_res(start_tid, end_tid,experiment_name):
    
    for tid in range(start_tid,end_tid):
        eid = expert_selector(tid)
        fname = f"./logs/{experiment_name}/evaluation/columns_{min(tid,4)}/"
        ade_fn = fname+"ADE-task-{}-exp-{}.npy".format(tid,eid)
        fde_fn = fname+"FDE-task-{}-exp-{}.npy".format(tid,eid)
        
        ade_single = np.load(ade_fn)
        fde_single = np.load(fde_fn)
        if tid == start_tid:
            ade_data = np.zeros_like(ade_single)[np.newaxis,:].repeat(end_tid-start_tid,0)
            fde_data = np.zeros_like(fde_single)[np.newaxis,:].repeat(end_tid-start_tid,0)
        ade_data[tid] = ade_single
        fde_data[tid] = fde_single
        
    return ade_data, fde_data

def visualize_result(data_dict:dict, axes):
    n = len(data_dict) # number of comparison models
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
    axes.set_xlabel("Trained dataset index")     
    axes.legend()   
        
fig, axs = plt.subplots(figsize=(8,6))
display_methods = ["GSM","vanilla"]
res_dict = dict()

for m in display_methods:
    ade_data, fde_data = get_eval_data(m, 1, start_model_id=0, end_model_id=3)
    res_dict[m] = ade_data
    
ade_data, fde_data = get_eval_data("DECIBL",1,start_model_id=0,end_model_id=3,experiment_name="experiment-3")
res_dict["DECIBL"] = ade_data

# print(len(data_dict["vanilla"]))
visualize_result(res_dict, axs)
xticks = [1,2,3]
axs.set_xticks(xticks)
xtickslabels = ["1-MA", "2-FT", "3-ZS"]
axs.set_xticklabels(xtickslabels)
axs.set_ylim(0,4.5)
axs.set_ylabel("ADE(m)")
plt.savefig("results/ADE-seq-tasks.png",dpi=300)

#*******************************************

fig, axs = plt.subplots(figsize=(8,6))
display_methods = ["GSM","vanilla","single"]
res_dict = dict()

for m in display_methods:
    ade_data, fde_data = get_eval_data(m, 0, start_model_id=0, end_model_id=3)
    res_dict[m] = ade_data
    
ade_data, fde_data = get_eval_data("DECIBL",0,start_model_id=0,end_model_id=3,experiment_name="experiment-3")
res_dict["DECIBL"] = ade_data

# print(len(data_dict["vanilla"]))
visualize_result(res_dict, axs)
xticks = [1,2,3]
axs.set_xticks(xticks)
xtickslabels = ["1-MA", "2-FT", "3-ZS"]
axs.set_xticklabels(xtickslabels)
axs.set_ylim(0,3)
axs.set_ylabel("ADE(m)")
plt.savefig("results/ADE-target-tasks.png",dpi=300)

#*******************************************
fig, axs = plt.subplots(figsize=(8,6))
display_methods = ["GSM","vanilla"]
res_dict = dict()

for m in display_methods:
    ade_data, fde_data = get_eval_data(m, 1, start_model_id=0, end_model_id=3)
    res_dict[m] = fde_data
    
ade_data, fde_data = get_eval_data("DECIBL",1,start_model_id=0,end_model_id=3,experiment_name="experiment-3")
res_dict["DECIBL"] = fde_data

# print(len(data_dict["vanilla"]))
visualize_result(res_dict, axs)
xticks = [1,2,3]
axs.set_xticks(xticks)
xtickslabels = ["1-MA", "2-FT", "3-ZS"]
axs.set_xticklabels(xtickslabels)
axs.set_ylim(0,12)
axs.set_ylabel("FDE(m)")
plt.savefig("results/FDE-seq-tasks.png",dpi=300)

#*******************************************

fig, axs = plt.subplots(figsize=(8,6))
display_methods = ["GSM","vanilla","single"]
res_dict = dict()

for m in display_methods:
    ade_data, fde_data = get_eval_data(m, 0, start_model_id=0, end_model_id=3)
    res_dict[m] = fde_data
    
ade_data, fde_data = get_eval_data("DECIBL",0,start_model_id=0,end_model_id=3,experiment_name="experiment-3")
res_dict["DECIBL"] = fde_data

# print(len(data_dict["vanilla"]))
visualize_result(res_dict, axs)
xticks = [1,2,3]
axs.set_xticks(xticks)
xtickslabels = ["1-MA", "2-FT", "3-ZS"]
axs.set_xticklabels(xtickslabels)
axs.set_ylim(0,8)
axs.set_ylabel("FDE(m)")
plt.savefig("results/FDE-target-tasks.png",dpi=300)

#*******************************************
# ade_data, fde_data = get_eval_data("DECIBL",0,start_model_id=0,end_model_id=6, experiment_name="experiment-split-3-no-init-bs64-ne150")
res_dict = dict()
display_methods = ["vanilla","single"]

for m in display_methods:
    ade_data, fde_data = get_eval_data(m, 0, start_model_id=0, end_model_id=5)
    res_dict[m] = ade_data
ade_data, fde_data = get_eval_data("GSM", 0, start_model_id=0, end_model_id=4)
res_dict["GSM"] = ade_data
ade_data,fde_data = get_DECIBL_res(0,7,"experiment-split-3-no-init-bs64-ne150")

fig, axs = plt.subplots(figsize=(8,6))

res_dict["DECIBL"] = ade_data
visualize_result(res_dict, axs)
# plt.show()
xticks = [1,2,3,4,5,6,7]
axs.set_xticks(xticks)
xtickslabels = ["MA-1", "FT-1", "ZS-1","EP-1","SR-1","MA-2", "FT-2"]
axs.set_xticklabels(xtickslabels)
# axs.set_ylim(0,)
axs.set_ylabel("ADE(m)")
plt.savefig("results/ADE-seq-15-tasks.png",dpi=300)
# plt.show()cv


# def get_forgetting_num(result_data, thre=0.5, before_id=0, current_id=1):
#     model_num, test_times, obs_case_num = result_data.shape
#     forgetting_num = 0
#     forgetting_idx = []
#     for i in range(obs_case_num): # for each observation case
#         current_case_ade = np.mean(result_data[current_id,:,i])   # ade data of the current trained model
#         last_case_ade = np.mean(result_data[before_id,:,i])    # ade data of the last_trained model
#         if current_case_ade - last_case_ade > thre:
#             forgetting_num += 1
#             forgetting_idx.append(i)

#     return forgetting_num, forgetting_idx

# f_n_vanilla, _ = get_forgetting_num(res_dict["vanilla"],0.5,before_id=0,current_id=2)
# print("vanilla: forgetting number is ", int(f_n_vanilla))
# f_n_gsm, _ = get_forgetting_num(res_dict["GSM"], 0.5,before_id=0,current_id=2)
# print("gsm: forgetting number is ", int(f_n_gsm))




