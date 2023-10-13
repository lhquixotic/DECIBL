import os
import numpy as np
import matplotlib.pyplot as plt

train_method = "psstgcnn"
model_name = "social-stgcnn"
data_dir = "./evaluations/" + train_method 
test_dataset_id = "1"

model_results = os.listdir(data_dir)
model_num = len(model_results)
print(model_num," results in toatal: ",model_results)

def get_eval_data(train_method,test_dataset_id,start_model_id=0,end_model_id=5,
                  model_name="social-stgcnn"):
    
    data_dir = "./evaluations/" + train_method
    model_results = os.listdir(data_dir)
    max_model_num = len(model_results)
    assert end_model_id - start_model_id <= max_model_num
    assert end_model_id <= max_model_num
    print("max model number: ", max_model_num," results in toatal: ",model_results,
          ", start with ", start_model_id, ", end with ", end_model_id)
    model_results = model_results[start_model_id:end_model_id]

    is_1st_result = True
    for model_result in model_results:
        # model_idx = model_result.count("-") - 1
        model_idx = int(model_result[0]) - 1
        print("model_idx:",model_idx)
        result_dir = data_dir + "/" + model_result
        np_files = os.listdir(result_dir)
        for np_file in np_files:
            np_dir = result_dir + "/" + np_file
            print("load {}".format(np_dir))
            if test_dataset_id in np_file:
                if "ADE" in np_file:
                    ade_single = np.load(np_dir)
                    print("ade single:",ade_single[:,0])
                else:
                    fde_single = np.load(np_dir)
        if is_1st_result:
            ade_data = np.zeros_like(ade_single)[np.newaxis,:]
            ade_data = ade_data.repeat(len(model_results),0)
            fde_data = np.zeros_like(fde_single)[np.newaxis,:].repeat(len(model_results),0)
            is_1st_result = False
            
        ade_data[model_idx] = ade_single
        fde_data[model_idx] = fde_single
    return ade_data, fde_data

# gsm_ade_data, gsm_fde_data = get_eval_data("GSM","1",start_model_id=0,end_model_id=4)
ade_data, fde_data = get_eval_data("psstgcnn","1",start_model_id=0,end_model_id=2)

print("ade_data",ade_data.shape)

# ade = np.load("./evaluation/vanilla/social-stgcnn/1-MA/ADE-1-MA.npy")
# ade_mean = np.mean(ade,axis=1)
# ade_mean = np.mean(ade_mean,axis=1)
# print(ade_mean)

model_num, test_times, obs_case_num = ade_data.shape
# gsm_model_num, gsm_test_times, gsm_obs_case_num = gsm_ade_data.shape

# Mean and var
ade_data_mean = np.mean(ade_data,axis=1)
# print("ade data mean:",ade_data_mean[:,0])
ade_data_var = np.var(ade_data,axis=1)
ade_dataset_mean = np.mean(ade_data_mean,axis=1)
ade_dataset_var_mean = np.mean(ade_data_var,axis=1)
print("ADE: ",ade_dataset_mean)

# gsm_ade_data_mean = np.mean(gsm_ade_data,axis=1)
# gsm_ade_data_var = np.var(gsm_ade_data,axis=1)
# gsm_ade_dataset_mean = np.mean(gsm_ade_data_mean,axis=1)
# gsm_ade_dataset_var_mean = np.mean(gsm_ade_data_var,axis=1)
# print("GSM-ADE: ",gsm_ade_dataset_mean)


fde_data_mean = np.mean(fde_data,axis=1)
print("shape",fde_data_mean.shape)
fde_data_var = np.var(fde_data,axis=1)
fde_dataset_mean = np.mean(fde_data_mean,axis=1)
fde_dataset_var_mean = np.mean(fde_data_var,axis=1)
print("FDE: ",fde_dataset_mean)

# gsm_fde_data_mean = np.mean(gsm_fde_data,axis=1)
# gsm_fde_data_var = np.var(gsm_fde_data,axis=1)
# gsm_fde_dataset_mean = np.mean(gsm_fde_data_mean,axis=1)
# gsm_fde_dataset_var_mean = np.mean(gsm_fde_data_var,axis=1)
# print("GSM-FDE: ",gsm_fde_dataset_mean)

# figure on dataset
fig,axs = plt.subplots(1,2,figsize=(10,4))

x = np.arange(1,1+model_num)

axs[0].plot(x,ade_dataset_mean)
axs[0].fill_between(x,ade_dataset_mean-ade_dataset_var_mean,ade_dataset_mean+ade_dataset_var_mean,alpha=0.3,color='blue')

# x = np.arange(1,1+gsm_model_num)
# axs[0].plot(x,gsm_ade_dataset_mean)
# axs[0].fill_between(x,gsm_ade_dataset_mean-gsm_ade_dataset_var_mean,gsm_ade_dataset_mean+gsm_ade_dataset_var_mean,alpha=0.3,color='blue')
# axs[0].set_title('ADE for {} times test on {}'.format(test_times, "1-MA"))
# axs[0].set_xlabel('Trained dataset index')
# axs[0].set_ylabel('ADE(m)')

x = np.arange(1,1+model_num)
axs[1].plot(x,fde_dataset_mean)
axs[1].fill_between(x,fde_dataset_mean-fde_dataset_var_mean,fde_dataset_mean+fde_dataset_var_mean,alpha=0.3,color='blue')

# x = np.arange(1,1+gsm_model_num)
# axs[1].plot(x,gsm_fde_dataset_mean)
# axs[1].fill_between(x,gsm_fde_dataset_mean-gsm_fde_dataset_var_mean,gsm_fde_dataset_mean+gsm_fde_dataset_var_mean,alpha=0.3,color='blue')

axs[1].set_title('FDE for {} times test on {}'.format(test_times, "1-MA"))
axs[1].set_xlabel('Trained dataset index')
axs[1].set_ylabel('FDE(m)')

plt.show()

def get_forgetting_num(result_data, thre=0.5, before_id=0, current_id=1):
    model_num, test_times, obs_case_num = result_data.shape
    forgetting_num = 0
    forgetting_idx = []
    for i in range(obs_case_num): # for each observation case
        current_case_ade = np.mean(result_data[current_id,:,i])   # ade data of the current trained model
        last_case_ade = np.mean(result_data[before_id,:,i])    # ade data of the last_trained model
        if current_case_ade - last_case_ade > thre:
            forgetting_num += 1
            forgetting_idx.append(i)

    return forgetting_num, forgetting_idx

f_n_vanilla, _ = get_forgetting_num(ade_data,0.5,before_id=0,current_id=2)
print("vanilla: forgetting number is ", int(f_n_vanilla))
# f_n_gsm, _ = get_forgetting_num(gsm_ade_data, 0.5,before_id=0,current_id=2)
# print("gsm: forgetting number is ", int(f_n_gsm))

# forgetting_thre = 0.8
# forgetting_num = np.zeros(model_num)
# forgetting_idx = [[],[],[],[],[]]

# how to define forgeting: given a threshold, if the error increasement over the threhold, we think forgetting occurs.
# for i in range(obs_case_num): # for each observation case
#     for j in range(1,model_num): # for each trained model using sequence data
#         if j > 1: # just evaluate after first train
#             break
#         current_case_ade = np.mean(ade_data[j,:,i])   # ade data of the current trained model
#         last_case_ade = np.mean(ade_data[j-1,:,i])    # ade data of the last_trained model
#         if current_case_ade - last_case_ade > forgetting_thre:
#             forgetting_num[j] += 1
#             forgetting_idx[j].append(i)
#         # else:
#         #     x = np.arange(2)
#         #     plt.plot(x,[last_case_ade,current_case_ade])
# print("forgetting number of mdoel 1->2 is:{:d}".format(int(forgetting_num[1])))

# # x = np.arange(model_num)
# # for i in range(1000):
# #     plt.plot(x,ade_data_mean[:,i])



