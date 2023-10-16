from argparse import Namespace
import torch
import copy
from typing import Tuple
from torch import nn
import os
import datetime
import numpy as np

from torch.utils.data import DataLoader
from DECIBL.train_eval import task_train, task_val, get_device, task_test


def get_data_loaders(args: Namespace, train_task, val_task, test_task) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_task.dataset, batch_size = 1,  shuffle=True,num_workers=16,drop_last = True)
    val_loader =  DataLoader(val_task.dataset, batch_size = 1,  shuffle=True,num_workers=16,drop_last = True)
    test_loader = DataLoader(test_task.dataset, batch_size = 1,  shuffle=False,num_workers=16,drop_last = True)
    return (train_loader, val_loader, test_loader)


class Learner():

    def __init__(self, args: Namespace, network, scenario) -> None:
        self.args = args
        self.optim_obj = getattr(torch.optim, args.optimizer)
        self.network = network.to(get_device())

        self.original_scenario = scenario

        self.metrics = {'train_loss':[], 'val_loss':[]}
        self.constant_metrics = {'min_val_epoch':[], 'min_val_loss':[]}
        self.ckpt_path = None
        
        self.best_model = None
        

    def learn_tasks(self):
        for task_index, (train_task, val_task, test_task) in enumerate(zip(self.original_scenario.train_stream,
                                                                           self.original_scenario.val_stream,  # type: ignore
                                                                           self.original_scenario.test_stream)):
            task_index = task_index + self.args.start_task
            
            print("*"*30+" Preparing task"+str(task_index)+" "+"*"*30)
            
            self.ckpt_path = "./logs/{}/checkpoint/val_best_model_{}.pth".format(self.args.experiment_name,task_index)
            
            # Before training, load the network as current knowledge
            self.load_existing_knowledge(task_index)
            
            # Create dir for saving results
            save_dir = "./logs/{}/evaluation/columns_{}/".format(self.args.experiment_name,len(self.network.columns))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Training, and save the model with best validating loss
            self.network = self.network.to(get_device())
            network = self.learn_next_task(task_index, train_task, val_task, test_task)

            print("*"*30+"  Testing task "+str(task_index)+" "+"*"*30)
            
            # After learning, test the model on current and before task
            
            # load best model at first   
            checkpoint = torch.load(self.ckpt_path)
            self.network.load_state_dict(checkpoint["model_state_dict"])  # load best model for current task
            self.network.batch_norm_para = checkpoint["batch_norm_para"]
            
            # test the the performance of previous experts on current task
            expert_res = dict()
            for expert_id in range(len(self.network.columns)): # test all experts on current task
                ade, fde = self.test_given_task(task_index, expert_id, save_dir)
                expert_res[(expert_id, task_index)] = [ade.mean(), fde.mean()]
            # Determine whether to expand an expert
            selected_expert_id = len(self.network.columns) - 1
            if self.args.dynamic_expand: # dynamic expansion
                best_eval = expert_res[(selected_expert_id, task_index)]
                for key, eval in expert_res.items():
                    if key[1] == task_index:
                        if eval[0] - best_eval[0] < self.args.expand_thres: # ade
                            if not selected_expert_id == key[0]:
                                print("No expansion since expert_{} performs well.".format(key[0]))
                                selected_expert_id = key[0]
                            break
            
            # Update expert selector
            print("Update expert selector: expert_{} -> task_{}".format(selected_expert_id, task_index))
            self.network.expert_selector[selected_expert_id].append(task_index)
            
            # if previous expert is selected
            if not selected_expert_id == len(self.network.columns) - 1:
                self.network.columns = self.network.columns[:-1]
                print("Current column number is {}".format(len(self.network.columns)))
            
            # Save ckpt
            self.store_checkpoint(self.network, self.network.batch_norm_para, self.network.expert_selector)
            
            # test the performance of current model on previous tasks
            for id, task in enumerate(self.original_scenario.test_stream): # test all tasks using task-specific expert
                task_id = id + self.args.start_task
                if task_id > task_index:
                    break
                expert_id = self.network.select_expert(task_id)
                if not (expert_id,task_id) in expert_res.keys():
                    ade, fde = self.test_given_task(task_id, expert_id=expert_id, save_dir=save_dir)
                    expert_res[(task_id, expert_id)] = [ade.mean(), fde.mean()]
                # else:
                    # print("(expert_{},task_{}) has been tested". format(expert_id,task_id))
                    
            # save the performance
            with open("./logs/{}/evaluation/expert_performance.txt".format(self.args.experiment_name),"w") as file:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Time:{current_time}\n")
                for key, value in expert_res.items():
                    file.write(f"{key}:{value}\n")
                file.write("*"*50+"\n")
                
            # print("*"*78)
        return network
    
    
    def load_existing_knowledge(self, task_index):
        if len(self.network.columns) == 0:
            self.network.new_task()
        else:
            # Load checkpoint to load previous knowledge, including model paras, expert selector, and bn paras
            ckpt_path = "./logs/{}/checkpoint/val_best_model_{}.pth".format(self.args.experiment_name,task_index-1)
            if os.path.exists(ckpt_path):
                print("Load existing model knowledge: {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            expert_num = len(checkpoint['expert_selector'])
            
            # trained_tasks = [seen_task for key, value in checkpoint["expert_selector"].items() for seen_task in value]     
            expand_times = expert_num - len(self.network.columns) + 1 
            print("For task_{} the model need to expand {} times.".format(task_index, expand_times))
            
            # expand if load pretrained model
            for t in range(expand_times):
                if t == expand_times - 1: # before last time for expansion, load the existing model
                        self.network.load_state_dict(checkpoint['model_state_dict'])
                        self.network.expert_selector = copy.deepcopy(checkpoint["expert_selector"])
                        self.network.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
                        self.network.freeze_columns() # freeze the existing knowledge
                        self.network.load_batch_norm_para(task_index-1)
                        
                self.network.new_task() # expand a new column

        print("Current column number is {}, expert-task dict is {}, bn para number is {}".format(
            len(self.network.columns), self.network.expert_selector,len(self.network.batch_norm_para))) 
    
    def test_given_task(self,test_task_id,expert_id, save_dir=None):
        assert expert_id < len(self.network.columns) # expert id should < column number
        self.network.load_batch_norm_para(test_task_id)

        for id,task in enumerate(self.original_scenario.test_stream):
            if id + self.args.start_task == test_task_id:
                test_loader = DataLoader(task.dataset, batch_size=1, shuffle=False,num_workers=16,drop_last = True)
                ade, fde = task_test(self.network, expert_id, test_loader, self.args)
                
                # save if save_dir is provided
                if save_dir is not None:        
                    np.save(save_dir+"ADE-task-{}-exp-{}.npy".format(test_task_id,expert_id),ade)
                    np.save(save_dir+"FDE-task-{}-exp-{}.npy".format(test_task_id,expert_id),fde)
                    
                    print("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                        len(self.network.columns), test_task_id, expert_id, ade.mean(), fde.mean()))
                return ade, fde
        return None,None
    
    def learn_next_task(self, task_index:int, train_task, val_task, test_task):
        print("*"*30+" Learning task "+str(task_index)+" "+"*"*30)
        train_loader, val_loader, test_loader = get_data_loaders(self.args,train_task,val_task,test_task)
        
        # Initialize the metrics container
        self.metrics['train_loss'] = []
        self.metrics['val_loss'] = []
        self.constant_metrics['min_val_epoch'] = -1
        self.constant_metrics['min_val_loss'] = 9999
        
        # Training process
        optimizer = self.optim_obj(self.network.parameters(), lr= self.args.learning_rate, weight_decay= 0)
        for ep_id in range(self.args.num_epochs):
            val_loss = task_val(self.network, val_loader, self.args)
            self.metrics['val_loss'].append(val_loss)
            self.network, train_loss = task_train(self.network, optimizer, train_loader, self.args)
            self.metrics['train_loss'].append(train_loss)
            print("[Loss] Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}".format(ep_id,train_loss,val_loss))
            if val_loss < self.constant_metrics["min_val_loss"]:
                self.constant_metrics['min_val_loss'] = val_loss
                self.constant_metrics['min_val_epoch'] = ep_id
                self.network.save_batch_norm_para(task_index)
                self.best_model = copy.deepcopy(self.network)
                self.store_checkpoint(self.best_model, self.best_model.batch_norm_para)
                print("** Save best model **")
            
        self.network = copy.deepcopy(self.best_model)
        
        # Save the training loss
        np_val_loss = np.array(self.metrics['val_loss'])
        np_tra_loss = np.array(self.metrics['train_loss'])
        np.save("./logs/{}/loss/val_task_{}.npy".format(self.args.experiment_name,task_index),np_val_loss)
        np.save("./logs/{}/loss/train_task_{}.npy".format(self.args.experiment_name,task_index),np_tra_loss)
        with open("./logs/{}/loss/constant_metrics.txt".format(self.args.experiment_name),"a") as file:
            file.write(f"[Task-{task_index}] Best model trained in epoch {self.constant_metrics['min_val_epoch']}, loss is {self.constant_metrics['min_val_loss']}\n")    
    
        return self.network
    
    def store_checkpoint(self, best_model, bn_para=None, expect_selector=None):
        f = dict()
        f['model_state_dict'] = best_model.state_dict()
        if bn_para is not None:
            f['batch_norm_para']  = bn_para
        if expect_selector is not None:
            f['expert_selector'] = expect_selector
        torch.save(f, self.ckpt_path)