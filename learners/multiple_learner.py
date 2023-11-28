from argparse import Namespace
import os
import logging

import torch
import torch.nn as nn

from learners.learner import Learner
from utils.train_eval import task_test_with_given_expert

class MultipleLearner(Learner):
    def __init__(self, model, scenarios, args: Namespace) -> None:
        super(MultipleLearner,self).__init__(model, scenarios, args)
        
    def before_task_learning(self, tid):
        if tid > 0:
            # reinitialization
            del self.model.columns[0]
            self.model.add_column()
        assert len(self.model.columns) == 1
        
    def after_task_learning(self, tid):
        # load best model
        self.load_best_model()
        
        # test all previous task
        res = dict()
        for id, task in enumerate(self.scenarios.test_stream):
            id += self.args.test_start_task
            if id != tid:
                continue
            if id > tid:
                break
            ade, fde = task_test_with_given_expert(self.model, id, expert_id=0,test_task=task,
                                                   args=self.args, save_dir=self.eval_path)
            logging.info("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(self.model.columns), id, 0, ade.mean(), fde.mean()))
            res[(id,0)] = [ade.mean(), fde.mean()]
            
        # store the performance
        with open(self.root_path+"/evaluation/performance.txt", "w") as file:
            for key, val in res.items():
                file.write(f"{key}: {val}\n")
            file.write("*"*40+"\n")
    
        