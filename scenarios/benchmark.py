from easydict import EasyDict as edict
import argparse
from scenarios.utils import *

class Benchmark():
    def __init__(self,train_datasets, test_datasets, val_datasets=None) -> None:
       self.train_stream = []
       self.test_stream = []
       self.val_stream = []
       self.initialize_streams(self.train_stream, train_datasets)
       self.initialize_streams(self.test_stream,test_datasets)
       if val_datasets is not None:
           self.initialize_streams(self.val_stream,val_datasets)
        

    def initialize_streams(self,stream,datasets):
        for id,dataset in enumerate(datasets):
            task = {"task_id": id, "dataset": dataset,"classes_in_this_experience":[0,1]}
            stream.append(edict(task))


def get_continual_scenario_benchmark(args: argparse.Namespace):
    
    # Load continual scenarios sequence
    task_seq = args.task_seq
    start_task_id = args.start_task
    task_dict = {1:'MA',2:'FT',3:'ZS',4:'EP',5:'SR'}
    dataset_ids = [int(task_id) for task_id in task_seq.split("-")]
    dataset_ids = [dataset_id for task_id,dataset_id in enumerate(dataset_ids) if task_id >= start_task_id]
    print("[Scenario] Task sequence:",dataset_ids)
 
    train_datasets = []
    val_datasets = []
    test_datasets  = []
    for dataset_id in dataset_ids:
        print("*" * 40 + " " + "Dataset: " +str(dataset_id)+"-" + task_dict[dataset_id] +" " + "*"*40)
        dataset_filename = "./scenarios/datasets/" + str(dataset_id)+"-" + task_dict[dataset_id]+"/"
        dir_names = ['train/','val/','test/']
        if args.is_demo:
            dir_names = ['tinydemo/','tinydemo/','tinydemo/']
        train_dir, val_dir, test_dir = [dataset_filename+dir_name for dir_name in dir_names]
        train_data = TrajectoryDataset(train_dir,
                                       obs_len=args.obs_seq_len,
                                       pred_len=args.pred_seq_len,
                                       skip=1,
                                       norm_lap_matr=True)
        val_data = TrajectoryDataset(val_dir,
                                    obs_len=args.obs_seq_len,
                                    pred_len=args.pred_seq_len,
                                    skip=1,
                                    norm_lap_matr=True)
        test_data = TrajectoryDataset(test_dir,
                                    obs_len=args.obs_seq_len,
                                    pred_len=args.pred_seq_len,
                                    skip=1,
                                    norm_lap_matr=True)
        train_datasets.append(train_data)
        test_datasets.append(test_data)
        val_datasets.append(val_data)
    benchmark = Benchmark(train_datasets,test_datasets,val_datasets=val_datasets)
    return benchmark