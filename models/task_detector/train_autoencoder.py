import pickle
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.task_detector.graph_autoencoder import GraphAutoEncoder, train, valid
from utils.utils import get_device

# args
parser = argparse.ArgumentParser(description='train autoencoder')
parser.add_argument('--task_id', type=int, default = 0)
parser.add_argument('--epochs_num', type=int, default=250)
parser.add_argument('--learning_rate',type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--load_trained_stgcn', type=int, default=0)

args = parser.parse_args()

# task info
task_dict = {0:"1-MA",1:"2-FT", 2:"3-ZS", 3:"4-EP", 4:"5-SR"}
task_id = args.task_id
print("Loading task_{}: {}".format(task_id,task_dict[task_id]))

# Data prepare
train_fname = "./scenarios/datasets/processed/{}/train.pkl".format(task_dict[task_id])
with open(train_fname, "rb") as file:
    train_data = pickle.load(file)
val_fname = "./scenarios/datasets/processed/{}/val.pkl".format(task_dict[task_id])
with open(val_fname, "rb") as file:
    val_data = pickle.load(file)
test_fname = "./scenarios/datasets/processed/{}/test.pkl".format(task_dict[task_id])
with open(test_fname, "rb") as file:
    test_data = pickle.load(file)

train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=16,drop_last=True)
val_loader = DataLoader(val_data,batch_size=1,shuffle=True,num_workers=16,drop_last=True)
test_loader = DataLoader(test_data,batch_size=1,shuffle=True,num_workers=16,drop_last=True)

# Model preparation
autoencoder = GraphAutoEncoder(seq_len=20)

# Load trained encoder
if args.load_trained_stgcn:
    trained_model_sd = torch.load("./networks/temp/val_best_{}.pth".format(task_id))
    # TODO: modify the trained model
    trained_state_dict = dict()
    needed_keys = [key for key in trained_model_sd.keys() if 'st_gcns' in key]
    for key in needed_keys:
        trained_state_dict[key] = trained_model_sd[key]
    # print("trained state dict keys: {}, length is {}".format(trained_state_dict.keys(),len(trained_state_dict.keys())))
    # print(trained_state_dict['st_gcns.0.tcn.0.bias'])
    autoencoder.load_trained_encoder(trained_state_dict)
    # print(autoencoder.encoder.state_dict()['tcn.0.bias'])

    # freeze the parameters of the encoder
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

autoencoder.to(get_device())

optimizer = torch.optim.SGD(autoencoder.parameters(), lr = args.learning_rate)
batch_size = args.batch_size

is_1st_loss = True
epoch_loss = 0
constant_metrics=dict()
constant_metrics["val_min_epoch"] = -1
constant_metrics["val_min_loss"] = 9999

for epoch in range(args.epochs_num):
    autoencoder, train_loss = train(autoencoder,optimizer,train_loader)
    val_loss = valid(autoencoder, val_loader)
    print("[Loss] Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}".format(epoch,train_loss,val_loss))
    if val_loss < constant_metrics["val_min_loss"]:
        constant_metrics["val_min_epoch"] = epoch
        constant_metrics["val_min_loss"] = val_loss
        torch.save(autoencoder.state_dict(),"./models/task_detector/checkpoint/autoencoder_{}.pth".format(task_id))
        print("save best model")
        if epoch - constant_metrics['val_min_epoch'] > 50:
            break
        