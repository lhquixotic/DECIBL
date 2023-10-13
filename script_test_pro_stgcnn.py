import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import quadprog
import torch.onnx

import argparse

from DECIBL.psstgcnn import *

parser = argparse.ArgumentParser()
# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,
                    help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5,
                    help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)

args = parser.parse_args()
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len

model = progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

model2 = progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

model3 = progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

tag = "test-1"

path1 = "./logs/{}/checkpoint/val_best_model_0.pth".format(tag)
path2 = "./logs/{}/checkpoint/val_best_model_1.pth".format(tag)
path3 = "./logs/{}/checkpoint/val_best_model_2.pth".format(tag)

# model.load_state_dict(torch.load('val_best.pth',map_location=torch.device('cpu')))
torch.manual_seed(1)

example_input = (torch.randn(1,2,20,4).to("cuda:0"),torch.randn(20,4,4).to("cuda:0"))

ckpt = torch.load(path1)
model.new_task()
model.load_state_dict(ckpt["model_state_dict"])
model.batch_norm_para = ckpt["batch_norm_para"]
model.expert_selector = ckpt["expert_selector"]
for key, value in ckpt["batch_norm_para"].items():
    print("task_{}: bn para len is {}".format(key, len(value[0])))
    
# print("es_{}:{}".format(0, model.batch_norm_para[0][0]))
model.load_batch_norm_para(task_id=0)
model = model.cuda()
model.eval()
res1 = model(example_input[0],example_input[1],expert_id=0)
print("res:",res1[0,0,0,0])

ckpt = torch.load(path2)
model2.new_task()
model2.new_task()
model2.load_state_dict(ckpt["model_state_dict"])
model2.batch_norm_para = ckpt["batch_norm_para"]
model2.expert_selector = ckpt["expert_selector"]

for key, value in ckpt["batch_norm_para"].items():
    print("task_{}: bn para len is {}".format(key, len(value[0])))
# print("es_{}:{}".format(0, model2.batch_norm_para[0][0]))

model2.load_batch_norm_para(task_id=0)
model2 = model2.cuda()
model2.eval()
res2 = model2(example_input[0],example_input[1],expert_id=0)
print("res:",res2[0,0,0,0])

model3.new_task()
model3.new_task()
model3.new_task()
ckpt = torch.load(path3)
model3.load_state_dict(ckpt["model_state_dict"])
model3.batch_norm_para = ckpt["batch_norm_para"]
model3.expert_selector = ckpt["expert_selector"]
model3 = model3.cuda()

for key, value in ckpt["batch_norm_para"].items():
    print("task_{}: bn para len is {}".format(key, len(value[0])))
# print("es_{}:{}".format(0, model3.batch_norm_para[0][0]))
    

model3.load_batch_norm_para(task_id=0)
model3.eval()
res3 = model3(example_input[0],example_input[1],expert_id=0)
print("res:",res3[0,0,0,0])

fxxk 

model.new_task()
model.freeze_columns()
model.load_state_dict(torch.load("./checkpoints/test/val_best_model_0.pth"))
model = model.to("cuda:0")
model.eval()

res = model(example_input[0],example_input[1])
print(res[0,0,0,0])
print("****"*20)
model2.new_task()
model2.new_task()
model2.load_state_dict(torch.load("./checkpoints/test/val_best_model_1.pth"))
model2.to("cuda:0")
model2.eval()

res = model2(example_input[0],example_input[1],task_id=0)
print(res[0,0,0,0])

# print("model1:",model.columns[0])
# print("model2:",model2.columns[0])

# fxxk
print("*"*40)
for para1,para2 in zip(model.columns[0].named_parameters(),model2.columns[0].parameters()):
    print(para1[0],":",(para1[1]==para2).all())

print("*"*40)
# print()
r1 = model.columns[0][0].block.residual[0]
r2 = model2.columns[0][0].block.residual[0]
n1 = model.columns[0][0].block.residual[1]
n2 = model2.columns[0][0].block.residual[1]
p1 = model.columns[0][0].block.residual.parameters()
p2 = model2.columns[0][0].block.residual.parameters()
d1 = r1(example_input[0])
d2 = r2(example_input[0])


model2.train()
model2.columns[0].eval()
# for id,module in enumerate(model2.columns[0].modules()):
#     print("module:",id,module)
#     if isinstance(module,nn.BatchNorm2d):
#         module.training = 

print("n1:",n1)
print("n2:",n2)
print("running_mean:",n1.running_mean,n2.running_mean)
print("training:",n2.training)
print("weight:",n2.weight)
print("d1d2:{}".format((n1(d1)==n1(d2)).all()))

# print("d1:",d1)
# print("d2:",d2)
# for pp1,pp2 in zip(p1,p2):
#     print("pp1:",pp1)
#     print("pp2:",pp2)
#     break



fxxk



# model.train()
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            print(f'Parameter {name} has gradient.')
        else:
            print(f'Parameter {name} does not have gradient.')
    else:
        print("false")

print("*"*40)

res = model(example_input[0],example_input[1])
print(res[0,0,0,0])
model.freeze_columns()


for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            print(f'Parameter {name} has gradient.')
        else:
            print(f'Parameter {name} does not have gradient.')
    else:
        print("false")
# print("model:", model)

model.new_task()
# model.freeze_columns()
model.load_state_dict(torch.load("./checkpoints/psstgcnn/val_best_model_1.pth"))

model = model.to("cuda:0")

# print("model:", model)
# print("*"*60)
# for module in model.modules():
    # if isinstance(module,nn.Conv2d):
        # print("module:",module)


# # torch.save(model,'stgcnn_model.pth')
# torch.onnx.export(model,eample_input,  "pro_stgcnn_model.onnx", input_names=["input"],output_names=["outputs"],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
model.eval()
res = model(example_input[0],example_input[1],task_id=0)
print(res[0,0,0,0])
# print(res.shape)
# print(res[:,:,0].shape,res[:,:,1].shape)
# print(res.shape)
# print(model)
# print("*"*60)
# print(model.st_gcns[0].residual)
# print("*"*60)
# pp = model.parameters
# cnt = 0
# model.train()
# for param in pp():
    
#     if param is not None:
#         print(cnt)
#         print(param)
#         print(param.grad)
        
#     cnt += 1