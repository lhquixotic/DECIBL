import copy
import pickle

import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader

from models.st_gcn import ConvTemporalGraphical,st_gcn
from utils.utils import get_device

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=32, kernel_size=3,seq_len=20):
        super(GraphAutoEncoder, self).__init__()
        
        middle_dim = int(latent_dim/2)
        self.encoder_1 = st_gcn(in_channels,middle_dim,(kernel_size,seq_len))
        self.encoder_2 = st_gcn(middle_dim,latent_dim,(kernel_size,seq_len))
        self.decoder_1 = st_gcn(latent_dim,middle_dim,(kernel_size,seq_len))
        self.decoder_2 = st_gcn(middle_dim,in_channels,(kernel_size,seq_len))
        
    def forward(self, v,a):
        v,a = self.encoder_1(v,a)
        ev,ea = self.encoder_2(v,a)
        v,a = self.decoder_1(ev,ea)
        dv,da = self.decoder_2(v,a)
        return ev,dv

    def load_trained_encoder(self,trained_state_dict):
        state_dict = copy.deepcopy(self.encoder.state_dict())
        self_keys = state_dict.keys()
        trained_keys = list(trained_state_dict.keys())
        for key_id, key in enumerate(self_keys):
            state_dict[key] = trained_state_dict[trained_keys[key_id]]
        self.encoder.load_state_dict(state_dict)

        
def train(model, optimizer, train_loader, batch_size=64):
    epoch_loss = 0
    is_1st_loss = True
    loss_func = nn.MSELoss()
    model.train()
    
    for step, case in enumerate(train_loader):
        optimizer.zero_grad()
        case = [tensor.to(get_device()) for tensor in case]
        _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
        V_obs_tmp = V_obs.permute(0,3,1,2)
        ev,dv = model(V_obs_tmp,A_obs.squeeze())
        loss = loss_func(dv,V_obs_tmp)
        
        if (step+1) % batch_size != 0:
            if is_1st_loss:
                 batch_loss = loss
                 is_1st_loss = False
            else:
                batch_loss += loss
        else:
            batch_loss += loss
            batch_loss = batch_loss / batch_size
            is_1st_loss = True
            batch_loss.backward()
            
            optimizer.step()
            epoch_loss += batch_loss.item()
    return model, epoch_loss/ (step+1)

def valid(model, val_loader, batch_size=64):
    epoch_loss = 0
    loss_func = nn.MSELoss()
    is_1st_loss = True
    model.eval()
    for step, case in enumerate(val_loader):
        case = [tensor.to(get_device()) for tensor in case]
        _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
        V_obs_tmp = V_obs.permute(0,3,1,2)
        ev,dv = model(V_obs_tmp,A_obs.squeeze())
        loss = loss_func(dv,V_obs_tmp) 

        if (step+1) % batch_size != 0:
            if is_1st_loss:
                batch_loss = loss
                is_1st_loss = False
            else:
                batch_loss += loss
        else:
            batch_loss += loss
            batch_loss = batch_loss / batch_size
            is_1st_loss = True
            epoch_loss += batch_loss.item()
        
    return epoch_loss/(step+1)
        


