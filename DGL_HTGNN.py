#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:21:40 2022

@author: yang.liu
"""
import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.model import HTGNN, NodePredictor
from utils.pytorchtools import EarlyStopping
from utils.HTGDataset import HTGDataset

from glob import glob
import random
# from utils.data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

mean_std = torch.load('mean_std.pt')
mean_std['loop_mean'] = mean_std['loop_mean']
mean_std['loop_std'] = mean_std['loop_std']

graph_list = glob('./data/processed/powerDrop*.bin')

full_index = range(len(graph_list))
train_index = random.sample(range(len(graph_list)),int(0.7*len(graph_list)))
test_index = list(set(full_index) - set(train_index))

train_dataset = HTGDataset(graph_list) 

n_hid = 32
n_input = {'loop':3, 'core':2, 'pump':1, 'hid':n_hid }
n_classes = {'loop':3, 'core':2, 'pump':1}
batch_size = 512
epochs = 500

htgnn = HTGNN(graph=train_feats[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)
predictor = NodePredictor(n_inp=n_hid , n_classes=n_classes,device = device)

model = nn.Sequential(htgnn, predictor).to(device)


early_stopping = EarlyStopping(patience=10, verbose=True, path='{model_out_path}/checkpoint_HTGNN.pt')
optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=False, **kwargs)

for epoch in range(epochs):
    model.train()
    mse = 0.
    for _, (G_feat, G_target) in enumerate(train_loader):
        
        G_feat, G_target = G_feat.to(device), G_target.to(device)
        
        for j in G_feat.ndata.keys():
            G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].to(device)
            G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].to(device)            
            G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].to(device)
            G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].to(device)            
            G_feat.nodes['core'].data[j] -= mean_std['core_mean'].to(device)
            G_feat.nodes['core'].data[j] /= mean_std['core_std'].to(device)
        
        for j in G_target.ndata.keys():
            G_target.nodes['loop'].data[j] -= mean_std['loop_mean'].to(device)
            G_target.nodes['loop'].data[j] /= mean_std['loop_std'].to(device)            
            G_target.nodes['core'].data[j] -= mean_std['core_mean'].to(device)
            G_target.nodes['core'].data[j] /= mean_std['core_std'].to(device)            
            G_target.nodes['pump'].data[j] -= mean_std['pump_mean'].to(device)
            G_target.nodes['pump'].data[j] /= mean_std['pump_std'].to(device)
        
        model.zero_grad()
        h = model[0](G_feat)
        pred = model[1](h)
        
        loss = F.mse_loss(pred['loop'], G_target.nodes['loop'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['pump'], G_target.nodes['pump'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['core'], G_target.nodes['core'].data['feat'], reduction = 'sum')
        
        # loss = loss/3
                         

        # train_mse_list.append(loss.item())
        # train_rmse_list.append(rmse.item())
        
        loss.backward()
        optim.step()
        mse += loss.item()
    
    mse = mse/(3*11+2*1+2*3)/batch_size
    
    rmse = np.sqrt(mse)
    print('rmse: ', rmse)

    # loss, rmse = evaluate(model, val_feats, val_labels)
    # early_stopping(loss, model)

    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break