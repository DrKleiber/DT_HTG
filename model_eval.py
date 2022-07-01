# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:13:43 2022

@author: Yang
"""
import dgl
import torch
import torch.nn as nn

from dgl.data.utils import load_graphs

from glob import glob
from model.model import NodeFuturePredictor, NodeSameTimePredictor
from utils.HTGDataset import EBRDataset

import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

device = torch.device('cuda')
graph_list = glob('../data/processed/powerDrop*.bin')
graph_template, _ = load_graphs(graph_list[0])

n_hid = 32
n_input = {'loop':3, 'core':2, 'pump':1}
model_0 = NodeSameTimePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)
model_1 = NodeFuturePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid, n_layers=2, n_heads=1, time_window=10, norm=False,device = device)

model_0 = nn.DataParallel(model_0)
model_0.load_state_dict(torch.load('powerDrop_dist_2parts_test/saved_model/model_0_epoch3000.pth', map_location=torch.device('cuda')))
model_0.eval()

model_1 = nn.DataParallel(model_1)
model_1.load_state_dict(torch.load('powerDrop_dist_2parts_test/saved_model/model_1_epoch3000.pth', map_location=torch.device('cuda')))
model_1.eval()

mean_std = torch.load('mean_std.pt')

CV_1_p = np.empty(shape =(0))
CV_1_T = np.empty(shape =(0))
CV_1_G = np.empty(shape =(0))

CV_2_p = np.empty(shape =(0))
CV_2_T = np.empty(shape =(0))
CV_2_G = np.empty(shape =(0))

CV_3_p = np.empty(shape =(0))
CV_3_T = np.empty(shape =(0))
CV_3_G = np.empty(shape =(0))

CV_4_p = np.empty(shape =(0))
CV_4_T = np.empty(shape =(0))
CV_4_G = np.empty(shape =(0))

IHX_G = np.empty(shape =(0))
IHX_T = np.empty(shape =(0))
IHX_p = np.empty(shape =(0))

Ch_A_G = np.empty(shape =(0))
Ch_A_T = np.empty(shape =(0))
Ch_A_p = np.empty(shape =(0))

Ch_A_T_fuel = np.empty(shape =(0))
Ch_A_P = np.empty(shape =(0))

Ch_B_G = np.empty(shape =(0))
Ch_B_T = np.empty(shape =(0))
Ch_B_p = np.empty(shape =(0))

Ch_B_T_fuel = np.empty(shape =(0))
Ch_B_P = np.empty(shape =(0))

Ch_P_G = np.empty(shape =(0))
Ch_P_T = np.empty(shape =(0))
Ch_P_p = np.empty(shape =(0))

Ch_P_P = np.empty(shape =(0))
Ch_P_T_fuel = np.empty(shape =(0))

Pump_0 =  np.empty(shape =(0))
Pump_1 =  np.empty(shape =(0))

G_feat, _ = load_graphs(graph_list[0])
G_input = copy.deepcopy(G_feat[0])
G_input = G_input.to(device)

# for j in G_input.ndata.keys():
#     G_input.nodes['loop'].data[j][(0,1,5,6,8,9),:] = 0
#     G_input.nodes['core'].data[j][:,1] = 0
#     G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
#     G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
#     G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
#     G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
#     G_input.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)
#     G_input.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)

# pred_0 = model_0(G_input.to(device))

# for j in pred_0['core'].keys():
#     pred_0['core'][j] *= mean_std['core_std'].to(device)
#     pred_0['core'][j] += mean_std['core_mean'].to(device)
    
#     Ch_A_P = np.append(Ch_A_P, pred_0['core'][j][0,0].detach().cpu().numpy())
#     Ch_B_P = np.append(Ch_B_P, pred_0['core'][j][1,0].detach().cpu().numpy())
#     Ch_P_P = np.append(Ch_P_P, pred_0['core'][j][2,0].detach().cpu().numpy())
#     Ch_A_T_fuel = np.append(Ch_A_T_fuel, pred_0['core'][j][0,1].detach().cpu().numpy())
#     Ch_B_T_fuel = np.append(Ch_B_T_fuel, pred_0['core'][j][1,1].detach().cpu().numpy())
#     Ch_P_T_fuel = np.append(Ch_P_T_fuel, pred_0['core'][j][2,1].detach().cpu().numpy())
    
# for j in pred_0['loop'].keys():
#     pred_0['loop'][j] *= mean_std['loop_std'].to(device)
#     pred_0['loop'][j] += mean_std['loop_mean'].to(device)    
    
#     CV_1_G = np.append(CV_1_G, pred_0['loop'][j][0,0].detach().cpu().numpy())
#     CV_2_G = np.append(CV_2_G, pred_0['loop'][j][1,0].detach().cpu().numpy())
#     Ch_A_G = np.append(Ch_A_G, pred_0['loop'][j][2,0].detach().cpu().numpy())
#     Ch_B_G = np.append(Ch_B_G, pred_0['loop'][j][3,0].detach().cpu().numpy())
#     Ch_P_G = np.append(Ch_P_G, pred_0['loop'][j][4,0].detach().cpu().numpy())
#     CV_3_G = np.append(CV_3_G, pred_0['loop'][j][5,0].detach().cpu().numpy())
#     IHX_G = np.append(IHX_G, pred_0['loop'][j][6,0].detach().cpu().numpy())
#     CV_4_G = np.append(CV_4_G, pred_0['loop'][j][8,0].detach().cpu().numpy())
    
#     CV_1_p = np.append(CV_1_p, pred_0['loop'][j][0,1].detach().cpu().numpy())
#     CV_2_p = np.append(CV_2_p, pred_0['loop'][j][1,1].detach().cpu().numpy())
#     Ch_A_p = np.append(Ch_A_p, pred_0['loop'][j][2,1].detach().cpu().numpy())
#     Ch_B_p = np.append(Ch_B_p, pred_0['loop'][j][3,1].detach().cpu().numpy())
#     Ch_P_p = np.append(Ch_P_p, pred_0['loop'][j][4,1].detach().cpu().numpy())
#     CV_3_p = np.append(CV_3_p, pred_0['loop'][j][5,1].detach().cpu().numpy())
#     IHX_p = np.append(IHX_p, pred_0['loop'][j][6,1].detach().cpu().numpy())
#     CV_4_p = np.append(CV_4_p, pred_0['loop'][j][8,1].detach().cpu().numpy())
    
#     CV_1_T = np.append(CV_1_T, pred_0['loop'][j][0,2].detach().cpu().numpy())
#     CV_2_T = np.append(CV_2_T, pred_0['loop'][j][1,2].detach().cpu().numpy())
#     Ch_A_T = np.append(Ch_A_T, pred_0['loop'][j][2,2].detach().cpu().numpy())
#     Ch_B_T = np.append(Ch_B_T, pred_0['loop'][j][3,2].detach().cpu().numpy())
#     Ch_P_T = np.append(Ch_P_T, pred_0['loop'][j][4,2].detach().cpu().numpy())
#     CV_3_T = np.append(CV_3_T, pred_0['loop'][j][5,2].detach().cpu().numpy())
#     IHX_T = np.append(IHX_T, pred_0['loop'][j][6,2].detach().cpu().numpy())
#     CV_4_T = np.append(CV_4_T, pred_0['loop'][j][8,2].detach().cpu().numpy())


# for j in pred_0['pump'].keys():
#     pred_0['pump'][j] *= mean_std['pump_std'].to(device)
#     pred_0['pump'][j] += mean_std['pump_mean'].to(device)
    
#     Pump_0 = np.append(Pump_0, pred_0['pump'][j][0].detach().cpu().numpy())
#     Pump_1 = np.append(Pump_1, pred_0['pump'][j][1].detach().cpu().numpy())

# G_input_future = copy.deepcopy(G_feat[0]).to(device)

# for j in G_input_future.ndata.keys():
#     G_input_future.nodes['loop'].data[j] = pred_0['loop'][j]
#     G_input_future.nodes['core'].data[j] = pred_0['core'][j]
#     G_input_future.nodes['pump'].data[j] = pred_0['pump'][j]
#     G_input_future.nodes['loop'].data[j] -= mean_std['loop_mean'].to(device)
#     G_input_future.nodes['loop'].data[j] /= mean_std['loop_std'].to(device)
#     G_input_future.nodes['core'].data[j] -= mean_std['core_mean'].to(device)
#     G_input_future.nodes['core'].data[j] /= mean_std['core_std'].to(device)
#     G_input_future.nodes['pump'].data[j] -= mean_std['pump_mean'].to(device)
#     G_input_future.nodes['pump'].data[j] /= mean_std['pump_std'].to(device)

# pred_1 = model_1(G_input_future)

# pred_1['core'] *= mean_std['core_std'].to(device)
# pred_1['core'] += mean_std['core_mean'].to(device)
# pred_1['loop'] *= mean_std['loop_std'].to(device)
# pred_1['loop'] += mean_std['loop_mean'].to(device)
# pred_1['pump'] *= mean_std['pump_std'].to(device)
# pred_1['pump'] += mean_std['pump_mean'].to(device)

for i in graph_list[0:300]:
    G_feat, _ = load_graphs(i)
    G_input = copy.deepcopy(G_feat[0])
    G_input = G_input.to(device)
    
    for j in G_input.ndata.keys():
        G_input.nodes['loop'].data[j][(0,1,5,6,8,9),:] = 0
        G_input.nodes['core'].data[j][:,1] = 0
        G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
        G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
        G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
        G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
        G_input.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)
        G_input.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)

    pred_0 = model_0(G_input.to(device))
    
    G_input_future = copy.deepcopy(G_feat[0]).to(device)

    for j in G_input_future.ndata.keys():
        G_input_future.nodes['loop'].data[j] = pred_0['loop'][j]
        G_input_future.nodes['core'].data[j] = pred_0['core'][j]
        G_input_future.nodes['pump'].data[j] = pred_0['pump'][j]
        
    pred_1 = model_1(G_input_future.to(device))
    # pred_1 = model_1(G_feat[0].to(device))

    pred_1['core'] *= mean_std['core_std'].to(device)
    pred_1['core'] += mean_std['core_mean'].to(device)
    pred_1['loop'] *= mean_std['loop_std'].to(device)
    pred_1['loop'] += mean_std['loop_mean'].to(device)
    pred_1['pump'] *= mean_std['pump_std'].to(device)
    pred_1['pump'] += mean_std['pump_mean'].to(device)
    
    Ch_A_P = np.append(Ch_A_P, pred_1['core'][0,0].detach().cpu().numpy())
    Ch_B_P = np.append(Ch_B_P, pred_1['core'][1,0].detach().cpu().numpy())
    Ch_P_P = np.append(Ch_P_P, pred_1['core'][2,0].detach().cpu().numpy())
    Ch_A_T_fuel = np.append(Ch_A_T_fuel, pred_1['core'][0,1].detach().cpu().numpy())
    Ch_B_T_fuel = np.append(Ch_B_T_fuel, pred_1['core'][1,1].detach().cpu().numpy())
    Ch_P_T_fuel = np.append(Ch_P_T_fuel, pred_1['core'][2,1].detach().cpu().numpy())
    
    CV_1_G = np.append(CV_1_G, pred_1['loop'][0,0].detach().cpu().numpy())
    CV_2_G = np.append(CV_2_G, pred_1['loop'][1,0].detach().cpu().numpy())
    Ch_A_G = np.append(Ch_A_G, pred_1['loop'][2,0].detach().cpu().numpy())
    Ch_B_G = np.append(Ch_B_G, pred_1['loop'][3,0].detach().cpu().numpy())
    Ch_P_G = np.append(Ch_P_G, pred_1['loop'][4,0].detach().cpu().numpy())
    CV_3_G = np.append(CV_3_G, pred_1['loop'][5,0].detach().cpu().numpy())
    IHX_G = np.append(IHX_G, pred_1['loop'][6,0].detach().cpu().numpy())
    CV_4_G = np.append(CV_4_G, pred_1['loop'][8,0].detach().cpu().numpy())
    
    CV_1_p = np.append(CV_1_p, pred_1['loop'][0,1].detach().cpu().numpy())
    CV_2_p = np.append(CV_2_p, pred_1['loop'][1,1].detach().cpu().numpy())
    Ch_A_p = np.append(Ch_A_p, pred_1['loop'][2,1].detach().cpu().numpy())
    Ch_B_p = np.append(Ch_B_p, pred_1['loop'][3,1].detach().cpu().numpy())
    Ch_P_p = np.append(Ch_P_p, pred_1['loop'][4,1].detach().cpu().numpy())
    CV_3_p = np.append(CV_3_p, pred_1['loop'][5,1].detach().cpu().numpy())
    IHX_p = np.append(IHX_p, pred_1['loop'][6,1].detach().cpu().numpy())
    CV_4_p = np.append(CV_4_p, pred_1['loop'][8,1].detach().cpu().numpy())
    
    CV_1_T = np.append(CV_1_T, pred_1['loop'][0,2].detach().cpu().numpy())
    CV_2_T = np.append(CV_2_T, pred_1['loop'][1,2].detach().cpu().numpy())
    Ch_A_T = np.append(Ch_A_T, pred_1['loop'][2,2].detach().cpu().numpy())
    Ch_B_T = np.append(Ch_B_T, pred_1['loop'][3,2].detach().cpu().numpy())
    Ch_P_T = np.append(Ch_P_T, pred_1['loop'][4,2].detach().cpu().numpy())
    CV_3_T = np.append(CV_3_T, pred_1['loop'][5,2].detach().cpu().numpy())
    IHX_T = np.append(IHX_T, pred_1['loop'][6,2].detach().cpu().numpy())
    CV_4_T = np.append(CV_4_T, pred_1['loop'][8,2].detach().cpu().numpy())
    
    Pump_0 = np.append(Pump_0, pred_1['pump'][0].detach().cpu().numpy())
    Pump_1 = np.append(Pump_1, pred_1['pump'][1].detach().cpu().numpy())

# for j in G_feat.ndata.keys():
#     G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
#     G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
#     G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
#     G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
#     G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)
#     G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)

# pred_1 = model_1(G_feat)

# initialize feature prediction
# CV_1_G.append()


# for i in graph_list:
#         G_feat, _ = load_graphs(i)
#         G_input = copy.deepcopy(G_feat)

#         for j in G_input.ndata.keys():
#             G_input.nodes['loop'].data[j][(0,1,5,6,8,9),:] = 0
#             G_input.nodes['core'].data[j][:,1] = 0
#             G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
#             G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
#             G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
#             G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
#             G_input.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)
#             G_input.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)

#         pred_0 = model_0(G_input)

#         for j in G_feat.ndata.keys():
#             G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
#             G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
#             G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
#             G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
#             G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)
#             G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)

#         pred_1 = model_1(G_input)

#         CV_1_G.append()

t = np.linspace(0.,299., 300)

sam_files = glob('C:/Users/Yang/Box/ProgramDevelopment/ebr2_sample/transients/ebr2_3chan_powerDrop/workdir.*/ebr2_3chan_powerDrop_csv.csv')
data_org = pd.read_csv('C:/Users/Yang/Box/ProgramDevelopment/ebr2_sample/transients/ebr2_3chan_powerDrop/workdir.1/ebr2_3chan_powerDrop_csv.csv')

fig,ax = plt.subplots(1,1)
ax.plot(t, Ch_A_T, color='darkblue', lw=1.5)
ax.plot(data_org['time'][401:772],data_org['CHA_outlet_temperature'][401:772])

fig,ax = plt.subplots(1,1)
ax.plot(t, Ch_B_T, color='darkblue', lw=1.5)
ax.plot(data_org['time'][401:772],data_org['CHB_outlet_temperature'][401:772])

fig,ax = plt.subplots(1,1)
ax.plot(t, Ch_P_T, color='darkblue', lw=1.5)
ax.plot(data_org['time'][401:772],data_org['CHP_outlet_temperature'][401:772])
