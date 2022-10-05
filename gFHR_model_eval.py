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
from model.model_dualwindow import NodeFuturePredictor, NodeSameTimePredictor
from utils.HTGDataset import SAMDataset

import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

device = torch.device('cpu')
graph_list = glob('../data/gFHR/processed/IHX_transient_01_*.bin')
graph_template, _ = load_graphs(graph_list[0])

n_hid = 32
n_input = {'loop':3, 'core':3, 'solid':1}
model_0 = NodeSameTimePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=1, n_heads=1, time_window=20, norm=False,device = device)
model_1 = NodeFuturePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid, n_layers=2, n_heads=1, time_window_inp=20, time_window_tar=5, norm=False,device = device)

# model_0 = nn.DataParallel(model_0)
model_0.load_state_dict(torch.load('cases/gFHR_test/saved_model/model_0_epoch1000.pth', map_location=torch.device('cpu')))
model_0.eval()

# model_1 = nn.DataParallel(model_1)
model_1.load_state_dict(torch.load('cases/gFHR_test/saved_model/model_1_epoch1000.pth', map_location=torch.device('cpu')))
model_1.eval()

mean_std = torch.load('mean_std.pt')

active_p = np.empty(shape =(0))  # loop node #2
active_T = np.empty(shape =(0))
active_G = np.empty(shape =(0))

pump_p = np.empty(shape =(0))  # loop node #7
pump_T = np.empty(shape =(0))
pump_G = np.empty(shape =(0))

IHX_P_p = np.empty(shape =(0)) # loop node #9
IHX_P_T = np.empty(shape =(0))
IHX_P_G = np.empty(shape =(0))

core_T_fuel = np.empty(shape =(0)) # core node
core_T_surface = np.empty(shape =(0)) 

active_hs_T = np.empty(shape =(0)) # solid heat structure node #5

# G_feat, _ = load_graphs(graph_list[0])
# G_input = copy.deepcopy(G_feat[0])
# G_input = G_input.to(device)

# for j in G_input.ndata.keys():
#     G_input.nodes['loop'].data[j][(0,1,2,4,5,7,8,9,11),:] = 0.
#     G_input.nodes['core'].data[j][:,(1,2)] = 0.
#     G_input.nodes['solid'].data[j][:,:] = 0.
#     G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].to(device)
#     G_input.nodes['loop'].data[j] /= mean_std['loop_std'].to(device)
#     G_input.nodes['core'].data[j] -= mean_std['core_mean'].to(device)
#     G_input.nodes['core'].data[j] /= mean_std['core_std'].to(device)
#     G_input.nodes['solid'].data[j] -= mean_std['solid_mean'].to(device)
#     G_input.nodes['solid'].data[j] /= mean_std['solid_std'].to(device)

# pred_0 = model_0(G_input.to(device))

# for j in pred_0['core'].keys():
#     pred_0['core'][j] *= mean_std['core_std'].to(device)
#     pred_0['core'][j] += mean_std['core_mean'].to(device)
    
#     core_T_fuel = np.append(core_T_fuel, pred_0['core'][j][0,0].detach().numpy())
#     core_T_surface = np.append(core_T_fuel, pred_0['core'][j][0,1].detach().numpy())
    
    
# for j in pred_0['loop'].keys():
#     pred_0['loop'][j] *= mean_std['loop_std'].to(device)
#     pred_0['loop'][j] += mean_std['loop_mean'].to(device)    
    
#     active_G = np.append(active_G, pred_0['loop'][j][2,0].detach().numpy())
#     pump_G = np.append(pump_G, pred_0['loop'][j][7,0].detach().numpy())
#     IHX_P_G = np.append(IHX_P_G, pred_0['loop'][j][9,0].detach().numpy())

    
#     active_p = np.append(active_p, pred_0['loop'][j][2,1].detach().numpy())
#     pump_p = np.append(pump_p, pred_0['loop'][j][7,1].detach().numpy())
#     IHX_P_p = np.append(IHX_P_p, pred_0['loop'][j][9,1].detach().numpy())

    
#     active_T = np.append(active_T, pred_0['loop'][j][2,2].detach().numpy())
#     pump_T = np.append(pump_T, pred_0['loop'][j][7,2].detach().numpy())
#     IHX_P_T = np.append(IHX_P_T, pred_0['loop'][j][9,2].detach().numpy())


# for j in pred_0['solid'].keys():
#     pred_0['solid'][j] *= mean_std['solid_std'].to(device)
#     pred_0['solid'][j] += mean_std['solid_mean'].to(device)
    
#     active_hs_T = np.append(active_hs_T, pred_0['solid'][j][5].detach().cpu().numpy())

# G_input_future = copy.deepcopy(G_feat[0]).to(device)

# for j in G_input_future.ndata.keys():
#     G_input_future.nodes['loop'].data[j] = pred_0['loop'][j]
#     G_input_future.nodes['core'].data[j] = pred_0['core'][j]
#     G_input_future.nodes['solid'].data[j] = pred_0['solid'][j]
#     G_input_future.nodes['loop'].data[j] -= mean_std['loop_mean'].to(device)
#     G_input_future.nodes['loop'].data[j] /= mean_std['loop_std'].to(device)
#     G_input_future.nodes['core'].data[j] -= mean_std['core_mean'].to(device)
#     G_input_future.nodes['core'].data[j] /= mean_std['core_std'].to(device)
#     G_input_future.nodes['solid'].data[j] -= mean_std['solid_mean'].to(device)
#     G_input_future.nodes['solid'].data[j] /= mean_std['solid_std'].to(device)

# pred_1 = model_1(G_input_future)

# pred_1['core']['t0'] *= mean_std['core_std'].to(device)
# pred_1['core']['t0'] += mean_std['core_mean'].to(device)
# pred_1['loop']['t0'] *= mean_std['loop_std'].to(device)
# pred_1['loop']['t0'] += mean_std['loop_mean'].to(device)
# pred_1['solid']['t0'] *= mean_std['solid_std'].to(device)
# pred_1['solid']['t0'] += mean_std['solid_mean'].to(device)

for i in graph_list[0:245]:
    G_feat, _ = load_graphs(i)
    # G_input = copy.deepcopy(G_feat[0])
    # G_input = G_input.to(device)
    
    # for j in G_input.ndata.keys():
    #     G_input.nodes['loop'].data[j][(0,1,2,4,5,7,8,9,11),:] = 0.
    #     G_input.nodes['core'].data[j][:,(1,2)] = 0.
    #     G_input.nodes['solid'].data[j][:,:] = 0.
        
    #     G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/12),1).to(device)
    #     G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/12),1).to(device)
    #     G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/1),1).to(device)
    #     G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/1),1).to(device)
    #     G_input.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(int(G_input.nodes['solid'].data[j].size()[0]/13),1).to(device)
    #     G_input.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(int(G_input.nodes['solid'].data[j].size()[0]/13),1).to(device)

    # pred_0 = model_0(G_input.to(device))
    
    G_input_future = copy.deepcopy(G_feat[0]).to(device)
    
    for j in G_input_future.ndata.keys():
        
        G_input_future.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input_future.nodes['loop'].data[j].size()[0]/12),1).to(device)
        G_input_future.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input_future.nodes['loop'].data[j].size()[0]/12),1).to(device)
        G_input_future.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input_future.nodes['core'].data[j].size()[0]/1),1).to(device)
        G_input_future.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input_future.nodes['core'].data[j].size()[0]/1),1).to(device)
        G_input_future.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(int(G_input_future.nodes['solid'].data[j].size()[0]/13),1).to(device)
        G_input_future.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(int(G_input_future.nodes['solid'].data[j].size()[0]/13),1).to(device)

    # for j in G_input_future.ndata.keys():
    #     G_input_future.nodes['loop'].data[j] = pred_0['loop'][j]
    #     G_input_future.nodes['core'].data[j] = pred_0['core'][j]
    #     G_input_future.nodes['solid'].data[j] = pred_0['solid'][j]
        
    pred_1 = model_1(G_input_future.to(device))
    # pred_1 = model_1(G_feat[0].to(device))

    pred_1['core']['t0'] *= mean_std['core_std'].to(device)
    pred_1['core']['t0'] += mean_std['core_mean'].to(device)
    pred_1['loop']['t0'] *= mean_std['loop_std'].to(device)
    pred_1['loop']['t0'] += mean_std['loop_mean'].to(device)
    pred_1['solid']['t0'] *= mean_std['solid_std'].to(device)
    pred_1['solid']['t0'] += mean_std['solid_mean'].to(device)
    
    core_T_fuel = np.append(core_T_fuel, pred_1['core']['t0'][0,1].detach().numpy())
    core_T_surface = np.append(core_T_fuel, pred_1['core']['t0'][0,2].detach().numpy())  
    
    active_G = np.append(active_G, pred_1['loop']['t0'][2,0].detach().numpy())
    pump_G = np.append(pump_G, pred_1['loop']['t0'][7,0].detach().numpy())
    IHX_P_G = np.append(IHX_P_G, pred_1['loop']['t0'][9,0].detach().numpy())
        
    active_p = np.append(active_p, pred_1['loop']['t0'][2,1].detach().numpy())
    pump_p = np.append(pump_p, pred_1['loop']['t0'][7,1].detach().numpy())
    IHX_P_p = np.append(IHX_P_p, pred_1['loop']['t0'][9,1].detach().numpy())
      
    active_T = np.append(active_T, pred_1['loop']['t0'][2,2].detach().numpy())
    pump_T = np.append(pump_T, pred_1['loop']['t0'][7,2].detach().numpy())
    IHX_P_T = np.append(IHX_P_T, pred_1['loop']['t0'][9,2].detach().numpy())
       
    active_hs_T = np.append(active_hs_T, pred_1['solid']['t0'][5].detach().numpy())



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

t = np.linspace(0.,488., 245)

# sam_files = glob('C:/Users/Yang/Box/ProgramDevelopment/ebr2_sample/transients/ebr2_3chan_powerDrop/workdir.*/ebr2_3chan_powerDrop_csv.csv')
data_org = pd.read_csv('C:/Users/Yang/Box/2022 Summer Projects/gFHR-DT/Simulation Results/IHX_Operational_transients/IHX1/gFHR/workdir.1/PB-FHR-multi-ss.csv')

# fig,ax = plt.subplots(1,1)
# ax.plot(t, Ch_A_T, color='darkblue', lw=1.5)
# ax.plot(data_org['time'][401:772],data_org['CHA_outlet_temperature'][401:772])

# from matplotlib.animation import FuncAnimation
# from IPython import display
# from IPython.display import HTML
# import base64
# import matplotlib.animation as animation

time = data_org['time'][406:651].to_numpy()
data_active_T = data_org['active-CTbar'][406:651].to_numpy()
data_core_T_fuel = data_org['MaxFuel'][406:651].to_numpy()
data_IHX_P_T = data_org['HX-STbar'][406:651].to_numpy()
data_active_hs_T = data_org['active-R-HSbar'][406:651].to_numpy()

fig,ax = plt.subplots(2,2)

ax[0,0].plot(time, active_T,'r-', label='DT prediction')
ax[0,0].plot(time, data_active_T,'r--',label='original SAM results')
ax[0,0].set_xlabel('time')
ax[0,0].set_ylabel('active coolant T')
ax[0,1].plot(time, core_T_fuel,'r-')
ax[0,1].plot(time, data_core_T_fuel,'r--')
ax[0,1].set_xlabel('time')
ax[0,1].set_ylabel('maximum fuel T')
ax[1,0].plot(time, IHX_P_T,'r-')
ax[1,0].plot(time, data_IHX_P_T,'r--')
ax[1,0].set_xlabel('time')
ax[1,0].set_ylabel('IHX outlet T')
ax[1,1].plot(time, active_hs_T, 'r-')
ax[1,1].plot(time, data_active_hs_T, 'r--')
ax[1,1].set_xlabel('time')
ax[1,1].set_ylabel('active heat structure T')
ax[0,0].legend(bbox_to_anchor=(1.2, 1.2), loc='center')


# fig = plt.figure(figsize=(3,3))
# graph, = plt.plot([], [],'r-')
# plt.xlim(-10, 300)
# plt.ylim(0, 52)
# plt.xlabel('time, s')
# plt.ylabel('power, MW')
# plt.title('power measuerment: channel A')
# def animate(i):
#     graph.set_data(time[:i+1], Ch_A_P[:i+1]*1e-6)
#     return graph
# ani = FuncAnimation(fig, animate, frames=310, interval=50)
# plt.tight_layout()
# plt.show()
# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=30) 
# ani.save('powerMeansurement_A.mp4', writer=writervideo)
# plt.close()



# fig = plt.figure(figsize=(3,3))
# graph, = plt.plot([], [],'b-')
# plt.xlim(-10, 300)
# plt.ylim(420, 480)
# plt.xlabel('time, s')
# plt.ylabel('flow rate, kg/s')
# plt.title('flow measuerment: hot leg')
# def animate(i):
#     graph.set_data(time[:i+1], CV_3_G[:i+1])
#     return graph
# ani = FuncAnimation(fig, animate, frames=310, interval=50)
# plt.tight_layout()
# plt.show()
# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=30) 
# ani.save('flowMeansurement_CV3.mp4', writer=writervideo)
# plt.close()


# fig = plt.figure(figsize=(3.5,3.5))
# # graph_true, = plt.plot([], [],'r-')
# # graph_pred, = plt.plot([], [],'r--')
# # graph_uq, = plt.fill_between([],[],[],color='lightgray',alpha=0.5)
# plt.xlim(-10, 300)
# plt.ylim(600, 800)
# plt.xlabel('time, s')
# plt.ylabel('temperature, K')
# plt.title('Core max T_fuel prediction')
# def animate(i):
#     # graph_true.set_data(time[:i+1],Ch_P_outlet_T[:i+1])
#     # graph_pred.set_data(t[:i+1], Ch_P_T[:i+1])
#     graph_true = plt.plot(time[:i+1],Ch_P_T_fuel_sam[:i+1],'r-')
#     graph_pred =  plt.plot(t[:i+1],Ch_P_T_fuel[:i+1],'r--')
#     graph_uq = plt.fill_between(t[:i+1], 0.995*Ch_P_T_fuel[:i+1], 1.005*Ch_P_T_fuel[:i+1],color='lightgray',alpha=0.5, label='prediction uncertainty')
#     plt.legend(['ground truth','prediction','predictive uncertainty'])
#     return graph_true,graph_pred,graph_uq
# ani = FuncAnimation(fig, animate, frames=310, interval=50)
# plt.tight_layout()
# plt.show()
# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=30) 
# ani.save('Core_max_T_fuel_P.mp4', writer=writervideo)
# plt.close()


# fig = plt.figure(figsize=(3.5,3.5))
# # graph_true, = plt.plot([], [],'r-')
# # graph_pred, = plt.plot([], [],'r--')
# # graph_uq, = plt.fill_between([],[],[],color='lightgray',alpha=0.5)
# plt.xlim(-10, 300)
# plt.ylim(600, 720)
# plt.xlabel('time, s')
# plt.ylabel('temperature, K')
# plt.title('Core outlet coolant teperature')
# def animate(i):
#     # graph_true.set_data(time[:i+1],Ch_P_outlet_T[:i+1])
#     # graph_pred.set_data(t[:i+1], Ch_P_T[:i+1])
#     graph_true = plt.plot(time[:i+1],Ch_P_outlet_T[:i+1],'r-')
#     graph_pred =  plt.plot(t[:i+1],Ch_P_T[:i+1],'r--')
#     graph_uq = plt.fill_between(t[:i+1], 0.995*Ch_P_T[:i+1], 1.005*Ch_P_T[:i+1],color='lightgray',alpha=0.5, label='prediction uncertainty')
#     plt.legend(['ground truth','prediction','predictive uncertainty'])
#     return graph_true,graph_pred,graph_uq
# ani = FuncAnimation(fig, animate, frames=310, interval=50)
# plt.tight_layout()
# plt.show()
# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=30) 
# ani.save('CoreOutlet_T_P.mp4', writer=writervideo)
# # writergif = animation.PillowWriter(fps=30)
# # ani.save('CoreOutlet_T_P.gif', writer=writergif)
# plt.close()




# fig,ax = plt.subplots(1,1)
# ax.plot(t, Ch_A_P*1e-6, color='red', lw=1.5)
# ax.set_xlabel('time, s')
# ax.set_ylabel('power, MW')
# ax.set_title('power measuerment: channel A')
# #ax.plot(data_org['time'][401:772],data_org['CHA_outlet_temperature'][401:772])

# fig,ax = plt.subplots(1,1)
# ax.plot(t, Ch_B_T, color='darkblue', lw=1.5)
# ax.plot(data_org['time'][401:772],data_org['CHB_outlet_temperature'][401:772])

# fig,ax = plt.subplots(1,1)
# ax.plot(t, Ch_P_T, color='darkblue', lw=1.5)
# ax.plot(data_org['time'][401:772],data_org['CHP_outlet_temperature'][401:772])
