# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:13:43 2022

@author: Yang
"""
import dgl
import torch
from dgl.data.utils import load_graphs

from glob import glob
from model.model import NodeFuturePredictor, NodeSameTimePredictor
from utils.HTGDataset import EBRDataset

device = torch.device('cpu')
graph_list = glob('../data/processed/powerDrop*.bin')
graph_template, _ = load_graphs(graph_list[0])

n_hid = 32
n_input = {'loop':3, 'core':2, 'pump':1}
model_0 = NodeSameTimePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)
model_1 = NodeFuturePredictor((graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device))

model_0.load_state_dict(torch.load('powerDrop_dist_2parts_test/saved_model/model_0_epoch3000.pth', map_location=torch.device('cpu')),strict=False)
model_0.eval()
model_1.load_state_dict(torch.load('powerDrop_dist_2parts_test/saved_model/model_1_epoch3000.pth', map_location=torch.device('cpu')),strict=False)
model_1.eval()

mean_std = torch.load('mean_std.pt')

CV_1_P = []
CV_1_T = []
CV_1_G = []

CV_2_P = []
CV_2_T = []
CV_2_G = []

CV_3_P = []
CV_3_T = []
CV_3_G = []

CV_4_P = []
CV_4_T = []
CV_4_G = []

IHX_G = []
IHX_T = []
IHX_P = []

Ch_A_G = []
Ch_A_T = []
Ch_A_P = []

Ch_B_G = []
Ch_B_T = []
Ch_B_P = []

Core_P_G = []
Core_P_T = []
Core_P_P = []

G_feat, _ = load_graphs(graph_list[0])
G_input = copy.deepcopy(G_feat)

for j in G_input.ndata.keys():
    G_input.nodes['loop'].data[j][(0,1,5,6,8,9),:] = 0
    G_input.nodes['core'].data[j][:,1] = 0
    G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
    G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
    G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
    G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
    G_input.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)
    G_input.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)

pred_0 = model_0(G_input)

for j in G_feat.ndata.keys():
    G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
    G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
    G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
    G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
    G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)
    G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)

pred_1 = model_1(G_feat)

# initialize feature prediction
CV_1_G.append()


for i in graph_list:
        G_feat, _ = load_graphs(i)
        G_input = copy.deepcopy(G_feat)

        for j in G_input.ndata.keys():
            G_input.nodes['loop'].data[j][(0,1,5,6,8,9),:] = 0
            G_input.nodes['core'].data[j][:,1] = 0
            G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
            G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_input.nodes['loop'].data[j].size()[0]/11),1).to(device)
            G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
            G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_input.nodes['core'].data[j].size()[0]/3),1).to(device)
            G_input.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)
            G_input.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_input.nodes['pump'].data[j].size()[0]/2),1).to(device)

        pred_0 = model_0(G_input)

        for j in G_feat.ndata.keys():
            G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
            G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(int(G_feat.nodes['loop'].data[j].size()[0]/11),1).to(device)
            G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
            G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(int(G_feat.nodes['core'].data[j].size()[0]/3),1).to(device)
            G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)
            G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(int(G_feat.nodes['pump'].data[j].size()[0]/2),1).to(device)

        pred_1 = model_1(G_input)

        CV_1_G.append()
