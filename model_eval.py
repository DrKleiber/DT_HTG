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
model = NodeSameTimePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)