# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:53:51 2022

@author: Yang
"""
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

class HGTDataset(DGLDataset):
    # def __init__(self):
    #     super().__init__()
    #     self.graph_feat = load_graphs('./data/processed/powerDrop_feat.bin')
    #     self.graph_target = load_graphs('./data/processed/powerDrop_target.bin')    
    #     if len(self.graph_feat) != len(self.graph_target) :
    #         raise ValueError('length of input list and output list are not the same!')
            
    # def process(self):
    #     pass
    
    # def __getitem__(self, i): 
    #     feat = self.graph_feat[i]
    #     target = self.graph_target[i]
    #     return feat, target
    
    # def __len__(self):
    #     return len(self.graph_feat)

    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list

    def process(self):
        pass
        
    def __getitem__(self, i):
        graph, _ = load_graphs(self.graph_list[i])
        self.feat = graph[0]
        self.target = graph[1]
        return self.feat, self.target

    def __len__(self):
        return len(self.graph_list)
