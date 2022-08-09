#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:23:12 2022

@author: yang.liu
"""
import dgl 
import torch
from glob import glob
import pandas as pd
import re
import numpy as np
# from dgl.data import DGLDataset

# class powerDropDataset(DGLDataset):

sam_files = glob('ebr2_3chan_powerDrop/workdir.*/ebr2_3chan_powerDrop_csv.csv')
dakota_files = glob('ebr2_3chan_powerDrop/workdir.*/params.in')    

pre_step = [-17.0,-15.0,-13.0,-11.0,-9.0,-7.0,-6.0,-5.0,-3.0,-1.0]
time_step = np.linspace(0,300,num=301).tolist()
time_step = pre_step + time_step

for i in range(len(sam_files)):
    data_org = pd.read_csv(sam_files[i]).round(decimals=4)
    params = open(dakota_files[i],'r').read().splitlines()
    
    data_org = data_org[data_org['time'].isin(time_step)].reset_index()

    IHX_sec_T = round(float(re.search("=  (.*?) }",params[4]).group(1)),4)
    IHX_sec_v = round(float(re.search("= -(.*?) }",params[5]).group(1)),4)
    
    seq_length = data_org['CV1:velocity'].values.shape[0]
    
    hetero_data = []
    
    for j in range(seq_length):
        
        hetero_dict = {
            ('loop','flow','loop') : (torch.tensor([0,0,1,2,3,4,5,6,8,8,9,9,10,10]), torch.tensor([2,4,3,5,5,5,6,8,9,10,0,1,0,1])),
            
            ('loop','heatLiquid','loop') : (torch.tensor([6]), torch.tensor([7])),
            
            ('loop','heatLiquidRev','loop') : (torch.tensor([7]), torch.tensor([6])),
            
            ('loop', 'heatSolidRev', 'core') : (torch.tensor([2,3,4]), torch.tensor([0,1,2])),
            
            ('core', 'heatSolid', 'loop') : (torch.tensor([0,1,2]), torch.tensor([2,3,4])),
            
            ('core', 'heatSource', 'core') : (torch.tensor([0,1,2]), torch.tensor([0,1,2])),
                                        
            ('pump', 'momentum', 'loop') : (torch.tensor([0,1]), torch.tensor([9,10])),
            
            ('loop', 'momentumRev', 'pump') : (torch.tensor([9,10]), torch.tensor([0,1]))
            }

        g = dgl.heterograph(hetero_dict)
        
        loop_node_0 = torch.tensor([data_org['CV1:velocity'][j]*data_org['CV1:rho'][j]*1.5592, 
                                    data_org['CV1:pressure'][j], data_org['CV1:pressure'][j],
                                    data_org['CV1:temperature'][j], data_org['CV1:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_1 = torch.tensor([data_org['CV2:velocity'][j]*data_org['CV2:rho'][j]*1.5592, 
                                    data_org['CV2:pressure'][j], data_org['CV2:pressure'][j],
                                    data_org['CV2:temperature'][j], data_org['CV2:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_2 = torch.tensor([data_org['CHA_outlet_flow'][j], 
                                    data_org['CV1:pressure'][j], data_org['CV3:pressure'][j],
                                    data_org['CV1:temperature'][j], data_org['CHA_outlet_temperature'][j]]).unsqueeze(0).float()
    
        loop_node_3 = torch.tensor([data_org['CHB_outlet_flow'][j], 
                                    data_org['CV2:pressure'][j], data_org['CV3:pressure'][j],
                                    data_org['CV2:temperature'][j], data_org['CHB_outlet_temperature'][j]]).unsqueeze(0).float()
    
        loop_node_4 = torch.tensor([data_org['CHP_outlet_flow'][j], 
                                    data_org['CV1:pressure'][j], data_org['CV3:pressure'][j],
                                    data_org['CV1:temperature'][j], data_org['CHP_outlet_temperature'][j]]).unsqueeze(0).float()
    
        loop_node_5 = torch.tensor([data_org['CV3:velocity'][j]*data_org['CV3:rho'][j]*2.6656, 
                                    data_org['CV3:pressure'][j], data_org['CV3:pressure'][j],
                                    data_org['CV3:temperature'][j], data_org['CV3:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_6 = torch.tensor([data_org['IHX_primaryflow'][j], 
                                   data_org['HX-primary_in:pressure'][j], data_org['HX-primary_out:pressure'][j],
                                   data_org['HX-primary_in:temperature'][j], data_org['HX-primary_out:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_7 = torch.tensor([IHX_sec_v*866*0.4267, 
                                    2e5, 2e5,
                                    IHX_sec_T, data_org['IHX_Soutlet_temperature'][j]]).unsqueeze(0).float()
        
        loop_node_8 = torch.tensor([data_org['IHX_primaryflow'][j],
                                    data_org['CV4:pressure'][j], data_org['CV4:pressure'][j],
                                    data_org['CV4:temperature'][j], data_org['CV4:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_9 = torch.tensor([data_org['Pump1_flow'][j],
                                    data_org['CV4:pressure'][j], data_org['E12-pump:pressure'][j],
                                    data_org['E12-pump:temperature'][j], data_org['E12-pump:temperature'][j]]).unsqueeze(0).float()
    
        loop_node_10 = torch.tensor([data_org['Pump2_flow'][j],
                                    data_org['CV4:pressure'][j], data_org['E23-pump:pressure'][j],
                                    data_org['E23-pump:temperature'][j], data_org['E23-pump:temperature'][j]]).unsqueeze(0).float()
    
        core_node_0 = torch.tensor([data_org['reactor:power'][j]*0.8999, data_org['max_Tf-ChA'][j]]).unsqueeze(0).float()
    
        core_node_1 = torch.tensor([data_org['reactor:power'][j]*9.2268e-02, data_org['max_Tf-ChB'][j]]).unsqueeze(0).float()
    
        core_node_2 = torch.tensor([data_org['reactor:power'][j]*7.8273e-03, data_org['max_Tf-ChP'][j]]).unsqueeze(0).float()
    
        pump_node_0 = torch.tensor([data_org['E12-pump:pump_head'][j]]).unsqueeze(0).float()
    
        pump_node_1 = torch.tensor([data_org['E23-pump:pump_head'][j]]).unsqueeze(0).float()
        
        # loop_tensor = torch.cat((loop_node_0, loop_node_1, loop_node_2, loop_node_3,
        #                         loop_node_4, loop_node_5, loop_node_6, loop_node_7,
        #                         loop_node_8, loop_node_9, loop_node_10),0)
        
        # core_tensor = torch.cat((core_node_0, core_node_1, core_node_2),0)
        
        # pump_tensor = torch.cat((pump_node_0, pump_node_1),0).float()
        
        g.nodes['loop'].data['feat'] = torch.cat((loop_node_0, loop_node_1, loop_node_2, loop_node_3,
                                loop_node_4, loop_node_5, loop_node_6, loop_node_7,
                                loop_node_8, loop_node_9, loop_node_10),0)[:,[0,1,4]]
        g.nodes['core'].data['feat'] = torch.cat((core_node_0, core_node_1, core_node_2),0)
        g.nodes['pump'].data['feat'] = torch.cat((pump_node_0, pump_node_1),0).float()
        
  #       g.nodes['loop'].data['massFlow'] = loop_tensor[:,0].unsqueeze(-1)
  #       g.nodes['loop'].data['p'] = loop_tensor[:,1].unsqueeze(-1)
  # #      g.nodes['loop'].data['p_out'] = loop_tensor[:,2]
  # #      g.nodes['loop'].data['T_in'] = loop_tensor[:,3]
  #       g.nodes['loop'].data['T'] = loop_tensor[:,4].unsqueeze(-1)
        
  #       g.nodes['core'].data['power'] = core_tensor[:,0].unsqueeze(-1)
  #       g.nodes['core'].data['maxTFuel'] = core_tensor[:,1].unsqueeze(-1)
        
  #       g.nodes['pump'].data['pumpHead'] = pump_tensor
                
        hetero_data.append(g)
        
#        del g
        
    if i < 10:
        dgl.save_graphs('../../GCN/temporal/data/powerDrop0' + str(i) + '.bin', hetero_data)
    else: 
        dgl.save_graphs('../../GCN/temporal/data/powerDrop' + str(i) + '.bin', hetero_data)