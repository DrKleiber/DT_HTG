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

# sam_files = glob('Simulation Results/Power_Operational_transients/Power_transient*/gFHR/workdir.*/PB-FHR-multi-ss.csv')
sam_files = glob('PB-FHR-multi-ss.csv')
# dakota_files = glob('ebr2_3chan_powerDrop/workdir.*/params.in')

# pre_step = [-17.0,-15.0,-13.0,-11.0,-9.0,-7.0,-6.0,-5.0,-3.0,-1.0]
# time_step = np.linspace(0,300,num=301).tolist()
# time_step = pre_step + time_step

# time_step = 

for i in range(len(sam_files)):
    data_org = pd.read_csv(sam_files[i]).round(decimals=4)

    data_org = data_org.loc[data_org['time']>=100].reset_index()

    seq_length = data_org['time'].values.shape[0]

    hetero_data = []

    for j in range(seq_length):

        hetero_dict = {
            ('loop','flow','loop') : (torch.tensor([0,1,2,3,4,5,6,7,8,9, 11,12,13,14,15,16]), 
                                      torch.tensor([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,0 ])),

            ('loop','liquid_heat_liquid','loop') : (torch.tensor([9,10]), torch.tensor([10,9])),
            
            ('solid','solid_heat_liquid','loop') : (torch.tensor([0,1,2,3,4,5,6,7,3 ,4 ,5 ,6 ,7 , 8 ,9,10,11,12]), 
                                                    torch.tensor([0,1,3,0,1,2,3,4,16,15,14,13,12, 12,13,14,15,16])),
            
            ('loop','liquid_heat_solid','solid') : (torch.tensor([0,1,3,0,1,2,3,4,16,15,14,13,12, 12,13,14,15,16]),
                                                    torch.tensor([0,1,2,3,4,5,6,7,3 ,4 ,5 ,6 ,7 , 8 ,9,10,11,12])),

            ('core', 'core_heat_liquid', 'loop') : (torch.tensor([0]), torch.tensor([2])),
            
            ('loop', 'liquid_heat_core', 'core') : (torch.tensor([2]), torch.tensor([0])),

            ('core', 'heatSource', 'core') : (torch.tensor([0]), torch.tensor([0])),

            }
        
        g = dgl.heterograph(hetero_dict)


        loop_node_0 = torch.tensor([data_org['Core_flow'][j], data_org['fueling-CPbar'][j], data_org['fueling-CTbar'][j]]).unsqueeze(0).float()   # fueling coolant
        loop_node_1 = torch.tensor([data_org['Core_flow'][j], data_org['expansion-CPbar'][j], data_org['expansion-CTbar'][j]]).unsqueeze(0).float()   # expansion coolant
        loop_node_2 = torch.tensor([data_org['Core_flow'][j], data_org['active-CPbar'][j], data_org['active-CTbar'][j]]).unsqueeze(0).float()   # active coolant
        loop_node_3 = torch.tensor([data_org['Core_flow'][j], data_org['contraction-CPbar'][j], data_org['contraction-CTbar'][j]]).unsqueeze(0).float()   # contraction coolant
        
        loop_node_4 = torch.tensor([data_org['pipe040-Vbar'][j]*data_org['Branch030:rho'][j]*0.47,
                                    data_org['pipe040-Pbar'][j], data_org['pipe040-Tbar'][j]]).unsqueeze(0).float() # pipe-040
        
        loop_node_5 = torch.tensor([data_org['pipe050-Vbar'][j]*data_org['Branch260:rho'][j]*0.47,
                                    data_org['pipe050-Pbar'][j], data_org['pipe050-Tbar'][j]]).unsqueeze(0).float() # pipe-050
        
        loop_node_6 = torch.tensor([data_org['pipe060-Vbar'][j]*data_org['Branch501:rho'][j]*0.47,
                                    data_org['pipe060-Pbar'][j], data_org['pipe060-Tbar'][j]]).unsqueeze(0).float() # pipe-060
        
        loop_node_7 = torch.tensor([data_org['Pump:velocity'][j]*data_org['Pump:rho'][j]*0.47,
                                    data_org['Pump:pressure'][j], data_org['Pump:temperature'][j]]).unsqueeze(0).float() # pump
        
        loop_node_8 = torch.tensor([data_org['pipe070-Vbar'][j]*data_org['Branch601:rho'][j]*0.47,
                                    data_org['pipe070-Pbar'][j], data_org['pipe070-Tbar'][j]]).unsqueeze(0).float() # pipe-070
        
        loop_node_9 = torch.tensor([data_org['pipe070-Vbar'][j]*data_org['Branch601:rho'][j]*0.47,
                                   data_org['HX-SPbar'][j], data_org['HX-STbar'][j]]).unsqueeze(0).float()
        
        loop_node_10 = torch.tensor([data_org['secondary-flow'][j],
                                    data_org['HX-PPbar'][j],
                                    data_org['HX-PTbar'][j]]).unsqueeze(0).float()
        
        loop_node_11 = torch.tensor([data_org['pipe070-Vbar'][j]*data_org['Branch605:rho'][j]*0.47,
                                    data_org['pipe070-Pbar'][j], data_org['pipe070-Tbar'][j]]).unsqueeze(0).float() # pipe-110
        
        loop_node_12 = torch.tensor([data_org['down-40-Vbar'][j]*data_org['Branch607:rho'][j]*0.58,
                                    data_org['down-40-Pbar'][j], data_org['down-40-Tbar'][j]]).unsqueeze(0).float() # down-40
        
        loop_node_13 = torch.tensor([data_org['down-cont-Vbar'][j]*data_org['down-chain:1:junction:rho'][j]*0.58,
                                    data_org['down-cont-Pbar'][j], data_org['down-cont-Tbar'][j]]).unsqueeze(0).float() # down-cont
        
        loop_node_14 = torch.tensor([data_org['down-active-Vbar'][j]*data_org['down-chain:2:junction:rho'][j]*0.58,
                                    data_org['down-active-Pbar'][j], data_org['down-active-Tbar'][j]]).unsqueeze(0).float() # down-active
        
        loop_node_15 = torch.tensor([data_org['down-expan-Vbar'][j]*data_org['down-chain:3:junction:rho'][j]*0.58,
                                    data_org['down-expan-Pbar'][j], data_org['down-expan-Tbar'][j]]).unsqueeze(0).float() # down-expan
        
        loop_node_16 = torch.tensor([data_org['down-fuel-Vbar'][j]*data_org['Branch280:rho'][j]*0.58,
                                    data_org['down-fuel-Pbar'][j], data_org['down-fuel-Tbar'][j]]).unsqueeze(0).float() # down-fueling
        
        core_node = torch.tensor([data_org['reactor:power'][j], data_org['MaxFuel'][j], data_org['MaxSurface'][j]]).unsqueeze(0).float()
        
        solid_node_0 = torch.tensor([data_org['fueling-CTsolidbar'][j]]).unsqueeze(0).float()
        solid_node_1 = torch.tensor([data_org['expansion-CTsolidbar'][j]]).unsqueeze(0).float()
        solid_node_2 = torch.tensor([data_org['contraction-CTsolidbar'][j]]).unsqueeze(0).float()
        solid_node_3 = torch.tensor([data_org['fueling-R-HSbar'][j]]).unsqueeze(0).float()
        solid_node_4 = torch.tensor([data_org['expansion-R-HSbar'][j]]).unsqueeze(0).float()
        solid_node_5 = torch.tensor([data_org['active-R-HSbar'][j]]).unsqueeze(0).float()
        solid_node_6 = torch.tensor([data_org['contract-R-HSbar'][j]]).unsqueeze(0).float()
        solid_node_7 = torch.tensor([data_org['contract-R-HSbar'][j]]).unsqueeze(0).float()
        solid_node_8 = torch.tensor([data_org['RV-40-HSbar'][j]]).unsqueeze(0).float()
        solid_node_9 = torch.tensor([data_org['RV-cont-HSbar'][j]]).unsqueeze(0).float()
        solid_node_10 = torch.tensor([data_org['RV-active-HSbar'][j]]).unsqueeze(0).float()
        solid_node_11 = torch.tensor([data_org['RV-expan-HSbar'][j]]).unsqueeze(0).float()
        solid_node_12 = torch.tensor([data_org['RV-fuel-HSbar'][j]]).unsqueeze(0).float()
        
        g.nodes['loop'].data['feat'] = torch.cat((loop_node_0, loop_node_1, loop_node_2, loop_node_3,
                                    loop_node_4, loop_node_5, loop_node_6, loop_node_7,
                                    loop_node_8, loop_node_9, loop_node_10, loop_node_11, loop_node_12,
                                    loop_node_13, loop_node_14, loop_node_15, loop_node_16),0)
        
        g.nodes['core'].data['feat'] = core_node
        
        g.nodes['solid'].data['feat'] = torch.cat((solid_node_0, solid_node_1, solid_node_2, solid_node_3,
                                                   solid_node_4, solid_node_5, solid_node_6, solid_node_7,
                                                   solid_node_8, solid_node_9, solid_node_10, solid_node_11,
                                                   solid_node_12),0)
    
        hetero_data.append(g)
    
    dgl.save_graphs('./example.bin', hetero_data)



#         loop_node_0 = torch.tensor([data_org['CV1:velocity'][j]*data_org['CV1:rho'][j]*1.5592,
#                                     data_org['CV1:pressure'][j], data_org['CV1:pressure'][j],
#                                     data_org['CV1:temperature'][j], data_org['CV1:temperature'][j]]).unsqueeze(0).float()

#         loop_node_1 = torch.tensor([data_org['CV2:velocity'][j]*data_org['CV2:rho'][j]*1.5592,
#                                     data_org['CV2:pressure'][j], data_org['CV2:pressure'][j],
#                                     data_org['CV2:temperature'][j], data_org['CV2:temperature'][j]]).unsqueeze(0).float()

#         loop_node_2 = torch.tensor([data_org['CHA_outlet_flow'][j],
#                                     data_org['CV1:pressure'][j], data_org['CV3:pressure'][j],
#                                     data_org['CV1:temperature'][j], data_org['CHA_outlet_temperature'][j]]).unsqueeze(0).float()

#         loop_node_3 = torch.tensor([data_org['CHB_outlet_flow'][j],
#                                     data_org['CV2:pressure'][j], data_org['CV3:pressure'][j],
#                                     data_org['CV2:temperature'][j], data_org['CHB_outlet_temperature'][j]]).unsqueeze(0).float()

#         loop_node_4 = torch.tensor([data_org['CHP_outlet_flow'][j],
#                                     data_org['CV1:pressure'][j], data_org['CV3:pressure'][j],
#                                     data_org['CV1:temperature'][j], data_org['CHP_outlet_temperature'][j]]).unsqueeze(0).float()

#         loop_node_5 = torch.tensor([data_org['CV3:velocity'][j]*data_org['CV3:rho'][j]*2.6656,
#                                     data_org['CV3:pressure'][j], data_org['CV3:pressure'][j],
#                                     data_org['CV3:temperature'][j], data_org['CV3:temperature'][j]]).unsqueeze(0).float()

#         loop_node_6 = torch.tensor([data_org['IHX_primaryflow'][j],
#                                    data_org['HX-primary_in:pressure'][j], data_org['HX-primary_out:pressure'][j],
#                                    data_org['HX-primary_in:temperature'][j], data_org['HX-primary_out:temperature'][j]]).unsqueeze(0).float()

#         loop_node_7 = torch.tensor([IHX_sec_v*866*0.4267,
#                                     2e5, 2e5,
#                                     IHX_sec_T, data_org['IHX_Soutlet_temperature'][j]]).unsqueeze(0).float()

#         loop_node_8 = torch.tensor([data_org['IHX_primaryflow'][j],
#                                     data_org['CV4:pressure'][j], data_org['CV4:pressure'][j],
#                                     data_org['CV4:temperature'][j], data_org['CV4:temperature'][j]]).unsqueeze(0).float()

#         loop_node_9 = torch.tensor([data_org['Pump1_flow'][j],
#                                     data_org['CV4:pressure'][j], data_org['E12-pump:pressure'][j],
#                                     data_org['E12-pump:temperature'][j], data_org['E12-pump:temperature'][j]]).unsqueeze(0).float()

#         loop_node_10 = torch.tensor([data_org['Pump2_flow'][j],
#                                     data_org['CV4:pressure'][j], data_org['E23-pump:pressure'][j],
#                                     data_org['E23-pump:temperature'][j], data_org['E23-pump:temperature'][j]]).unsqueeze(0).float()

#         core_node_0 = torch.tensor([data_org['reactor:power'][j]*0.8999, data_org['max_Tf-ChA'][j]]).unsqueeze(0).float()

#         core_node_1 = torch.tensor([data_org['reactor:power'][j]*9.2268e-02, data_org['max_Tf-ChB'][j]]).unsqueeze(0).float()

#         core_node_2 = torch.tensor([data_org['reactor:power'][j]*7.8273e-03, data_org['max_Tf-ChP'][j]]).unsqueeze(0).float()

#         pump_node_0 = torch.tensor([data_org['E12-pump:pump_head'][j]]).unsqueeze(0).float()

#         pump_node_1 = torch.tensor([data_org['E23-pump:pump_head'][j]]).unsqueeze(0).float()

#         # loop_tensor = torch.cat((loop_node_0, loop_node_1, loop_node_2, loop_node_3,
#         #                         loop_node_4, loop_node_5, loop_node_6, loop_node_7,
#         #                         loop_node_8, loop_node_9, loop_node_10),0)

#         # core_tensor = torch.cat((core_node_0, core_node_1, core_node_2),0)

#         # pump_tensor = torch.cat((pump_node_0, pump_node_1),0).float()

#         g.nodes['loop'].data['feat'] = torch.cat((loop_node_0, loop_node_1, loop_node_2, loop_node_3,
#                                 loop_node_4, loop_node_5, loop_node_6, loop_node_7,
#                                 loop_node_8, loop_node_9, loop_node_10),0)[:,[0,1,4]]
#         g.nodes['core'].data['feat'] = torch.cat((core_node_0, core_node_1, core_node_2),0)
#         g.nodes['pump'].data['feat'] = torch.cat((pump_node_0, pump_node_1),0).float()

#   #       g.nodes['loop'].data['massFlow'] = loop_tensor[:,0].unsqueeze(-1)
#   #       g.nodes['loop'].data['p'] = loop_tensor[:,1].unsqueeze(-1)
#   # #      g.nodes['loop'].data['p_out'] = loop_tensor[:,2]
#   # #      g.nodes['loop'].data['T_in'] = loop_tensor[:,3]
#   #       g.nodes['loop'].data['T'] = loop_tensor[:,4].unsqueeze(-1)

#   #       g.nodes['core'].data['power'] = core_tensor[:,0].unsqueeze(-1)
#   #       g.nodes['core'].data['maxTFuel'] = core_tensor[:,1].unsqueeze(-1)

#   #       g.nodes['pump'].data['pumpHead'] = pump_tensor

#         hetero_data.append(g)

# #        del g

#     if i < 10:
#         dgl.save_graphs('../../GCN/temporal/data/powerDrop0' + str(i) + '.bin', hetero_data)
#     else:
#         dgl.save_graphs('../../GCN/temporal/data/powerDrop' + str(i) + '.bin', hetero_data)
