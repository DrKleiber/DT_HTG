# -*- coding: utf-8 -*-
"""
Created on Sun May 29 22:16:38 2022

@author: Yang
"""

import dgl
from glob import glob
from dgl.data.utils import load_graphs

def construct_htg_data(glist, idx, time_window_feat, time_window_target):
    sub_glist_feat = glist[idx-time_window_feat:idx]
    
    sub_glist_target = glist[idx:idx+time_window_target]

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist_feat):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
    G_feat = dgl.heterograph(hetero_dict)
    for (t, g_s) in enumerate(sub_glist_feat):
        for ntype in G_feat.ntypes:
            G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist_target):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
    G_target = dgl.heterograph(hetero_dict)
    for (t, g_s) in enumerate(sub_glist_target):
        for ntype in G_target.ntypes:
            G_target.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']
            
    return G_feat, G_target

temporalData_id = 0

graph_list = glob('./*.bin')

time_window_feat = 10
time_window_target = 5

feat_list, target_list = [], []

for j in graph_list:
    transient_data, _ = load_graphs(j)
    print('Now processing: ', j)
    for i in range(len(transient_data)):
#         if i >= time_window:
        if i >= time_window_feat and i <= len(transient_data) - time_window_target:
            G_feat, G_target = construct_htg_data(transient_data, i, time_window_feat, time_window_target)
            
#             feat_list.append(G_feat)
#             target_list.append(G_target)

# dgl.save_graphs('./processed/powerDrop_feat.bin',feat_list)
# dgl.save_graphs('./processed/powerDrop_target.bin',target_list)

            
            if temporalData_id< 10:
                dgl.save_graphs('./processed/test_0000' + str(temporalData_id) + '.bin', [G_feat, G_target])
            elif temporalData_id>=10 and temporalData_id<100:
                dgl.save_graphs('./processed/test_000' + str(temporalData_id) + '.bin', [G_feat, G_target])
            elif temporalData_id >=100 and temporalData_id<1000:
                dgl.save_graphs('./processed/test_00' + str(temporalData_id) + '.bin', [G_feat, G_target])
            elif temporalData_id >= 1000 and temporalData_id < 10000:
                dgl.save_graphs('./processed/test_0' + str(temporalData_id) + '.bin', [G_feat, G_target])
            else:
                dgl.save_graphs('./processed/test_' + str(temporalData_id) + '.bin', [G_feat, G_target])
            
            temporalData_id += 1
