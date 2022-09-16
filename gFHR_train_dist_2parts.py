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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model.model_dualwindow import NodeFuturePredictor, NodeSameTimePredictor
from utils.HTGDataset import SAMDataset
from utils.pytorchtools import EarlyStopping

import os
import numpy as np
from glob import glob
import random
import pickle
import copy

def main(rank, world_size, graph_list, seed=0):
    init_process_group(world_size, rank)
    print('distributed training initiated with {} gpus'.format(world_size))
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    model_0 = init_model_current(seed, graph_list[0], device)
    model_1 = init_model_future(seed, graph_list[0], device)

    mean_std = torch.load('mean_std.pt')
    
    full_index = range(len(graph_list))
    train_index = random.sample(range(len(graph_list)),int(0.7*len(graph_list)))
    test_index = list(set(full_index) - set(train_index))
    
    graph_list_train = [graph_list[i] for i in sorted(train_index)]
    graph_list_test = [graph_list[i] for i in sorted(test_index)]
    
    train_dataset = SAMDataset(graph_list_train)
    test_dataset = SAMDataset(graph_list_test)
    
    batch_size = 512
    epochs = 1000

    ckpt_freq = 200
    log_freq = 1
    ckpt_dir = './cases/gFHR_test/saved_model'
    log_dir = './cases/gFHR_test'
    
    optim_0 = torch.optim.Adam(model_0.parameters(), lr=2e-4, weight_decay=5e-4)
    optim_1 = torch.optim.Adam(model_1.parameters(), lr=2e-4, weight_decay=5e-4)
        
    kwargs = {'num_workers': 8,
                  'pin_memory': True} if torch.cuda.is_available() else {}   

    train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=True, use_ddp = True, **kwargs)
    test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=batch_size,shuffle=True, drop_last=True, use_ddp = True, **kwargs)
    
    print('data loaded!')

    logger = {}
    logger['rmse_0_train'] = []
    logger['rmse_0_test'] = []
    logger['rmse_1_train'] = []
    logger['rmse_1_test'] = []
    
    print('Start training........................................................')
    for epoch in range(epochs + 1):
        train_loader.set_epoch(epoch)
        model_0.train()
        model_1.train()
        mse_0 = 0.
        mse_1 = 0.

        for _, (G_feat, G_target) in enumerate(train_loader):
    
            G_feat, G_target = G_feat.to(device), G_target.to(device)
            
            # zero out unknown node information
            # first part of the model is to infer node feature of the current time window
            G_input = copy.deepcopy(G_feat)
            
            for j in G_input.ndata.keys():
                G_input.nodes['loop'].data[j][(0,1,2,4,5,7,8,9,11),:] = 0.
                G_input.nodes['core'].data[j][:,(1,2)] = 0.
                G_input.nodes['solid'].data[j] = 0.
                G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                G_input.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                G_input.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
    
            for j in G_feat.ndata.keys():
                G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                G_feat.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                G_feat.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
                
            model_0.zero_grad()
            pred_0 = model_0(G_input)

            loss_0 = 0.
            for j in G_feat.ndata.keys():
                loss_0 += F.mse_loss(pred_0['loop'][j], G_feat.nodes['loop'].data[j], reduction = 'sum')
                loss_0 += F.mse_loss(pred_0['solid'][j], G_feat.nodes['solid'].data[j], reduction = 'sum')
                loss_0 += F.mse_loss(pred_0['core'][j], G_feat.nodes['core'].data[j], reduction = 'sum')
    
            loss_0.backward()
            optim_0.step()
            mse_0 += loss_0.item()
    
            for j in G_target.ndata.keys():
                G_target.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                G_target.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                G_target.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                G_target.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                G_target.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                G_target.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
            
            model_1.zero_grad()
            pred_1 = model_1(G_feat)
            
            loss_1 = 0.
            for j in G_target.ndata.keys():
                loss_1 += F.mse_loss(pred_1['loop'][j], G_target.nodes['loop'].data[j], reduction = 'sum')
                loss_1 += F.mse_loss(pred_1['solid'][j], G_target.nodes['solid'].data[j], reduction = 'sum')
                loss_1 += F.mse_loss(pred_1['core'][j], G_target.nodes['core'].data[j], reduction = 'sum')
            
            loss_1.backward()
            optim_1.step()
            mse_1 += loss_1.item()
                
        mse_0 = mse_0/(3*17+3*1+1*13)/len(graph_list_train)/10
        mse_1 = mse_1/(3*17+3*1+1*13)/len(graph_list_train)/5
    
        rmse_0 = np.sqrt(mse_0)
        rmse_1 = np.sqrt(mse_1)
        print("epoch: {}, train rmse, current time window: {:.6f}".format(epoch, rmse_0))
        print("epoch: {}, train rmse, future_state: {:.6f}".format(epoch, rmse_1))

        if (epoch) % ckpt_freq == 0 and epoch != 0:
            torch.save(model_0.state_dict(), ckpt_dir + "/model_0_epoch{}.pth".format(epoch))
            torch.save(model_1.state_dict(), ckpt_dir + "/model_1_epoch{}.pth".format(epoch))

        if epoch % log_freq == 0:
            logger['rmse_0_train'].append(rmse_0)
            logger['rmse_1_train'].append(rmse_1)
            f = open(log_dir + '/' + 'rmse_0_train.pkl',"wb")
            pickle.dump(logger['rmse_0_train'],f)
            f.close()
            f = open(log_dir + '/' + 'rmse_1_train.pkl',"wb")
            pickle.dump(logger['rmse_1_train'],f)
            f.close()
            
        with torch.no_grad():
            model_0.eval()
            model_1.eval()
            mse_0 = 0.
            mse_1 = 0.
            for _, (G_feat, G_target) in enumerate(test_loader):
        
                G_feat, G_target = G_feat.to(device), G_target.to(device)
                
                # zero out unknown node information
                # first part of the model is to infer node feature of the current time window
                G_input = copy.deepcopy(G_feat)
                
                for j in G_input.ndata.keys():
                    G_input.nodes['loop'].data[j][(0,1,2,4,5,7,8,9,11),:] = 0.
                    G_input.nodes['core'].data[j][:,(1,2)] = 0.
                    G_input.nodes['solid'].data[j] = 0.
                    G_input.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                    G_input.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                    G_input.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                    G_input.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                    G_input.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                    G_input.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
        
                for j in G_feat.ndata.keys():
                    G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                    G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                    G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                    G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                    G_feat.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                    G_feat.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
                    
                pred_0 = model_0(G_input)

                loss_0 = 0.
                for j in G_feat.ndata.keys():
                    loss_0 += F.mse_loss(pred_0['loop'][j], G_feat.nodes['loop'].data[j], reduction = 'sum')
                    loss_0 += F.mse_loss(pred_0['solid'][j], G_feat.nodes['solid'].data[j], reduction = 'sum')
                    loss_0 += F.mse_loss(pred_0['core'][j], G_feat.nodes['core'].data[j], reduction = 'sum')
        
                mse_0 += loss_0.item()
        
                for j in G_target.ndata.keys():
                    G_target.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
                    G_target.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
                    G_target.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
                    G_target.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
                    G_target.nodes['solid'].data[j] -= mean_std['solid_mean'].repeat(batch_size,1).to(device)
                    G_target.nodes['solid'].data[j] /= mean_std['solid_std'].repeat(batch_size,1).to(device)
                
                pred_1 = model_1(G_feat)
                
                loss_1 = 0.
                for j in G_target.ndata.keys():
                    loss_1 += F.mse_loss(pred_1['loop'][j], G_target.nodes['loop'].data[j], reduction = 'sum')
                    loss_1 += F.mse_loss(pred_1['solid'][j], G_target.nodes['solid'].data[j], reduction = 'sum')
                    loss_1 += F.mse_loss(pred_1['core'][j], G_target.nodes['core'].data[j], reduction = 'sum')
                
                loss_1.backward()
                optim_1.step()
                mse_1 += loss_1.item()
                    
            mse_0 = mse_0/(3*17+3*1+1*13)/len(graph_list_train)/10
            mse_1 = mse_1/(3*17+3*1+1*13)/len(graph_list_train)/5
            
            rmse_0 = np.sqrt(mse_0)
            rmse_1 = np.sqrt(mse_1)
            print("epoch: {}, test rmse, current time window: {:.6f}".format(epoch, rmse_0))
            print("epoch: {}, test rmse, future_state: {:.6f}".format(epoch, rmse_1))

            if epoch % log_freq == 0:
                logger['rmse_0_test'].append(rmse_0)
                logger['rmse_1_test'].append(rmse_1)
                f = open(log_dir + '/' + 'rmse_0_test.pkl',"wb")
                pickle.dump(logger['rmse_0_test'],f)
                f.close()
                f = open(log_dir + '/' + 'rmse_1_test.pkl',"wb")
                pickle.dump(logger['rmse_1_test'],f)
                f.close()

    dist.destroy_process_group()            

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl',     # change to 'nccl' for multiple GPUs
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank)

def init_model_current(seed, graph_name, device):
    torch.manual_seed(seed)
    graph_template, _ = load_graphs(graph_name)
    n_hid = 32
    n_input = {'loop':3, 'core':3, 'solid':1}
    model = NodeSameTimePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)
    model = model.to(device)
    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    return model


def init_model_future(seed, graph_name, device):
    torch.manual_seed(seed)
    graph_template, _ = load_graphs(graph_name)
    n_hid = 32
    n_input = {'loop':3, 'core':3, 'solid':1}
    model = NodeFuturePredictor(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window_inp=10, time_window_tar=5, norm=False, device = device)
    model = model.to(device)
    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    return model


if __name__ == '__main__':
   import torch.multiprocessing as mp

   num_gpus = 6
   procs = []
   graph_list = glob('../data/gFHR/processed/*.bin')
   mp.spawn(main, args=(num_gpus, graph_list), nprocs=num_gpus)
