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
from utils.HTGDataset import EBRDataset
from utils.pytorchtools import EarlyStopping

from glob import glob
import random
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main(rank, world_size, graph_list, seed=0):

    init_process_group(world_size, rank)
    print('distributed training initiated!')
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    model = init_model(seed, device)

    mean_std = torch.load('mean_std.pt')
    mean_std['loop_mean'] = mean_std['loop_mean']
    mean_std['loop_std'] = mean_std['loop_std']

    full_index = range(len(graph_list))
    train_index = random.sample(range(len(graph_list)),int(0.7*len(graph_list)))
    test_index = list(set(full_index) - set(train_index))

    graph_list_train = [graph_list[i] for i in sorted(train_index)]
    graph_list_test = [graph_list[i] for i in sorted(test_index)]

    train_dataset = EBRDataset(graph_list_train)
    test_dataset = EBRDataset(graph_list_test)

    n_hid = 32
    n_input = {'loop':3, 'core':2, 'pump':1}
    n_classes = {'loop':3, 'core':2, 'pump':1}
    batch_size = 512
    epochs = 1000

    ckpt_freq = 100
    log_freq = 1
    ckpt_dir = './powerDrop_test/saved_model'
    log_dir = './powerDrop_test/'

    optim = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)

    kwargs = {'num_workers': 8,
                  'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=True, use_ddp=True, **kwargs)
    test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=batch_size,shuffle=True, drop_last=True, **kwargs)

    print('data loaded!')

    logger = {}
    logger['rmse_train'] = []
    logger['rmse_test'] = []

    print('Start training........................................................')
    for epoch in range(epochs + 1):
        train(epoch)
        with torch.no_grad():
            test(epoch)
    dist.destroy_process_group()

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl',     # change to 'nccl' for multiple GPUs
#        init_method='tcp://127.0.0.1:12345',
        world_size=world_size,
        rank=rank)

def init_model(seed, device):
    torch.manual_seed(seed)
    graph_template,_ = load_graphs(graph_list[0])
    model = HTGNN(graph=graph_template[0], n_inp=n_input, n_hid=n_hid , n_layers=2, n_heads=1, time_window=10, norm=False,device = device)
    model = model.to(device)
    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    return model

def train(epoch):
    model.train()
    train_loader.set_epoch(epoch)
    mse = 0.
    for _, (G_feat, G_target) in enumerate(train_loader):

        G_feat, G_target = G_feat.to(device), G_target.to(device)

        for j in G_feat.ndata.keys():
            G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
            G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(batch_size,1).to(device)
            G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)

        for j in G_target.ndata.keys():
            G_target.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
            G_target.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
            G_target.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(batch_size,1).to(device)

        model.zero_grad()
        pred = model(G_feat)

        loss = F.mse_loss(pred['loop'], G_target.nodes['loop'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['pump'], G_target.nodes['pump'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['core'], G_target.nodes['core'].data['feat'], reduction = 'sum')

        loss.backward()
        optim.step()
        mse += loss.item()

    mse = mse/(3*11+2*1+2*3)/len(graph_list_train)

    rmse = np.sqrt(mse)
    print("epoch: {}, train rmse: {:.6f}".format(epoch, rmse))

    if (epoch) % ckpt_freq == 0 and epoch != 0:
        torch.save(model.state_dict(), ckpt_dir + "/model_epoch{}.pth".format(epoch))

    if epoch % log_freq == 0:
#        logger['r2_train'].append(r2_train)
        logger['rmse_train'].append(rmse)
        f = open(log_dir + '/' + 'rmse_train.pkl',"wb")
        pickle.dump(logger['rmse_train'],f)
        f.close()

def test(epoch):
    model.eval()
    mse = 0.
    for _, (G_feat, G_target) in enumerate(test_loader):
        G_feat, G_target = G_feat.to(device), G_target.to(device)
        for j in G_feat.ndata.keys():
            G_feat.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
            G_feat.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(batch_size,1).to(device)
            G_feat.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
            G_feat.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)

        for j in G_target.ndata.keys():
            G_target.nodes['loop'].data[j] -= mean_std['loop_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['loop'].data[j] /= mean_std['loop_std'].repeat(batch_size,1).to(device)
            G_target.nodes['core'].data[j] -= mean_std['core_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['core'].data[j] /= mean_std['core_std'].repeat(batch_size,1).to(device)
            G_target.nodes['pump'].data[j] -= mean_std['pump_mean'].repeat(batch_size,1).to(device)
            G_target.nodes['pump'].data[j] /= mean_std['pump_std'].repeat(batch_size,1).to(device)

        pred = model(G_feat)

        loss = F.mse_loss(pred['loop'], G_target.nodes['loop'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['pump'], G_target.nodes['pump'].data['feat'], reduction = 'sum')
        loss += F.mse_loss(pred['core'], G_target.nodes['core'].data['feat'], reduction = 'sum')
        mse += loss.item()
    mse = mse/(3*11+2*1+2*3)/len(graph_list_train)
    rmse = np.sqrt(mse)
    print("epoch: {}, test rmse: {:.6f}".format(epoch, rmse))

    if epoch % log_freq == 0:
        logger['rmse_test'].append(rmse)
        f = open(log_dir + '/' + 'rmse_test.pkl',"wb")
        pickle.dump(logger['rmse_test'],f)
        f.close()

if __name__ == '__main__':
   import torch.multiprocessing as mp

   num_gpus = 2
   procs = []
   graph_list = glob('../data/processed/powerDrop*.bin')
   mp.spawn(main, args=(num_gpus, graph_list), nprocs=num_gpus)
