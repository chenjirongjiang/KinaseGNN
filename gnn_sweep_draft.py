import sys
import yaml
from gnns.data.dataset import PocketLigandDataset_test
from gnns.config import DatasetConfig, GraphConfig

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import wandb
from scipy import stats
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


def swish(x):
    return x * torch.sigmoid(x)
class GCN(torch.nn.Module):
    """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

    def __init__(self, conv_layers, lin_layers, activation, batch_normalization, dropout):
        super().__init__()

        if conv_layers[-1] != lin_layers[0]:
            raise ValueError(f'The first node size of lin_layers needs to be same as the last node size of conv_layer.')
        if lin_layers[-1] != 1:
            raise ValueError(f'The last node size of the lin_layers needs to be a 1, to form an affinity prediction.')
        self.layers = nn.ModuleList()
        if batch_normalization:
            self.batch_norms = nn.ModuleList()
        self.batch_normalization = batch_normalization
        self.conv_len = len(conv_layers)-1
        self.lin_len = len(lin_layers)-1
        self.dropout=dropout

        if activation =='relu':
            self.activation = F.relu
        elif activation =='swish':
            self.activation = swish
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = F.tanh
        
        
        for i in range(self.conv_len):
            self.layers.append(GCNConv(conv_layers[i], conv_layers[i+1]))

        for i in range(self.lin_len):
            self.layers.append(nn.Linear(lin_layers[i],lin_layers[i+1]))
            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(lin_layers[i+1]))

    def forward(self, data):
        e = data.edge_index
        x = data.x
        batch = data.batch

        # Pass data through convolution layers with activation function except last one
        for i in range(self.conv_len-1):
            x = self.layers[i](x,e)

            x = self.activation(x)

        x = self.layers[self.conv_len-1](x,e)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for j in range(self.lin_len-1):
            x = self.layers[self.conv_len+j](x)
            if self.batch_normalization:
                x = self.batch_norms[j](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)

            
        
        x = self.layers[-1](x)

        return x
    
    
class GIN(torch.nn.Module):
    """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""

    def __init__(self, conv_layers, lin_layers, activation, batch_normalization, dropout):
        super(GIN, self).__init__()

        if conv_layers[-1] != lin_layers[0]:
            raise ValueError(f'The first node size of lin_layers needs to be same as the last node size of conv_layer.')
        
        self.layers = nn.ModuleList()
        if batch_normalization:
            self.batch_norms = nn.ModuleList()
        self.batch_normalization = batch_normalization
        
        self.conv_len = len(conv_layers)-1
        self.lin_len = len(lin_layers)-1
        self.dropout = dropout
        
        if activation =='relu':
            self.activation = F.relu
        elif activation =='swish':
            self.activation = swish
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = F.tanh

        for i in range(self.conv_len):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(conv_layers[i], conv_layers[i+1]), 
                        nn.BatchNorm1d(conv_layers[i+1]), 
                        nn.ReLU(), 
                        nn.Linear(conv_layers[i+1],conv_layers[i+1]),
                        nn.ReLU()
                    )
                )
            )
            
        for i in range(self.lin_len):
            self.layers.append(nn.Linear(lin_layers[i],lin_layers[i+1]))
            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(lin_layers[i+1]))


    def forward(self, data):
        x = data.x
        e = data.edge_index
        batch = data.batch

        # Pass data through convolution layers with activation function except last one
        # No batch norm between conv layers
        for i in range(self.conv_len-1):
            x = self.layers[i](x,e)

            x = self.activation(x)
        
        x = self.layers[self.conv_len-1](x,e)

        # Graph-level readout
        x = global_add_pool(x, batch)

        for j in range(self.lin_len-1):
            x = self.layers[self.conv_len+j](x)
            if self.batch_normalization:
                x = self.batch_norms[j](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.activation(x)
        
        x = self.layers[-1](x)

        return x
    
class GAT(torch.nn.Module):
    def __init__(self, conv_layers, lin_layers, activation, batch_normalization, dropout, heads=10):
        super().__init__()

        if conv_layers[-1] != lin_layers[0]:
            raise ValueError(f'The first node size of lin_layers needs to be same as the last node size of conv_layer.')
        
        self.layers = nn.ModuleList()
        if batch_normalization:
            self.batch_norms = nn.ModuleList()
        self.batch_normalization = batch_normalization
        
        self.conv_len = len(conv_layers)-1
        self.lin_len = len(lin_layers)-1
        self.dropout = dropout
        
        if activation =='relu':
            self.activation = F.relu
        elif activation =='swish':
            self.activation = swish
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = F.tanh
        
        for i in range(self.conv_len):
            self.layers.append(
                GATConv(conv_layers[i],conv_layers[i+1], heads=heads, dropout=self.dropout,concat=True, bias=True)
            )

        for i in range(self.lin_len):
            self.layers.append(nn.Linear(lin_layers[i],lin_layers[i+1]))
            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(lin_layers[i+1]))


    def forward(self, data):
        x = data.x
        e = data.edge_index

        for i in range(self.conv_len-1):
            x = self.layers[i](x,e)
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)
        
        x = self.layers[self.conv_len-1](x,e)

        for j in range(self.lin_len-1):
            x = self.layers[self.conv_len+j](x)
            if self.batch_normalization:
                x = self.batch_norms[j](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.activation(x)
        
        x = self.layers[-1](x)

        return x

def set_model(config, dataset):
    config.conv_layers[0] = dataset[0].num_node_features
    if config.model_type == 'GCN':
        model = GCN(config.conv_layers, config.lin_layers, config.activation, config.batch_normalization,config.dropout)
    elif config.model_type == 'GIN':
        model = GIN(config.conv_layers, config.lin_layers, config.activation, config.batch_normalization,config.dropout)
    elif config.model_type == 'GAT':
        model = GAT(config.conv_layers, config.lin_layers, config.activation, config.batch_normalization,config.dropout)
    return model

def set_dataset(config, partition):
    data_folder = '/home/jiangcjr1/kinaseGNN/gnns/data/'
    if partition == 'train':
        data_path = f'{data_folder}comply_mols_5p_nonan_random_train.parquet'
    elif partition == 'val':
        data_path = f'{data_folder}comply_mols_5p_nonan_random_val.parquet'
    node_features = dict(
        atom_degree = config.atom_degree,
        atom_symbol = config.atom_symbol,
        atom_hybridization = config.atom_hybridization,
        atom_charge = config.atom_charge,
        atom_valence = config.atom_valence,
        atom_is_aromatic = config.atom_is_aromatic,
        atom_is_in_ring = config.atom_is_in_ring,
        atom_residue = config.atom_residue,
    )
    graph_config=GraphConfig(node_features=node_features, bond_type=True, one_hot_node_type=config.one_hot) #when bond type false needs to have binary edge
    dataset=PocketLigandDataset_test(config=DatasetConfig(dataframe_path = data_path ,graph_config=graph_config, pad_dim=config.pad_dim, pocket_radius= config.radius, max_pose_rank= config.max_rank, drop_nan=True))
    return dataset

def set_optimizer(optimizer_name,model,lr):
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    return optimizer

def train_epoch(train_loader, model, criterion, optimizer):
    model.train()

    current_loss = 0
    for data in tqdm(train_loader):
        #target = target.to(device)
        optimizer.zero_grad()
        data.x = data.x.float().to(device)
        data.y = data.y.float().to(device)

        out = model(data)

        l = criterion(out, torch.reshape(data.y, (len(data.y), 1)))
        current_loss += l / len(train_loader)
        l.backward()
        optimizer.step()
    return current_loss, model

def test_epoch(loader, model, criterion):
    test_loss = 0
    preds = np.empty((0))
    targets = np.empty((0))
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            data.x = data.x.float()
            data.y = data.y.float()
            #target = target.to(device)
            out = model(data)
            # NOTE
            # out = out.view(d.y.size())
            l = criterion(out, torch.reshape(data.y, (len(data.y), 1)))
            test_loss += l / len(loader)

            # save prediction vs ground truth values for plotting
            preds = np.concatenate((preds, out.cpu().detach().numpy()[:, 0]))
            targets = np.concatenate((targets, data.y.cpu().detach().numpy()))
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets, preds)
    r2 = r_value**2
    rmse = root_mean_squared_error(targets, preds)
    mse=rmse**2
    mae=mean_absolute_error(targets, preds)

    return test_loss, r2, rmse, mse, mae, preds, targets
    

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        writer = SummaryWriter(f'runs/{config.exp_name}')
        # set trainloader, val loader and test loader 
        train_dataset = set_dataset(config,'train')
        val_dataset = set_dataset(config,'val')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
        model=set_model(config,train_dataset).to(device)
        
        
        optimizer = set_optimizer(config.optimizer, model, config.lr)
        criterion = torch.nn.MSELoss()

        best_r2 = 0
        
        for epoch in range(config.epochs):
            print('Epoch:', epoch, '/', config.epochs)
            #########
            # TRAIN #
            ######### 
            train_loss, model = train_epoch(train_loader, model, criterion, optimizer)

            ##############
            # VALIDATION #
            ##############
            val_loss, r2,rmse,mse,mae,preds,targets = test_epoch(val_loader, model, criterion)
            
            wandb.log({'train_loss': train_loss,
                    'val_loss': val_loss,
                        'r2': r2,
                        'rmse': rmse,
                        'mse':mse,
                        'mae':mae})

            # tensorboard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("R2", r2, epoch)
            writer.add_scalar("RMSE", rmse, epoch)
            writer.add_scalar("MSE", mse, epoch)
            writer.add_scalar("MAE", mae, epoch)

            if r2 >= best_r2:
                best_r2 = r2

                savepath = 'GNN_checkpoints/%s/best_model.t7' % config.exp_name
                torch.save(model.state_dict(), savepath)

            data = pd.DataFrame({'real': targets, 'preds': preds})
            data.to_csv(f'GNN_checkpoints/{config.exp_name}/results.csv', index=False)

        writer.flush()
        writer.close()


def test(config):
    dataset = set_dataset(config)
    ## change later
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
    model=set_model(config.model_type,dataset,config.dim_h, config.dropout)
    loss = torch.nn.MSELoss()
    
    checkpoint = torch.load('GNN_checkpoints/' + config.exp_name + '/best_model.t7')
    model.load_state_dict(checkpoint)
    model = model.eval()

    test_loss,r2,rmse,mse,mae,preds,targets = test_epoch(test_loader, model, loss)

    df = pd.DataFrame({'real': targets, 'preds': preds})
    df.to_csv(f'GNN_checkpoints/{config.exp_name}/results.csv', index=False)
    
    # (Later) Add some statistics on performance per kinase
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='failsave',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--model_type', type=str, default='GCN',
                        help='specific model used')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (adam)') 
    parser.add_argument('--dim_h', type=int, default=128,
                        help='number of nodes in message passing layers')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function used (e.g. relu, elu, swish)') 
    parser.add_argument('--batch_normalization', type=bool, default=False,
                        help='Apply batch normalization in network.') 
    parser.add_argument('--conv_layers', type=list, default=[0,128,128,128],
                        help='Number of nodes in the convolution layers') 
    parser.add_argument('--lin_layers', type=list, default=[128,128,1],
                        help='Number of nodes in the convolution layers') 

     
    parser.add_argument('--atom_degree', type=bool, default=True,
                        help='atom_degree')
    parser.add_argument('--atom_symbol', type=bool, default=True,
                        help='atom_symbol')
    parser.add_argument('--atom_hybridization', type=bool, default=True,
                        help='atom_hybridization')
    parser.add_argument('--atom_charge', type=bool, default=True,
                        help='atom_charge')
    parser.add_argument('--atom_valence', type=bool, default=True,
                        help='atom_valence')
    parser.add_argument('--atom_is_aromatic', type=bool, default=True,
                        help='atom_is_aromatic')
    parser.add_argument('--atom_is_in_ring', type=bool, default=True,
                        help='atom_is_in_ring')
    parser.add_argument('--atom_residue', type=bool, default=True,
                        help='atom_residue')
    parser.add_argument('--pad_dim', type=int, default=350,
                        help='pad_dim')
    parser.add_argument('--radius', type=int, default=3,
                        help='radius')
    parser.add_argument('--max_rank', type=int, default=1,
                        help='max_rank')
    
    config = parser.parse_args()

    with open('sweep_config.yaml') as file:
        sweep_config=yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep_config, project="GNN", entity="chenjirongjiang")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() and not sweep_config['parameters']['no_cuda']['value'] else 'cpu')
    print(device)

    if not os.path.exists('GNN_checkpoints'):
        os.makedirs('GNN_checkpoints')
    if not os.path.exists('GNN_checkpoints/' + sweep_config['parameters']['exp_name']['value']):
        os.makedirs('GNN_checkpoints/' + sweep_config['parameters']['exp_name']['value'])
    elif not sweep_config['parameters']['eval']:
        check = input('This model already exists, do you wish to overwrite it? (y/n) ')

        if not check == 'y':
            print('Cancelling...')
            exit()

    
    wandb.agent(sweep_id, train)
    #else:
    #    test(config)