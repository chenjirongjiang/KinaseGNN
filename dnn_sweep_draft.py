import argparse
import math
import os
from collections import defaultdict
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datasets import CustomDatasetDB, CustomDatasetMaster, CustomDatasetSubset
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

class NN(nn.Module):
    '''
    Structure based on PLEC paper. (and Drugex)?
    '''
    def __init__(self, dropout,layer_sizes, activation,batch_normalization,device):
        super(NN, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.batch_normalization = batch_normalization
        self.device = device

        for i in range(len(layer_sizes) - 1):
            self.linears.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        
    def forward(self, x):

        for layer in range(len(self.linears)-1):
            x = self.linears[layer](x)

            if self.batch_normalization:
                x = self.batch_norms[layer](x)

            if self.activation =='relu':
                x = F.relu(x)
            elif self.activation =='swish':
                x=swish(x)
            elif self.activation == 'elu':
                x = F.elu(x)
            elif self.activation == 'prelu':
                weight=torch.tensor(0.25).to(self.device)
                x= F.prelu(x,weight)
            elif self.activation == 'tanh':
                x= F.tanh(x)

            if self.training:
                x = self.dropout(x)
        
        x = self.linears[-1](x)

        return x

# Swish Function 
def swish(x):
    return x * torch.sigmoid(x)

def set_dataset(dataset_type,input_file,docking_type,batch_size):
    if dataset_type == 'DB':
        dataset = CustomDatasetDB
    elif dataset_type == 'master':
        dataset = CustomDatasetMaster
    else:
        dataset = CustomDatasetSubset
    
    if dataset_type == 'DB':
        train_loader = DataLoader(dataset('train', input_file, docking_type), num_workers=0,
                                batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset('test', input_file, docking_type), num_workers=0,
                                batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(dataset('train'), num_workers=0,
                                batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset('test'), num_workers=0,
                                batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader,test_loader

def set_model(device,dropout,layer_sizes,activation,batch_normalization):
    model = NN(dropout=dropout, layer_sizes=layer_sizes,activation=activation,batch_normalization=batch_normalization,device=device).to(device)

    return model

def train_epoch(model, train_loader, criterion, train_losses, optimizer):
    model.train()
    for data, targets in tqdm(train_loader):
        targets = torch.as_tensor(targets[:, 0].numpy(), dtype=torch.float)

        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        scores = scores.flatten()

        loss = criterion(scores, targets)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def test_epoch(model,test_loader):
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for data, targets, poses in tqdm(test_loader):
            data = data.to(device)

            scores = model(data)
            scores = scores.detach().cpu().numpy()

            preds = scores.flatten()

            test_preds.extend(preds)
            test_true.extend(targets.numpy().flatten())

    slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_preds)
    r2 = r_value**2
    rmse = mean_squared_error(test_true, test_preds, squared=False)
    mse=rmse**2
    mae=mean_absolute_error(test_true, test_preds)
    return r2,rmse,mse,mae

def train(config=None):
    
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        writer = SummaryWriter(f'runs/{config.exp_name}')

        print('Retrieving data...')
        # Pick dataset
        train_loader, test_loader = set_dataset(config.dataset,config.input_file,config.docking_type,config.batch_size)

        # Init model
        model = set_model(device,config.dropout,config.layer_sizes,config.activation,config.batch_normalization)
        
        # Loss and optim
        criterion = nn.MSELoss()
        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)
        elif config.optimizer == 'adamax':
            optimizer = optim.Adamax(model.parameters(), lr=config.lr,weight_decay=config.weight_decay) 

        best_r2 = 0

        # Train network
        for epoch in range(config.epochs):
            print('Epoch:', epoch, '/', config.epochs)
            train_losses = []

            #########
            # TRAIN #
            ######### 
            train_epoch(model, train_loader, criterion, train_losses, optimizer)

            ########
            # TEST #
            ########
            r2,rmse,mse,mae=test_epoch(model,test_loader)

            wandb.log({'train_loss': np.mean(train_losses),
                    'r2': r2,
                    'rmse': rmse,
                    'mse':mse,
                    'mae':mae})

            # tensorboard
            writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
            writer.add_scalar("R2", r2, epoch)
            writer.add_scalar("RMSE", rmse, epoch)

            if r2 >= best_r2:
                best_r2 = r2

                model_state_dict = model.state_dict()
                savepath = 'DNN_checkpoints/%s/best_model.t7' % config.exp_name
                torch.save(model_state_dict, savepath)

        writer.flush()
        writer.close()

def test(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # Pick dataset
        train_loader, test_loader = set_dataset(config.dataset,config.input_file,config.docking_type,config.batch_size)

        # Init model
        model = set_model(device,config.dropout,config.layer_sizes,config.activation,config.batch_normalization)
            
        checkpoint = torch.load('DNN_checkpoints/' + config.exp_name + '/best_model.t7')
        model.load_state_dict(checkpoint)
        model = model.eval()

        test_true = []
        test_preds = []
        test_stds = []
        all_poses = []
        pose_dict = defaultdict(lambda: defaultdict(list)) # In order to calculate the mean of the poses

        if config.dataset == 'DB':
            ml_table = pd.read_csv(f'DNN_data/{config.input_file}').set_index('pose_ID').to_dict(orient='index')
        elif config.dataset == 'subset':
            ml_table = pd.read_csv('DNN_data/ML_table_diffdock.csv').set_index('pose_ID').to_dict(orient='index')

        with torch.no_grad():
            for data, targets, poses in tqdm(test_loader):
                data = data.to(device)
                targets = targets.numpy().flatten()

                scores = model(data)
                scores = scores.detach().cpu().numpy()

                preds = scores.flatten()

                for i, pose in enumerate(poses):
                    if config.dataset == 'master':
                        key = pose.split('_')[0]
                    else:
                        key = str(ml_table[pose]['klifs_ID']) + '_' + ml_table[pose]['SMILES_docked']

                    pose_dict[key]['preds'].append(preds[i])
                    pose_dict[key]['true'].append(targets[i])

                test_preds.extend(preds)
                test_true.extend(targets)
                all_poses.extend(poses)

        data = pd.DataFrame({'poseID': all_poses, 'real': test_true, 'preds': test_preds})
        data.to_csv(f'DNN_checkpoints/{config.exp_name}/results.csv', index=False)

        slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_preds)
        r2 = r_value**2
        rmse = mean_squared_error(test_true, test_preds, squared=False)
        mse = rmse **2
        mae = mean_absolute_error(test_true, test_preds)

        print(f'R2: {r2}')
        print(f'RMSE: {rmse}')
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')

        mean_true = []
        mean_pred = []
        max_true = []
        max_pred = []
        poses = []

        for pose, sub_dict in pose_dict.items():
            poses.append(pose)
            mean_true.append(mean(sub_dict['true']))
            mean_pred.append(mean(sub_dict['preds']))
            max_true.append(max(sub_dict['true']))
            max_pred.append(max(sub_dict['preds']))

        mean_data = pd.DataFrame({'poseID': poses, 'mean_real': mean_true, 'mean_pred': mean_pred})
        max_data = pd.DataFrame({'poseID': poses, 'max_real': max_true, 'max_pred': max_pred})

        mean_data.to_csv(f'DNN_checkpoints/{config.exp_name}/mean_results.csv', index=False)
        max_data.to_csv(f'DNN_checkpoints/{config.exp_name}/max_results.csv', index=False)

        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_true, mean_pred)
        r2 = r_value**2
        rmse = mean_squared_error(mean_true, mean_pred, squared=False)
        mse = rmse**2
        mae = mean_absolute_error(mean_true, mean_pred)

        print(f'R2 (mean of poses): {r2}')
        print(f'RMSE (mean of poses): {rmse}')
        print(f'MSE (mean of poses): {mse}')
        print(f'MAE (mean of poses): {mae}')

        slope, intercept, r_value, p_value, std_err = stats.linregress(max_true, max_pred)
        r2 = r_value**2
        rmse = mean_squared_error(max_true, max_pred, squared=False)
        mse = rmse**2
        mae = mean_absolute_error(max_true, max_pred)

        print(f'R2 (max of poses): {r2}')
        print(f'RMSE (max of poses): {rmse}')
        print(f'MSE (mean of poses): {mse}')
        print(f'MAE (mean of poses): {mae}')

def mean_std(x, y):
    '''
    Calculate mean and SD from numpy arrays.
    '''
    mu = (1/sum(y))*(sum(x*y))
    stddev = math.sqrt(sum(((x-mu)**2)*y))

    return mu, stddev

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='failsave',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='dropout rate')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')         
    parser.add_argument('--dataset', type=str, default='DB',
                        help='Dataset type (DB, subset, master)')
    parser.add_argument('--docking_type', type=str, default='vina',
                        help='Docking software (vina, diffdock)')                                            
    parser.add_argument('--input_file', type=str,
                        help='Input file name')
    parser.add_argument('--layer_sizes', type=list_of_ints, default=[65536,4000,1000,1],
                        help='List of nodes per layer')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (adam)') 
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function used (e.g. relu, elu, swish)') 
    parser.add_argument('--batch_normalization', type=bool, default=False,
                        help='Apply batch normalization in network.') 
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    with open('sweep_config.yaml') as file:
        sweep_config=yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep_config, project="DNN", entity="chenjirongjiang")
    
    if not os.path.exists('DNN_checkpoints'):
        os.makedirs('DNN_checkpoints')
    if not os.path.exists('DNN_checkpoints/' + sweep_config['parameters']['exp_name']['value']):
        os.makedirs('DNN_checkpoints/' + sweep_config['parameters']['exp_name']['value'])
    elif not sweep_config['parameters']['eval']:
        check = input('This model already exists, do you wish to overwrite it? (y/n) ')

        if not check == 'y':
            print('Cancelling...')
            exit()

    if not args.eval:
        wandb.agent(sweep_id, train)
        
    #wandb.agent(sweep_id, test)

    
