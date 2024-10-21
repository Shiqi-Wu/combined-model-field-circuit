from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/shiqi/code/Project2-sensor-case/model_combination_Argos/utils')
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import yaml
from load_dataset import cut_slices, load_dataset
from sklearn.decomposition import PCA
import copy
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

### Read config file
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def read_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

### Load data
def data_preparation(config, data_dir):
    window_size = config['window_size']
    print(f'window_size: {window_size}')
    x_dataset, y_dataset, u_dataset = [], [], []
    # Load data
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file exists before trying to load it
        
        if os.path.exists(data_file_path) and data_file_path.endswith('.npy'):
            
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u_data, _ = load_dataset(data_dict)
            # print(x_data.shape, y_data.shape, u_data.shape)
            x_dataset.append(x_data[1:window_size])
            y_dataset.append(y_data[1:window_size])
            u_dataset.append(u_data[1:window_size])
            # print(f"Loaded data from {data_file_path}")
        else:
            print(f"File not found: {data_file_path}")

    print(x_dataset[0].shape, y_dataset[0].shape, u_dataset[0].shape)
    # Concatenate data
    x_data = np.concatenate(x_dataset, axis=0)
    y_data = np.concatenate(y_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    print(f'x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, u_data shape: {u_data.shape}')

    return x_data, y_data, u_data

def prepare_data_residual(x_data, y_data, u_data, model, device, batch_size=128, shuffle=True):
    
    
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    else:
        x_data = x_data.to(device)

    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    else:
        y_data = y_data.to(device)

    if not isinstance(u_data, torch.Tensor):
        u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    else:
        u_data = u_data.to(device)

    if model is not None:
        model.to(device)
        x_data_reshape = x_data.view(-1, x_data.shape[-1])
        u_data_reshape = u_data.view(-1, u_data.shape[-1])

        
        # Calculate err_data
        pred_data = model(x_data_reshape, u_data_reshape)
        pred_data = pred_data.detach()
        
        # Reshape err_data to match y_data shape
        pred_data = pred_data.view(y_data.shape)
        err_data = y_data - pred_data
    else:
        err_data = y_data

     # Create a TensorDataset and DataLoader
    dataset = TensorDataset(x_data, err_data, u_data)
    data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

    return data_loader, err_data

### Train the residual model
def train_residual_model(linear_model, residual_model, dataset, config, num_epoches = 100, learning_rate = 0.001, device = 'cpu', err_scaler = False):

    if linear_model is not None:
        linear_model.to(device)
        linear_model.eval()
    residual_model.to(device)

    # Data Loader
    x_data, y_data, u_data = dataset

    # print(config['window_size'], config['predict_num'])

    x_data_slices = cut_slices(x_data, config['window_size'] - 1, config['predict_num'])
    y_data_slices = cut_slices(y_data, config['window_size'] - 1, config['predict_num'])
    u_data_slices = cut_slices(u_data, config['window_size'] - 1, config['predict_num'])
    x_data_slices = torch.cat(x_data_slices, dim=0).to(device)
    y_data_slices = torch.cat(y_data_slices, dim=0).to(device)
    u_data_slices = torch.cat(u_data_slices, dim=0).to(device)

    data_loader, _ = prepare_data_residual(x_data_slices, y_data_slices, u_data_slices, None, device)

    if err_scaler:
        _, err_data = prepare_data_residual(x_data, y_data, u_data, linear_model, device)
        err_data = err_data.detach()

        # scale the err_data
        std_layer_err = StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))
        # print(std_layer_err.mean, std_layer_err.std)

    else:
        std_layer_err = StdScalerLayer(torch.mean(x_data, dim=0), torch.std(x_data, dim=0))

    print(std_layer_err.mean, std_layer_err.std)

    # Optimizer
    optimizer = Adam(residual_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epoches):
        total_loss = 0     
        for x_batch, _, u_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epoches}", unit="batch"):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)

            loss = hybrid_loss(linear_model, residual_model, std_layer_err, x_batch, u_batch)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1)
            total_loss += loss.item()
        total_loss = total_loss / len(data_loader)
        scheduler.step()

        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state = copy.deepcopy(residual_model.state_dict())

    if best_model_state is not None:
        residual_model.load_state_dict(best_model_state)
    else:
        print('No improvement during training.')
    
    print(f'Training Residual Model: Epoch {epoch+1}/{num_epoches}, Loss: {best_loss}')
    return residual_model, std_layer_err

def hybrid_loss(linear_model, residual_model, std_layer_err, x_data, u_data):
    mse = torch.nn.MSELoss()
    N = x_data.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    x0 = x_data[:, 0, :]
    x_pred = [x0]
    # print(x0.shape)

    for i in range(1, N):
        if residual_model is not None and linear_model is not None:
            x_pred_cur = linear_model(x_pred[-1], u_data[:, i-1, :]) + residual_model.latent_to_latent_forward(x_pred[-1], u_data[:, i-1, :], std_layer_err)
        elif residual_model is not None:
            x_pred_cur = residual_model.latent_to_latent_forward(x_pred[-1], u_data[:, i-1, :], std_layer_err)
        else:
            x_pred_cur = linear_model(x_pred[-1], u_data[:, i-1, :])
        x_pred.append(x_pred_cur)

    x_pred = torch.stack(x_pred, dim=1)
    loss = mse(x_pred, x_data)
    return loss


### Linear regression
def linear_regression(dataset, linear_model, device='cpu'):
    x, y, u = dataset
    x = x.to(device)
    y = y.to(device)
    u = u.to(device)
        
    x_dic = linear_model.x_dict(x)
    y_dic = linear_model.x_dict(y)
    u_dic = linear_model.u_dict(u)

    z = torch.cat((x_dic, u_dic), dim=1)
    z_pseudo_inv = torch.pinverse(z)

    param_pseudo_inv = torch.matmul(z_pseudo_inv, y_dic)
    A = param_pseudo_inv[:x_dic.shape[1], :]
    B = param_pseudo_inv[x_dic.shape[1]:, :]

    with torch.no_grad():
        linear_model.register_parameter('A', nn.Parameter(A))
        linear_model.register_parameter('B', nn.Parameter(B))
    
    return linear_model


### Iterative training
def iterative_training(dataset, linear_model, residual_model, config, num_iter=10, num_epoches=100, learning_rate=0.001, device='cpu'):
    # Initialize the linear model
    x_data, y_data, u_data = dataset
    linear_model = linear_regression(dataset, linear_model, device)
    # linear_model = train_linear_model_singlestep([x_data, y_data, u_data], linear_model, device)

    # Iterative training
    iterative_losses = []
    for i in range(num_iter):
        residual_model, std_layer_err = train_residual_model(linear_model, residual_model, dataset, config, num_epoches, learning_rate, device, config['err_scaler'])
        print(x_data.shape, u_data.shape, std_layer_err.mean.shape, std_layer_err.std.shape)

        err_data = y_data - residual_model.latent_to_latent_forward(x_data, u_data, std_layer_err).detach()
        linear_model = linear_regression([x_data, err_data, u_data], linear_model, device)
        # linear_model = train_linear_model_singlestep([x_data, err_data, u_data], linear_model, device)
        linear_pred = linear_model(x_data, u_data).detach()
        print(x_data.shape, u_data.shape, std_layer_err.mean.shape, std_layer_err.std.shape)
        residual_pred = residual_model.latent_to_latent_forward(x_data, u_data, std_layer_err).detach()
        y_pred = linear_pred + residual_pred
        y_data = y_data.detach()
        mse = torch.nn.MSELoss()
        loss = mse(y_pred, y_data)
        print(f'Iterative Training: Iteration {i+1}/{num_iter}, Loss: {loss}')
        iterative_losses.append(loss.item())
    return linear_model, residual_model, std_layer_err, iterative_losses

### Models
class Linear_model(torch.nn.Module):
    def __init__(self, state_dim, control_dim):
        super(Linear_model, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(state_dim, state_dim))

        self.B = torch.nn.Parameter(torch.randn(control_dim, state_dim))
    
    def x_dict(self, x):
        ones = torch.ones(x.shape[0], 1).to(x.device)
        return torch.cat((x, ones), dim=1)
    
    def u_dict(self, u):
        return u
    
    def forward(self, x, u):
        x = self.x_dict(x)
        u = self.u_dict(u)
        y = torch.matmul(x, self.A) + torch.matmul(u, self.B)
        return y[:, 1:]
class PCAKoopman(torch.nn.Module):
    def __init__(self, params, std_layer_1, pca_transform, std_layer_2, std_layer_u, x_dict, u_dict):
        super(PCAKoopman, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.pca_transform = pca_transform
        self.std_layer_2 = std_layer_2
        self.std_layer_u = std_layer_u
        self.x_dict = x_dict
        self.u_dict = u_dict
        self.A = torch.nn.Parameter(torch.randn(params.x_dic_dim, params.x_dic_dim))
        self.B = torch.nn.Parameter(torch.randn(params.u_dic_dim, params.x_dic_dim))

    def forward(self, x, u, std_layer_err):
        x = self.std_layer_1.transform(x)
        x = self.pca_transform.transform(x)
        x = self.std_layer_2.transform(x)
        u = self.std_layer_u.transform(u)
        y = self.latent_to_latent_forward(x, u, std_layer_err)
        y = self.std_layer_2.inverse_transform(y)
        y = self.pca_transform.inverse_transform(y)
        y = self.std_layer_1.inverse_transform(y)
        return y
        
    def latent_to_latent_forward(self, x, u, std_layer_err):
        x = self.x_dict(x)
        u = self.u_dict(u)
        y = torch.matmul(x, self.A) + torch.matmul(u, self.B)
        y = y[:, 1:self.params.pca_dim + 1]
        # print(y.shape)
        y = std_layer_err.inverse_transform(y)
        return y
        
class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pca_matrix):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.transform.weight = nn.Parameter(pca_matrix, requires_grad=False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        self.inverse_transform.weight = nn.Parameter(pca_matrix.T, requires_grad=False)

class StdScalerLayer(nn.Module):
    def __init__(self, mean, std):
        super(StdScalerLayer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, input):
        return input * self.std + self.mean

class State_Encoder(nn.Module):
    "Implements State dictionary"
    def __init__(self, params):
        super(State_Encoder, self).__init__()
        self.dic_model = params.dic_model
        if params.dic_model != 0:
            self.input_layer = nn.Linear(params.pca_dim, params.dd_model)
            self.Layer = FeedForwardLayerConnection(params.dd_model, FeedForward(params.dd_model, params.dd_ff), params.dropout)
            self.layers = clones(self.Layer, params.N_State)
            self.output_layer = nn.Linear(params.dd_model, params.dic_model)

    def forward(self, x):
        if self.dic_model == 0:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            return torch.cat((ones, x), dim = 1)
        else:
            y = self.input_layer(x)
            y = F.relu(y)
            for layer in self.layers:
                y = layer(y)
            y = self.output_layer(y)
            ones = torch.ones(x.shape[0], 1).to(x.device)
            y = torch.cat((ones, x, y), dim = -1)
            return y

class Control_Encoder(nn.Module):
    "Implements Control dictionary"
    def __init__(self, params):
        super(Control_Encoder, self).__init__()
        self.dic_model = params.u_dic_model
        if params.u_dic_model != 0:
            self.input_layer = nn.Linear(params.u_dim, params.u_model)
            self.Layer = FeedForwardLayerConnection(params.u_model, FeedForward(params.u_model, params.u_ff), params.dropout)
            self.layers = clones(self.Layer, params.N_Control)
            self.output_layer = nn.Linear(params.u_model, params.u_dic_model)

    def forward(self, x):
        if self.dic_model == 0:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            return torch.cat((ones, x), dim = 1)
        else:
            y = self.input_layer(x)
            y = F.relu(y)
            for layer in self.layers:
                y = layer(y)
            y = self.output_layer(y)
            ones = torch.ones(x.shape[0], 1).to(x.device)
            y = torch.cat((ones, x, y), dim = 1)
            return y
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        
class FeedForwardLayerConnection(nn.Module):
    def __init__(self, size, feed_forward, dropout):
        super(FeedForwardLayerConnection, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
    
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.feed_forward(x))
        return x

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout = 0.2): 
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Params():
    "A class to hold model hyperparameters."
    def __init__(self, x_dim, u_dim, config):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.pca_dim = config.get('pca_dim', 10)
        self.dic_model = config.get('dic_model', 20)
        self.x_dic_dim = 1 + self.pca_dim + self.dic_model
        self.dd_model = config.get('dd_model', 64)
        self.dd_ff = config.get('dd_ff', 128)
        self.N_State = config.get('N_State', 6)
        self.u_dic_model = config.get('u_dic_model', 20)
        self.u_dic_dim = 1 + u_dim + self.u_dic_model
        self.u_model = config.get('u_model', 64)
        self.u_ff = config.get('u_ff', 128)
        self.N_Control = config.get('N_Control', 6)
        self.dropout = config.get('dropout', 0.2)

### Evaluation
def load_evaluation_data(begin, end, data_dir):
    x_dataset = []
    u_dataset = []

    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file exists before trying to load it
        if os.path.exists(data_file_path) and item.endswith('.npy'):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, _, u_data, _ = load_dataset(data_dict)
            x_dataset.append(x_data[begin:end, :])
            u_dataset.append(u_data[begin:end, :])
        else:
            print(f"File not found: {data_file_path}")
    
    return x_dataset, u_dataset

def encode_state(x, std_layer_1, pca_transformer, std_layer_2):
    x = std_layer_1.transform(x)
    x = pca_transformer.transform(x)
    x = std_layer_2.transform(x)
    return x

def decode_state(x, std_layer_1, pca_transformer, std_layer_2):
    x = std_layer_2.inverse_transform(x)
    x = pca_transformer.inverse_transform(x)
    x = std_layer_1.inverse_transform(x)
    return x

def generate_linear_trajectories(x_dataset, u_dataset, linear_model, std_layer_1, pca_transformer, std_layer_2, std_layer_u, device):
    x_data_pred_traj = []
    x_data_pca_traj = []
    x_data_pca_pred_traj = []
    window_size = len(x_dataset[0])
    steps = window_size

    for x_data, u_data in zip(x_dataset, u_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
        x_pred = torch.zeros_like(x_data).to(device)
        x_pred[0, :] = x_data[0, :]
        x0 = encode_state(x_data[0, :].reshape(1, -1), std_layer_1, pca_transformer, std_layer_2)

        for step in range(1, steps):
            
            u = std_layer_u.transform(u_data[step - 1, :].reshape(1, -1))
            x1 = linear_model(x0, u)
            x_pred[step,:] = decode_state(x1, std_layer_1, pca_transformer, std_layer_2)
            x0 = x1
        
        x_data_pred_traj.append(x_pred.detach().cpu().numpy())

        x_pca_true = encode_state(x_data.detach(), std_layer_1, pca_transformer, std_layer_2)
        x_pca_pred = encode_state(x_pred.detach(), std_layer_1, pca_transformer, std_layer_2)

        x_data_pca_traj.append(x_pca_true)
        x_data_pca_pred_traj.append(x_pca_pred)
    
    return x_data_pred_traj, x_data_pca_traj, x_data_pca_pred_traj

def generate_residual_trajectories(x_dataset, u_dataset, residual_model, std_layer_1, pca_transformer, std_layer_2, std_layer_err, device):
    x_data_pred_traj = []
    x_data_pca_traj = []
    x_data_pca_pred_traj = []
    window_size = len(x_dataset[0])
    steps = window_size

    for x_data, u_data in zip(x_dataset, u_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
        x_pred = torch.zeros_like(x_data).to(device)
        x_pred[0, :] = x_data[0, :]
        x0 = encode_state(x_data[0:1, :].reshape(1, -1), std_layer_1, pca_transformer, std_layer_2)
        for step in range(1, steps):
            u = residual_model.std_layer_u.transform(u_data[step - 1, :].reshape(1, -1))
            x1 = residual_model.latent_to_latent_forward(x0, u, std_layer_err)
            x_pred[step, :] = decode_state(x1, std_layer_1, pca_transformer, std_layer_2)
            x0 = x1
        
        x_data_pred_traj.append(x_pred.detach().cpu().numpy())

        x_pca_true = encode_state(x_data.detach(), std_layer_1, pca_transformer, std_layer_2)
        x_pca_pred = encode_state(x_pred.detach(), std_layer_1, pca_transformer, std_layer_2)

        x_data_pca_traj.append(x_pca_true)
        x_data_pca_pred_traj.append(x_pca_pred)
    
    return x_data_pred_traj, x_data_pca_traj, x_data_pca_pred_traj

def generate_hybrid_trajectories(x_dataset, u_dataset, linear_model, residual_model, std_layer_1, pca_transformer, std_layer_2, std_layer_u, std_layer_err, device):
    x_data_pred_traj = []
    x_data_pca_traj = []
    x_data_pca_pred_traj = []
    window_size = len(x_dataset[0])
    steps = window_size
    # print(f'window_size: {window_size}')

    for x_data, u_data in zip(x_dataset, u_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
        # print(x_data.shape, u_data.shape)
        x_pred = torch.zeros_like(x_data).to(device)
        x_pred[0, :] = x_data[0, :]
        x0 = encode_state(x_data[0:1, :], std_layer_1, pca_transformer, std_layer_2)

        for step in range(1, steps):
            u = std_layer_u.transform(u_data[step - 1, :].reshape(1, -1))
            x1 = linear_model(x0, u) + residual_model.latent_to_latent_forward(x0, u, std_layer_err)
            x_pred[step, :] = decode_state(x1, std_layer_1, pca_transformer, std_layer_2)
            x0 = x1
        
        x_data_pred_traj.append(x_pred.detach().cpu().numpy())

        x_pca_true = encode_state(x_data.detach(), std_layer_1, pca_transformer, std_layer_2)
        x_pca_pred = encode_state(x_pred.detach(), std_layer_1, pca_transformer, std_layer_2)

        x_data_pca_traj.append(x_pca_true)
        x_data_pca_pred_traj.append(x_pca_pred)
    
    return x_data_pred_traj, x_data_pca_traj, x_data_pca_pred_traj


def calculate_relative_diff(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, axis=1, ord=2)
    max_norm = np.max(np.linalg.norm(x_true, axis=1, ord=2))
    relative_diff = row_norm_diff / max_norm
    return relative_diff

def calculate_mean_relative_diff_set(x_true_traj, x_pred_traj):
    relative_diffs = [calculate_relative_diff(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)]
    mean_relative_diffs = np.mean(relative_diffs, axis=0)
    return mean_relative_diffs

def calculate_relative_error(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, ord='fro')
    total_norm_true = np.linalg.norm(x_true, ord='fro')
    return row_norm_diff / total_norm_true

def calculate_mean_relative_error_set(x_true_traj, x_pred_traj):
    relative_errors = [calculate_relative_error(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)]
    return relative_errors

### Main function
def main():
    ## Parse arguments
    args = parse_arguments()
    config = read_config_file(args.config)

    ## Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## Load data
    x_data, y_data, u_data = data_preparation(config, config['train_data_dir'])
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    dataset = [x_data, y_data, u_data]

    ## Dimension
    x_dim = x_data.shape[-1]
    u_dim = u_data.shape[-1]


    ## PCA
    # Standardize data
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)
    x_data_scaled = std_layer_1.transform(x_data)

    # PCA layer
    pca = PCA(n_components=config['pca_dim'])
    # Ensure x_data_scaled is converted back to a NumPy array for PCA
    pca.fit(x_data_scaled.detach().cpu().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32).to(device)
    print(f'PCA matrix shape: {pca_matrix.shape}')
    pca_layer = PCALayer(x_dim, config['pca_dim'], pca_matrix)

    # Standardize data 2
    x_pca = pca_layer.transform(x_data_scaled)
    mean_2 = torch.mean(x_pca, dim=0)
    std_2 = torch.std(x_pca, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # Build pca dataset
    x_pca_scaled = std_layer_2.transform(x_pca)
    y_data_scaled = std_layer_1.transform(y_data)
    y_pca = pca_layer.transform(y_data_scaled)
    y_pca_scaled = std_layer_2.transform(y_pca)
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)
    u_data_scaled = std_layer_u.transform(u_data)
    dataset = [x_pca_scaled, y_pca_scaled, u_data_scaled]
    print(f'PCA data shape: {x_pca_scaled.shape}, {y_pca_scaled.shape}, {u_data_scaled.shape}')

    ## Models
    # Linear model
    linear_model = Linear_model(config['pca_dim']+1, u_dim)


    # Residual model
    params = Params(x_dim, u_dim, config)
    state_dict = State_Encoder(params)
    control_dict = Control_Encoder(params)
    residual_model = PCAKoopman(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dict, control_dict)

    # Evaluation data
    x_dataset_train, u_dataset_train = load_evaluation_data(config['begin'], config['end'], config['train_data_dir'])
    x_dataset_test, u_dataset_test = load_evaluation_data(config['begin'], config['end'], config['test_data_dir'])

    # ## Baseline 1: Train linear model
    # linear_model = linear_regression(dataset, linear_model, device)

    # # Evaluation
    # x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_linear_trajectories(x_dataset_train, u_dataset_train, linear_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, device)
    # x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_linear_trajectories(x_dataset_test, u_dataset_test, linear_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, device)

    # # Calculate mean relative error
    # mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)

    # # Calculate mean relative diff
    # mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)

    # # Save results
    # np.save(config['save_dir'] + '/linear_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/linear_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/linear_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/linear_mean_relative_diffs_test.npy', mean_relative_diffs_test)
    # torch.save(linear_model.state_dict(), config['save_dir'] + '/linear_model.pth')

    # # Baseline 2: Train residual model only
    # residual_model_2 = PCAKoopman(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dict, control_dict)
    # residual_model_2, std_layer_err = train_residual_model(None, residual_model_2, dataset, config, config['num_epoches'], config['learning_rate'], device, config['err_scaler'])

    # # Evaluation 
    # x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_residual_trajectories(x_dataset_train, u_dataset_train, residual_model_2, std_layer_1, pca_layer, std_layer_2, std_layer_err, device)
    # x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_residual_trajectories(x_dataset_test, u_dataset_test, residual_model_2, std_layer_1, pca_layer, std_layer_2, std_layer_err, device)

    # # Calculate mean relative error
    # mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)

    # # Calculate mean relative diff
    # mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)

    # # Save results
    # np.save(config['save_dir'] + '/residual_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/residual_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/residual_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/residual_mean_relative_diffs_test.npy', mean_relative_diffs_test)
    # torch.save(residual_model_2.state_dict(), config['save_dir'] + '/residual_model.pth')

    ## Iterative training
    start_time = datetime.now()
    linear_model, residual_model, std_layer_err, iterative_losses = iterative_training(dataset, linear_model, residual_model, config, config['num_iterations'], config['num_epoches'], config['learning_rate'], device)
    end_time = datetime.now()
    training_time = end_time - start_time
    # np.save(config['save_dir'] + '/iterative_losses.npy', iterative_losses)

    # Evaluation
    state_time = datetime.now()
    x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_hybrid_trajectories(x_dataset_train, u_dataset_train, linear_model, residual_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, std_layer_err, device)
    x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_hybrid_trajectories(x_dataset_test, u_dataset_test, linear_model, residual_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, std_layer_err, device)
    end_time = datetime.now()
    evaluation_time = end_time - state_time

    with open(config['save_dir'] + '/time_cost.txt', 'w') as f:
        f.write(f'Training time: {training_time}')
        f.write(f'Evaluation time: {evaluation_time}')
    


    # Calculate mean relative error
    mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)

    # Calculate mean relative diff
    mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)

    # # Save results
    # np.save(config['save_dir'] + '/hybrid_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_test.npy', mean_relative_diffs_test)
    # torch.save(linear_model.state_dict(), config['save_dir'] + '/hybrid_linear_model.pth')
    # torch.save(residual_model.state_dict(), config['save_dir'] + '/hybrid_residual_model.pth')
    # torch.save(std_layer_err.state_dict(), config['save_dir'] + '/hybrid_std_layer_err.pth')

    return

def main_evaluate():
    ## Parse arguments
    args = parse_arguments()
    config = read_config_file(args.config)

    ## Device
    device = torch.device('cpu')
    
    ## Load data
    x_data, y_data, u_data = data_preparation(config, config['train_data_dir'])
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    dataset = [x_data, y_data, u_data]

    ## Dimension
    x_dim = x_data.shape[-1]
    u_dim = u_data.shape[-1]


    ## PCA
    # Standardize data
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)
    x_data_scaled = std_layer_1.transform(x_data)

    # PCA layer
    pca = PCA(n_components=config['pca_dim'])
    # Ensure x_data_scaled is converted back to a NumPy array for PCA
    pca.fit(x_data_scaled.detach().cpu().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32).to(device)
    print(f'PCA matrix shape: {pca_matrix.shape}')
    pca_layer = PCALayer(x_dim, config['pca_dim'], pca_matrix)

    # Standardize data 2
    x_pca = pca_layer.transform(x_data_scaled)
    mean_2 = torch.mean(x_pca, dim=0)
    std_2 = torch.std(x_pca, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # Build pca dataset
    x_pca_scaled = std_layer_2.transform(x_pca)
    y_data_scaled = std_layer_1.transform(y_data)
    y_pca = pca_layer.transform(y_data_scaled)
    y_pca_scaled = std_layer_2.transform(y_pca)
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)
    u_data_scaled = std_layer_u.transform(u_data)
    dataset = [x_pca_scaled, y_pca_scaled, u_data_scaled]
    print(f'PCA data shape: {x_pca_scaled.shape}, {y_pca_scaled.shape}, {u_data_scaled.shape}')

    
    ## Models
    # Linear model
    linear_model = Linear_model(config['pca_dim']+1, u_dim)

    # Residual model
    params = Params(x_dim, u_dim, config)
    state_dict = State_Encoder(params)
    control_dict = Control_Encoder(params)
    residual_model = PCAKoopman(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dict, control_dict)

    # Evaluation data
    x_dataset_train, u_dataset_train = load_evaluation_data(config['begin'], config['end'], config['train_data_dir'])
    x_dataset_test, u_dataset_test = load_evaluation_data(config['begin'], config['end'], config['test_data_dir'])
    np.save(config['save_dir'] + '/x_dataset_train.npy', x_dataset_train)
    np.save(config['save_dir'] + '/u_dataset_train.npy', u_dataset_train)
    np.save(config['save_dir'] + '/x_dataset_test.npy', x_dataset_test)
    np.save(config['save_dir'] + '/u_dataset_test.npy', u_dataset_test)
    x_dataset_train_Ip = [arr[:, -1:] for arr in x_dataset_train]
    x_dataset_test_Ip = [arr[:, -1:] for arr in x_dataset_test]
    x_dataset_train_Phi = [arr[:, :-1] for arr in x_dataset_train]
    x_dataset_test_Phi = [arr[:, :-1] for arr in x_dataset_test]
    np.save(config['save_dir'] + '/x_dataset_train_Ip.npy', x_dataset_train_Ip)
    np.save(config['save_dir'] + '/x_dataset_test_Ip.npy', x_dataset_test_Ip)
    np.save(config['save_dir'] + '/x_dataset_train_Phi.npy', x_dataset_train_Phi)
    np.save(config['save_dir'] + '/x_dataset_test_Phi.npy', x_dataset_test_Phi)

    ## Baseline 1: Train linear model
    linear_model.load_state_dict(torch.load(config['save_dir'] + '/linear_model.pth'))

    # Evaluation
    x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_linear_trajectories(x_dataset_train, u_dataset_train, linear_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, device)
    x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_linear_trajectories(x_dataset_test, u_dataset_test, linear_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, device)
    # np.save(config['save_dir'] + '/x_data_pred_traj_train_linear.npy', x_data_pred_traj_train)
    # np.save(config['save_dir'] + '/x_data_pred_traj_test_linear.npy', x_data_pred_traj_test)


    # # Calculate mean relative error
    # mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)

    # # Calculate mean relative diff
    # mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)



    # # Save results
    # np.save(config['save_dir'] + '/linear_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/linear_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/linear_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/linear_mean_relative_diffs_test.npy', mean_relative_diffs_test)

    x_data_pred_traj_train_Ip = [arr[:, -1:] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Ip = [arr[:, -1:] for arr in x_data_pred_traj_test]
    x_data_pred_traj_train_Phi = [arr[:, :-1] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Phi = [arr[:, :-1] for arr in x_data_pred_traj_test]
    np.save(config['save_dir'] + '/x_data_pred_traj_train_linear_Ip.npy', x_data_pred_traj_train_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_linear_Ip.npy', x_data_pred_traj_test_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_train_linear_Phi.npy', x_data_pred_traj_train_Phi)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_linear_Phi.npy', x_data_pred_traj_test_Phi)

    mean_relative_errors_train_Ip = calculate_mean_relative_error_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_errors_test_Ip = calculate_mean_relative_error_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip)
    mean_relative_errors_train_Phi = calculate_mean_relative_error_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_errors_test_Phi = calculate_mean_relative_error_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)

    mean_relative_diffs_train_Ip = calculate_mean_relative_diff_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_diffs_test_Ip = calculate_mean_relative_diff_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip)
    mean_relative_diffs_train_Phi = calculate_mean_relative_diff_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_diffs_test_Phi = calculate_mean_relative_diff_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)

    np.save(config['save_dir'] + '/linear_mean_relative_errors_train_Ip.npy', mean_relative_errors_train_Ip)
    np.save(config['save_dir'] + '/linear_mean_relative_errors_test_Ip.npy', mean_relative_errors_test_Ip)
    np.save(config['save_dir'] + '/linear_mean_relative_errors_train_Phi.npy', mean_relative_errors_train_Phi)
    np.save(config['save_dir'] + '/linear_mean_relative_errors_test_Phi.npy', mean_relative_errors_test_Phi)
    np.save(config['save_dir'] + '/linear_mean_relative_diffs_train_Ip.npy', mean_relative_diffs_train_Ip)
    np.save(config['save_dir'] + '/linear_mean_relative_diffs_test_Ip.npy', mean_relative_diffs_test_Ip)
    np.save(config['save_dir'] + '/linear_mean_relative_diffs_train_Phi.npy', mean_relative_diffs_train_Phi)
    np.save(config['save_dir'] + '/linear_mean_relative_diffs_test_Phi.npy', mean_relative_diffs_test_Phi)

    # Baseline 2: Train residual model only
    residual_model_2 = PCAKoopman(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dict, control_dict)
    mean_y = torch.mean(y_pca_scaled, dim=0)
    std_y = torch.std(y_pca_scaled, dim=0)
    std_layer_err = StdScalerLayer(mean_y, std_y)
    residual_model_2.load_state_dict(torch.load(config['save_dir'] + '/residual_model.pth'))

    # Evaluation 
    x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_residual_trajectories(x_dataset_train, u_dataset_train, residual_model_2, std_layer_1, pca_layer, std_layer_2, std_layer_err, device)
    x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_residual_trajectories(x_dataset_test, u_dataset_test, residual_model_2, std_layer_1, pca_layer, std_layer_2, std_layer_err, device)
    # np.save(config['save_dir'] + '/x_data_pred_traj_train_residual.npy', x_data_pred_traj_train)
    # np.save(config['save_dir'] + '/x_data_pred_traj_test_residual.npy', x_data_pred_traj_test)

    # # Calculate mean relative error
    # mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)

    # # Calculate mean relative diff
    # mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)

    # # Save results
    # np.save(config['save_dir'] + '/residual_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/residual_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/residual_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/residual_mean_relative_diffs_test.npy', mean_relative_diffs_test)

    x_data_pred_traj_train_Ip = [arr[:, -1:] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Ip = [arr[:, -1:] for arr in x_data_pred_traj_test]
    x_data_pred_traj_train_Phi = [arr[:, :-1] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Phi = [arr[:, :-1] for arr in x_data_pred_traj_test]
    np.save(config['save_dir'] + '/x_data_pred_traj_train_residual_Ip.npy', x_data_pred_traj_train_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_residual_Ip.npy', x_data_pred_traj_test_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_train_residual_Phi.npy', x_data_pred_traj_train_Phi)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_residual_Phi.npy', x_data_pred_traj_test_Phi)

    mean_relative_errors_train_Ip = calculate_mean_relative_error_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_errors_test_Ip = calculate_mean_relative_error_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip)
    mean_relative_errors_train_Phi = calculate_mean_relative_error_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_errors_test_Phi = calculate_mean_relative_error_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)
    
    mean_relative_diffs_train_Ip = calculate_mean_relative_diff_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_diffs_test_Ip = calculate_mean_relative_diff_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip)
    mean_relative_diffs_train_Phi = calculate_mean_relative_diff_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_diffs_test_Phi = calculate_mean_relative_diff_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)

    np.save(config['save_dir'] + '/residual_mean_relative_errors_train_Ip.npy', mean_relative_errors_train_Ip)
    np.save(config['save_dir'] + '/residual_mean_relative_errors_test_Ip.npy', mean_relative_errors_test_Ip)
    np.save(config['save_dir'] + '/residual_mean_relative_errors_train_Phi.npy', mean_relative_errors_train_Phi)
    np.save(config['save_dir'] + '/residual_mean_relative_errors_test_Phi.npy', mean_relative_errors_test_Phi)
    np.save(config['save_dir'] + '/residual_mean_relative_diffs_train_Ip.npy', mean_relative_diffs_train_Ip)
    np.save(config['save_dir'] + '/residual_mean_relative_diffs_test_Ip.npy', mean_relative_diffs_test_Ip)
    np.save(config['save_dir'] + '/residual_mean_relative_diffs_train_Phi.npy', mean_relative_diffs_train_Phi)
    np.save(config['save_dir'] + '/residual_mean_relative_diffs_test_Phi.npy', mean_relative_diffs_test_Phi)

    ## Iterative training
    linear_model.load_state_dict(torch.load(config['save_dir'] + '/hybrid_linear_model.pth'))
    residual_model.load_state_dict(torch.load(config['save_dir'] + '/hybrid_residual_model.pth'))
    std_layer_err = copy.deepcopy(std_layer_2)
    std_layer_err.load_state_dict(torch.load(config['save_dir'] + '/hybrid_std_layer_err.pth'))

    # Evaluation
    x_data_pred_traj_train, x_data_pca_traj_train, x_data_pca_pred_traj_train = generate_hybrid_trajectories(x_dataset_train, u_dataset_train, linear_model, residual_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, std_layer_err, device)
    x_data_pred_traj_test, x_data_pca_traj_test, x_data_pca_pred_traj_test = generate_hybrid_trajectories(x_dataset_test, u_dataset_test, linear_model, residual_model, std_layer_1, pca_layer, std_layer_2, std_layer_u, std_layer_err, device)
    # np.save(config['save_dir'] + '/x_data_pred_traj_train_hybrid.npy', x_data_pred_traj_train)
    # np.save(config['save_dir'] + '/x_data_pred_traj_test_hybrid.npy', x_data_pred_traj_test)
    
    # # Calculate mean relative error
    # mean_relative_errors_train = calculate_mean_relative_error_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_errors_test = calculate_mean_relative_error_set(x_dataset_test, x_data_pred_traj_test)
    # # print(x_data_pred_traj_train)

    # # Calculate mean relative diff
    # mean_relative_diffs_train = calculate_mean_relative_diff_set(x_dataset_train, x_data_pred_traj_train)
    # mean_relative_diffs_test = calculate_mean_relative_diff_set(x_dataset_test, x_data_pred_traj_test)

    # # Save results
    # np.save(config['save_dir'] + '/hybrid_mean_relative_errors_train.npy', mean_relative_errors_train)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_errors_test.npy', mean_relative_errors_test)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_train.npy', mean_relative_diffs_train)
    # np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_test.npy', mean_relative_diffs_test)

    x_data_pred_traj_train_Ip = [arr[:, -1:] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Ip = [arr[:, -1:] for arr in x_data_pred_traj_test]
    x_data_pred_traj_train_Phi = [arr[:, :-1] for arr in x_data_pred_traj_train]
    x_data_pred_traj_test_Phi = [arr[:, :-1] for arr in x_data_pred_traj_test]
    np.save(config['save_dir'] + '/x_data_pred_traj_train_hybrid_Ip.npy', x_data_pred_traj_train_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_hybrid_Ip.npy', x_data_pred_traj_test_Ip)
    np.save(config['save_dir'] + '/x_data_pred_traj_train_hybrid_Phi.npy', x_data_pred_traj_train_Phi)
    np.save(config['save_dir'] + '/x_data_pred_traj_test_hybrid_Phi.npy', x_data_pred_traj_test_Phi)

    mean_relative_errors_train_Ip = calculate_mean_relative_error_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_errors_test_Ip = calculate_mean_relative_error_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip)
    mean_relative_errors_train_Phi = calculate_mean_relative_error_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_errors_test_Phi = calculate_mean_relative_error_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)

    mean_relative_diffs_train_Ip = calculate_mean_relative_diff_set(x_dataset_train_Ip, x_data_pred_traj_train_Ip)
    mean_relative_diffs_test_Ip = calculate_mean_relative_diff_set(x_dataset_test_Ip, x_data_pred_traj_test_Ip) 
    mean_relative_diffs_train_Phi = calculate_mean_relative_diff_set(x_dataset_train_Phi, x_data_pred_traj_train_Phi)
    mean_relative_diffs_test_Phi = calculate_mean_relative_diff_set(x_dataset_test_Phi, x_data_pred_traj_test_Phi)

    np.save(config['save_dir'] + '/hybrid_mean_relative_errors_train_Ip.npy', mean_relative_errors_train_Ip)
    np.save(config['save_dir'] + '/hybrid_mean_relative_errors_test_Ip.npy', mean_relative_errors_test_Ip)
    np.save(config['save_dir'] + '/hybrid_mean_relative_errors_train_Phi.npy', mean_relative_errors_train_Phi)
    np.save(config['save_dir'] + '/hybrid_mean_relative_errors_test_Phi.npy', mean_relative_errors_test_Phi)

    np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_train_Ip.npy', mean_relative_diffs_train_Ip)
    np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_test_Ip.npy', mean_relative_diffs_test_Ip)
    np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_train_Phi.npy', mean_relative_diffs_train_Phi)
    np.save(config['save_dir'] + '/hybrid_mean_relative_diffs_test_Phi.npy', mean_relative_diffs_test_Phi)

    return

if __name__ == '__main__':
    main_evaluate()