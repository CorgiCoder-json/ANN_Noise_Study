from sklearn.datasets import make_regression
from model_utils import save_model_parameters, save_network_heatmap, create_layers, model_string_generator, train, test, model_str_file_name, NetworkSkeleton
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import os.path as path
import random

activation_pool = {'relu': nn.ReLU(), 
                   'sig': nn.Sigmoid(), 
                   'tanh': nn.Tanh(), 
                   'splus': nn.Softplus(), 
                   'swish': nn.SiLU(), 
                   'smax': nn.Softmax(), 
                   'llrelu': nn.LeakyReLU(), 
                   'hlrelu': nn.LeakyReLU(1.01), 
                   }

optimizers = [
    torch.optim.SGD,
    torch.optim.Adam,
    torch.optim.RMSprop,
    torch.optim.Rprop
]

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

def run_tests(data, device, num_test, epochs, min_error, error_func, optimize, activation_pool, insert_path, base_model = None):
    data_load_train, data_load_test = None, None
    if isinstance(data, np.ndarray):    
        data_set_train = GeneratedDataset(data[0][int(len(data[0])*.2):], data[1][int(len(data[1])*.2):])
        data_set_test = GeneratedDataset(data[0][:int(len(data[0])*.2)], data[1][:int(len(data[1])*.2)])
        data_load_train = DataLoader(data_set_train, batch_size=50, shuffle=True)
        data_load_test = DataLoader(data_set_test, batch_size=50, shuffle=True)
    if isinstance(data, tuple):
        data_load_train = data[0]
        data_load_test = data[1]
    lr = 1e-4
    for i in range(num_test):
        if base_model == None:
            model_str = model_string_generator(100, random.choice([0,1,2]), 1, list(activation_pool.keys()), (40, 450))
            model = NetworkSkeleton(create_layers(model_str, activation_pool))
        save_model_parameters(model, model_str, insert_path, 'pre', device)
        j = 0
        acc = min_error + 1
        while j < epochs and min_error < acc:
            train(data_load_train, model, error_func, optimize(model.parameters(), lr=lr),device)
            acc = test(data_load_test, model, error_func, device)
            j += 1
        save_model_parameters(model, model_str, insert_path, 'post', device)
        with open(path.join(insert_path, model_str_file_name(model_str) + "_report.txt"), 'wt') as file:
            file.write(f"Model String: {model_str}\nModel Accuracy: {acc}\nEpochs Ran: {j}\nOptimizer: {optimize}\nLearning Rate: {1e-4}\nError Function: {error_func}")
            
if __name__ == "__main__":
    data = make_regression(n_samples = 5000, n_features=100, n_informative=10)
    run_tests(data, 'cuda', 1, 25,   100, nn.MSELoss(), torch.optim.SGD, activation_pool)
    print("Hello world")