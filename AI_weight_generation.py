from sklearn.datasets import make_regression
from model_utils import save_model_parameters, save_network_heatmap, create_layers, model_string_generator, train, test, model_str_file_name, NetworkSkeleton
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import os.path as path

activation_pool = {'relu': nn.ReLU(), 'sig': nn.Sigmoid(), 'tanh': nn.Tanh(), 'splus': nn.Softplus(), 'swish': nn.SiLU(), 'smax': nn.Softmax()}

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

def run_tests_regression(data, device, num_test, epochs, min_error, error_func, optimize):
    global activation_pool
    data_set_train = GeneratedDataset(data[0][int(len(data[0])*.2):], data[1][int(len(data[1])*.2):])
    data_set_test = GeneratedDataset(data[0][:int(len(data[0])*.2)], data[1][:int(len(data[1])*.2)])
    data_load_train = DataLoader(data_set_train, batch_size=50, shuffle=True)
    data_load_test = DataLoader(data_set_test, batch_size=50, shuffle=True)
    tracker = 0
    for i in range(num_test):
        model_str = model_string_generator(100, 1, 1, list(activation_pool.keys()), (40, 450))
        model = NetworkSkeleton(create_layers(model_str, activation_pool))
        save_network_heatmap(model, 50, (0,0), model_str + '_pre', device, tracker,  path.join("regression","heatmaps",model_str_file_name(model_str) + "_pre"))
        save_model_parameters(model, model_str, 'pre', 'regression', device)
        j = 0
        acc = min_error + 1
        while j < epochs and min_error < acc:
            train(data_load_train, model, error_func, optimize(model.parameters(), lr=1e-4),device)
            acc = test(data_load_test, model, error_func, device)
            print(acc)
            j += 1
        save_network_heatmap(model, 50, (0,0),  model_str + '_post', device, tracker, path.join("regression","heatmaps",model_str_file_name(model_str) + "_post"))
        save_model_parameters(model, model_str, 'post', 'regression', device)
        with open(path.join("regression", "reports", model_str_file_name(model_str) + "_report.txt"), 'wt') as file:
            file.write(f"Model String: {model_str}\nModel Accuracy: {acc}\nEpochs Ran: {j}\nOptimizer: {optimize}\nError Function: {error_func}")
            
if __name__ == "__main__":
    data = make_regression(n_samples = 5000, n_features=100, n_informative=10)
    run_tests_regression(data, 'cuda', 1, 25, 500, nn.MSELoss(), torch.optim.SGD)
    print("Hello world")