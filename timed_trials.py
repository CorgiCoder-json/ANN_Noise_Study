import time
from model_utils import create_layers, NetworkSkeleton, train, model_string_generator
from One_Pass_Update import train_model
from dropout_pass_update import train_model_one_loop, train_model_torch_boost
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import pandas as pd
from scipy.stats import zscore

#NOTE: The copy is so that the timings can be accurate. The imported version will be used as a "dirty" run,
# where the copy is included. The first runs will include this timing. It could be that the copy is negligible, but who knows

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

def one_epoch_test_same_model(train_data, train_data_loader):
    my_method = [] 
    gradient_method = []
    model_string = model_string_generator(100, 1, 1, ['relu', 'silu'], (100, 200))
    for i in range(30):
        model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
        model_copy = copy.deepcopy(model)
        start_time = time.time()
        train_model(model, train_data, model_string, 'cpu', {'relu': nn.ReLU(), 'silu': nn.SiLU()}, 0.00004)
        end_time = time.time()
        my_method.append(end_time - start_time)
        start_time = time.time()
        train(train_data_loader, model_copy, nn.MSELoss(), torch.optim.Adam(model_copy.parameters(), lr=1e-5), 'cpu')
        end_time = time.time()
        gradient_method.append(end_time - start_time)
    return my_method, gradient_method   

def one_epoch_test_random_model(train_data, train_data_loader):
    my_method = [] 
    gradient_method = []
    for i in range(30):
        model_string = model_string_generator(100, 1, 1, ['relu', 'silu'], (100, 200))
        model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
        model_copy = copy.deepcopy(model)
        start_time = time.time()
        train_model_torch_boost(model, train_data, model_string, 'cpu', 0.00004, 100)
        end_time = time.time()
        my_method.append(end_time - start_time)
        start_time = time.time()
        train(train_data_loader, model_copy, nn.MSELoss(), torch.optim.SGD(model_copy.parameters(), lr=1e-5), 'cpu')
        end_time = time.time()
        gradient_method.append(end_time - start_time)
    return my_method, gradient_method

def ramp_data_random_model():
    pass   

if __name__ == "__main__":
    imp_dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    x_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns != 'y']].to_numpy()
    y_vals =  imp_dataset[imp_dataset.columns[imp_dataset.columns == 'y']].to_numpy()
    dataset = pd.DataFrame(x_vals[int(len(x_vals)*.2):])
    dataset["y"] = y_vals[int(len(y_vals)*.2):]
    dataset["z_answers"] = zscore(dataset['y'])
    dataset["z_answers"] = dataset['z_answers'].abs()
    sorted_data = dataset.sort_values(by='z_answers', ascending=True).drop(["z_answers"], axis=1)
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train, batch_size=100)
    data_loader_test = DataLoader(formatted_data_test, batch_size=100)
    print(one_epoch_test_random_model(sorted_data, data_loader_train))