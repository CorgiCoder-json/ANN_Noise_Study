import time
from model_utils import create_layers, NetworkSkeleton, train, model_string_generator
from One_Pass_Update import train_model, get_percent_imporvement
from dropout_pass_update import train_model_one_loop, train_model_torch_boost, train_model_torch_thread
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

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
    my_method = 0 
    gradient_method = 0
    model_string = model_string_generator(100, 1, 1, ['relu', 'silu'], (100, 200))
    model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
    model_copy = copy.deepcopy(model)
    start_time = time.time()
    train_model(model, train_data, model_string, 'cpu', {'relu': nn.ReLU(), 'silu': nn.SiLU()}, 0.00004)
    end_time = time.time()
    my_method = end_time - start_time
    start_time = time.time()
    train(train_data_loader, model_copy, nn.MSELoss(), torch.optim.Adam(model_copy.parameters(), lr=1e-5), 'cpu')
    end_time = time.time()
    gradient_method = end_time - start_time
    return my_method, gradient_method   

def one_epoch_test_random_model(train_data, train_data_loader):
    my_method = 0 
    gradient_method = 0
    model_string = model_string_generator(100, 1, 1, ['relu', 'silu'], (100, 200))
    model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
    model_copy = copy.deepcopy(model)
    start_time = time.time()
    train_model_torch_thread(model, train_data, model_string, 'cpu', 0.00004, 20)
    end_time = time.time()
    my_method = end_time - start_time
    start_time = time.time()
    train(train_data_loader, model_copy, nn.MSELoss(), torch.optim.SGD(model_copy.parameters(), lr=1e-5, foreach=False, fused=False), 'cpu')
    end_time = time.time()
    gradient_method = end_time - start_time
    return my_method, gradient_method

def one_epoch_test_random_model_2(train_data):
    my_method = 0 
    gradient_method = 0
    model_string = model_string_generator(100, 1, 1, ['relu', 'silu'], (100, 200))
    model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
    model_copy = copy.deepcopy(model)
    start_time = time.time()
    train_model_torch_thread(model, train_data, model_string, 'cpu', 0.00004, 100)
    end_time = time.time()
    my_method = end_time - start_time
    start_time = time.time()
    train_model_torch_boost(model_copy, train_data, model_string, 'cpu', 0.00004, 100)
    end_time = time.time()
    gradient_method = end_time - start_time
    return my_method, gradient_method

def ramp_data_random_model(files):
    my_method = []
    gradient_method = []
    for file in files:
        imp_dataset = pd.read_csv(file)
        if imp_dataset.shape[1] > 101:
            imp_dataset = imp_dataset.drop([imp_dataset.columns[0]], axis=1)
        x_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns != 'y']].to_numpy()
        y_vals =  imp_dataset[imp_dataset.columns[imp_dataset.columns == 'y']].to_numpy()
        dataset = pd.DataFrame(x_vals[int(len(x_vals)*.2):])
        dataset["y"] = y_vals[int(len(y_vals)*.2):]
        dataset["z_answers"] = zscore(dataset['y'])
        dataset["z_answers"] = dataset['z_answers'].abs()
        sorted_data = dataset.sort_values(by='z_answers', ascending=True).drop(["z_answers"], axis=1)
        formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
        formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
        data_loader_train = DataLoader(formatted_data_train, batch_size=20)
        data_loader_test = DataLoader(formatted_data_test, batch_size=20)
        results = one_epoch_test_random_model(sorted_data, data_loader_train)
        my_method.append(results[0])
        gradient_method.append(results[1])
    return my_method, gradient_method
           

if __name__ == "__main__":
    # imp_dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    # x_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns != 'y']].to_numpy()
    # y_vals =  imp_dataset[imp_dataset.columns[imp_dataset.columns == 'y']].to_numpy()
    # dataset = pd.DataFrame(x_vals[int(len(x_vals)*.2):])
    # dataset["y"] = y_vals[int(len(y_vals)*.2):]
    # dataset["z_answers"] = zscore(dataset['y'])
    # dataset["z_answers"] = dataset['z_answers'].abs()
    # sorted_data = dataset.sort_values(by='z_answers', ascending=True).drop(["z_answers"], axis=1)
    # formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    # formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    # data_loader_train = DataLoader(formatted_data_train, batch_size=100)
    # data_loader_test = DataLoader(formatted_data_test, batch_size=100)
    # for i in range(30):
    #     thread_boost, torch_boost = one_epoch_test_random_model_2(sorted_data)
    #     percent_boost = get_percent_imporvement(torch_boost, thread_boost)
    #     print(f"ROUND {i}:\n   THREAD + TORCH: {thread_boost}\n   TORCH: {torch_boost}\nROUND INCREASE PERCENT: {percent_boost}\n")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    files = [
        './generated_data_sets/2000_100_10_regression_generated.csv',
        './generated_data_sets/small_5000_100_10_regression_generated.csv',
        './generated_data_sets/7000_100_10_regression_generated.csv',
        './generated_data_sets/10000_100_10_regression_generated.csv',
        './generated_data_sets/12000_100_10_regression_generated.csv',
        './generated_data_sets/15000_100_10_regression_generated.csv',
        './generated_data_sets/17000_100_10_regression_generated.csv',
        './generated_data_sets/20000_100_10_regression_generated.csv'
    ]
    ramp_test = ramp_data_random_model(files)
    graph_x = range(len(files))
    print(ramp_test)
    fig = plt.figure(0)
    plt.plot(graph_x, ramp_test[0], 'o:r', label='My Method')
    plt.plot(graph_x, ramp_test[1], 's:b', label='Gradient')
    plt.legend()
    plt.show()