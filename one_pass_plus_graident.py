from One_Pass_Update import train_model 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_utils import NetworkSkeleton, create_layers, test, train
import copy
global_device = 'cuda'

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])


if __name__ == "__main__":
    dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    x_vals = dataset[dataset.columns[dataset.columns != 'y']].to_numpy()
    y_vals =  dataset[dataset.columns[dataset.columns == 'y']].to_numpy()
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train)
    data_loader_test = DataLoader(formatted_data_test)
    percent_improvements = []
    trained_min_loss = []
    losses = []
    model_string = '100|200->silu->200|150->silu->150|1'
    temp_model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
    model_copy = copy.deepcopy(temp_model)
    model_copy.to(global_device)
    temp_model.to(global_device)
    trained_model = train_model(temp_model, dataset, model_string, global_device)
    min_acc = np.inf
    trained_rounds = 0
    minimum_model: NetworkSkeleton = NetworkSkeleton([])
    #save_model_parameters(temp_model, '100|128->relu->128|128->relu->128|1', f'pre_round_{j}', 'D:\\regression', global_device)
    for i in range(8):
        print(f"MSE OF THE TRAINED MODEL AFTER TRAINING ROUND {i}: ")
        acc = test(data_loader_test, trained_model, nn.MSELoss(), device=global_device)
        print(f"Loss: {acc}")
        losses.append(acc)
        if acc < min_acc:
            minimum_model = trained_model
            min_acc = acc
            trained_rounds = i
        trained_model = train_model(trained_model, dataset, model_string, global_device)
    print("One pass step completed. Testing gradient descent...")
    for i in range(8):
        train(data_loader_train, minimum_model, nn.MSELoss(), torch.optim.SGD(minimum_model.parameters(), lr=1e-3), global_device)
        print(test(data_loader_test, minimum_model, nn.MSELoss(), device=global_device))
    print("Gradient Descent + one pass completed. Testing random model...")
    for i in range(8):
        train(data_loader_train, model_copy, nn.MSELoss(), torch.optim.SGD(model_copy.parameters(), lr=1e-3), global_device)
        print(test(data_loader_test, model_copy, nn.MSELoss(), device=global_device))
    print("Experiments complete! Review.")
        