from One_Pass_Update import train_model 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_utils import NetworkSkeleton, create_layers, test, train, display_network, save_model_parameters
import copy
import matplotlib.pyplot as plt
import math
import seaborn as sns
global_device = 'cpu'

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

def get_percent_imporvement(start_loss, min_loss):
    return (math.fabs(start_loss - min_loss) / start_loss) * 100.0

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
    trained_copy_loss = []
    losses = []
    loss_fn = nn.MSELoss()
    model_string = '100|200->relu->200|150->relu->150|1'
    for j in range(32):
        temp_model = NetworkSkeleton(create_layers(model_string, {'relu': nn.ReLU(), 'silu': nn.SiLU()}))
        model_copy = copy.deepcopy(temp_model)
        model_copy.to(global_device)
        temp_model.to(global_device)
        trained_model = train_model(temp_model, dataset, model_string, global_device)
        min_acc = np.inf
        trained_rounds = 0
        minimum_model: NetworkSkeleton = NetworkSkeleton([])
        save_model_parameters(temp_model, model_string, f'base_round_{j}', 'D:\\pass_gradient_exp\\regression', global_device)
        for i in range(8):
            print(f"MSE OF THE TRAINED MODEL AFTER TRAINING ROUND {i}: ")
            acc = test(data_loader_test, trained_model, loss_fn, device=global_device)
            print(f"Loss: {acc}")
            losses.append(acc)
            if acc < min_acc:
                minimum_model = copy.deepcopy(trained_model.cpu())
                trained_model.to(global_device)
                min_acc = acc
                trained_rounds = i
            trained_model = train_model(trained_model, dataset, model_string, global_device)
        print("One pass step completed. Testing gradient descent...")
        minimum_optim = torch.optim.SGD(minimum_model.parameters(), lr=1e-5)
        copy_optim = torch.optim.SGD(model_copy.parameters(), lr=1e-5)
        for i in range(8):
            train(data_loader_train, minimum_model, loss_fn, minimum_optim, global_device)
            print(f"MSE Loss for One Pass + Gradient: {test(data_loader_test, minimum_model, loss_fn, device=global_device)}")
            train(data_loader_train, model_copy, loss_fn, copy_optim, global_device)
            print(f"MSE Loss for Gradient: {test(data_loader_test, model_copy, loss_fn, device=global_device)}")
        save_model_parameters(temp_model, model_string, f'one_pass_grad_round_{j}', 'D:\\pass_gradient_exp\\regression', global_device)
        save_model_parameters(temp_model, model_string, f'just_grad_round_{j}', 'D:\\pass_gradient_exp\\regression', global_device)
        min_loss = test(data_loader_test, minimum_model, loss_fn, global_device)
        trained_min_loss.append(min_loss)
        copy_loss = test(data_loader_test, model_copy, loss_fn, device=global_device)
        trained_copy_loss.append(copy_loss)
        improvement = get_percent_imporvement(copy_loss, min_loss)
        percent_improvements.append(improvement)
        print("Round {j} complete.")
    plt.show()
        