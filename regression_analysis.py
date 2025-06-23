"""
pytorch docs were used to find the loss functions for numpy calculation
BCE: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#bceloss
MAE (L1Loss): https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
MSE: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from model_utils import NetworkSkeleton, create_layers, GeneratedDataset, train, test
from dropout_pass_update import train_model_torch_boost
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import zscore
from sklearn.metrics import r2_score

def numpy_mse(pred, true):
    return (pred - true) ** 2

def numpy_mae(pred, true):
    return np.abs(pred - true)

def numpy_bce(pred, true):
    return true*(np.log(pred)) + (1-true)*np.log(1-pred)

def display_loss_pred_graph(trained_predictions, base_predictions, true_values, reg_loss_fn):
    for index in range(len(trained_predictions)):
        trained_loss = reg_loss_fn(trained_predictions[index], true_values[index])
        base_loss = reg_loss_fn(base_predictions[index], true_values[index])
        trained_regression = np.polyfit(trained_predictions[index], trained_loss, 1)
        base_regression = np.polyfit(base_predictions[index], base_loss, 1)
        trained_line = trained_regression[0] * trained_predictions[index] + trained_regression[1]
        base_line = base_regression[0] * base_predictions[index] + base_regression[1]
        plt.plot(trained_predictions[index], trained_line, label=f'Trained Regression {index}')
        plt.scatter(trained_predictions[index], trained_loss, label=f'Trained points {index}')
        plt.plot(base_predictions[index], base_line, label=f'Trained Regression {index}')
        plt.scatter(base_predictions[index], base_loss, label=f'Trained points {index}')
    

if __name__ == "__main__":
    # Define the Network Variables
    net_string = '100|150->silu->150|120->relu->120|200->relu->200|1'
    string_to_activations = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'llrelu': nn.LeakyReLU(0.1), 'hlrelu': nn.LeakyReLU(1.1), 'sig': nn.Sigmoid(), 'tanh': nn.Tanh()}
    device = 'cuda'
    model = NetworkSkeleton(create_layers(net_string, string_to_activations)).to(device)
    loss = nn.L1Loss()
    optim = torch.optim.RMSprop(model.parameters(), 1e-3)
    epochs = 8
    
    #Load and format the data into test and training sets
    dataset = pd.read_csv("generated_data_sets/5000_100_10_regression_generated.csv")
    # dataset.drop(dataset.columns[0], axis=1, inplace=True)
    x_vals = dataset[dataset.columns[dataset.columns != 'y']].to_numpy()
    y_vals =  dataset[dataset.columns[dataset.columns == 'y']].to_numpy()
    dataset = pd.DataFrame(x_vals[int(len(x_vals)*.2):])
    dataset["y"] = y_vals[int(len(y_vals)*.2):]
    dataset["z_answers"] = zscore(dataset['y'])
    dataset["z_answers"] = dataset['z_answers'].abs()
    sorted_data = dataset.sort_values(by='z_answers', ascending=True).drop(["z_answers"], axis=1)
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train, 2)
    data_loader_test = DataLoader(formatted_data_test, 2)
    base_layers = copy.deepcopy(model.cpu().layers)
    model.to(device)
    
    #Run the training loop
    for i in range(epochs):
        trained_layers = copy.deepcopy(model.cpu().layers)
        model.to(device)
        samp_x, samp_y = next(iter(data_loader_test))
        x_train = samp_x.to('cpu').type(torch.float)
        x_base = samp_x.to('cpu').type(torch.float)
        samp_y = samp_y.to('cpu').type(torch.float).numpy()
        row_tracker = 1
        for index in range(len(trained_layers)):
            trained_pred = np.squeeze(trained_layers[index].forward(x_train).detach().numpy())
            base_pred = np.squeeze(base_layers[index].forward(x_base).detach().numpy())
            try:
                display_loss_pred_graph(trained_pred, base_pred, samp_y, numpy_mae)
                plt.title(f"Layer Guessed Values v. Loss")
                plt.xlabel('Guessed Value (#)')
                plt.ylabel('Loss (BCE)')
                plt.show()
            except:
                pass
            x_train = torch.from_numpy(trained_pred).type(torch.float)
            x_base = torch.from_numpy(base_pred).type(torch.float)
        row_tracker += 1
        train(data_loader_train, model, loss, optim, device)
        print(f"Loss: {test(data_loader_test, model, loss, device)}")

        
        
        
    #Seperate the layers of the network and format a sample group
    # trained_layers = copy.deepcopy(model.cpu().layers)
    # model.to(device)
    # step_sample = dataset.sample(n=10)
    # samp_x_vals = step_sample[step_sample.columns[step_sample.columns != 'y']].to_numpy()
    # samp_y_vals =  step_sample[step_sample.columns[step_sample.columns == 'y']].to_numpy()
    # formatted_step_test = GeneratedDataset(samp_x_vals, samp_y_vals)
    # formatted_loader = DataLoader(formatted_step_test)
    # row_tracker = 1
    # for x, y in formatted_loader:
    #     x_train = x.to('cpu').type(torch.float)
    #     x_base = x.to('cpu').type(torch.float)
    #     y = np.squeeze(y.to('cpu').type(torch.float).numpy())
    #     for index in range(len(trained_layers)):
    #         print(type(trained_layers[index]))
    #         trained_pred = np.squeeze(trained_layers[index].forward(x_train).detach().numpy())
    #         base_pred = np.squeeze(base_layers[index].forward(x_base).detach().numpy())
    #         trained_loss = (trained_pred - y) ** 2
    #         base_loss = (base_pred - y) ** 2
    #         fig = plt.figure(0)
    #         plt.scatter(trained_loss, trained_pred, color='red', label='Trained Neurons')
    #         plt.scatter(base_loss, base_pred, color='blue', label='Base Neurons')
    #         plt.title(f"Neuron Loss for Row {row_tracker}")
    #         plt.xlabel('Loss (MSE)')
    #         plt.ylabel('Guessed Value (#)')
    #         plt.legend()
    #         plt.show()
    #         if isinstance(trained_loss, np.float32):
    #             print(f"Final Trained Loss: {trained_loss}")
    #             print(f"Final Base Loss: {base_loss}")
    #         x_train = torch.from_numpy(trained_pred).type(torch.float)
    #         x_base = torch.from_numpy(base_pred).type(torch.float)
    #     row_tracker += 1
    
    print("Hello World!")