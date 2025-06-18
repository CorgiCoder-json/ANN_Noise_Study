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

if __name__ == "__main__":
    # Define the Network Variables
    net_string = '100|150->llrelu->150|120->relu->120|200->hlrelu->200|1'
    string_to_activations = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'llrelu': nn.LeakyReLU(0.1), 'hlrelu': nn.LeakyReLU(1.1), 'sig': nn.Sigmoid(), 'tanh': nn.Tanh()}
    device = 'cuda'
    model = NetworkSkeleton(create_layers(net_string, string_to_activations)).to(device)
    loss = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), 1e-4)
    epochs = 8
    
    #Load and format the data into test and training sets
    dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
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
    data_loader_train = DataLoader(formatted_data_train, 50)
    data_loader_test = DataLoader(formatted_data_test, 50)
    base_layers = copy.deepcopy(model.cpu().layers)
    model.to(device)
    
    #Run the training loop
    for i in range(epochs):
        trained_layers = copy.deepcopy(model.cpu().layers)
        model.to(device)
        samp_x, samp_y = next(iter(data_loader_test))
        x_train = samp_x[0].to('cpu').type(torch.float)
        x_base = samp_x[0].to('cpu').type(torch.float)
        samp_y = np.squeeze(samp_y[0].to('cpu').type(torch.float).numpy())
        row_tracker = 1
        for index in range(len(trained_layers)):
            trained_pred = np.squeeze(trained_layers[index].forward(x_train).detach().numpy())
            base_pred = np.squeeze(base_layers[index].forward(x_base).detach().numpy())
            trained_loss = numpy_mse(trained_pred, samp_y)
            base_loss = numpy_mse(base_pred, samp_y)
            fig = plt.figure(0)
            try:
                reg_fit_trained = np.polyfit(trained_pred, trained_loss, 1)
                reg_fit_base = np.polyfit(base_pred, base_loss, 1)
                trained_line = reg_fit_trained[0] * trained_pred + reg_fit_trained[1]
                base_line = reg_fit_base[0] * base_pred + reg_fit_base[1]
                print(f"Trained R^2 score: {r2_score(trained_loss, trained_line)}")
                print(f"Base R^2 score: {r2_score(base_loss, base_line)}")
                plt.plot(trained_pred, trained_line, color='red', label='Trained Line')
                plt.plot(base_pred, base_line, color='blue', label='Base line')
            except:
                pass
            print(reg_fit_trained)
            print(reg_fit_base)
            print(type(trained_layers[index]))
            plt.scatter(trained_pred, trained_loss, color='red', label='Trained Neurons')
            plt.scatter(base_pred, base_loss, color='blue', label='Base Neurons')
            plt.title(f"Neuron Loss for Row {row_tracker}")
            plt.xlabel('Guessed Value (#)')
            plt.ylabel('Loss (BCE)')
            plt.legend()
            plt.show()
            if isinstance(trained_loss, np.float32):
                print(f"Final Trained Loss: {trained_loss}")
                print(f"Final Base Loss: {base_loss}")
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