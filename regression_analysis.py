import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from model_utils import NetworkSkeleton, create_layers, GeneratedDataset, train, test
import matplotlib.pyplot as plt
import numpy as np
import copy

if __name__ == "__main__":
    # Define the Network Variables
    net_string = '100|150->hlrelu->150|120->silu->120|200->llrelu->200|1'
    string_to_activations = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'llrelu': nn.LeakyReLU(0.1), 'hlrelu': nn.LeakyReLU(1.1)}
    device = 'cuda'
    model = NetworkSkeleton(create_layers(net_string, string_to_activations)).to(device)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    epochs = 8
    
    #Load and format the data into test and training sets
    dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    x_vals = dataset[dataset.columns[dataset.columns != 'y']].to_numpy()
    y_vals =  dataset[dataset.columns[dataset.columns == 'y']].to_numpy()
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train, 50)
    data_loader_test = DataLoader(formatted_data_test, 50)
    base_layers = copy.deepcopy(model.cpu().layers)
    model.to(device)
    
    #Run the training loop
    for i in range(epochs):
        train(data_loader_train, model, loss, optim, device)
        print(f"Loss: {test(data_loader_train, model, loss, device)}")
        
    #Seperate the layers of the network and format a sample group
    trained_layers = copy.deepcopy(model.cpu().layers)
    model.to(device)
    step_sample = dataset.sample(n=10)
    samp_x_vals = step_sample[step_sample.columns[step_sample.columns != 'y']].to_numpy()
    samp_y_vals =  step_sample[step_sample.columns[step_sample.columns == 'y']].to_numpy()
    formatted_step_test = GeneratedDataset(samp_x_vals, samp_y_vals)
    formatted_loader = DataLoader(formatted_step_test)
    row_tracker = 1
    for x, y in formatted_loader:
        x_train = x.to('cpu').type(torch.float)
        x_base = x.to('cpu').type(torch.float)
        y = np.squeeze(y.to('cpu').type(torch.float).numpy())
        for index in range(len(trained_layers)):
            print(type(trained_layers[index]))
            trained_pred = np.squeeze(trained_layers[index].forward(x_train).detach().numpy())
            base_pred = np.squeeze(base_layers[index].forward(x_base).detach().numpy())
            trained_loss = (trained_pred - y) ** 2
            base_loss = (base_pred - y) ** 2
            fig = plt.figure(0)
            plt.scatter(trained_loss, trained_pred, color='red', label='Trained Neurons')
            plt.scatter(base_loss, base_pred, color='blue', label='Base Neurons')
            plt.title(f"Neuron Loss for Row {row_tracker}")
            plt.xlabel('Loss (MSE)')
            plt.ylabel('Guessed Value (#)')
            plt.legend()
            plt.show()
            if isinstance(trained_loss, np.float32):
                print(f"Final Trained Loss: {trained_loss}")
                print(f"Final Base Loss: {base_loss}")
            x_train = torch.from_numpy(trained_pred).type(torch.float)
            x_base = torch.from_numpy(base_pred).type(torch.float)
        row_tracker += 1
    
    print("Hello World!")