"""
Created: 6/28/2025

Purpose: Test a system where weights are the same except for a shape, seeing how the shape effects the network
"""
from model_utils import model_string_generator, create_layers, NetworkSkeleton, GeneratedDataset, train, test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt

def make_net_blank(network: NetworkSkeleton):
    state_copy = copy.deepcopy(network.cpu().state_dict())
    for key in state_copy:
        state_copy[key] = torch.from_numpy(np.ones(state_copy[key].numpy().shape) * .1)
    network.load_state_dict(state_copy)

def apply_eye_shape(model: NetworkSkeleton):
    state_copy = copy.deepcopy(model.cpu().state_dict())
    for key in state_copy:
        try:
            if state_copy[key].numpy().shape[1] <= 2:
                print("too small!")
            elif "weight" in key:
                x_coord_pupil = random.randint(1, len(state_copy[key].numpy())-2)
                y_coord_pupil = random.randint(1, len(state_copy[key].numpy()[0])-2)
                temp_mat = state_copy[key].numpy()
                temp_mat[x_coord_pupil][y_coord_pupil] *= -1
                temp_mat[x_coord_pupil+1][y_coord_pupil] *= -.5
                temp_mat[x_coord_pupil-1][y_coord_pupil] *= -.5
                temp_mat[x_coord_pupil][y_coord_pupil+1] *= -.5
                temp_mat[x_coord_pupil][y_coord_pupil-1] *= -.5
                temp_mat[x_coord_pupil+1][y_coord_pupil+1] *= -.5
                temp_mat[x_coord_pupil+1][y_coord_pupil-1] *= -.5
                temp_mat[x_coord_pupil-1][y_coord_pupil+1] *= -.5
                temp_mat[x_coord_pupil-1][y_coord_pupil-1] *= -.5
                state_copy[key] = torch.from_numpy(temp_mat)
        except:
            pass
    model.load_state_dict(state_copy)

def apply_staircase_shape(model: NetworkSkeleton, diag_len: int = 3, descend: bool = False):
    state_copy = copy.deepcopy(model.cpu().state_dict())
    for key in state_copy:
        try:
            if state_copy[key].numpy().shape[1] <= 2:
                print("too small!")
            weight_numpy = state_copy[key].numpy()
            x_init = random.randint(0+diag_len, weight_numpy.shape[0]-1) if not descend else random.randint(0, weight_numpy.shape[0]-1-diag_len)
            y_init = random.randint(0, weight_numpy.shape[1]-1-diag_len)
            for i in reversed(range(diag_len)):
                weight_numpy[x_init][y_init] *= -1
                weight_numpy[x_init][y_init+1] *= -1
                if not descend:
                    x_init -= 1
                else:
                    x_init += 1
                y_init += 1
            state_copy[key] = torch.from_numpy(weight_numpy)
        except Exception as e:
            pass
    model.load_state_dict(state_copy)
            

def randomize_final_layer(model: NetworkSkeleton):
    state_copy = copy.deepcopy(model.cpu().state_dict())
    final_key = list(state_copy.keys())[-2]
    state_copy[final_key] = torch.rand(state_copy[final_key].shape)
    model.load_state_dict(state_copy) 

def display_net(state_dict, tracker):
    for key in state_dict:
        if "weight" in key:
            fig = plt.figure(tracker)
            plt.imshow(state_dict[key].numpy())
            plt.title(f"Weight Heatmap for Layer {tracker}")
            plt.colorbar()
            tracker += 1
    plt.show()

if __name__ == "__main__":
    epochs = 8
    tracker = 0
    loss = nn.BCEWithLogitsLoss()
    shape_funcs = [apply_eye_shape, apply_staircase_shape]
    num_shapes = 40
    activation_list = {"sig": nn.Sigmoid(), 'tanh': nn.Tanh()}
    model_str = model_string_generator(100, 1, 1, list(activation_list.keys()), (100, 300))
    print(f"Model String: {model_str}")
    base_model = NetworkSkeleton(create_layers(model_str,activation_list))
    make_net_blank(base_model)
    for i in range(num_shapes):
        choice = random.choice(shape_funcs)
        if choice == apply_staircase_shape:
            choice(base_model, random.randint(3, 13))
        else:
            choice(base_model)
    # randomize_final_layer(base_model)
    optimizer = torch.optim.SGD(base_model.parameters(), lr=7e-1)
    dataset = pd.read_csv("generated_data_sets/7000_100_10_classification_generated.csv")
    dataset.drop(dataset.columns[0], axis=1, inplace=True)
    x_vals = dataset[dataset.columns[dataset.columns != 'y']].to_numpy()
    y_vals =  dataset[dataset.columns[dataset.columns == 'y']].to_numpy()
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train, 50)
    data_loader_test = DataLoader(formatted_data_test, 50)
    states = base_model.state_dict()
    display_net(states, tracker)
    for i in range(epochs):
        train(data_loader_train, base_model, loss, optimizer, 'cpu')
        print(f"MSE Loss: {test(data_loader_test, base_model, loss, 'cpu')}")
        display_net(base_model.state_dict(), tracker)