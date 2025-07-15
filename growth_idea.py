"""
Created: 6/28/2025

Purpose: Test a system where weights are the same except for a shape, seeing how the shape effects the network
"""
from unittest.main import MODULE_EXAMPLES
from networkx import generate_gexf
from model_utils import model_string_generator, create_layers, NetworkSkeleton, GeneratedDataset, train, test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt

class Shape:
    def __init__(self, shape_array: list[list[float]], shape_center: tuple[int, int], placement_coords: None | tuple[int, int] = None, layer_limit: None | list[int] = None):
        self.template = np.array(shape_array)
        self.template_center = shape_center
        self.set_coordinates = placement_coords
        self.layer_limits = layer_limit
    def apply_to_model(self, model: NetworkSkeleton):
        state_copy = copy.deepcopy(model.cpu().state_dict())
        layer_tracker = 1
        was_none = True if self.set_coordinates is None else False
        for key in state_copy:
            if "weight" in key:
                if (1 in state_copy[key].numpy().shape and self.template.shape[0] != 1) or (self.layer_limits is not None and layer_tracker not in self.layer_limits):
                    continue
                else:
                    if self.set_coordinates == None:
                        gen_x = random.randint(self.template_center[0], len(state_copy[key].numpy()) - 1 - self.template.shape[0] - self.template_center[0])
                        gen_y = random.randint(self.template_center[1], len(state_copy[key].numpy()[0]) - 1 - self.template.shape[1] - self.template_center[1])
                        self.set_coordinates = (gen_x, gen_y)
                    for i in range(len(self.template)):
                        for j, item in enumerate(self.template[i]):
                            model_x = self.set_coordinates[0] - (self.template_center[0] - i)
                            model_y = self.set_coordinates[1] - (self.template_center[1] - j)
                            state_copy[key].numpy()[model_x][model_y] *= self.template[i][j]
            else:
                layer_tracker += 1
            if was_none:
                self.set_coordinates = None
        model.load_state_dict(state_copy)

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
    inv_eye_pattern = [[-1,-1,-1], [-1,-0.5,-1], [-1,-1,-1]]
    eye_pattern = [[-0.5,-0.5,-0.5], [-0.5,-1,-0.5], [-0.5,-0.5,-0.5]]
    random_pattern = (-1 * np.random.random((2, 4))).tolist()
    eye_shape = Shape(eye_pattern, (1, 1), None, [1])
    inv_eye_shape = Shape(inv_eye_pattern, (1, 1), None, [2])
    random_shape = Shape(random_pattern, (1, 3), None, [1, 2])
    epochs = 8
    tracker = 0
    loss = nn.BCEWithLogitsLoss()
    shape_funcs = [apply_eye_shape, apply_staircase_shape]
    num_shapes = 40
    activation_list = {"sig": nn.Sigmoid(), 'tanh': nn.Tanh()}
    model_str = '100|200->sig->200|150->tanh->150|1'
    print(f"Model String: {model_str}")
    base_model = NetworkSkeleton(create_layers(model_str,activation_list))
    make_net_blank(base_model)
    for i in range(num_shapes):
        eye_shape.apply_to_model(base_model)
        inv_eye_shape.apply_to_model(base_model)
        random_shape.apply_to_model(base_model)
    sgd_copy = copy.deepcopy(base_model)
    rms_copy = copy.deepcopy(base_model)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    optimizer_sgd = torch.optim.SGD(sgd_copy.parameters(), 1e-2)
    optimizer_rms = torch.optim.RMSprop(rms_copy.parameters(), 1e-3)
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
        print(f"BCE Adam Loss: {test(data_loader_test, base_model, loss, 'cpu')}")
        train(data_loader_train, sgd_copy, loss, optimizer_sgd, 'cpu')
        print(f"BCE SGD Loss: {test(data_loader_test, sgd_copy, loss, 'cpu')}")
        train(data_loader_train, rms_copy, loss, optimizer_rms, 'cpu')
        print(f"BCE RMSProp Loss: {test(data_loader_test, rms_copy, loss, 'cpu')}")
    display_net(base_model.state_dict(), tracker)
    display_net(sgd_copy.state_dict(), tracker)
    display_net(rms_copy.state_dict(), tracker)