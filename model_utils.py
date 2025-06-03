from pyexpat import model
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import random
import pandas as pd
import os.path as path

class NetworkSkeleton(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers.forward(x.type(torch.float))

def save_network_heatmap(model: NetworkSkeleton, window_size: int, starting_weight: tuple[int, int], label: str, starting_device: str, figure_tracker: int, save_file: str):
    model_copy = copy.deepcopy(model.cpu())
    fig_number = copy.deepcopy(figure_tracker)
    layer = 1
    for item in model_copy.state_dict():
        if item[-6:] == "weight":
            weight_slice = model_copy.state_dict()[item][starting_weight[0]:starting_weight[0]+window_size]
            fig = plt.figure(fig_number)
            plt.imshow([temp[starting_weight[1]:starting_weight[1]+window_size] for temp in weight_slice])
            plt.colorbar()
            plt.title(label)
            plt.savefig(f"{save_file}_layer_{layer}.png")
            fig_number += 1
            layer += 1
        else:
            continue
    model.to(starting_device)

def display_network(model: NetworkSkeleton, window_size: int, starting_weight: tuple[int, int], label: str, starting_device: str, figure_tracker: int, immediate_diplay: bool = False) -> None:
    model_copy = copy.deepcopy(model.cpu())
    fig_number = copy.deepcopy(figure_tracker)
    for item in model_copy.state_dict():
        if item[-6:] == "weight":
            weight_slice = model_copy.state_dict()[item][starting_weight[0]:starting_weight[0]+window_size]
            fig = plt.figure(fig_number)
            plt.imshow([temp[starting_weight[1]:starting_weight[1]+window_size] for temp in weight_slice])
            plt.colorbar()
            plt.title(label)
            fig_number += 1
        else:
            continue
    if immediate_diplay:
        plt.show()
    model.to(starting_device)
    


def create_layers(layer_str: str, str_to_activ: dict[str, nn.Module]) -> list[nn.Module]:
    """Converts a model string into a list of layers to be applied to a SkeletonNetwork

    Args:
        layer_str (str): The description of the model as a string (ex. 128|1->relu->1|14)
        str_to_activ (dict[str, nn.Module]): a dictionary with a str to match a function in the string to a real activation function module (ex. {'relu': nn.Relu()})

    Returns:
        list[nn.Module]: the nn.Modules that are from the layer_str
    """
    layers_list: list[str] = layer_str.split('->')
    converted_layers: list[nn.Module] = []
    for command in layers_list:
        try:
            dimensions = tuple(command.split('|'))
            in_dim: int = int(dimensions[0])
            out_dim: int = int(dimensions[1])
            converted_layers.append(nn.Linear(in_dim, out_dim))
        except:
            converted_layers.append(str_to_activ[command])
    return converted_layers
    
def model_string_generator(input_dim: int, num_hidden: int, num_output: int, activations: list[str], size_range: tuple[int, int]) -> str:
    """Generates a model string from an input dimension, number of hidden layers, and output dimension to be used
    for creating models, with some control of activation functions and number of weights in each linear layer

    Args:
        input_dim (int): the size of the data input
        num_hidden (int): number of hidden layers (SHOULD NOT INCLUDE INPUT/OUTPUT LAYERS)
        num_output (int): the dimension of the final answer
        activations (list[str]): the activation functions that can be randomly chosen
        size_range (tuple[int, int]): the min and max size of randomly generated layers

    Returns:
        str: a model string that can be used in the create_layers function
    """
    output_str = f"{input_dim}|"
    prev_output = random.randint(size_range[0], size_range[1])
    output_str += f"{prev_output}->"
    for i in range(num_hidden):
        new_out = random.randint(size_range[0], size_range[1])
        output_str += f"{random.choice(activations)}->{prev_output}|{new_out}->"
        prev_output = new_out
    output_str += f"{random.choice(activations)}->{prev_output}|{num_output}"
    return output_str

def model_str_file_name(model_str):
    file_path = model_str.replace('->', '_').replace('|', '-')
    return file_path 

def save_model_parameters(model, model_str, pre_or_post, problem_type, device):
    ID = 1
    model_copy = copy.deepcopy(model.cpu())
    params = model_copy.state_dict()
    layer = 1
    for item in params:
        if item[-6:] == "weight":
            table = pd.DataFrame(params[item])
            table.to_csv(path.join(f"{problem_type}", "weights", f"{model_str_file_name(model_str)}_{pre_or_post}_layer_{layer}.csv"),index=False)
        else:
            series = pd.Series(params[item])
            series.to_csv(path.join(f"{problem_type}", "biases", f"{model_str_file_name(model_str)}_{pre_or_post}_layer_{layer}.csv"),index=False)
            layer += 1
        ID += 1
    model.to(device)       
    
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
    test_loss /= num_batches
    return test_loss

if __name__ == "__main__":
    # net = NetworkSkeleton(create_layers('128|450->relu->450|130->relu->130|1', {'relu': nn.ReLU()}))
    # print(net)
    # display_network(net, 40, (0,0), "Test", 'cpu', 0, True)
    for i in range(10):
        print(model_string_generator(128, 1, 1, ['relu', 'sig', 'lrelu', 'silu'], (40, 450)))
    
    