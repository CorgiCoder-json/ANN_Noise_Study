"""
Created: 6/4/2025

Purpose: Add a dropout parmeter p to the onepass update system
"""
import numpy as np
import torch
import torch.nn as nn
import copy
import random
import pandas as pd
from scipy.stats import zscore
from model_utils import NetworkSkeleton, create_layers

def decay_constant(step: int, total_data_size: int):
    return np.exp(-1 * (1/total_data_size) * step)

def scaled_weight(data: np.ndarray):
    return 1 / (1 + np.exp(-1 * data))

def convert_to_weight(data: np.ndarray, volitility: float, total_data_size: int, step: int):
    """
    converts a given row of data into updateable weights

    Args:
        data (np.ndarray): the row of data that is to be converted. Should already be in z-score form
        volitility (float): controls how much change each weight should have. Higher values result in higher delta weights and vice versa for lower values 
        total_data_size (int): this size of the data set that the row comes from
        step (int): the row number of the data input
    """
    return np.sign(data) * ((decay_constant(step, total_data_size) * scaled_weight(data))) * volitility

def get_acitvations(model_str) -> list[str]:
    split_str = model_str.split('->')
    activation_functions = []
    for string in split_str:
        if "|" in string:
            continue
        else:
            activation_functions.append(string)
    return activation_functions

def train_model_dropout(model, dataset: pd.DataFrame, model_str, device, drop_out = 0.2):
    global global_device
    model.eval()
    with torch.no_grad():
        num_rows = len(dataset)
        v = 0.0001

        #Extract the weights
        model_copy = copy.deepcopy(model.cpu())
        data_copy = copy.deepcopy(dataset)
        weights = []
        biases = []
        result_sets = [[], [], []]
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                weights.append(model_copy.state_dict()[item].numpy())
            else:
                biases.append(model_copy.state_dict()[item].numpy())
        model.to(device)

        #TODO: Sort the data set by "outlierness" (sort by zscore of answer)
        data_copy["z_answers"] = zscore(data_copy['y'])
        data_copy["z_answers"] = data_copy['z_answers'].abs()
        sorted_data = data_copy.sort_values(by='z_answers')
        x = sorted_data.loc[:, sorted_data.columns != 'y'].drop(["z_answers"], axis=1).to_numpy()
        # data_copy = data_copy.sample(frac=1)
        # x = data_copy.loc[:, data_copy.columns != 'y'].to_numpy()

        #Step 1: move the data into the weight dimension
        for i, item in enumerate(x):
            new_item = item
            for j, weight in enumerate(weights):
                temp = torch.from_numpy(np.dot(weight, new_item) + biases[j])
                if j != 2:
                    new_item = nn.functional.silu(temp).numpy()
                result_sets[j].append(new_item)

        #Step 2: convert all of the data into zscores for their respective columns
        converted_sets = []
        for result in result_sets:
            converted_sets.append(zscore(result, axis=0))
        
        #pop the final row and add the initial set
        converted_sets.pop()
        converted_sets.insert(0, zscore(x, axis=0))   
        
        #Step 3: convert the zscores into delta weights
        #Step 4: apply the delta weights to ALL neurons in the layer
        for i, set in enumerate(converted_sets):
            for j, row in enumerate(set):
                if np.isnan(row.sum()):
                    row = np.nan_to_num(row)
                calculated_weights = convert_to_weight(row, v, num_rows, j)
                if(random.random() < drop_out):
                    weights[i] -= calculated_weights 
                else:
                    weights[i] += calculated_weights     

        #Step 5: reapply the weights to a new model and return
        index = 0
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                model_copy.state_dict()[item] = weights[i]
                index += 1
            else:
                continue
        model_copy.to(device)
        return model_copy

def train_model_one_loop(model, dataset: pd.DataFrame, model_str, device, str_to_activation, v):
    global global_device
    model.eval()
    with torch.no_grad():
        num_rows = len(dataset)

        #Extract the weights
        model_copy = copy.deepcopy(model.cpu())
        data_copy = copy.deepcopy(dataset)
        weights = []
        biases = []
        result_sets = [[], [], []]
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                weights.append(model_copy.state_dict()[item].numpy())
            else:
                biases.append(model_copy.state_dict()[item].numpy())
        model.to(device)

        #TODO: Sort the data set by "outlierness" (sort by zscore of answer)
        data_copy["z_answers"] = zscore(data_copy['y'])
        data_copy["z_answers"] = data_copy['z_answers'].abs()
        sorted_data = data_copy.sort_values(by='z_answers', ascending=True)
        x = sorted_data.loc[:, sorted_data.columns != 'y'].drop(["z_answers"], axis=1).to_numpy()
        # data_copy = data_copy.sample(frac=1)
        # x = data_copy.loc[:, data_copy.columns != 'y'].to_numpy()

        #Step 1: move the data into the weight dimension
        activations = get_acitvations(model_str)
        for i, item in enumerate(x):
            new_item = item
            for j, weight in enumerate(weights):
                item_zscore = zscore(new_item)
                delta_weights = convert_to_weight(item_zscore, v, num_rows, i)
                new_item = torch.from_numpy(np.dot(weight, new_item) + biases[j])
                if j < len(activations):
                    new_item = str_to_activation[activations[j]](new_item).numpy()
                weights[j] += delta_weights

        #Step 5: reapply the weights to a new model and return
        index = 0
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                model_copy.state_dict()[item] = weights[index]
                index += 1
            else:
                continue
        return_model = NetworkSkeleton(create_layers(model_str, {'relu': nn.ReLU(), 'silu': nn.SiLU()})).to(device)
        return_model.load_state_dict(model_copy.state_dict())
        return return_model