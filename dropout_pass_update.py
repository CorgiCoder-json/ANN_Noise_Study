"""
Created: 6/4/2025

Purpose: Add a dropout parmeter p to the onepass update system
"""
from hmac import new
import numpy as np
import torch
import torch.nn as nn
import copy
import random
import pandas as pd
from scipy.stats import zscore
from model_utils import NetworkSkeleton, create_layers, test
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

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
    total_iters = 0
    model.eval()
    with torch.no_grad():
        num_rows = len(dataset)

        #Extract the weights
        model_copy = copy.deepcopy(model.cpu())
        data_copy = copy.deepcopy(dataset)
        weights = []
        biases = []
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
                total_iters += 1

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
        print(f"The train model algorithm  using only oen loop had {total_iters} iterations")
        return return_model
    
def train_model_torch_boost(model, dataset: pd.DataFrame, model_str, device, v, batch_size):
    global global_device
    total_iters = 0
    model.eval()
    with torch.no_grad():
        num_rows = len(dataset)

        #Extract the weights
        model_copy = copy.deepcopy(model.cpu())
        data_copy = copy.deepcopy(dataset)
        x = data_copy.loc[:, data_copy.columns != 'y'].to_numpy()
        y_vals = data_copy.loc[:, data_copy.columns == 'y'].to_numpy()
        result_sets = [zscore(x, axis=0)]
        weights = []
        model.to(device)

        #TODO: Sort the data set by "outlierness" (sort by zscore of answer)
        # data_copy = data_copy.sample(frac=1)
        # x = data_copy.loc[:, data_copy.columns != 'y'].to_numpy()

        #Step 1: move the data into the weight dimension
        index = 1
        temp_layers = model_copy.layers
        end_flag = False
        for layer in range(0,len(temp_layers),2):
            new_data = result_sets[-1] if len(result_sets) > 0 else x
            loader = DataLoader(GeneratedDataset(new_data, y_vals), batch_size=batch_size)
            weights.append(temp_layers[layer].weight.numpy())
            for X, y in loader:
                try:
                    result = temp_layers[layer+1](temp_layers[layer](X.type(torch.float))).numpy()
                except Exception as e:
                    end_flag = True
                    break
                try:
                    result_sets[index] = np.append(result_sets[index], result, axis=0)
                except Exception as e:
                    result_sets.append(result)
            if not end_flag:
                result_sets[index] = zscore(result_sets[index])
            index += 1               
            
        #Step 3: convert the zscores into delta weights
        #Step 4: apply the delta weights to ALL neurons in the layer
        for i, item_set in enumerate(result_sets):
            for j, row in enumerate(item_set):
                if np.isnan(row.sum()):
                    row = np.nan_to_num(row)
                calculated_weights = convert_to_weight(row, v, num_rows, j)
                weights[i] += calculated_weights

        #Step 5: reapply the weights to a new model and return
        index = 0
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                model_copy.state_dict()[item] = weights[index]
                index += 1
        return_model = NetworkSkeleton(create_layers(model_str, {'relu': nn.ReLU(), 'silu': nn.SiLU()})).to(device)
        return_model.load_state_dict(model_copy.state_dict())
        return return_model

if __name__ == "__main__":
    imp_dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    x_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns != 'y']].to_numpy()
    y_vals =  imp_dataset[imp_dataset.columns[imp_dataset.columns == 'y']].to_numpy()
    dataset = pd.DataFrame(x_vals[int(len(x_vals)*.2):])
    dataset["y"] = y_vals[int(len(y_vals)*.2):]
    dataset["z_answers"] = zscore(dataset['y'])
    dataset["z_answers"] = dataset['z_answers'].abs()
    sorted_data = dataset.sort_values(by='z_answers', ascending=True).drop(["z_answers"], axis=1)
    print(sorted_data)
    formatted_data_train = GeneratedDataset(x_vals[int(len(x_vals)*.2):], y_vals[int(len(y_vals)*.2):])
    formatted_data_test = GeneratedDataset(x_vals[:int(len(x_vals)*.2)], y_vals[:int(len(y_vals)*.2)])
    data_loader_train = DataLoader(formatted_data_train)
    data_loader_test = DataLoader(formatted_data_test)
    model_string = '100|200->silu->200|150->silu->150|1'
    string_to_activation = {'relu': nn.ReLU(), 'silu': nn.SiLU()}
    temp_model = NetworkSkeleton(create_layers(model_string, string_to_activation)).to('cpu')
    print(test(data_loader_test, temp_model, nn.MSELoss(), 'cpu'))
    new_model = train_model_torch_boost(temp_model, sorted_data, model_string, 'cpu', 0.00004, 20)
    print(test(data_loader_test, new_model, nn.MSELoss(), 'cpu'))