"""
This algorithm is meant to partially challenge backpropigation in time and space complexity. The foundations for 
backpropigation time complexity can be found in the link here:
https://ai.stackexchange.com/questions/5728/what-is-the-time-complexity-for-training-a-neural-network-using-back-propagation

This approach has an advantage in that there is no need to store and calculate gradients, which can
greatly save both time and space.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from scipy.stats import zscore
import copy
import matplotlib.pyplot as plt
from model_utils import NetworkSkeleton, display_network, create_layers, save_model_parameters, test
import math
import seaborn as sns
from dropout_pass_update import train_model_dropout, train_model_one_loop, train_model_torch_boost

global_device = 'cpu'

class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = copy.deepcopy(x_data)
        self.y = copy.deepcopy(y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], np.float32(self.y[index])

class SmallRegressNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.l1_l2_active = nn.ReLU()
        self.l2_l3_active = nn.ReLU()

    def forward(self, x):
        l1_res = self.l1(x.type(torch.float))
        logits = self.l1_l2_active(l1_res)
        l2_res = self.l2(logits)
        logits = self.l2_l3_active(l2_res)
        return self.l3(logits)


"""
Update algorithm steps:

#1 run the data set through the network. record the answers at each individual layer. - done
#2 convert each new dataset to zscores of their respective columns
#3 convert each row into a weight update using the convert_to_weight - done-ish
#4 apply each delta to each row of the weights matrix
#5 repeat for all layers
#6 test the accuracy. Profit (hopefully)
"""

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

def train_model(model, dataset: pd.DataFrame, model_str, device, str_to_activation, v):
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
                new_item = torch.from_numpy(np.dot(weight, new_item) + biases[j])
                if j < len(activations):
                    new_item = str_to_activation[activations[j]](new_item).numpy()
                result_sets[j].append(new_item)
                total_iters += 1

        #Step 2: convert all of the data into zscores for their respective columns
        converted_sets = []
        for result in result_sets:
            converted_sets.append(zscore(result, axis=0))
            total_iters += 1
        
        #pop the final row and add the initial set
        converted_sets.pop()
        converted_sets.insert(0, zscore(x, axis=0))   
        
        #Step 3: convert the zscores into delta weights
        #Step 4: apply the delta weights to ALL neurons in the layer
        for i, item_set in enumerate(converted_sets):
            for j, row in enumerate(item_set):
                if np.isnan(row.sum()):
                    row = np.nan_to_num(row)
                calculated_weights = convert_to_weight(row, v, num_rows, j)
                weights[i] += calculated_weights
                total_iters += 1      

        #Step 5: reapply the weights to a new model and return
        index = 0
        for item in model_copy.state_dict():
            if item[-6:] == "weight":
                model_copy.state_dict()[item] = weights[i]
                index += 1
            else:
                continue
        return_model = NetworkSkeleton(create_layers(model_str, str_to_activation)).to(device)
        return_model.load_state_dict(model_copy.state_dict())
        print(f"The standard train model algorithm had {total_iters} iterations")
        return return_model

def get_percent_imporvement(start_loss, min_loss):
    return (math.fabs(start_loss - min_loss) / start_loss) * 100.0

if __name__ == "__main__":
    imp_dataset = pd.read_csv("generated_data_sets/small_5000_100_10_regression_generated.csv")
    x_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns != 'y']].to_numpy()
    y_vals = imp_dataset[imp_dataset.columns[imp_dataset.columns == 'y']].to_numpy()
    min_ratio = 0.2
    dataset = pd.DataFrame(x_vals[int(len(x_vals)*min_ratio):])
    dataset["y"] = y_vals[int(len(y_vals)*min_ratio):]
    formatted_data = GeneratedDataset(x_vals[:int(len(x_vals)*min_ratio)], y_vals[:int(len(y_vals)*min_ratio)])
    data_loader = DataLoader(formatted_data)
    percent_improvements = []
    trained_min_loss = []
    losses = []
    model_string = '100|200->relu->200|150->relu->150|1'
    string_to_activation = {'relu': nn.ReLU(), 'silu': nn.SiLU()}
    for j in range(1):
        temp_model = NetworkSkeleton(create_layers(model_string, string_to_activation)).to(global_device)
        trained_model = train_model_torch_boost(temp_model, dataset, model_string, global_device, 0.00006, 20)
        min_acc = np.inf
        trained_rounds = 0
        minimum_model: NetworkSkeleton = NetworkSkeleton([])
        #save_model_parameters(temp_model, '100|128->relu->128|128->relu->128|1', f'pre_round_{j}', 'D:\\regression', global_device)
        steps = 8
        for i in range(steps):
            #volitility = math.exp(-1.0*(i+1.0)/6.0)*0.00004
            volitility = 0.00006
            print(f"MSE OF THE TRAINED MODEL AFTER TRAINING ROUND {i}: ")
            acc = test(data_loader, trained_model, nn.MSELoss(), device=global_device)
            print(f"Loss: {acc}")
            losses.append(acc)
            if acc < min_acc:
                minimum_model = trained_model
                min_acc = acc
                trained_rounds = i
            trained_model = train_model_torch_boost(trained_model, dataset, model_string, global_device, volitility, 20)
        untrained_acc = test(data_loader, temp_model, nn.MSELoss(), device=global_device)
        trained_min_acc = test(data_loader, minimum_model, nn.MSELoss(), device=global_device)
        #save_model_parameters(minimum_model, '100|128->relu->128|128->relu->128|1', f'post_round_{j}', 'D:\\regression', global_device)
        improvement = get_percent_imporvement(untrained_acc, trained_min_acc)
        percent_improvements.append(improvement)
        trained_min_loss.append(trained_rounds)
        print("MSE OF THE NON-TRAINED MODEL: ")
        print(f"Loss: {untrained_acc}")
        print("MSE OF THE MINIMUM MSE MODEL: ")
        print(f"Loss: {trained_min_acc}")
        print("PERCENT IMPROVEMENT: ")
        print(f"Percent Change: {improvement:.4f}%")
    temp = pd.DataFrame({"percentages": percent_improvements})
    temp2 = pd.DataFrame({"rounds": trained_min_loss})
    plt.plot(losses)
    plt.title("Loses over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    # temp.to_csv("regression\\reports\\percentage_change.csv")
    # temp2.to_csv("regression\\reports\\min_rounds.csv")
    sns.histplot(temp, x='percentages', kde=True).set_title("Percent Change between untrained and Trained netowrks")
    # sns.histplot(temp2, x='rounds', kde=True).set_title("Rounds to meet minimum loss")
    plt.show()