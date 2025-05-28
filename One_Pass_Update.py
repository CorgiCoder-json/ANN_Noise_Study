import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from scipy.stats import zscore
import copy

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
        self.l2_l3_active = nn.SiLU()

    def forward(self, x):
        logits = self.l1_l2_active(self.l1(x.type(torch.float)))
        logits = self.l2_l3_active(self.l2(logits))
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
    return 1 / (1 + np.exp(-1 * (data * data)))

def convert_to_weight(data: np.ndarray, volitility: float, d: int, total_data_size: int, step: int):
    """
    converts a given row of data into updateable weights

    Args:
        data (np.ndarray): the row of data that is to be converted. Should already be in z-score form
        volitility (float): controls how much change each weight should have. Higher values result in higher delta weights and vice versa for lower values 
        d (int): The "dimension" the model weights reside in. For example, if weights are between 0.8 and -0.8, d should be 10, since the weights exsist in the "tenths dimension"
        total_data_size (int): this size of the data set that the row comes from
        step (int): the row number of the data input
    """
    return np.sign(data) * ((decay_constant(step, total_data_size) * scaled_weight(data))/d) * volitility

def train_model(model, dataset):
    
    dimension = 100
    num_rows = len(dataset)
    v = 0.3
    
    #Extract the weights
    model_copy = copy.deepcopy(model)
    weights = []
    result_sets = [[], [], []]
    for item in model_copy.state_dict():
        if item[-6:] == "weight":
            weights.append(model_copy.state_dict()[item].numpy())
        else:
            continue
    #Step 1: move the data into the weight dimension
    for i, item in enumerate(dataset):
        new_item = item
        for j, weight in enumerate(weights):
            new_item = np.dot(weight, new_item) 
            result_sets[j].append(new_item)
    #Step 2: convert all of the data into zscores for their respective columns
    converted_sets = []
    for result in result_sets:
        converted_sets.append(zscore(result, axis=1))
        
    #pop the final row and add the initial set
    converted_sets.pop()
    converted_sets.insert(0, zscore(dataset, axis=1))   
    
    #TODO: Sort the data set by "outlierness" (sort by zscore of answer)
        
    #Step 3: convert the zscores into delta weights
    delta_weights = []
    for set in converted_sets:
        delta_weights_holder = []
        for i, row in enumerate(set):
            calculated_weights = convert_to_weight(row, v, dimension, num_rows, i)
            delta_weights_holder.append(calculated_weights)
        delta_weights.append(delta_weights_holder)
        
    #step 4: apply the delta weights to the weight matricies of their respective layers
    for i, deltas in enumerate(delta_weights):
        for delta in deltas:
            weights[i] += delta         
    
    #Step 5: reapply the weights to a new model and return
    index = 0
    for item in model_copy.state_dict():
        if item[-6:] == "weight":
            model_copy.state_dict()[item] = weights[i]
            index += 1
        else:
            continue
    model_copy.load_state_dict(model_copy.state_dict())
    return model_copy
def test_accuracy(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    small_regression_problem = make_regression(n_samples = 5000, n_features=100, n_informative=10)
    temp_model = SmallRegressNetwork().to('cpu')
    trained_model = train_model(temp_model, small_regression_problem[0])
    print(test_accuracy())