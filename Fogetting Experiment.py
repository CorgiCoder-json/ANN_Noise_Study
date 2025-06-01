"""
Created: 6/1/2025

Purpose: To determine if training a Neural Network on two seperate datasets causes the heatmap
of each of the network layers to change from the base distribution    
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import copy
import matplotlib.pyplot as plt
device = 'cuda'
fig_number = 0

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

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        """
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            #pred = (pred > 0.5).type(torch.float)
            #correct += (pred == y.unsqueeze(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def display_model(model: SmallRegressNetwork, window_size: int = 50, neuron_start = 0, column_start = 0, label=''):
    global fig_number
    model_copy = copy.deepcopy(model.cpu())
    for item in model_copy.state_dict():
        if item[-6:] == "weight":
            weight_slice = model_copy.state_dict()[item][neuron_start:neuron_start+window_size]
            fig = plt.figure(fig_number)
            plt.imshow([temp[column_start:column_start+window_size] for temp in weight_slice])
            plt.colorbar()
            plt.title(label)
            fig_number += 1
        else:
            continue
    model.to(device)

if __name__ == "__main__":
    dataset_1 = make_regression(n_samples = 5000, n_features=100, n_informative=10)
    dataset_2 = make_regression(n_samples = 5000, n_features=100, n_informative=10)
    data_1_train = GeneratedDataset(dataset_1[0][int(len(dataset_1[0])*.2):], dataset_1[1][int(len(dataset_1[1])*.2):])
    data_1_test = GeneratedDataset(dataset_1[0][:int(len(dataset_1[0])*.2)], dataset_1[1][:int(len(dataset_1[1])*.2)])
    data_2_train = GeneratedDataset(dataset_2[0][int(len(dataset_2[0])*.2):], dataset_2[1][int(len(dataset_2[1])*.2):])
    data_2_test = GeneratedDataset(dataset_2[0][:int(len(dataset_2[0])*.2)], dataset_2[1][:int(len(dataset_2[1])*.2)])
    data_1_train_loader = DataLoader(data_1_train, batch_size=50, shuffle=True)
    data_2_train_loader = DataLoader(data_2_train, batch_size=50, shuffle=True)
    data_1_test_loader = DataLoader(data_1_test, batch_size=50, shuffle=True)
    data_2_test_loader = DataLoader(data_2_test, batch_size=50, shuffle=True)
    small_model_regress = SmallRegressNetwork().to(device)
    loss_fn_regress = nn.MSELoss()
    optimizer_regress = torch.optim.SGD(small_model_regress.parameters(), lr=1e-4)
    new_state = copy.deepcopy(small_model_regress.cpu().state_dict())
    old_state = copy.deepcopy(small_model_regress.cpu().state_dict())
    for key in new_state:
        new_state[key] = old_state[key] * 10
    small_model_regress.load_state_dict(new_state)
    small_model_regress.to(device)
    display_model(small_model_regress, label="Before Training")
    data_1_acc = []
    data_2_acc = []
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data_1_train_loader, small_model_regress, loss_fn_regress, optimizer_regress)
        test(data_1_test_loader, small_model_regress, loss_fn_regress)
    display_model(small_model_regress, label="Trained on Dataset 1")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data_2_train_loader, small_model_regress, loss_fn_regress, optimizer_regress)
        test(data_2_test_loader, small_model_regress, loss_fn_regress)
    display_model(small_model_regress, label="Trained on Dataset 2")
    print("MEMORY TEST: DATASET 1")
    test(data_1_test_loader, small_model_regress, loss_fn_regress)
    plt.show()
    
    #Co training:
    new_model = SmallRegressNetwork().to(device)
    new_loss= nn.MSELoss()
    new_optimize = torch.optim.SGD(new_model.parameters(), 1e-4)
    new_state = copy.deepcopy(new_model.cpu().state_dict())
    old_state = copy.deepcopy(new_model.cpu().state_dict())
    for key in new_state:
        new_state[key] = old_state[key] * 10
    new_model.load_state_dict(new_state)
    new_model.to(device)
    display_model(new_model, label="Co-training prior")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data_2_train_loader, new_model, new_loss, new_optimize)
        data_2_acc.append(test(data_2_test_loader, new_model, new_loss))
        train(data_1_train_loader, new_model, new_loss, new_optimize)
        data_1_acc.append(test(data_1_test_loader, new_model, new_loss))
    display_model(new_model, label="Co-training after")
    plt.show()