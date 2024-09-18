"""
Programmer: Jacob Maurer
Date: 9/17/2024
Description: Main running file for ANN weight pattern testing
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# model_dict = torch.jit.load('AI_Model/seeds/experiment_seed_1.pt')
# model_dict.eval().to(device)
# model.load_state_dict(model_dict.state_dict())

# state_dict_copy = model.state_dict()

# for item in model.cpu().state_dict():
#     shape = state_dict_copy[item].cpu().numpy().shape
#     new_state = []
#     if item[20:] == "weight":
#         for i in range(shape[0]):
#             space = np.linspace(-0.02, 0.02, num=shape[1])
#             new_state.append(space)
#         new_state = np.array(new_state)
#     else:
#         new_state = np.linspace(-0.02, 0.02, num=shape[0])
#     print(new_state.shape)
#     state_dict_copy[item] = torch.from_numpy(new_state)

# model.load_state_dict(state_dict_copy)
ID = 1
for item in model.cpu().state_dict():
    if item[20:] == "weight":
        table = pd.DataFrame(model.cpu().state_dict()[item])
        table.to_csv("results/ss_s_weight_pre_" + str(ID) + ".csv",index=False)
    if item[20:] == "bias":
        series = pd.Series(model.cpu().state_dict()[item])
        series.to_csv("results/ss_s_bias_pre_" + str(ID) + ".csv",index=False)
    ID += 1

model.to(device)
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

ID = 1
for item in model.cpu().state_dict():
    if item[20:] == "weight":
        table = pd.DataFrame(model.cpu().state_dict()[item])
        table.to_csv("results/ss_s_weight_post_" + str(ID) + ".csv",index=False)
    if item[20:] == "bias":
        series = pd.Series(model.cpu().state_dict()[item])
        series.to_csv("results/ss_s_bias_post_" + str(ID) + ".csv",index=False)
    ID += 1

