"""
Created: 6/19/2025

purpose: Generate many models and collect key data on it 
"""
import concurrent.futures
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model_utils as util
import pandas as pd
import random
import concurrent
import copy
import os
from AI_weight_generation import run_tests

def prep(files_list: list[str], test_percent: float, batch_size: int = 50) -> dict[int, tuple[DataLoader, DataLoader]]:
    preped_data = {}
    for file in files_list:
        import_frame = pd.read_csv(file)
        row_amount = file.split('/')[2].split('_')[0]
        if import_frame.shape[1] > 101:
            import_frame.drop(import_frame.columns[0], axis=1, inplace=True)
        x_vals = import_frame[import_frame.columns[import_frame.columns != 'y']].to_numpy()
        y_vals = import_frame[import_frame.columns[import_frame.columns == 'y']].to_numpy()
        train_dataset = util.GeneratedDataset(x_vals[int(len(x_vals)*test_percent):], y_vals[int(len(y_vals)*test_percent):])
        test_dataset = util.GeneratedDataset(x_vals[:int(len(x_vals)*test_percent)], y_vals[:int(len(y_vals)*test_percent)])
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        preped_data.update({int(row_amount): (train_loader, test_loader)})
    return preped_data

def custom_train(model: util.NetworkSkeleton, prep_data: tuple[DataLoader, DataLoader], row_count: int, optim, err, problem: str, epochs: int):
    for epoch in range(epochs):
        util.train(prep_data[0], model, err, optim, 'cpu')
        # save_model parameters
        util.test(prep_data[1], model, err, 'cpu')
    #open matadata file in write:
        # record final accuarcy
        # record optimizer
        # record loss function
    # DONT SAVE NEURON AS REGRESSION DATA; can be done post model training. Make a seperate saving folder
        

if __name__ == "__main__":
    classification_dict = {'relu': nn.ReLU(), 'sig': nn.Sigmoid(), 'tanh': nn.Tanh(), 'silu': nn.SiLU()}
    regression_dict = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'llrelu': nn.LeakyReLU(0.01), 'hlrelu': nn.LeakyReLU(1.01)}
    num_tests = 1
    in_dim = 100
    num_hidden = [0, 1, 2]
    labels = [2000, 5000, 7000, 10000, 12000, 15000, 17000, 20000]
    num_out = 1
    size_range = (100, 300)
    main_path_regression = "D:\\model_dataset\\regression"
    main_path_classification = "D:\\model_dataset\\classification"
    optimizer_list = [
        torch.optim.SGD,
        torch.optim.Adam,
        torch.optim.RMSprop,
        torch.optim.ASGD
    ]
    regress_files = [
        "./generated_data_sets/2000_100_10_regression_generated.csv",
        "./generated_data_sets/5000_100_10_regression_generated.csv",
        "./generated_data_sets/7000_100_10_regression_generated.csv",
        "./generated_data_sets/10000_100_10_regression_generated.csv",
        "./generated_data_sets/12000_100_10_regression_generated.csv",
        "./generated_data_sets/15000_100_10_regression_generated.csv",
        "./generated_data_sets/17000_100_10_regression_generated.csv",
        "./generated_data_sets/20000_100_10_regression_generated.csv"
    ]
    class_files = [
        "./generated_data_sets/2000_100_10_classification_generated.csv",
        "./generated_data_sets/5000_100_10_classification_generated.csv",
        "./generated_data_sets/7000_100_10_classification_generated.csv",
        "./generated_data_sets/10000_100_10_classification_generated.csv",
        "./generated_data_sets/12000_100_10_classification_generated.csv",
        "./generated_data_sets/15000_100_10_classification_generated.csv",
        "./generated_data_sets/17000_100_10_classification_generated.csv",
        "./generated_data_sets/20000_100_10_classification_generated.csv"
    ]
    regress_data = prep(regress_files, 0.2)
    class_data = prep(class_files, 0.2)
    all_files = copy.deepcopy(regress_files)
    all_files.extend(class_files)
    model_id = 1
    for i in range(model_id, model_id + num_tests):
        print(f"Entering Testing loop for try {i}")
        class_str = util.model_string_generator(in_dim, random.choice(num_hidden), num_out, list(classification_dict.keys()), size_range)
        regress_str = util.model_string_generator(in_dim, random.choice(num_hidden), num_out, list(regression_dict.keys()), size_range)
        base_model_class = util.NetworkSkeleton(util.create_layers(class_str, classification_dict))
        base_model_regress = util.NetworkSkeleton(util.create_layers(regress_str, regression_dict))
        regress_loss = nn.MSELoss()
        class_loss = nn.BCEWithLogitsLoss()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_files)) as execut:
            futures = []
            for label in labels:
                try:
                    os.mkdir(main_path_regression + f"\\{label}_rows\\model_{model_id}")
                    os.mkdir(main_path_classification + f"\\{label}_rows\\model_{model_id}")
                except FileExistsError:
                    print("Error has occured! directory overwrite happened, should not be the case")
                futures.append(execut.submit(run_tests, regress_data[label], 'cpu', 1, 30, 16, regress_loss, random.choice(optimizer_list), regression_dict, main_path_regression + f"\\{label}_rows\\model_{model_id}",copy.deepcopy(base_model_regress), regress_str))
                futures.append(execut.submit(run_tests, class_data[label], 'cpu', 1, 30, 0.2, class_loss, random.choice(optimizer_list), classification_dict, main_path_classification + f"\\{label}_rows\\model_{model_id}",copy.deepcopy(base_model_class), class_str))
            for future in concurrent.futures.as_completed(futures):
                print("Gathering all the data...")
                future.result()
        model_id += 1
    print("Exited! check the results to ensure correct output")