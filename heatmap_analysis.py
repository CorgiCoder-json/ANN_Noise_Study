import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

@dataclass
class Report:
    model_str: str
    accuracy: float
    epochs: int
    optimizer: str
    learning_rate: float
    error_function: str
    def __str__(self):
        return f"Model String: {self.model_str}\nOptimizer Used: {self.optimizer}\nError Function: {self.error_function}\nLearning Rate: {self.learning_rate}\nEpochs Ran: {self.epochs}\nAccuracy: {self.accuracy}"

class MemModel:
    def __init__(self, fpath):
        self.model_path = fpath
    def load_model(self):
        return pd.read_csv(self.model_path)

def sort_pre_post(files)-> tuple[list[str], list[str]]:
    pre_files = []
    post_files = []
    for file in files:
        if "pre" in file:
            pre_files.append(file)
        elif "post" in file:
            post_files.append(file)        
        else:
            print("Other file found... possibly report file?")
    return pre_files, post_files

def open_network(base_path, layer_files):
    network = []
    for file in layer_files:
        network.append(pd.read_csv(base_path+"\\"+file))
    return network

def open_log(base_path, layer_files):
    file_str = ""
    for file in layer_files:
        if "report" in file:
            file_str = base_path + '\\' + file
            break
    with open(file_str, 'r') as f:
        lines = f.readlines()
        report_obj: Report = Report(lines[0].strip().split(":")[-1].strip(), float(lines[1].strip().split(":")[-1]), int(lines[2].strip().split(":")[-1]), lines[3].strip().split(":")[-1].strip(), float(lines[4].strip().split(":")[-1]), lines[5].strip().split(":")[-1].strip())
        return report_obj

def step_file_system(fpath):
    model_paths = []
    for root, dirs, files in os.walk(fpath):
        for dir in dirs:
            model_paths.append(fpath + "\\" + dir)
        break
    return model_paths

def gather_model_paths(class_path, regress_path):
    data_dirs_regress = step_file_system(regress_path)
    data_dirs_class =  step_file_system(class_path)
    model_dirs_regress = []
    model_dirs_class = []
    for index, model_dir in enumerate(data_dirs_regress):
        model_dirs_regress.extend(step_file_system(model_dir))
        model_dirs_class.extend(step_file_system(data_dirs_class[index]))
    
def load_references()

if __name__ =='__main__':
    class_path = "D:\\model_dataset\\classification"
    regress_path = "D:\\model_dataset\\regression"
    model_path = "D:\\model_dataset\\classification\\2000_rows\\model_1"
    roots, directories, file_names = [], [], []
    for root, dirs, files in os.walk(model_path):
        roots.append(root)
        directories.append(dirs)
        file_names.append(files)
        break
    gather_data(class_path, regress_path)
    logs = open_log(model_path, file_names[0])
    print(logs)
    pre_train, post_train = sort_pre_post(file_names[0])
    pre_net = open_network(model_path, pre_train)
    post_net = open_network(model_path, post_train)
    fig, axs = plt.subplots(1,3)
    im1 = axs[0].imshow(pre_net[4].to_numpy())
    im2 = axs[1].imshow(post_net[4].to_numpy())
    im3 = axs[2].imshow((post_net[4].to_numpy() - pre_net[4].to_numpy()))
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    fig.colorbar(im3, ax=axs[2])    
    plt.show()
    
    print("Hello World")