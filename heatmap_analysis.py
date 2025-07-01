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
        self.training_path = fpath + "\\training"
    def load_base_model(self, pre_train: bool = True):
        weights = [] 
        biases = []
        for root, dir, files in os.walk(self.model_path):
            for file in files:
                if file.split('.')[-1] != 'csv':
                    continue
                elif pre_train and "pre" in file:
                    if "weight" in file:
                        weights.append(pd.read_csv(self.model_path + "\\" + file))
                    if "bias" in file:
                        biases.append(pd.read_csv(self.model_path + "\\" + file))
                elif not pre_train and "post" in file:
                    if "weight" in file:
                        weights.append(pd.read_csv(self.model_path + "\\" + file))
                    if "bias" in file:
                        biases.append(pd.read_csv(self.model_path + "\\" + file))
            break
        return weights, biases
    def load_training_model(self):
        pass

def open_log(base_path):
    file_str = ""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if "report" in file:
                file_str = base_path + '\\' + file
                break
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
    return model_dirs_regress, model_dirs_class
    
def load_references(model_dirs):
    model_classes = []
    for model_dir in model_dirs:
        model_classes.append(MemModel(model_dir))
    return model_classes

if __name__ =='__main__':
    class_path = "D:\\model_dataset\\classification"
    regress_path = "D:\\model_dataset\\regression"
    model_paths = gather_model_paths(class_path, regress_path)
    class_models = load_references(model_paths[1])
    regress_models = load_references(model_paths[0])
    for model in class_models:
        logs = open_log(model.model_path)
        print(logs.model_str) 
        weights_pre, biases_pre = model.load_base_model()
        weights_post, biases_post = model.load_base_model(False)
        fig_num = 0 
        for index, weight in enumerate(weights_pre):
            fig = plt.figure(fig_num)
            plt.imshow(weight)
            plt.colorbar()
            plt.title(f"Heatmap Layer {index} Pre")
            fig_num += 1
            fig = plt.figure(fig_num)
            plt.imshow(weights_post[index])
            plt.colorbar()
            plt.title(f"Heatmap Layer {index} Post")
            fig_num += 1
        print(logs)  
        plt.show() 