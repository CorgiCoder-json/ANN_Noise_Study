import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimilarityObject:
    layer_similarities: list[float]
    model_str: str = ""
    optimizer: str = ""
    loss_func: str = ""
    lr: float = 0.0
    epochs: int = 0
    final_loss: float = 0.0

def info_chunks(file_path: str) -> list[list[list[str]]]:
    chunks = []
    with open(file_path, 'r') as file:
        temp_chunk = []
        try:
            temp = file.readline().strip().split()
            while True:
                while temp[0] != "Average":
                    temp_chunk.append(temp)
                    temp = file.readline().strip().split()
                while len(temp) == 0 or temp[0] != 'For':
                    temp = file.readline()
                    if temp == "":
                        raise EOFError("Hit the end of the file!")
                    temp = temp.strip().split()
                chunks.append(temp_chunk)
                temp_chunk = []       
        except:
            pass
    return chunks

def chunk_to_obj(chunk_list: list[list[list[str]]]) -> list[SimilarityObject]:
    reports = []
    for chunk in chunk_list:
        temp_dict = {"mod_str": "", "optim": "", "loss": "", "lr": 0.0, "epo": 0, "fin_loss": 0.0, "sim_layers": []}
        for chunk_row in chunk:
            if chunk_row[0] == "For":
                temp_dict["sim_layers"].append(float(chunk_row[-1]))
            elif chunk_row[0] == "Model":
                temp_dict["mod_str"] = chunk_row[-1]
            elif chunk_row[0] == "Optimizer":
                temp_dict["optim"] = chunk_row[-1]
            elif chunk_row[0] == "Error":
                temp_dict["loss"] = chunk_row[-1]
            elif chunk_row[0] == "Accuracy:":
                temp_dict["fin_loss"] = float(chunk_row[-1])
            elif chunk_row[0] == "Epochs":
                temp_dict["epo"] = int(chunk_row[-1])
            elif chunk_row[0] == "Learning":
                temp_dict["lr"] = float(chunk_row[-1])
        reports.append(SimilarityObject(temp_dict["sim_layers"], temp_dict["mod_str"], temp_dict["optim"], temp_dict["loss"], temp_dict["lr"], temp_dict["epo"], temp_dict["fin_loss"]))
    return reports

def encode_optim(optim_string: str) -> int:
    if "Adam" in optim_string:
        return 0
    if "RMSprop" in optim_string:
        return 1
    if "ASGD" in optim_string:
        return 2
    if "SGD" in optim_string:
        return 3
    return -1

if __name__ == "__main__":
    chunks = info_chunks('D:\\similarity_scores.txt')
    reports = chunk_to_obj(chunks)
    regress_class_sim = {"classification": [], "regression": []}
    acc_avg_sim_data = {"fin_acc": [], "avg_score": [], "optim": []}
    for i in reports:
        if i.loss_func == "BCEWithLogitsLoss()":
            regress_class_sim["classification"].append(i)
            acc_avg_sim_data["fin_acc"].append(i.final_loss)
            acc_avg_sim_data["avg_score"].append(np.mean(i.layer_similarities))
            acc_avg_sim_data["optim"].append(encode_optim(i.optimizer)) 
        else:
            regress_class_sim["regression"].append(i)
    class_scores = []
    regress_scores = []
    for key in regress_class_sim:
        for report in regress_class_sim[key]:
            if key == "classification":
                class_scores.append(np.mean(report.layer_similarities))
            else:
                regress_scores.append(np.mean(report.layer_similarities))
    print(f"For classification problems, the average of average similarities is: {np.mean(class_scores)}")
    print(f"For regression problems, the average of average similarities is: {np.mean(regress_scores)}")
    acc_sim_plot_data = pd.DataFrame(acc_avg_sim_data)
    sgd_scores = acc_sim_plot_data[acc_sim_plot_data["optim"] == 3]
    asgd_scores= acc_sim_plot_data[acc_sim_plot_data["optim"] == 2]
    rms_scores = acc_sim_plot_data[acc_sim_plot_data["optim"] == 1]
    adam_scores = acc_sim_plot_data[acc_sim_plot_data["optim"] == 0]
    fig, axes = plt.subplots(2, 2)
    sns.histplot(sgd_scores, x="avg_score", ax=axes[0, 0])
    sns.histplot(asgd_scores, x="avg_score", ax=axes[0, 1])
    sns.histplot(rms_scores, x="avg_score", ax=axes[1, 0])
    sns.histplot(adam_scores, x="avg_score", ax=axes[1, 1])
    plt.show()
    