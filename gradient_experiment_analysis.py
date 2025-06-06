import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def open_logs(fpath):
    round_results = []
    with open(fpath, 'rt') as file:
        for line in file.readlines():
            line = line.strip()
            if line == "IMPROVED":
                round_results.append(0)
            elif line == "NOT IMPROVED":
                round_results.append(1)
            elif line == "MARKED":
                round_results.append(2)
            else:
                continue
    return round_results
                

if __name__ == "__main__":
    logs = open_logs("./regression/reports/gradient_experimental_logs_test_1.txt")
    print(len(logs))
    imperfection_tracker = 0
    wierd_tracker = 0
    no_takeover_tracker = 0
    for item in logs:
        if item == 1:
            no_takeover_tracker += 1
            imperfection_tracker += 1
        elif item == 2:
            wierd_tracker += 1
            imperfection_tracker += 1
        else:
            continue
    print(f"There are {imperfection_tracker} imperfections, which means that in {((len(logs)-imperfection_tracker)/len(logs)) * 100}% of cases, there is improvement")
    print(f"There are {wierd_tracker} wierd results, and {no_takeover_tracker} no improvement results.")
    loss_stats = pd.read_csv("D:\\pass_gradient_exp\\regression_test_1\\stats.csv")
    print(loss_stats)
    for i in range(len(logs)):
        if logs[i] == 2:
            loss_stats.drop(i, axis=0, inplace=True)
    temp_percent = loss_stats["percent"]
    temp_percent *= np.sign(loss_stats["differences"].to_numpy())
    loss_stats["percent"] = temp_percent
    print(loss_stats)
    print(loss_stats["percent"].to_numpy())
    sns.histplot(loss_stats, x="percent", kde=True).set_title("Percent Improvement Distribution")
    plt.show()