import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

if __name__ == "__main__":
    window_size = 50
    start_weight = (20, 0)
    weights = pd.read_csv("D:\\pass_gradient_exp\\regression_test_1\\weights\\100-200_relu_200-150_relu_150-1_one_pass_grad_round_0_layer_1.csv").to_numpy()
    region = [temp[start_weight[1]:start_weight[1]+window_size] for temp in weights[start_weight[0]:start_weight[0]+window_size]]
    x = np.arange(0, len(region), 1)
    y = np.arange(0, len(region[0]), 1)
    x, y = np.meshgrid(x, y)
    z = np.squeeze(region)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False) # type: ignore
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    print("Hello World!")