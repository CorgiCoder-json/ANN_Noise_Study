"""
Created: 6/11/2025

Run this file to make generated datasets with scikit learn and make them into csv files for later use.
"""
from sklearn.datasets import make_regression
import pandas as pd
import os

if __name__ == "__main__":
    num_samples = 17000
    num_features = 100
    num_inform = 10
    problem_x, problem_y = make_regression(num_samples, n_features=num_features, n_informative=num_inform)
    dataset = pd.DataFrame(problem_x)
    dataset['y'] = problem_y
    print(dataset)
    dataset.to_csv(os.path.join(f'./generated_data_sets/{num_samples}_{num_features}_{num_inform}_regression_generated.csv'))