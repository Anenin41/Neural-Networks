# Assignment 1 - Exercise (c) #
# Generate one random dataset and run the Rosenblatt algo on it. #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 13:13:15 +0100
# Modified: Mon 01 Dec 2025 @ 22:14:38 +0100

# Packages
from typing import Tuple
import numpy as np

from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result

def run_single_dataset(P : int,
                       N : int,
                       n_max : int = 100,
                       seed : int | None = None,
                       ) -> Tuple[np.ndarray, np.ndarray, Result]:
    """
    This funtion generates one data set and trains a perceptron on it.

    Args:
        P : int
            Number of patterns.
        N : int
            Pattern dimension.
        n_max : int
            Maximum number of sweeps (epochs) through the data.
        seed : int or None
            Random seed for data generation. 

    Returns:
        X : np.ndarray
        y : np.ndarray
        result : TypedDict data structure defined in 'sequential_perceptron.py'
    """
    # Ensure that everything is encoded through numpy ndarrays (efficient & fast)
    X: np.ndarray
    y: np.ndarray
    X, y = generate_dataset(P, N, seed=seed)
    result: Result = rosenblatt_train(X, y, n_max=n_max)
    
    return X, y, result

if __name__ == "__main__":
    P = 50
    N = 20
    n_max = 100
    seed = 0
    X, y, res = run_single_dataset(P, N, n_max=n_max, seed=seed)

    print(f"P = {P}, N = {N}")
    print(f"Converged: {res['converged']}")
    print(f"Sweeps: {res['sweeps']}")
    print(f"Number of updates: {res['updates']}")
