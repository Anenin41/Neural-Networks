import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part (A) code
from part_a import soft_committee_forward, mse_loss, init_weights, sgd_step
import part_b as pb2
from config import *

#loading the bodyfat dataset
X_PATH=r"assignment_3\xzscore.csv"
T_PATH=r"assignment_3\tshift.csv"
OUTDIR = "results_bonus"
def split_data(X:np.ndarray,
               y:np.ndarray,
               P:int,
               Q:int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split the bodyfat dataset into training and test sets.

    Args:
        X : np.ndarray
        Y: np.ndarray
        P: int
            Number of training examples.
        Q: int
            Number of test examples.

    Returns:
        X_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.
        X_test : np.ndarray
            Test input data.
        y_test : np.ndarray
            Test target data.
    """
    # Split into training and test sets
    X_train = X[:P]
    y_train = y[:P]
    X_test = X[P:P + Q]
    y_test = y[P:P + Q]

    return X_train, y_train, X_test, y_test

def training_bonus(X_train : np.ndarray, 
                           y_train : np.ndarray,
                           X_test : np.ndarray, 
                           y_test : np.ndarray, 
                           K : int, 
                           eta : float, 
                           tmax : int, 
                           seed : int,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        X_train, y_train, X_test, y_test : np.ndarrays
            Train/test splits.
        K : int
            Number of hidden units.
        eta : int
            Learning rate.
        tmax : int
            Number of maximum epochs allowed.
        seed : int
            RNG seed for reproducable results.

    Returns:
        W : np.ndarray with shape (K, N)
            Array of final weights
        t_values : np.ndarray with shape (tmax+1,)
            Epochs (used for plotting later)
        E_train : np.ndarray with shape (tmax+1,)
            Training error for each epoch
        E_test : np.ndarray with shape (tmax+1,)
            Generalization error for each epoch
    """
    # Fetch dimentions
    P = X_train.shape[0]
    N = X_train.shape[1]

    # Set RNG
    rng = np.random.default_rng(seed)

    # Initialize weights as normals with unit norm, see part_a.init_weights
    W = init_weights(K, N, rng)

    # Initialize storage arrays
    t_values = np.arange(tmax + 1, dtype=int)
    E_train = np.zeros(tmax + 1, dtype=float)
    E_test = np.zeros(tmax + 1, dtype=float)

    # t = 0
    E_train[0], E_test[0] = pb2.compute_errors(X_train, y_train, X_test, y_test, W)

    # t = 1, ..., tmax
    for t in range(1, tmax + 1):
        # One epoch is P random single-example SGD updates
        for _ in range(P):
            # Pick the lucky one
            idx = int(rng.integers(low=0, high=P))

            # Update
            sgd_step(X_train[idx], (y_train[idx]), W, eta)

        # Track error behaviour across the epochs
        E_train[t], E_test[t] = pb2.compute_errors(X_train, y_train, X_test, y_test, W)

    return W, t_values, E_train, E_test
def main():
    pb2.ensure_outdir(OUTDIR)
    X=pd.read_csv(X_PATH, header=None).to_numpy().T
    y=pd.read_csv(T_PATH, header=None).to_numpy().ravel()
    print(X.shape, y.shape)
    X_train, y_train, X_test, y_test = split_data(X, y, P_TRAIN, Q_TEST)

    # Run learning scheme
    W_final, t, E_train, E_test = training_bonus(
            X_train, y_train, X_test, y_test,
            K = K_HIDDEN, eta=ETA, tmax=TMAX, seed=SEED
            )

    # Plot fancy curves
    pb2.plotter_learning_curves(t, E_train, E_test, os.path.join(OUTDIR, "learning_curves_%d_%.3f.png" % (P_TRAIN, ETA)))
    pb2.plot_weights(W_final, OUTDIR)

    # CLI Summary
    print("Training complete.")
    print(f"P={P_TRAIN}\tQ={Q_TEST}\teta={ETA:.5F}\ttmax={TMAX}\tseed={SEED}")
    print(f"Final E: {E_train[-1]:.6f}")
    print(f"Final E_test: {E_test[-1]:.6f}")
    print("Saved results in:", os.path.abspath(OUTDIR))

if __name__ == "__main__":
    main()

