# Part B - Assignment 3 - Neural Networks #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 5 Jan 2026 @ 16:11:10 +0100
# Modified: Sun 25 Jan 2026 @ 18:30:52 +0100

# Packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Part (A) code
from part_a import soft_committee_forward, mse_loss, init_weights, sgd_step

# Import GLOBALS
from config import *

def load_xi_tau(xi_path, tau_path):
    """
    Load xi and tau from the CSV files. CSV stands for Comma Separated Values.

    The assignment provides:
        - xi as a 50 x 5000 array (N=50 features, 5000 examples)
        - tau as a vector of length 5000

    This function standardizes them to the following dimensions:
        - X of shape (num_examples, N)
        - y of shape (num_examples,)

    If in the case xi is stored as (N, num_examples) due to human error, the code
    uses the transpose, i.e. (num_examples, N).

    Returns:
        X : np.ndarray of shape (num_examples, N)
        y : np.ndarray of shape (num_examples,)
    """
    # Load them CSVs. An easier choice is to use the pandas package, but I find 
    # it to be overkill for just 2 commands.
    X_raw = np.loadtxt(xi_path, delimiter=",")
    y = np.loadtxt(tau_path, delimiter=",")
    
    # Ensure y is 1D
    if y.ndim != 1:
        y = y.reshape(-1)

    # Raise error if X is not 2D
    if X_raw.ndim != 2:
        raise ValueError("xi must be a 2D array, current shape: %s" % (X_raw.shape,))

    # This is a heuristic check on the dimensions of X. If you want to be 
    # absolutely sure about what X looks like, run some pandas command in an 
    # interactive Python environment, like the kernel.
    if X_raw.shape[0] == 50 and X_raw.shape[1] != 50:
        X = X_raw.T
    else:
        X = X_raw

    if X.shape[0] != y.shape[0]:
        raise ValueError(
                "Mismatch: X has %d examples but tau has %d values!"
                % (X.shape[0], y.shape[0])
                )

    return X.astype(float), y.astype(float)

def split_train_test(X, y, P, Q):
    """
    Split the dataset as is required in the assignment sheet:
        - Training set: the first P examples (idx 0...P-1)
        - Test set: next Q examples (idx P...Q-1)

    This guarantees that there is no leakage when training the model.

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Some proper guardrails on the user-defined parameters to ensure sanity
    if P <= 0 or Q <= 0:
        raise ValueError("P and Q must be positive.")
    if P + Q > X.shape[0]:
        raise ValueError(
                "Need at least P+Q=%d examples, but X has %d." % (P + Q, 
                                                                  X.shape[0])
                )

    X_train = X[:P]
    y_train = y[:P]
    X_test = X[P:P + Q]
    y_test = y[P:P + Q]

    return X_train, y_train, X_test, y_test

def compute_errors(X_train, y_train, X_test, y_test, W):
    """
    Compute:
        E       = 0.5 * mean((sigma_train - y_train)^2)
        E_test  = 0.5 * mean((sigma_test - y_test)^2)

    Use the same mse_loss() function as in part_a.py
    
    Returns:
        E, E_test : floats
    """
    # Training & Testing set prediction
    pred_train = soft_committee_forward(X_train, W)
    pred_test = soft_committee_forward(X_test, W)
    
    # Call mse_loss() to compute the respective errors
    E = mse_loss(pred_train, y_train)
    E_test = mse_loss(pred_test, y_test)

    return E, E_test

def training_with_tracking(X_train, y_train, X_test, y_test, K, eta, tmax, seed):
    """
    SGD training with a per-epoch tracking of E(t) and E_test(t). Practically 
    very useful to decide when to stop the training of the model for an optimal
    balance between training and generalization error.

    Time, what is time?
        - One epoch = P stochastic updates (sampling is done with replacement).
        - Training time t is measured in epochs.
        - Total updates = tmax * P.

    What is being tracked?
        - Errors at t=0 (before any update takes place)
        - Erros after each epoch, up to t=tmax (maximum iteration budget)

    Returns:
        - W : np.ndarray with shape (K, N)
            Array of final weights
        - t_values : np.ndarray with shape (tmax+1,)
            Epochs (used for plotting later)
        - E_train : np.ndarray with shape (tmax+1,)
            Training error for each epoch
        - E_test : np.ndarray with shape (tmax+1,)
            Generalization error for each epoch
    """
    # Fetch dimentions
    P = X_train.shape[0]
    N = X_train.shape[1]

    # Set RNG
    rng = np.random.default_rng(seed)

    # Initialize weights are normals with unit norm
    W = init_weights(K, N, rng)

    # Initialize storage arrays
    t_values = np.arange(tmax + 1, dtype=int)
    E_train = np.zeros(tmax + 1, dtype=float)
    E_test = np.zeros(tmax + 1, dtype=float)

    # t = 0
    E_train[0], E_test[0] = compute_errors(X_train, y_train, X_test, y_test, W)

    # t = 1, ..., tmax
    for t in range(1, tmax + 1):
        # One epoch is P random single-example SGD updates
        for _ in range(P):
            # Pick the lucky one
            idx = int(rng.integers(low=0, high=P))

            # Update
            sgd_step(X_train[idx], float(y_train[idx]), W, eta)

        # Track error behaviour across the epochs
        E_train[t], E_test[t] = compute_errors(X_train, y_train, X_test, y_test, W)

    return W, t_values, E_train, E_test

def ensure_outdir(path):
    """
    Create output directory if it doesn't exist.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def plotter_learning_curves(t, E_train, E_test, outpath):
    """
    Plot E(t) and E_test(t) and save the figures to outpath.
    """
    plt.figure()
    plt.plot(t, E_train, label="E(t) train")
    plt.plot(t, E_test, label="E_test(t)")
    plt.xlabel("Training time t (epochs = P updates)")
    plt.ylabel("Error (0.5 * MSE)")
    plt.title("Training vs. Generalization Errors vs. Training Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_weights(W, outpath):
    """
    Plot the weight matrix as bar plots of the final weight vectors. (Fancy :D)

    W has shape (K, N). For the assignment the default is K=2, so this saves:
        - weights_w1.png
        - weights_w2.png

    If K is increased, this will create K many .png files.
    """
    K = W.shape[0]
    N = W.shape[1]

    for k in range(K):
        plt.figure()
        plt.bar(np.arange(N), W[k])
        plt.xlabel("Component Index j")
        plt.ylabel("w%d[j]" % (k + 1))
        plt.title("Final Weight Vector w%d" % (k + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, "weights_w%d.png" % (k + 1)), dpi=200)
        plt.close()

def main():
    """
    Entry point for Part (B) of the assignment.

    Steps this function performs (if needed):
        1) Config override in case you need different parameters
        2) Load xi/tau into scope
        3) Split into train/test (first use P, then Q)
        4) Train with a per-epoch tracker of the errors
        5) Save plots
        6) Print a short summary in CLI

    Note: variable names in CAPITAL letters denote global variables stored in
    config.py. DO NOT modify them from here, instead use the aforementioned file.
    """
    # Ensure storage directory exists
    ensure_outdir(OUTDIR)

    # Load the dataset into scope
    X, y = load_xi_tau(XI_PATH, TAU_PATH)

    # Split into training and test set
    X_train, y_train, X_test, y_test = split_train_test(X, y, P_TRAIN, Q_TEST)

    # Run learning scheme
    W_final, t, E_train, E_test = training_with_tracking(
            X_train, y_train, X_test, y_test,
            K = K_HIDDEN, eta=ETA, tmax=TMAX, seed=SEED
            )

    # Plot fancy curves
    plotter_learning_curves(t, E_train, E_test, os.path.join(OUTDIR, "learning_curves.png"))
    plot_weights(W_final, OUTDIR)

    # CLI Summary
    print("Training complete.")
    print("P=%d\tQ=%d\teta=%.5f\ttmax=%d\tseed=%d" % (P_TRAIN, Q_TEST, ETA, TMAX, SEED))
    print("Final E: %.6f" % (E_train[-1],))
    print("Final E_test: %.6f" % (E_test[-1],))
    print("Saved results in:", os.path.abspath(OUTDIR))

if __name__ == "__main__":
    main()
