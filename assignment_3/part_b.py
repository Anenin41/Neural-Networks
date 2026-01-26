# Part B - Assignment 3 - Neural Networks #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 5 Jan 2026 @ 16:11:10 +0100
# Modified: Mon 26 Jan 2026 @ 21:04:34 +0100

# Packages
import os
import numpy as np
import matplotlib.pyplot as plt

# Part (A) code
from part_a import soft_committee_forward, mse_loss, init_weights, sgd_step

# Import GLOBALS
from config import *

def load_xi_tau(xi_path, tau_path):
    """
    Load xi (inputs) and tau (targets) from the CSV files and normalize shapes.

    The assignment provides xi as a (50, 5000) array: (N, num_examples).
    Internally, a more common convention is used:
        X: (num_examples, N)
        y: (num_examples,)

    Heuristic check:
        If X_raw looks like (N, num_examples) with N=50, then transpose it.

    Args:
        xi_path : str
            Path to xi.csv.
        tau_path : str
            Path to tau.csv.

    Returns:
        X : np.ndarray of shape (num_examples, N), with floats for elements.
        y : np.ndarray of shape (num_examples,), with floats for elements.
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
        raise ValueError(f"xi must be 2D. Current shape={X_raw.shape}.")

    # This is a heuristic check on the dimensions of X. If you want to be 
    # absolutely sure about what X looks like, run some pandas command in an 
    # interactive Python environment, like the kernel.
    if X_raw.shape[0] == 50 and X_raw.shape[1] != 50:
        X = X_raw.T
    else:
        X = X_raw

    if X.shape[0] != y.shape[0]:
        raise ValueError(
                f"Mismatch: X has {X.shape[0]} examples but tau has {y.shape[0]} values."
                )

    return X.astype(float), y.astype(float)

def split_train_test(X : np.ndarray,
                     y : np.ndarray,
                     P : int, 
                     Q : int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset as is required in the assignment sheet:
        - Training set: the first P examples (idx 0...P-1)
        - Test set: next Q examples (idx P...Q-1)

    This guarantees that there is no leakage when training the model.

    Args:
        X : np.ndarray, y : np.ndarray
            Full dataset with X shape (num_examples, N) and y shape (num_examples,).
        P : int
            Number of training examples.
        Q : int
            Number of test examples.

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Some proper guardrails on the user-defined parameters to ensure sanity
    if P <= 0 or Q <= 0:
        raise ValueError("P and Q must be positive.")
    if P + Q > X.shape[0]:
        raise ValueError(
                f"Need at least P+Q={P+Q} examples, but X has {X.shape[0]}."
                )
    
    # Split the sets accordingly
    X_train = X[:P]
    y_train = y[:P]
    X_test = X[P:P + Q]
    y_test = y[P:P + Q]

    return X_train, y_train, X_test, y_test

def compute_errors(X_train : np.ndarray, 
                   y_train : np.ndarray, 
                   X_test : np.ndarray,
                   y_test : np.ndarray, 
                   W : np.ndarray
                   ) -> tuple[float, float]:
    """
    Compute training and test errors using standard MSE loss function.
        E       = 0.5 * mean((sigma_train - y_train)^2)
        E_test  = 0.5 * mean((sigma_test - y_test)^2)

    Use the same mse_loss() function as in part_a.py
    
    Args:
        X_train, y_train, X_test, y_test : np.ndarrays
            Train/test split of the dataset.
        W : np.ndarray
            Weight matrix of shape (K, N) for the hidden layer (K hidden units).
    
    Returns:
        E, E_test : floats
    """
    # Training & Testing set prediction
    pred_train = soft_committee_forward(X_train, W)
    pred_test = soft_committee_forward(X_test, W)
    
    # Call mse_loss() to compute the respective errors
    E = mse_loss(pred_train, y_train)
    E_test = mse_loss(pred_test, y_test)

    return float(E), float(E_test)

def training_with_tracking(X_train : np.ndarray, 
                           y_train : np.ndarray,
                           X_test : np.ndarray, 
                           y_test : np.ndarray, 
                           K : int, 
                           eta : float, 
                           tmax : int, 
                           seed : int,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    Entry point for Part (B) of the assignment. Variables at runtime are 
    introduced from config.py as globals. If you manually change them in this 
    function, you risk breaking the code.

    Pipeline:
        1) Create directory to store the results.
        2) Load xi/tau from CSV.
        3) Split into train/test using P and Q globals.
        4) Train with online SGD and track E(t), E_test(t)
        5) Save plots and print a short summary in CLI.
    
    Note: variable names in CAPITAL letters denote global variables stored in
    config.py. Please update them using that file.
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
    print(f"P={P_TRAIN}\tQ={Q_TEST}\teta={ETA:.5F}\ttmax={TMAX}\tseed={SEED}")
    print(f"Final E: {E_train[-1]:.6f}")
    print(f"Final E_test: {E_test[-1]:.6f}")
    print("Saved results in:", os.path.abspath(OUTDIR))

if __name__ == "__main__":
    main()
