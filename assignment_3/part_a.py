# Part A - Assignment 3 - Neural Networks #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 5 Jan 2026 @ 11:50:51 +0100
# Modified: Sun 25 Jan 2026 @ 18:31:05 +0100

# Packages
from __future__ import annotations
import numpy as np

def soft_committee_forward(X : np.ndarray, W : np.ndarray) -> np.ndarray:
    """
    Forward pass for the soft committee machine with a fixed v_k = 1.

    Args:
        X : np.ndarray
            Input matrix of shape (P, N), where:
            - P = number of examples
            - N = number of features
        W : np.ndarray
            Weight matrix of shape (K, N), where:
            - K = number of hidden units
            - W[k] is the weight vector w_k

    Returns:
        np.ndarray
            Model output sigma(X) of shape (P,).
    """
    U = X @ W.T             # shape (P, K)
    H = np.tanh(U)          # shape (P, K)
    sigma = H.sum(axis=1)   # shape (P,)
    return sigma

def mse_loss(y_pred : np.ndarray, y_true : np.ndarray) -> float:
    """
    Compute the average half squared error:
        E = 0.5 * mean(( y_prediction - y_true )^2)

    Args:
        y_pred : np.ndarray
            Predictions with shape (P,) (or broadcastable to (P,)).
        y_true : np.ndarray
            Targets with shape (P,) (also broadcastable).

    Returns:
        float
            Loss value.
    """
    err = y_pred - y_true
    output = 0.5 * np.mean(err * err)
    return output

def sgd_step(x : np.ndarray, tau : float, W : np.ndarray, eta : float) -> None:
    """
    Perform a single-example stochastic gradient update on the hidden weights W
    using one example (x, tau).

    Model:
        sigma(x) = sum_k tanh(w_k * x)

    Single-sample loss:
        e = 0.5 * (sigma(x) - tau)^2

    Gradient:
        de/dw_k = (sigma(x) - tau) * (1 - tanh^2(w_k * x)) * x

    Update:
        w_k <- w_k - eta * de/dw_k

    Args:
        x : np.ndarray
            Single input vector with shape (N,).
        tau : float
            Target value for this example.
        W : np.ndarray
            Hidden weights, shape (K, N). Updated in place, hence why no output
            in the function's signature.
        eta : float
            Learning rate (this is constant for Part (a) of the assignment).
    """
    # Forward update for one example
    u = W @ x                           # (K,)
    h = np.tanh(u)                      # (K,)
    sigma = h.sum()                     # scalar
    delta = sigma - tau                 # scalar

    # Derivative of tanh(u) = 1 - tanh^2(u)
    dh_du = 1.0 - h * h                 # (K,)
    
    # In-place update: (K, 1) * (1, N) -> (K, N)
    W -= eta * (delta * dh_du)[:, None] * x[None, :]

    # Since W is updated here, and then this function is called inside the nested
    # loop, there is no need for an output.

def init_weights(K : int, N : int, rng : np.random.Generator) -> np.ndarray:
    """
    Initialize K weight vectors in N-dim. real space with unit Euclidean norm.
    
    Args:
        K : int
            Number of hidden units.
        N : int
            Input dimensionality (number of features)
        rng : np.random.Generator
            NumPy random generator in case reproducability is needed.

    Returns:
        np.ndarray
            Weight matrix W of shape (K, N), where each row has norm(w_k) = 1. 
            Once again the metric here is the standard Euclidean norm.
    """
    # Normally distributed initialization
    W = rng.normal(loc=0.0, scale=1.0, size=(K, N))

    # Compute the norm of each row vector
    # Keep dimensions so it is (K, 1) instead of (K,)
    norms = np.linalg.norm(W, axis=1, keepdims=True)

    # Guard against division by zero, so I don't go insane
    norms = np.where(norms == 0.0, 1.0, norms)

    return W / norms

def train_machine(
        X_train : np.ndarray,
        y_train : np.ndarray,
        *,
        K : int = 2,
        eta : float = 0.05,
        tmax : int = 200,
        seed : int = 0,
        verbose : bool = True) -> dict:
    """
    Train the soft committee machine using SGD. This effectively solves Part (a)
    of the assignment.

    Time, what is time?
    - One epoch = P single-example SGD updated (sampling is done with replacement)
    - Total updates = tmax * P

    Args:
        X_train : np.ndarray
            Training inputs with shape (P, N).
        y_train : np.ndarray
            Training targets with shape (P,).
        K : int, optional (default is 2 as per the assignment's instructions)
            Number of hidden units.
        eta : float, optional (default is 0.05)
            A constant learning rate.
        tmax : int, optinal (default is 200)
            Number of epochs. Total SGD steps are tmax * P.
        seed : int, optional (default is 0)
            RNG seed for a reproducable initialization and sampling.
        verbose : bool, optional (default is True)
            If true, it logs E(t) once per epoch with t=0 included.

    Returns:
        dict
            Dictionary with:
            - "W_final": np.ndarray, final weights with shape (K, N).
            - "E_train": np.ndarray, training error history (list of lengths
            tmax+1 if verbose is turned on, otherwise this is empty)
            - "eta", "tmax", "seed": training configuration used.
    """
    # Transform everything to numpy datastructures if they are not one yet
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # Fetch the shape
    P, N = X_train.shape

    # Decide on rng
    rng = np.random.default_rng(seed)

    # Initialize weight using the function from before
    W = init_weights(K, N, rng)

    # Initialize the storage list and ensure that it will only contain floats
    E_hist: list[float] = []

    # Compute E(t=0) and add it to the list
    if verbose:
        y_pred_0 = soft_committee_forward(X_train, W)
        E_hist.append(mse_loss(y_pred_0, y_train))

    # One epoch is P SGD updates with replacement
    for epoch in range(1, tmax + 1):
        # Omit iterable
        for _ in range(P):
            # Decide the special sample
            idx = rng.integers(low = 0, high = P)       

            # Do an SGD step
            sgd_step(X_train[idx], float(y_train[idx]), W, eta)
        
        # Log the result onto the list once the epoch is done
        if verbose:
            y_pred = soft_committee_forward(X_train, W)
            E_hist.append(mse_loss(y_pred, y_train))

    return {
            "W_final" : W,
            "E_train" : np.array(E_hist, dtype=float),
            "eta"     : eta,
            "tmax"    : tmax,
            "seed"    : seed,
            }

if __name__ == "__main__":
    # Dummy test set to see if everything is working properly
    P, N = 100, 50
    rng = np.random.default_rng(21)
    X = rng.normal(size=(P, N))

    # Teacher (I made it up)
    y = np.tanh(X[:, 0]) + 0.5 * np.tanh(0.7 * X[:, 1])

    out = train_machine(X, y, K=2, eta=0.05, tmax=50, seed=1, verbose=True)

    print("Final training E:", out["E_train"][-1])
    print("Final W norms:", np.linalg.norm(out["W_final"], axis=1))
