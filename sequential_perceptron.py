# Assignment 1 - Exercise (b) + (c) #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Sun 30 Dec 2025 @ 10:30:38 +0100
# Modified: Fri 12 Dec 2025 @ 20:06:29 +0100

# Packages
import numpy as np
from typing import Dict

# Custom dictionary structure to store the results of the perceptron algorithm.
# Defined as global here to also be imported in other scripts as well.
Result = Dict[str, np.ndarray | bool | int]

def rosenblatt_train(X : np.ndarray, 
                     y : np.ndarray,
                     n_max : int,
                     learning_rate : float | None = None,
                     ) -> Result:
    """
    This function runs the Rosenblatt training algorithm as is explained in
    exercise (b) of the assignment.

    Args:
        X : np.ndarray, shape (P, N)
            Storage matrix for the feature vectors
        y : np.ndarray, shape (P,)
            Binary labels chosen equally from {+1, -1}
        n_max : int
            Maximum number of allowed sweeps (epochs) through the data
        learning_rate : float or None
            This is the coefficient in front of the second term at the update
            rule. If None, then 1/N is chosen, the default choice from the 
            assignment.

    Returns:
        result : dictionary with keys
            "weights"   : np.ndarray
            "converged" : bool
            "sweeps"    : int
            "updates"   : int
    """
    # Make sure that input is a numpy nd.array hosting floats
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Fetch the shape of the feature matrix
    P, N = X.shape
    
    # Possibility to also define a custom learning rate, or default back to the
    # one that was presented on the lectures.
    if learning_rate is None:
        learning_rate = 1.0 / N

    # Initialize the storage vector for the weights, ensure floats
    w: np.ndarray = np.zeros(N, dtype=float)
    n_updates: int = 0                          # same for the number of updates

    # Run through the epochs. Nested loop O(n^2). 
    for sweep in range(1, n_max + 1):
        for mu in range(P):
            xi = X[mu]                      # fetch feature vector
            S = float(y[mu])                # fetch its label
            E = float(np.dot(w, xi) * S)    # compute local potential
            
            # Update the weights based on the local potential criterion
            if E <= 0.0:
                w = w + learning_rate * xi * S
                n_updates += 1
                updated_this_sweep = True

    # Check separability after each full sweep
    margins: np.ndarray = y * (X @ w)           # Vector * (Matrix * Vector)
    if np.all(margins > 0.0):
        return {
                "weights"   : w, 
                "converged" : True,
                "sweeps"    : sweep,
                "updates"   : n_updates,
                }

    # No separating solution was reached within n_max sweeps
    return {
            "weights"   : w,
            "converged" : False,
            "sweeps"    : n_max,
            "updates"   : n_updates,
            }
