# Assignment 1 - Exercise (b) + (c) #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Sun 30 Dec 2025 @ 10:30:38 +0100
# Modified: Fri 12 Dec 2025 @ 18:31:03 +0100

# Packages
import numpy as np
from typing import Dict

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Custom dictionary structure to store the results of the perceptron algorithm.
# Defined as global here to also be imported in other scripts as well.
Result = Dict[str, np.ndarray | bool | int]


if NUMBA_AVAILABLE:

    @njit
    def _rosenblatt_train_numba(
        X: np.ndarray,
        y: np.ndarray,
        n_max: int,
        learning_rate: float,
    ):
        """
        Low-level Rosenblatt training loop compiled with Numba.

        Returns
        -------
        w : np.ndarray
        converged : bool
        sweeps_done : int
        n_updates : int
        """
        P, N = X.shape
        w = np.zeros(N)
        n_updates = 0
        sweeps_done = 0
        converged = False

        for sweep in range(1, n_max + 1):
            sweeps_done = sweep

            # one sweep through all patterns
            for mu in range(P):
                xi = X[mu]
                S = y[mu]
                E = np.dot(w, xi) * S
                if E <= 0.0:
                    w += learning_rate * xi * S
                    n_updates += 1

            # check convergence: all margins positive
            margins = y * (X @ w)
            if np.all(margins > 0.0):
                converged = True
                break

        return w, converged, sweeps_done, n_updates

else:

    def _rosenblatt_train_numba(
        X: np.ndarray,
        y: np.ndarray,
        n_max: int,
        learning_rate: float,
    ):
        raise RuntimeError("Numba is not available; install numba or set use_numba=False.")


def rosenblatt_train(X : np.ndarray, 
                     y : np.ndarray,
                     n_max : int,
                     learning_rate : float | None = None,
                     use_numba: bool = True,
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
    
    if use_numba and NUMBA_AVAILABLE:
        w, converged, sweeps_done, n_updates = _rosenblatt_train_numba(
                X, y, n_max, float(learning_rate)
                )
    else:
        # Initialize the storage vector for the weights, ensure floats
        w: np.ndarray = np.zeros(N, dtype=float)
        n_updates: int = 0                          # same for the number of updates

        # Run through the epochs. Nested loop O(n^2). 
        for sweep in range(1, n_max + 1):
            updated_this_sweep = False

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
            # Early stop, perfect classification or no more changes
            margins: np.ndarray = y * (X @ w)           # Vector * (Matrix * Vector)
            if np.all(margins > 0.0) or not updated_this_sweep:
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
