# Testing & Sanity Checks for each Perceptron Script #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 09:03:25 +0100
# Modified: Mon 01 Dec 2025 @ 13:19:49 +0100

# Packages
from typing import List, Tuple
import numpy as np

from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result

def main() -> None:
    """
    Run all tests in this script.
    """
    print("Running tests...\n")
    test_generate_data()
    test_rosenblatt_train()
    print("\nAll tests completed successfully.")

def test_generate_data() -> None:
    """
    Sanity check for generate_dataset.
    """
    P: int = 10
    N: int = 5
    X: np.ndarray
    y: np.ndarray
    X, y = generate_dataset(P, N, seed=0)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (P, N), f"X shape mismatch: {X.shape}"
    assert y.shape == (P,), f"y shape mismatch: {y.shape}"

    unique_labels = np.unique(y)
    assert set(unique_labels).issubset({1, -1}), f"Unexpected labels: {unique_labels}"

    print("test_generate_data: OK")

def test_rosenblatt_train() -> None:
    """
    Sanity check for rosenblatt_train on a small problem.
    """
    P: int = 20
    N: int = 10
    X: np.ndarray
    y: np.ndarray
    X, y = generate_dataset(P, N, seed=1)
    
    result: Result = rosenblatt_train(X, y, n_max=100)

    # Check structure of the result
    for key in ("weights", "converged", "sweeps", "updates"):
        assert key in result, f"Missing key '{key}' in result"

    w: np.ndarray = result["weights"]

    assert isinstance(w, np.ndarray)
    assert w.shape == (N,), f"Weight vector shape mismatch: {w.shape}"
    assert isinstance(result["converged"], bool)
    assert isinstance(result["sweeps"], int)
    assert isinstance(result["updates"], int)

    print("test_rosenblatt_train: OK ( converged =", result["converged"], ")")

if __name__ == "__main__":
    main()
