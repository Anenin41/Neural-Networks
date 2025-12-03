# Testing & Sanity Checks for each Perceptron Script #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 09:03:25 +0100
# Modified: Wed 03 Dec 2025 @ 11:38:54 +0100

# Packages
from typing import List, Tuple
import numpy as np

from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result
from run_perceptron import run_single_dataset
from capacity_experiment import estimate_Q

def main() -> None:
    """
    Run all tests in this script.
    """
    print("Running tests...\n")
    test_generate_data()
    test_rosenblatt_train()
    test_run_single_dataset()
    test_estimate_Q()
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

def test_run_single_dataset() -> None:
    """
    Sanity check for the run_perceptron script.
    """
    P: int = 30
    N: int = 15

    X: np.ndarray
    y: np.ndarray
    res: Result
    X, y, res = run_single_dataset(P, N, n_max=100, seed=2)

    assert X.shape == (P, N)
    assert y.shape == (P,)
    assert "weights" in res
    assert isinstance(res["weights"], np.ndarray)

    print("test_run_single_dataset: OK", f"( converged = {res['converged']}, sweeps = {res['sweeps']} )")

def test_estimate_Q() -> None:
    """
    Sanity check estimate_Q for a small experiment.
    """
    N: int = 10
    P_values: List[int] = [5, 10, 15]

    alphas: List[float]
    q_ls_vals: List[float]

    alphas, q_ls_values = estimate_Q(
        N=N,
        P_values=P_values,
        n_datasets=5,       # keep it small and fast
        n_max=50,
        base_seed=3,
        plot=False,
        verbose=False,
    )

    assert len(alphas) == len(P_values)
    assert len(q_ls_values) == len(P_values)

    # All probabilities must be in [0, 1]
    assert all(0.0 <= q <= 1.0 for q in q_ls_values)

    print("test_estimate_q_ls: OK")
    print(" - alphas =", alphas)
    print(" - Q_ls(alpha) =", q_ls_values)


if __name__ == "__main__":
    main()
