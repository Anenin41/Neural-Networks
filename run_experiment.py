# Assignment 1 - Exercise (d) #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 19:13:54 +0100
# Modified: Mon 01 Dec 2025 @ 22:09:45 +0100

# Packages
from typing import Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result

def estimate_Q(N : int,
               P_values : Iterable[int],
               n_datasets : int,
               n_max : int,
               base_seed : int | None = None,
               plot: bool = False,
               verbose: bool = True,
               ) -> Tuple[List[float], List[float]]:
    """
    This is the main script of the assignment. It call all the other functions of
    the repository, and performs the computer experiments as described in (d) of
    the Assignment.

    Args:
        N : int
            Input dimension.
        P_values : iterable of int
            List or other iterable data structure of P values to test.
        n_datasets : int
            Number of independent datasets for each P.
        n_max : int
            Maximum sweeps per training run.
        base_seed : int or None
            Posibility to parse static seed for reproducibility of results. If no
            seed is parsed, it used the global RNG state.
        plot : bool
            If true, plots Q_ls(alpha) vs alpha.
        verbose : bool
            When true, it prints the results of the experiment on the CLI.

    Returns:
        alphas: list of float
        q_ls_vals : list of float
            Empirical probabilities achieved for each alpha.
    """
    # Hubris
    rng = np.random.default_rng(base_seed)

    # Storage lists
    alphas: List[float] = []
    q_ls_vals: List[float] = []

    if verbose:
        print("P\talpha\tQ_ls\t(successes / n_datasets)")

    for P in P_values:
        successes: int = 0

        for _ in range(n_datasets):
            # Fresh seed per dataset for reproducibility
            seed_dataset: int | None
            if base_seed is not None:
                seed_dataset = int(rng.integers(0, 2**32 - 1))
            else:
                seed_dataset = None

            X: np.ndarray
            y: np.ndarray
            X, y = generate_dataset(P, N, seed=seed_dataset)
            result: Result = rosenblatt_train(X, y, n_max=n_max)

            if result["converged"]:
                successes += 1

        alpha: float = P / float(N)
        q_ls: float = successes / float(n_datasets)

        alphas.append(alpha)
        q_ls_vals.append(q_ls)
        
        if verbose:
            print(f"{P}\t{alpha:.3f}\t{q_ls:.3f}\t({successes} / {n_datasets})")

    if plot:
        plt.figure()
        plt.plot(alphas, q_ls_vals, marker="o")
        plt.xlabel("alpha = P/N")
        plt.ylabel("Q(alpha)")
        plt.title("Empirical probability of linear separability")
        plt.grid(True)
        plt.show()

    return alphas, q_ls_vals

if __name__ == "__main__":
    N = 20
    P_values = [5, 10, 15, 20, 25, 30]
    estimate_Q(
            N,
            P_values,
            n_datasets=100,
            n_max=500,
            base_seed=None,
            plot=True,
            )
