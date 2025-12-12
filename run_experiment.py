# Assignment 1 - Exercise (d) #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 19:13:54 +0100
# Modified: Fri 12 Dec 2025 @ 18:31:01 +0100

# Packages
from typing import Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Custom modules
from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result

def _single_run(args: tuple[int, int, int, int | None]) -> int:
    """
    Litle helper function for debugging, nothing interesting here.
    """
    P, N, n_max, seed = args
    X, y = generate_dataset(P, N, seed=seed)
    result: Result = rosenblatt_train(X, y, n_max=n_max)
    return result["converged"]

def estimate_Q(N : int,
               P_values : Iterable[int],
               n_datasets : int,
               n_max : int,
               base_seed : int | None = None,
               plot: bool = False,
               verbose: bool = True,
               n_workers: int | None = None,
               ) -> Tuple[List[float], List[float]]:
    """
    This is the main script of the assignment. It call all the other functions of
    the repository, and performs the computer experiments necessary to solve the
    assignment sheet. 

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
            seed is parsed, it uses the global RNG state.
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
        print("P\talpha\tQ_ls\t(successes / n_datasets)")   #\t = 1 tab space
    
    # Number of feature vectors
    for P in P_values:
        successes: int = 0

        # Pre-generate seeds for all datasets
        seeds: list[int | None] = []

        # Omit iterable, not necessary here
        for _ in range(n_datasets):
            if base_seed is not None:
                seed_dataset = int(rng.integers(0, 2**32 - 1))  # pick one
                seeds.append(seed_dataset)
            else:
                seeds.append(None)

        args_list = [(P, N, n_max, s) for s in seeds]

        if n_workers is None or n_workers == 1:
            # sequential
            results = [_single_run(a) for a in args_list]
        else:
            # paraller across CPU cores
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_single_run, args_list))

        successes = int(sum(results))
        alpha = P / float(N)
        q_ls = successes / float(n_datasets)
            
        
        # Append them on the storage list
        alphas.append(alpha)
        q_ls_vals.append(q_ls)
        
        # Print results in CLI if verbose is True
        if verbose:
            print(f"{P}\t{alpha:.3f}\t{q_ls:.3f}\t({successes} / {n_datasets})")
    
    # Plot the alpha curve as well
    if plot:
        plt.figure()
        plt.plot(alphas, q_ls_vals, marker="o")
        plt.xlabel("alpha = P/N")
        plt.ylabel("Q(alpha)")
        plt.title("Empirical probability of linear separability")
        plt.grid(True)
        plt.show()

    return alphas, q_ls_vals

# To run the experiment for different values, simply modify the following numbers
if __name__ == "__main__":
    N = 40
    a = 0.20
    P_values = [] 
    while a < 3.0:
        p_val = int(a * N)
        P_values.append(p_val)
        a += 0.20
    estimate_Q(
            N,
            P_values,
            n_datasets=100,
            n_max=1000,
            base_seed=None,
            plot=True,
            verbose=True,
            n_workers=4
            )
