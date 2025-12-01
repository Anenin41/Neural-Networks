# Assignment 1 - Exercise (c) #
# Generate one random dataset and run the Rosenblatt algo on it. #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 01 Dec 2025 @ 13:13:15 +0100
# Modified: Mon 01 Dec 2025 @ 13:19:34 +0100

# Packages
from typing import Tuple
import numpy as np

from generate_data import generate_dataset
from sequential_perceptron import rosenblatt_train, Result

def run_single_datasett(P : int,
                        N : int,
                        n_max : int = 100,
                        seed : int | None = None,
                        ) -> Tuple[np.ndarray, np.ndarray, Result]:
    """
    This funtion generates one data set and trains a perceptron on it.

    Args:
        P : int
