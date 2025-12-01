# About
This repository contains the Python code that solves Assignment 1. In this `README` file, you will find instructions on how to install the dependencies needed to run the code, as well as documentation as to what each script does.

# How to Run
1. First create a local Python virtual environment to install the dependencies by running the command `python3 -m venv <your_name_choice>`.[^1]
2. Run `pip install -r requirements.txt`.
3. Run `python3 testing.py` to ensure that all scripts are running properly and that the commands above were completed successfully.
4. You are ready to go! Run each Python file with `python3 script.py`.

# `generate_data.py`
**Corresponding Exercise:** (a)

This script implements Exercise (a). In generates a synthetic dataset with i.i.d. Gaussian feature vectors and random labels in `{+1, -1}`. It also includes a helper function to save any generated dataset to CSV with one column per feature and a final `label` column.

# `sequential_perceptron.py`
**Corresponding Exercise:** (b) & (c)

This script implements the Rosenblatt perceptron learning rule as is asked for Exercises (b) and (c). Given `X` and `y`, it runs sequential training over at most `n_max` sweeps (epochs) with the same update formula as in the lecture notes. The weights update based on the value of the local potential. Finally, it returns a dictionary with the final weight vector, convergence flag, number of sweeps, and number of updates.

# `single_experiment.py`
**Corresponding Exercise:** (c)

This script wraps up the full pipeline of "generate one random dataset + train the perceptron once". It can be used to run the the learning procedure on custom-made datasets and when it is run as script, a set of basic statistics is returned on the command line interface.

# `run_experiment.py`
**Corresponding Exercise:** (d)

This file implements Exercise (d). The function `estimate_Q(...)` repeats the perceptron training procedure for many independently generated datasets. It estimates the empirical success probability `Q(a)` as a function of `a = P/N`. Optionally, it can print these results to the console, and/or plot `Q(a)`.

# `testing.py`

Contains lightweight sanity tests for all components. It checks dataset generation, the core Rosenblatt training routine, the single-dataset pipeline, and the capacity experiment, ensuring the logic of the code is sound, data types are correct and probability ranges don't violate the mathematics. Running this script executes all tests and reports success/failure in the terminal through a number of  `assert` commands. 

[^1]: You can skip this part if you are using a global virtual environment to handle your Python dependencies. Be aware that when you run the `pip install -r requirements.txt` command, the process will update the respective modules globally, and might cause compatibility issues with other scripts. 
