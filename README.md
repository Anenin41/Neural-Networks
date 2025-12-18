# About
This repository contains the Python code that solves Assignment 1. In this `README` file, you will find instructions on how to install the dependencies needed to run the code, as well as documentation as to what each script does.

# How to Run
Written & tested using Python 3.12.3. Python 3.10+ recommended.
1. First create a local Python virtual environment to install the dependencies by running the command `python3 -m venv <your_name_choice>`.[^1]
2. Run `pip install -r requirements.txt`.[^2]
3. Run `python3 testing.py` to ensure that all scripts are running properly and that the commands above were completed successfully.
4. You are ready to go! Run each Python file with `python3 script.py`.

# Project Structure

- `generate_data.py`
    Generates synthetic datasets: `X ~ N(0,1)` and "binary" labels `y in {+1, -1}`.

- `sequential_perceptron.py`
    Implements the Rosenblatt training algorithm:
    - Local potential per sample: `E_mu = y_mu * (w, xi_mu)`
    - Update condition: update if `E_mu <= c`
        - `c = 0.0` gives the standard perceptron update on misclassified points.
        - `c > 0.0` also updates on correctly classified but low-margin points.

- `single_experiment.py`
    At this stage of the development, this is a helper file and was used to manually play-test if the perceptron training algorithm was running properly. It can generate one dataset and run training on it once.

- `run_experiment.py`
    This file holds the main numerical experiments:
    - `estimate_Q(...)` estimates the empirical probability for linear separability by running many independent datasets for each `P`.
    - It supports (optional) parallel execution via `ProcessPoolExecutor`, which significantly reduces computational overhead.
    - `compare_c_values(...)` plots multiple empirical probability curves in one figure for different values of the threshold.

- `testing.py`
    Sanity checks that ensure the logic of these files is sound. Note that the output of these files are not really validated here. 

- `config.py`
    A Python file which allows fast access to the experiment variables used in `run_experiment.py`. These are passed around in the document as global variables.

[^1]: This command is different in Windows. I am not entirely familiar with it, but follow [this guide](https://python.land/virtual-environments/virtualenv) if you are having trouble.

[^2]: If you use a global Python virtual environment, this command will forcefully update your (probably) already installed modules. 
