# Assignment 3 - Soft Committee Machine

This folder contains the Python code that solves Assignment 3. In this `README` file, you will find instructions on how to install the dependencies needed to run the code, as well as documentation as to what each script does.

# How to Run

Written & tested using Python 3.12.3. Python 3.10+ recommended.

1. First create a local Python virtual environment to install the dependencies by running the command `python3 -m venv <your_name_choice>`.[^1]
2. Run 'pip install -r requirements.txt'.[^2]
3. You are ready to go! Run each Python file with `python3 script.py`.

# Project Structure

1. `part_a.py` Implements stochastic gradient descent training for a *soft committee machine* with `K=2` hidden units, fixed `v_k = 1` hidden-to-output weights, activation function `tanh` and quadratic deviation loss. Implementation is as follows:
    - Forward pass `sigma(x) = sum over k tanh(w_k * x)`.
    - Loss function `E = (1/P) sum 0.5 (sigma - tau)^2`.
    - Single-sample SGD update.
    - Training loop with `tmax * P` updates, with `tmax` being the iteration budget defined by the user.

[^1]: This command is different in Windows. I am not entirely familiar with it, but follow [this guide](python.land/virtual-environments/virtualenv) if you are having trouble.

[^2]: If you use a global Python virtual environment, this command will forcefully update your (probably) already installed modules.
