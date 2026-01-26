# Assignment 3 - Soft Committee Machine

This folder contains the Python code that solves Assignment 3. In this `README` file, you will find instructions on how to install the dependencies needed to run the code, as well as documentation as to what each script does.

# How to Run

Written & tested using Python 3.12.3. Python 3.10+ recommended.

1. First create a local Python virtual environment to install the dependencies by running the command `python3 -m venv <your_name_choice>`.[^1]
2. Run `pip install -r requirements.txt`.[^2]
3. You are ready to go! Run each Python file with `python3 script.py`.

# Part A

The file `part_a.py` is dedicated to the solution of the first part of the assignment with the goal of reusing its code in the second part as well. More specifically, it implements five different building blocks:

1. **Forward pass** of a soft committee machine (`soft_committee_forward`)
2. **Loss function** as a half mean-squared error (`mse_loss`)
3. **Online SGD update** for the weights (`sgd_step`)
4. **Weight initialization** as  unit-norm rows (`init_weights`)
5. **A training loop** that runs an epoch-based SGD and (optionally) logs the training error (`train_machine`)

There is also a small `__main__` block that specifically tests these aforementioned functions on a small synthetic dataset.

## Model: Soft Committee Machine

The model is a shallow neural network with:
- One hidden layer with `K` hidden units.
- `tanh` as the activation function of the hidden units.
- Output computed as the sum of all the hidden unit activations.
- Hidden-to-output weights fixed to 1, thus the only parameters are the input-to-hidden weights.

## Forward Pass

The forward pass basically computes the model prediction for each example. It works as follows:
1. Compute the dot product between the input vector and each hidden unit's weight vector.
2. Apply `tanh` to each of those dot products (one value per hidden unit).
3. Sum the hidden unit outputs to get the final scalar prediction for that example.

## Error Metric

The error function is the typical mean squared error. In code it is implemented as `(0.5 * MSE)`. 

## SGD Update Step

Function `sgd_step` performs exactly one stochastic gradient descent update using a single training example:
- `x`: one input vector
- `tau`: the corresponding target value
- `W`: the weight matrix updated **in place**
- `eta`: the learning rate

It computes and tries to minimize the squared prediction error on the sampled example using the same error metric as discussed before. It is noteworthy that the update is vectorized so all K hidden weight vectors can be updated without an explicit loop.

## Weight initialization

The goal here is to create an initial `(K, N)` weight matrix with random entries, then normalize each row to have a unit Euclidean norm. Normalization is performed to keep the hidden units on a comparable scale at the start, and to avoid accidental "large" initial weights that will dominate early training dynamics.

Note: after initialization, the norms are not forced to stay at 1, and so SGD can change them.

## Training Loop

The goal here is to run an epoch-based SGD training over the training set. One epoch is defined as `P` SGD updates, where `P` is the number of training examples. Then, each update samples one training example uniformly at random (sampling is done with replacement). A maximum iteration budget is enforced as `updates = tmax * P`.

If `verbose=True`, the function computes the training error once at `t=0` and then once after each epoch. This yields a training error array of length `tmax + 1`. 

## Returned Object

When the whole pipeline has finished running, a dictionary is returned containing at least:
- the final weights,
- the training error history (if enabled),
- the hyper-parameters used.

# Part B

The file `part_b.py` implements the second part of the assignment and is basically the experiment runner. To be more specific, it uses the primitive code of Part A, and applies it on the target dataset in question. As a deliverable it produces:
1. A learning curve plot, showcasing training error versus generalization error.
2. Bar plots of the final learned weight vectors `w_1, w_2`.

## Pipeline
1. Load the provided data files `xi.csv` and `tau.csv` into scope.
2. Split the data into a fixed training and test set partition following the desire of the user.
3. Train the model with online SGD for a specified number of epochs.
4. Measure training and test error once per epoch.
5. Generate the required plots to showcase to the user the respective learning curves and final weights.


[^1]: This command is different in Windows. I am not entirely familiar with it, but follow [this guide](python.land/virtual-environments/virtualenv) if you are having trouble.

[^2]: If you use a global Python virtual environment, this command will forcefully update your (probably) already installed modules.
