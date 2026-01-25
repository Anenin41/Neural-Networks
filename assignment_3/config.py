# Config File to store Globals #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Sun 25 Jan 2026 @ 17:10:43 +0100
# Modified: Sun 25 Jan 2026 @ 18:24:44 +0100

# CSV Files
XI_PATH = "xi.csv"
TAU_PATH = "tau1.csv"

# Number of training examples
P_TRAIN = 100

# Number of test examples
Q_TEST = 100

# Number of hidden units
K_HIDDEN = 2

# Training Configuration
ETA = 0.05                  # constant learning rate
TMAX = 200                  # number of epochs, each one = P SGD updates
SEED = 21                   # RNG seed

# Where to save the figures?
OUTDIR = "results"
