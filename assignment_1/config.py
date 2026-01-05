# Config File to store Globals #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Thu 18 Dec 2025 @ 14:32:27 +0100
# Modified: Thu 18 Dec 2025 @ 14:46:00 +0100

# IMPORTANT: In general, be mindful of increasing the following numbers too high.
# The Python module `run_experiment.py` was built with convenience in mind to 
# play test simple perceptrons and datasets. Even though it is capable of so much
# more, the triple nested loop [O(n^3), best case] in lines 237-250, will signi-
# ficantly increase your computational uptime.

# I suggest running it on a cluster or on a powerful desktop computer, if you 
# want to test higher values in the following lists.


N_values = [20, 40]                 # Input dimension
c_values = [0.0, 0.1, 0.5, 1.0]     # Margin thresholds
a_min = 0.75                        # Min a = P/N to consider
a_step = 0.25                       # Step for alpha's increments
a_max = 3.25                        # Max a = P/N to consider (exluding R. bound)
n_datasets = 50                     # How many datasets to generate on each run
n_max = 100                         # Max computational budget
base_seed = None                    # Pick your static seed 
n_workers = None                    # None & 1 = sequential, >1 parallel comps.
save = True                         # Save resulting plots?
show = True                         # Show plots?

# REMARK 1: If you run the experiment in a non-GUI environment (like Habrok), then
# the show global should be turned off.

# REMARK 2: To transfer files from a cluster or a virtual environment to which you
# have access through ssh, simple use the scp protocol.
