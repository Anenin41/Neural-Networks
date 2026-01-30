# Assignment Repository

This repository contains the assignments of the course, organized into self-contained folders. Each assignment directory includes its own implementation, documentation, and dependency list, as well as instruction on how to properly run the individual pieces of code.

## Repository Structure

```
.
├── assignment_1
│   ├── config.py
│   ├── generate_data.py
│   ├── README.md
│   ├── requirements.txt
│   ├── run_experiment.py
│   ├── sequential_perceptron.py
│   ├── single_experiment.py
│   └── testing.py
├── assignment_3
│   ├── config.py
│   ├── part_a.py
│   ├── part_b.py
│   ├── bonus.py
│   ├── README.md
│   ├── requirements.txt
│   ├── results
│   │   ├── learning_curves.png
│   │   ├── weights_w1.png
│   │   └── weights_w2.png
│   ├── tau1.csv
│   └── xi.csv
└── README.md
```

Notes:
1. When running `part_b.py`, the code will try to import files `tau.csv` and `xi.csv` into scope. Make sure to properly update the paths in the respective `config.py` file, or just "dump" these CSVs into the respective folder as is showcased above.
2. When `part_b.py` is finished, it will create `results/` folder in which the output `.png` images will be stored. 

## Assignments

- **Assignment 1**: See [`assignment_1/README.md`](assignment_1/README.md)
- **Assignment 3**: See [`assignment_3/README.md`](assignment_3/README.md)

Each assignment README documents:
- Short but detailed instructions on how to properly install the dependencies needed to run the code. 
- What each code file actually implements, its most important features, those being technical or theoretical. 
