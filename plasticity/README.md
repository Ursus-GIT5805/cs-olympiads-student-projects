# Evaluating Plasticity Loss through a Teacher–Student Setup on MNIST
## Project Description

[📑 Check the slides](https://docs.google.com/presentation/d/1bA_67PF6VYzYw1BacBRaPVGQXOaH8yPcao7MzLU4W54/edit?usp=sharing)

This project aims to evaluate **plasticity loss** in neural networks and explore possible solutions.
We train a **Teacher Network** on the MNIST dataset, while a **Student Network** (same size and initialization) follows the teacher during training, attempting to minimize the [KL divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between the two.
For each teacher epoch (era), the student performs multiple epochs of training.

## Installation

1. Navigate into the project folder:
   ```bash
   cd plasticity
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

There are several files, which specifically plots something:

- `differentdata.py`: Experiments with students training on "noise" vs "training data"
- `differentlearners.py`: Experiments with different optimizer
- `good_teacher.py`: Experiments with a fast learning teacher
- `live_bright.py`: Compares live student (continual) and bright student (from scratch)

Each of those files have a `--- Variables ---` section, which contain the used hyperparameters. The other files contain utility methods, which are further specified in the files itself.

The `plots` directory contain plots created by the scripts.
