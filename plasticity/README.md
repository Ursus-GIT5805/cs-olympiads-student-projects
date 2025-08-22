# Evaluating Plasticity Loss through a Teacher–Student Setup on MNIST
## Project Description

[📑 Check the slides](https://docs.google.com/presentation/d/1bA_67PF6VYzYw1BacBRaPVGQXOaH8yPcao7MzLU4W54/edit?usp=sharing)

This project aims to evaluate **plasticity loss** in neural networks and explore possible solutions.  
We train a **Teacher Network** on the MNIST dataset, while a **Student Network** (same size and initialization) follows the teacher during training, attempting to minimize the [KL divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between the two.
For each teacher epoch, the student performs `N` epochs of training, where `N` can vary depending on the configuration.


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

```tree
plasticity/
├── plots/                          plots images from experiments
├── differentdata.py                random data vs training data
├── differentlearners.py            experiments with different optimizer
├── good_teacher.py                 different setup with a better teacher
├── linear.py                       implementation of a linear layer
├── live_bright.py                  compares a live student (follows teacher) vs a bright student (trained from scratch)
├── loader.py                       MNIST dataset loader
├── model.py                        main model definitions (teacher & student)
├── plotter.py                      helper for generating plots
├── presets.py                      predefined neural network architectures
└── requirements.txt                different setup with a better teacher
```
---
