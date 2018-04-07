# README

## About

A library for gaussian process regression, I write this library for my Ph.D research of Bayesian optimization

- ARD Gaussian Kernel
- Constant mean function
- MLE to estimate the hyperparameters

## Dependencies

- NLOPT
- [MVMO](https://github.com/Alaya-in-Matrix/MVMO)

## TODO

- VFE and FITC approximation


## Usage

The training data should be stored in text files `train_x` and `train_y`, the test data should be written in `test_x`, see `data/` for example

To use the first `num_train` data in the `train_x` and `train_y` as training set:

```bash
$ gp   num_train
$ fitc num_train num_inducing
```

