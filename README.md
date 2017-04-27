# tf-examples
TensorFlow tutorials.

# Requirements
The code has been tested with Python 3.5 and TensorFlow 1.0.1.

# Notes

## Datasets

* `datasets/iris/iris.txt` is a subset of the full [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) used to illustrate simple linear and logistic regression. It contains the sepal length, petal length, and species label for _Iris versicolor_ and _Iris virginica_.

## Simple linear regression with the Iris dataset

* `iris_linreg_np.py` implements linear regression in NumPy with batch gradient descent, i.e., each gradient is computed on the entire dataset.
* `iris_linreg_tf.py` does the same with TensorFlow.

## Simple binary logistic regression with the Iris dataset

## Multiclass logistic regression with MNIST

## Convolutional neural network with MNIST

* `mnist_cnn.py` implements a convolutional neural network.
* `mnist_cnn_model.py`, `mnist_cnn_train.py`, and `mnist_cnn_test` do the same but demonstrate a more canonical way to organize code for such projects. They also demonstrate how to use variable scopes, saving and restoring models, and using TensorBoard to monitor training progress.

## Variational autoencoder for MNIST

* `vae.py` is a simplified version of the code from https://jmetzen.github.io/2015-11-27/vae.html.
