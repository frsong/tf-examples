# tf-examples
TensorFlow tutorials.

# Requirements
The code has been tested with Python 3.5 and TensorFlow 1.0.1.

# Notes

## Datasets

* `datasets/iris/iris.txt` is a subset of the full [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) used to illustrate simple linear and logistic regression. It contains the sepal length, petal length, and species label for _Iris versicolor_ and _Iris virginica_.

## Linear regression with the Iris flower dataset.

* `iris_linreg_np.py` implements linear regression in NumPy with batch gradient descent, i.e., each gradient is computed on the entire dataset.
* `iris_linreg_tf.py` does the same with TensorFlow.

## Logistic regression for MNIST handwritten digit classification.

## Convolutional neural network (CNN) for MNIST handwritten digit classification.

* `mnist_cnn.py` implements a CNN.
* `mnist_cnn_model.py`, `mnist_cnn_train.py`, and `mnist_cnn_test` do the same but demonstrate a more canonical way to organize files for such projects. They also demonstrate how to use `tf.get_variable()` instead of `tf.Variable()`, saving and restoring models, and using TensorBoard to monitor training progress.

## Variational autoencoder for MNIST.

* `vae.py` is a simplified version of the code from https://jmetzen.github.io/2015-11-27/vae.html.
