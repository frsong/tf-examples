# TensorFlow examples

It seemed to me while learning TensorFlow that some of the tutorials online provide code that is more complicated and difficult to read than necessary, in particular obscuring the shared structure of many deep learning models. This is my attempt to remedy the situation, I hope it helps.

# Requirements

The code has been tested with Python 3.5 and TensorFlow 1.0.1.

# Notes

The results are reproducible on a CPU but not on GPUs.

## Datasets

* `datasets/iris/iris.txt` is a subset of the full [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) used to illustrate simple linear and logistic regression. It contains the sepal length, petal length, and species label for _Iris versicolor_ and _Iris virginica_.

* MNIST will download itself as needed.

## Simple linear regression with the Iris dataset

* `iris_linreg_np.py` implements linear regression in NumPy with batch gradient descent, i.e., each gradient is computed on the entire dataset.

* `iris_linreg_tf.py` does the same with TensorFlow.

## Simple binary logistic regression with the Iris dataset

* `iris_logreg_np.py` implements logistic regression in NumPy with stochastic gradient descent, i.e., each gradient is computed on one random example from the dataset.

* `iris_logreg_tf.py` does the same with TensorFlow.

## Multiclass logistic regression with MNIST

## Convolutional neural network with MNIST

* `mnist_cnn.py` is a simplified version of the code from https://www.tensorflow.org/get_started/mnist/pros.

```
Variables
---------
conv1/W:0 (5, 5, 1, 32)
conv1/b:0 (32,)
conv2/W:0 (5, 5, 32, 64)
conv2/b:0 (64,)
fc/W:0 (3136, 1024)
fc/b:0 (1024,)
softmax/W:0 (1024, 10)
softmax/b:0 (10,)
=> Total number of parameters = 3274634
After 1 epochs, validation accuracy = 0.97079998254776
After 2 epochs, validation accuracy = 0.9829999804496765
After 3 epochs, validation accuracy = 0.9779999852180481
After 4 epochs, validation accuracy = 0.9825999736785889
After 5 epochs, validation accuracy = 0.9876000285148621
After 6 epochs, validation accuracy = 0.9876000285148621
After 7 epochs, validation accuracy = 0.9876000285148621
After 8 epochs, validation accuracy = 0.9914000034332275
After 9 epochs, validation accuracy = 0.9887999892234802
After 10 epochs, validation accuracy = 0.9855999946594238
Test accuracy = 0.9858999848365784
```

* `mnist_cnn_model.py`, `mnist_cnn_train.py`, and `mnist_cnn_test` do the same but with dropout, and demonstrate a more canonical way of organizing the code so that the model can be used for both training and extracting predictions from the saved model. Includes variable scopes, saving and restoring checkpoints, and using TensorBoard to monitor training progress.

## Variational autoencoder with MNIST

* `vae.py` is a simplified version of the code from https://jmetzen.github.io/2015-11-27/vae.html.
