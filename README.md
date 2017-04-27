# TensorFlow examples

It seemed to me while learning TensorFlow that much of the code found online is more complicated than necessary, in particular hiding the shared structure underlying the training of many deep learning models. I tried to make the code as easy to read as possible, and only used "advanced" features in a few places when they seemed appropriate.

# Requirements

The code has been tested with Python 3.5 and TensorFlow 1.0.1.

# Notes

The results are reproducible on a CPU but not on GPUs.

## Datasets

* `datasets/iris/iris.txt` is a subset of the full [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) used to illustrate simple linear and logistic regression. It contains the sepal length, petal length, and species label for _Iris versicolor_ and _Iris virginica_.

## Simple linear regression with the Iris dataset

* `iris_linreg_np.py` implements linear regression in NumPy with batch gradient descent, i.e., each gradient is computed on the entire dataset.

* `iris_linreg_tf.py` does the same with TensorFlow.

```
After 1000 epochs, loss = 0.11595218628644943
After 2000 epochs, loss = 0.11327244341373444
After 3000 epochs, loss = 0.11110898852348328
After 4000 epochs, loss = 0.10936232656240463
After 5000 epochs, loss = 0.10795214772224426
After 6000 epochs, loss = 0.10681366920471191
After 7000 epochs, loss = 0.10589452087879181
After 8000 epochs, loss = 0.10515245795249939
After 9000 epochs, loss = 0.10455334186553955
After 10000 epochs, loss = 0.10406967252492905
W = 0.952523
b = -1.04301
```

![](https://github.com/frsong/tf-examples/blob/master/figs/iris_linreg_tf.png)

## Simple binary logistic regression with the Iris dataset

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
Test accuracy = 0.9733999967575073
```

* `mnist_cnn_model.py`, `mnist_cnn_train.py`, and `mnist_cnn_test` do the same but demonstrate a more canonical way to organize code for such projects. They also demonstrate how to use variable scopes, saving and restoring models, and using TensorBoard to monitor training progress.

## Variational autoencoder for MNIST

* `vae.py` is a simplified version of the code from https://jmetzen.github.io/2015-11-27/vae.html.
