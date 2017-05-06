# TensorFlow examples

It seemed to me while learning TensorFlow that many example codes are not as consistent or intuitive as they could be for beginners, often obscuring the shared structure of deep learning models. This is my attempt to remedy the situation. That said, I'm grateful to the authors of the original code for doing the hard part of getting things to work. I'm still learning, so please let me know if you have suggestions for making the code even clearer.

# Notes

This code has been tested with Python 3.5 and TensorFlow 1.0.1. Many examples generate a figure or two so take a look at `figs/` to see what to expect. Results should be consistent between runs on a single CPU, but not necessarily on a GPU. An "epoch" means one full presentation of the dataset.

## Datasets

* `datasets/iris/iris.txt`, a subset of the full [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris), is used to illustrate simple linear and logistic regression. It contains the sepal length, petal length, and species label for _Iris versicolor_ and _Iris virginica_.

* The MNIST handwritten digit dataset downloads and/or extracts itself as needed.

* The Tiny Shakespeare corpus, which is originally from https://github.com/karpathy/char-rnn and consists of a subset of Shakespeare's works, is used for the char-rnn example.

* `datasets/images/wsp_arch.png` is a black-and-white photograph of Washington Square Park in New York City used for the random filter example.

## Linear regression with the Iris dataset

* `iris_linreg_np.py` implements linear regression in NumPy with batch gradient descent, i.e., each gradient is computed on the entire dataset. It hardly needs to be said (but I'll say it anyway) that this is not a good way to do linear regression for real data.

* `iris_linreg_tf.py` does the same with TensorFlow.

<img src="https://github.com/frsong/tf-examples/blob/master/figs/iris_linreg_tf.png" width=400 />

* `visualize_graph.py` writes the computational graph for the linear regression loss to an event file so it can be visualized by running `tensorboard --logdir=logs/visualize`. Open the indicated IP address in a browser and click on the "Graphs" tab.

## Binary logistic regression with the Iris dataset

* `iris_logreg_np.py` implements logistic regression in NumPy with stochastic gradient descent, i.e., each gradient is computed on one random example from the dataset.

* `iris_logreg_tf.py` does the same with TensorFlow.

<img src="https://github.com/frsong/tf-examples/blob/master/figs/iris_logreg_tf.png" width=400 />

## Multiclass logistic regression with MNIST

* `mnist_logreg.py` implements logistic regression based on the code from https://www.tensorflow.org/get_started/mnist/beginners.

## Convolutional neural network with MNIST

* `filter.py` demonstrates the effect of a convolution with a random filter on a black-and-white image.

* `mnist_cnn.py` implements a CNN based on the code from https://www.tensorflow.org/get_started/mnist/pros. Includes name scopes and code for displaying a list of trainable variables and counting the number of parameters in the model, which I find useful to know.

* `mnist_cnn_model.py`, `mnist_cnn_train.py`, and `mnist_cnn_test` together do the same with dropout in the fully connected layer, and demonstrate a more canonical way of organizing the code so that the model can be simultaneously used for training and extracting predictions. Includes variable scopes, saving and restoring checkpoints, and using TensorBoard to monitor training progress (for which, run ``tensorboard --logdir=logs/mnist`` in another terminal). Run `python mnist_cnn_train.py` to train, and `python mnist_cnn_test.py` to extract a prediction.

```
$ python mnist_cnn_train.py

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
After 1 epochs, validation accuracy = 0.9574000239372253
After 2 epochs, validation accuracy = 0.9782000184059143
After 3 epochs, validation accuracy = 0.9818000197410583
After 4 epochs, validation accuracy = 0.9832000136375427
After 5 epochs, validation accuracy = 0.9887999892234802
After 6 epochs, validation accuracy = 0.9891999959945679
After 7 epochs, validation accuracy = 0.9890000224113464
After 8 epochs, validation accuracy = 0.989799976348877
After 9 epochs, validation accuracy = 0.9900000095367432
After 10 epochs, validation accuracy = 0.989799976348877
After 11 epochs, validation accuracy = 0.9909999966621399
After 12 epochs, validation accuracy = 0.991599977016449
After 13 epochs, validation accuracy = 0.9909999966621399
After 14 epochs, validation accuracy = 0.9911999702453613
After 15 epochs, validation accuracy = 0.9926000237464905
After 16 epochs, validation accuracy = 0.9922000169754028
After 17 epochs, validation accuracy = 0.9927999973297119
After 18 epochs, validation accuracy = 0.9922000169754028
After 19 epochs, validation accuracy = 0.9926000237464905
After 20 epochs, validation accuracy = 0.9937999844551086
Test accuracy = 0.9924
```

## Variational autoencoder (VAE) with MNIST

* `vae.py` implements a VAE based on the code from https://jmetzen.github.io/2015-11-27/vae.html. Note that the KL term is computed using Equation (10) in the original [paper](https://arxiv.org/abs/1312.6114).

```
$ python vae.py

Variables
---------
encoder/hidden_1/W:0 (784, 500)
encoder/hidden_1/b:0 (500,)
encoder/hidden_2/W:0 (500, 500)
encoder/hidden_2/b:0 (500,)
encoder/mean/W:0 (500, 2)
encoder/mean/b:0 (2,)
encoder/log_var/W:0 (500, 2)
encoder/log_var/b:0 (2,)
decoder/hidden_1/W:0 (2, 500)
decoder/hidden_1/b:0 (500,)
decoder/hidden_2/W:0 (500, 500)
decoder/hidden_2/b:0 (500,)
decoder/reconstruction/W:0 (500, 784)
decoder/reconstruction/b:0 (784,)
=> Total number of parameters = 1289788
After 5 epochs, loss = 155.88664106889203
After 10 epochs, loss = 148.58128617720172
After 15 epochs, loss = 145.61274827436966
After 20 epochs, loss = 143.976779535467
After 25 epochs, loss = 142.8190042669123
After 30 epochs, loss = 142.0120587019487
After 35 epochs, loss = 141.4040315662731
After 40 epochs, loss = 140.78155033458364
After 45 epochs, loss = 140.41188788674094
After 50 epochs, loss = 139.98751409357246
After 55 epochs, loss = 139.62178749778053
After 60 epochs, loss = 139.27646353981712
After 65 epochs, loss = 139.04612417047673
After 70 epochs, loss = 138.74444517655806
After 75 epochs, loss = 138.50106091586025
```

<img src="https://github.com/frsong/tf-examples/blob/master/figs/vae_embedding.png" width=400 /><img src="https://github.com/frsong/tf-examples/blob/master/figs/vae_samples.png" width=400 />

## char-rnn with Shakespeare

* `char_rnn_reader.py`, `char_rnn_model.py`, `char_rnn_train.py`, and `char_rnn_test.py` together implement an LSTM character-level language model based on https://github.com/sherjilozair/char-rnn-tensorflow. Includes RNN cells, `dynamic_rnn`, dropout for RNNs, gradient clipping, embeddings (and pinning to the CPU), and a demonstration of how to use the TensorFlow flag system for command-line arguments. The `Reader` class in `char_rnn_reader.py` is an interface to the data; each batch is a list of words (inputs) and the list of words that follow each of those words (targets), as you can see by running `python char_rnn_reader.py` on its own. Run `python char_rnn_train.py` to train, then try different start texts with `char_rnn_test.py` to see where the model takes you (but some samples are more plausible than others):

```
$ python char_rnn_test.py --start_text="The meaning of life is "
The meaning of life is service: and it
false ready to the liberal, in my meaning judgment
his resign unless sorrow from nothing:
All absent and you here to someware believe.

HENRY BOLINGBROKE:
See you besides, too worn; but a day will not.

KING EDWARD IV:
Richard, ho! Then all the troth,
When you kill'd, we still desire to England.

GLOUCESTER:
Are they better on him, make fear those that
did hear our old suitors for the sacrament,
Now I grief follow'd by the choices of marriage?

JULIET:
Go, spy up in reproved trai
```

## Generative adversarial network (GAN) for a normal distribution

* `normal_gan.py` implements a GAN based on the code from http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html and http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/. Some of the tricks mentioned in those blog posts didn't seem to be necessary for this simple example so are not included. Demonstrates the use of an exponentially decaying learning rate and optimizing with respect to subsets of the model parameters.

<img src="https://github.com/frsong/tf-examples/blob/master/figs/normal_gan.png" width=400 />

## Asynchronous advantage actor-critic (A3C) reinforcement learning for Atari

* `a3c_*.py` implement A3C for Atari games using the OpenAI Gym environment. It requires Gym, Universe, and OpenCV, which you can install, for example, by running

```
pip install "gym[atari]"
pip install universe
pip install opencv-python
```

For the most part, this is just a stripped-down version of the already excellent code at https://github.com/openai/universe-starter-agent. I hope it's a bit more readable and therefore easier to modify, but it's also not as general as the original code - for instance, it cannot play games over VNC and doesn't work for earlier versions of TensorFlow. Use `a3c_train.py` to train (note that this version uses nohup to launch processes by default, but as in the original code you can use `--mode=tmux` for tmux if you have it). `a3c_train.py` calls `a3c_worker.py` to launch a parameter server called ps and `num-workers` workers that experience the environment. Use TensorBoard to monitor progress during training and `a3c_test.py` at any time to generate a video of the agent playing. For the examples below only 2 workers were used but basically the more workers (with more cores) the better. The code was tuned for Pong so it learns Pong pretty easily; Breakout takes much longer.

**Pong:**
```
$ python a3c_train.py --env-id=PongDeterministic-v3 --num-workers=2 --log-dir=/tmp/pong
$ tensorboard --logdir=/tmp/pong
$ python a3c_test.py --env-id=PongDeterministic-v3 --log-dir=/tmp/pong --movie-path=movies/pong
```

<img src="https://github.com/frsong/tf-examples/blob/develop/images/pong_reward.png" width=350 /> <img src="https://github.com/frsong/tf-examples/blob/develop/images/pong.gif" />

**Breakout:**
```
$ python a3c_train.py --env-id=BreakoutDeterministic-v3 --num-workers=2 --log-dir=/tmp/breakout
$ tensorboard --logdir=/tmp/breakout
$ python a3c_test.py --env-id=BreakoutDeterministic-v3 --log-dir=/tmp/breakout --movie-path=movies/breakout
```

<img src="https://github.com/frsong/tf-examples/blob/develop/images/breakout_reward.png" width=350 /> <img src="https://github.com/frsong/tf-examples/blob/develop/images/breakout.gif" />

Note that the original mp4 files were converted to animated gifs so they could be included here.
