"""
Asynchronous advantage actor-critic based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple

def normalized_columns_initializer(rng, stddev):
    if rng is None:
        rng = np.random.RandomState(0)
    def _initializer(shape, dtype=None, partition_info=None):
        out = rng.normal(size=shape).astype(np.float32)
        out *= stddev / np.sqrt(np.sum(np.square(out), axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv(name, x, num_filters, filter_size=[3, 3], stride=[1, 1],
         padding='SAME', collections=None):
    with tf.variable_scope(name):
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [x.get_shape()[-1].value, num_filters]

        # Get W
        fan_in  = np.prod(filter_shape[:2]) * filter_shape[2]
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        bound   = np.sqrt(6.0 / (fan_in + fan_out))
        init    = tf.random_uniform_initializer(-bound, bound)
        W       = tf.get_variable('W', filter_shape, initializer=init,
                                  collections=collections)

        # Get b
        init = tf.constant_initializer(0.0)
        b    = tf.get_variable('b', [num_filters], initializer=init,
                            collections=collections)

        return tf.nn.conv2d(x, W, stride_shape, padding) + b

def linear(name, x, size, rng, stddev=1.0):
    with tf.variable_scope(name):
        init = normalized_columns_initializer(rng, stddev)
        W    = tf.get_variable('W', [x.get_shape()[1], size], initializer=init)

        init = tf.constant_initializer(0.0)
        b    = tf.get_variable('b', [size], initializer=init)
    return tf.matmul(x, W) + b

def categorical_sample(logits, d):
    logits = logits - tf.reduce_max(logits, [1], keep_dims=True)
    action = tf.multinomial(logits, 1)
    action = tf.squeeze(action, [1])
    return tf.one_hot(action, d)

class Policy(object):
    def __init__(self, ob_space, ac_space, rng=None, rnn_size=256):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # Convolutional layers
        for i in range(4):
            x = conv('l{}'.format(i + 1), x, 32, [3, 3], [2, 2])
            x = tf.nn.elu(x)

        # x.shape -> [batch_size=1, time, seq_length, rnn_size]
        x = tf.reshape(x, [1, -1, np.prod(x.get_shape().as_list()[1:])])

        # LSTM
        lstm = BasicLSTMCell(rnn_size)

        # Initial state
        c_init = np.zeros((1, rnn_size), np.float32)
        h_init = np.zeros((1, rnn_size), np.float32)
        self.state_init = LSTMStateTuple(c_init, h_init)

        # For feeding in the last state
        c_in = tf.placeholder(tf.float32, [1, rnn_size])
        h_in = tf.placeholder(tf.float32, [1, rnn_size])
        self.state_in = LSTMStateTuple(c_in, h_in)

        # Run the RNN
        outputs, state = tf.nn.dynamic_rnn(lstm, x, initial_state=self.state_in)
        self.state_out = LSTMStateTuple(state.c[:1], state.h[:1])

        # x.shape -> [seq_length, rnn_size]
        x = tf.reshape(outputs, [-1, rnn_size])

        # Readouts
        self.logits = linear('action', x, ac_space, rng, stddev=0.01)
        self.action = categorical_sample(self.logits, ac_space)[0]
        self.vf     = tf.reshape(linear('value', x, 1, rng), [-1])

        print(x.shape, self.logits.shape, self.action.shape, self.vf.shape)

        # Variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           tf.get_variable_scope().name)

    def get_initial_state(self):
        return self.state_init

    def act(self, obs, state):
        sess      = tf.get_default_session()
        feed_dict = {self.x: [obs], self.state_in: state}
        return sess.run([self.action, self.vf, self.state_out], feed_dict)

    def value(self, obs, state):
        sess      = tf.get_default_session()
        feed_dict = {self.x: [obs], self.state_in: state}
        return sess.run(self.vf, feed_dict)[0]
