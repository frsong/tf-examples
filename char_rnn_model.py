"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_layers', 2, "number of LSTM layers")
tf.app.flags.DEFINE_integer('rnn_size', 128, "LSTM size")
tf.app.flags.DEFINE_float('learning_rate', 0.002, "learning rate")
tf.app.flags.DEFINE_float('keep_prob', 0.5, "dropout probability")

class Model(object):
    def __init__(self, vocab_size, training=False):
        if training:
            batch_size = FLAGS.batch_size
            seq_length = FLAGS.seq_length
        else:
            batch_size = 1
            seq_length = 1

        # Seed random number generator for reproducible initialization
        tf.set_random_seed(0)

        # For feeding in data
        self.inputs  = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        # Input embedding with dropout
        with tf.device('/cpu:0'):
            init = tf.random_uniform_initializer(-1, 1)
            embedding = tf.get_variable('embedding',
                                        [vocab_size, FLAGS.rnn_size],
                                        initializer=init)
            inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

        # Multilayer RNN with output dropout
        cells = [BasicLSTMCell(FLAGS.rnn_size) for _ in range(FLAGS.num_layers)]
        if training:
            cells = [DropoutWrapper(cell, output_keep_prob=FLAGS.keep_prob)
                     for cell in cells]
        self.cell = MultiRNNCell(cells)

        # len(initial_state) = num_layers
        # state[i].c.shape   = [batch_size, rnn_size]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state
            )

        # Reshape outputs to [batch_size * seq_length, rnn_size]
        outputs = tf.reshape(outputs, [-1, FLAGS.rnn_size])

        # Readout
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [FLAGS.rnn_size, vocab_size])
            b = tf.get_variable('b', [vocab_size])
        self.logits = tf.matmul(outputs, W) + b
        self.probs  = tf.nn.softmax(self.logits)

        # loss.shape = [batch_size * seq_length]
        loss = sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones(self.logits.get_shape()[0])]
            )
        self.loss = tf.reduce_mean(loss)

        if not training:
            return

        #-----------------------------------------------------------------------
        # For training only
        #-----------------------------------------------------------------------

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads, variables = zip(*optimizer.compute_gradients(self.loss))
        grads, _ = tf.clip_by_global_norm(grads, 5)
        self.train_op = optimizer.apply_gradients(zip(grads, variables))

        # For TensorBoard
        tf.summary.scalar('loss', self.loss)

    def sample(self, sess, chars, vocab, start_text, seed=0, num_chars=500):
        # Run the LSTM through the start text
        # len(state) = num_layers, state[i].c.shape = [1, rnn_size]
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in start_text[:-1]:
            x = vocab[char]
            feed_dict = {self.inputs: [[x]], self.initial_state: state}
            state = sess.run(self.final_state, feed_dict)

        # Get random number generator
        rng = np.random.RandomState(seed)

        # Generate new text
        text = start_text
        char = start_text[-1]
        for _ in range(num_chars):
            x = vocab[char]
            feed_dict = {self.inputs: [[x]], self.initial_state: state}
            probs, state = sess.run([self.probs, self.final_state], feed_dict)

            p = probs[0]
            char = chars[rng.choice(len(p), p=p)]
            text += char

        return text
