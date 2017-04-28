"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_layers', 2, "number of LSTM layers")
tf.app.flags.DEFINE_integer('rnn_size', 128, "LSTM size")
tf.app.flags.DEFINE_float('learning_rate', 0.002, "learning rate")

class Model(object):
    def __init__(self, vocab_size, training=False):
        if training:
            batch_size = FLAGS.batch_size
            seq_length = FLAGS.seq_length
        else:
            batch_size = 1
            seq_length = 1

        # Seed random number generators
        tf.set_random_seed(0)
        np.random.seed(0)

        # Multilayer RNN
        cells = [BasicLSTMCell(FLAGS.rnn_size) for _ in range(FLAGS.num_layers)]
        self.cell = MultiRNNCell(cells)

        # For feeding in data
        self.inputs  = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        # len(initial_state) = num_layers
        # state[i].c.shape   = [batch_size, rnn_size]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        # Input embedding
        embedding = tf.get_variable('embedding', [vocab_size, FLAGS.rnn_size])

        # inputs.shape = [batch_size, seq_length, rnn_size]
        inputs = tf.nn.embedding_lookup(embedding, self.inputs)

        # inputs is list of seq_length x [batch_size, 1, rnn_size]
        inputs = tf.split(inputs, seq_length, 1)

        # inputs is list of seq_length x [batch_size, rnn_size]
        inputs = [tf.squeeze(i, [1]) for i in inputs]

        # outputs is list of seq_length x [batch_size, rnn_size]
        outputs, self.final_state = legacy_seq2seq.rnn_decoder(
            inputs, self.initial_state, self.cell
            )

        # concat  -> [batch_size, seq_length x rnn_size]
        # reshape -> [batch_size * seq_length, rnn_size]
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, FLAGS.rnn_size])

        # Readout
        softmax_W = tf.get_variable('softmax_W', [FLAGS.rnn_size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])
        self.logits = tf.matmul(outputs, softmax_W) + softmax_b
        self.probs  = tf.nn.softmax(self.logits)

        # loss.shape = [batch_size * seq_length]
        loss = legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones(batch_size * seq_length)]
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

    def sample(self, sess, chars, vocab, start_text, num_chars=500):
        # Run the LSTM through the start text
        # len(state) = num_layers, state[i].c.shape = [1, rnn_size]
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in start_text[:-1]:
            x = vocab[char]
            feed_dict = {self.inputs: [[x]], self.initial_state: state}
            state = sess.run(self.final_state, feed_dict)

        # Generate new text
        text = start_text
        char = start_text[-1]
        for _ in range(num_chars):
            x = vocab[char]
            feed_dict = {self.inputs: [[x]], self.initial_state: state}
            probs, state = sess.run([self.probs, self.final_state], feed_dict)

            p = probs[0]
            char = chars[np.random.choice(len(p), p=p)]
            text += char

        return text
