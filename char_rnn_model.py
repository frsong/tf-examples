"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_layers', 2, "number of LSTM layers")
tf.app.flags.DEFINE_integer('rnn_size', 128, "LSTM size")

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
        Cell  = rnn.BasicLSTMCell
        cells = [Cell(FLAGS.rnn_size) for _ in range(FLAGS.num_layers)]
        self.cell = cell = rnn.MultiRNNCell(cells)

        # For feeding in data
        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets    = tf.placeholder(tf.int32, [batch_size, seq_length])

        # len(initial_state) = num_layers
        # state[i].c.shape = [batch_size, rnn_size]
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # Readout
        softmax_W = tf.get_variable('softmax_W', [FLAGS.rnn_size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])

        # Input embedding
        embedding = tf.get_variable('embedding', [vocab_size, FLAGS.rnn_size])

        # inputs.shape = [batch_size, seq_length, rnn_size]
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # inputs is list of seq_length x [batch_size, 1, rnn_size]
        inputs = tf.split(inputs, seq_length, 1)

        # inputs is list of seq_length x [batch_size, rnn_size]
        inputs = [tf.squeeze(i, [1]) for i in inputs]

        if training:
            predict_char = None
        else:
            def predict_char(prev, _):
                prev = tf.matmul(prev, softmax_W) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embedding, prev_symbol)

        # outputs is list of seq_length x [batch_size, rnn_size]
        outputs, last_state = legacy_seq2seq.rnn_decoder(
            inputs, self.initial_state, cell,
            loop_function=predict_char
            )

        # concat  -> [batch_size, seq_length x rnn_size]
        # reshape -> [batch_size * seq_length, rnn_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, FLAGS.rnn_size])

        self.logits = tf.matmul(output, softmax_W) + softmax_b
        self.probs  = tf.nn.softmax(self.logits)

        # loss.shape = [batch_size * seq_length]
        loss = legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones(batch_size * seq_length)]
            )
        self.loss = tf.reduce_mean(loss)
        self.final_state = last_state

        if not training:
            return

        #-----------------------------------------------------------------------
        # For training only
        #-----------------------------------------------------------------------

        self.lr = tf.Variable(FLAGS.learning_rate, trainable=False)
        trainables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, 5)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, trainables))

        # For TensorBoard
        tf.summary.scalar('loss', self.loss)

    def sample(self, sess, chars, vocab, start_text, num_chars=500):
        # len(state) = num_layers, state[i].c.shape = [1, rnn_size]
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in start_text[:-1]:
            x = vocab[char]
            feed_dict = {self.input_data: [[x]], self.initial_state: state}
            state = sess.run(self.final_state, feed_dict)

        text = start_text
        char = start_text[-1]
        for _ in range(num_chars):
            x = vocab[char]
            feed_dict = {self.input_data: [[x]], self.initial_state: state}
            probs, state = sess.run([self.probs, self.final_state], feed_dict)

            p = probs[0]
            sample = np.random.choice(len(p), p=p)

            pred = chars[sample]
            text += pred
            char = pred

        return text
