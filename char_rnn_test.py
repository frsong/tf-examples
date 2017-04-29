"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import os
import pickle
import tensorflow as tf

from char_rnn_model import Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_dir', 'save/char-rnn', "save directory")
tf.app.flags.DEFINE_string('start_text', "KING ", "start text")
tf.app.flags.DEFINE_integer('seed', 0, "random number generator seed")

def sample():
    filename = os.path.join(FLAGS.save_dir, 'chars_vocab.pkl')
    with open(filename, 'rb') as f:
        chars, vocab = pickle.load(f)
    vocab_size = len(chars)

    # Model
    model = Model(vocab_size)

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # Load model
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint available.")

        # Generate sample
        return model.sample(sess, chars, vocab, FLAGS.start_text, FLAGS.seed)

#///////////////////////////////////////////////////////////////////////////////

def main(_):
    print(sample())

if __name__ == '__main__':
    tf.app.run()
