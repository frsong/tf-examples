"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import os
import pickle
import tensorflow as tf

from char_rnn_model import Model

def sample(save_dir, start_text):
    filename = os.path.join(save_dir, 'config.pkl')
    with open(filename, 'rb') as f:
        saved_args = pickle.load(f)

    filename = os.path.join(save_dir, 'chars_vocab.pkl')
    with open(filename, 'rb') as f:
        chars, vocab = pickle.load(f)

    # Model
    model = Model(saved_args)

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # Load model
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # Generate sample
        text = model.sample(sess, chars, vocab, start_text)
        print(text)

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    save_dir   = 'save/char-rnn'
    start_text = "Alas, "
    sample(save_dir, start_text)
