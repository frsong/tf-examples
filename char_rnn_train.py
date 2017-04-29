"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

"""
import os
import pickle
import numpy as np
import tensorflow as tf

from char_rnn_model import Model
import char_rnn_reader as reader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'datasets/tinyshakespeare',
                           "data directory")
tf.app.flags.DEFINE_string('save_dir', 'save/char-rnn', "save directory")
tf.app.flags.DEFINE_string('log_dir', 'logs/char-rnn', "log directory")
tf.app.flags.DEFINE_integer('batch_size', 50, "batch size")
tf.app.flags.DEFINE_integer('num_epochs', 100, "number of epochs to train")
tf.app.flags.DEFINE_integer('seq_length', 50, "sequence length")

def train():
    # Load data
    data = reader.Reader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.seq_length)
    vocab_size = len(data.chars)

    # Setup directories
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    filename = os.path.join(FLAGS.save_dir, 'chars_vocab.pkl')
    with open(filename, 'wb') as f:
        pickle.dump([data.chars, data.vocab], f)

    # Model
    model = Model(vocab_size, training=True)

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    # Summary
    summary_op = tf.summary.merge_all()

    # Print list of variables
    print("")
    print("Variables")
    print("---------")
    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    num_params = 0
    for v in variables:
        num_params += np.prod(v.get_shape().as_list())
        print("{} {}".format(v.name, v.get_shape()))
    print("=> Total number of parameters = {}".format(num_params))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # For TensorBoard
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Minimize the loss function
        for epoch in range(FLAGS.num_epochs):
            state = sess.run(model.initial_state)
            current_loss = 0
            for b in range(data.num_batches):
                batch = data.next_batch()
                feed_dict = {model.inputs: batch[0], model.targets: batch[1]}
                for layer, (c, h) in enumerate(model.initial_state):
                    feed_dict[c] = state[layer].c
                    feed_dict[h] = state[layer].h
                fetches = [model.train_op, model.loss, model.final_state,
                           summary_op]
                _, loss, state, summary = sess.run(fetches, feed_dict)
                current_loss += loss

                # Add to summary
                writer.add_summary(summary, epoch*data.num_batches + b)

            # Progress report
            print("After {} epochs, loss = {}"
                  .format(epoch+1, current_loss/data.num_batches))

            # Save
            ckpt_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=epoch+1)

#///////////////////////////////////////////////////////////////////////////////

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
