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

def train(args):
    # Load data
    data = reader.Reader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = len(data.chars)

    # Setup directories
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    filename = os.path.join(args.save_dir, 'config.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(args, f)

    filename = os.path.join(args.save_dir, 'chars_vocab.pkl')
    with open(filename, 'wb') as f:
        pickle.dump([data.chars, data.vocab], f)

    # Model
    model = Model(args, training=True)

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    # Summary
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # For TensorBoard
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

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

        for epoch in range(args.num_epochs):
            # Update learning rate for epoch
            update_lr = tf.assign(model.lr,
                                  args.learning_rate * args.decay_rate**epoch)
            sess.run(update_lr)

            data.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data.num_batches):
                batch = data.next_batch()
                feed_dict = {model.input_data: batch[0], model.targets: batch[1]}
                for i, (c, h) in enumerate(model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
                fetches = [model.train_op, model.loss, model.final_state,
                           summary_op]
                _, loss, state, summary = sess.run(fetches, feed_dict)

                # Add to summary
                writer.add_summary(summary, epoch*data.num_batches + b)

                # Progress report
                print("Epoch {}, {}/{}: loss = {}"
                      .format(epoch, b+1, data.num_batches, loss))

            # Save
            ckpt_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=epoch+1)

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    from argparse import Namespace as Parser

    args = Parser()

    args.num_layers    = 2
    args.rnn_size      = 128
    args.seq_length    = 50
    args.batch_size    = 50
    args.num_epochs    = 50
    args.learning_rate = 0.002
    args.decay_rate    = 0.97
    args.data_dir      = 'datasets/tinyshakespeare'
    args.save_dir      = 'save/char-rnn'
    args.log_dir       = 'logs/char-rnn'

    train(args)
