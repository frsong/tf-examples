"""
MNIST handwritten digit classification with a convolutional neural network.

"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_cnn_model import Model

def train(save_dir='save/mnist', log_dir='logs/mnist'):
    # Load data
    data = input_data.read_data_sets('datasets/mnist', one_hot=True)

    # Setup directories
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Model
    model = Model()

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    # Summary
    summary_op = tf.summary.merge_all()

    # Hyperparameters
    num_epochs = 1
    batch_size = 50

    # Seed the random number generator for reproducible batches
    np.random.seed(0)

    # Print list of variables
    print("")
    print("Variables")
    print("---------")
    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    num_params = 0
    for v in variables:
        num_params += np.prod(v.get_shape().as_list())
        print("{} {}".format(v.name, v.get_shape()))
    print("=> Total number of parameters =", num_params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # For TensorBoard
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # Minimize the loss function
        num_batches_per_epoch = data.train.num_examples // batch_size
        for epoch in range(num_epochs):
            # Present one mini-batch at a time
            for b in range(num_batches_per_epoch):
                batch = data.train.next_batch(batch_size)
                feed_dict = {model.x: batch[0],
                             model.y: batch[1]}#,
                             #model.keep_prob: 0.5}
                _, summary = sess.run([model.train_op, summary_op], feed_dict)

                # Add to summary
                writer.add_summary(summary, epoch*num_batches_per_epoch + b)

            # Progress report
            feed_dict = {model.x: data.validation.images,
                         model.y: data.validation.labels}#,
                         #model.keep_prob: 1.0}
            accuracy = sess.run(model.accuracy_op, feed_dict)
            print("After {} epochs, validation accuracy = {}"
                  .format(epoch+1, accuracy))

            # Save
            ckpt_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=epoch+1)

        # Test accuracy
        feed_dict = {model.x: data.test.images,
                     model.y: data.test.labels}#,
                     #model.keep_prob: 1.0}
        accuracy = sess.run(model.accuracy_op, feed_dict)
        print("Test accuracy = {}".format(accuracy))

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    train()
