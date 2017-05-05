"""
MNIST handwritten digit classification with a convolutional neural network,
based on

    https://www.tensorflow.org/get_started/mnist/pros

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the data
data = input_data.read_data_sets('datasets/mnist', one_hot=True)

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------

def weight(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W, name='W')

def bias(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b, name='b')

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def inference(x):
    # Reshape input
    x = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional + pooling layer
    with tf.name_scope('conv1'):
        W = weight([5, 5, 1, 32])
        b = bias([32])
    x = conv(x, W) + b
    x = tf.nn.relu(x)
    x = max_pool(x) # 14x14

    # Second convolutional + pooling layer
    with tf.name_scope('conv2'):
        W = weight([5, 5, 32, 64])
        b = bias([64])
    x = conv(x, W) + b
    x = tf.nn.relu(x)
    x = max_pool(x) # 7x7

    # Flatten feature planes
    x = tf.reshape(x, [-1, 7*7*64])

    # Fully connected layer
    with tf.name_scope('fc'):
        W = weight([7*7*64, 1024])
        b = bias([1024])
    x = tf.matmul(x, W) + b
    x = tf.nn.relu(x)

    # Softmax layer
    with tf.name_scope('softmax'):
        W = weight([1024, 10])
        b = bias([10])
    x = tf.matmul(x, W) + b

    return x

# Seed the TF random number generator for reproducible initialization
tf.set_random_seed(0)

# For feeding in data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Define the model
logits = inference(x)

# Prediction
predict_op  = tf.argmax(logits, 1)
correct_op  = tf.equal(predict_op, tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32))

#-------------------------------------------------------------------------------
# Train
#-------------------------------------------------------------------------------

# Loss function
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(loss)

# Hyperparameters
learning_rate = 1e-4
num_epochs    = 20
batch_size    = 50

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op  = optimizer.minimize(loss)

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
    print(v.name, v.get_shape())
print("=> Total number of parameters =", num_params)

# TF session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Minimize the loss function
num_batches_per_epoch = data.train.num_examples // batch_size
for epoch in range(num_epochs):
    # Present one mini-batch at a time
    for _ in range(num_batches_per_epoch):
        batch = data.train.next_batch(batch_size)
        feed_dict = {x: batch[0], y: batch[1]}
        sess.run(train_op, feed_dict)

    # Progress report
    feed_dict = {x: data.validation.images, y: data.validation.labels}
    accuracy = sess.run(accuracy_op, feed_dict)
    print("After {} epochs, validation accuracy = {}".format(epoch+1, accuracy))

# Test accuracy
feed_dict = {x: data.test.images, y: data.test.labels}
accuracy = sess.run(accuracy_op, feed_dict)
print("Test accuracy =", accuracy)

def predict(images):
    feed_dict = {x: images}
    return sess.run(predict_op, feed_dict)

#-------------------------------------------------------------------------------
# Use the model to make predictions
#-------------------------------------------------------------------------------

idx   = 0
image = data.test.images[idx]
label = data.test.labels[idx]

predicted_label = predict([image])[0]

# Plot image with label
plt.imshow(image.reshape((28, 28)), cmap='gray')
plt.title("True label: {}, predicted: {}"
          .format(label.argmax(), predicted_label))

# Save figure
plt.savefig('figs/mnist_cnn.png')
