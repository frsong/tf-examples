"""
MNIST handwritten digit classification with logistic regression based on

    https://www.tensorflow.org/get_started/mnist/beginners

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

# For feeding in data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Model parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the model
logits = tf.matmul(x, W) + b

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
learning_rate = 0.1
num_epochs    = 10
batch_size    = 100

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op  = optimizer.minimize(loss)

# Seed the random number generator for reproducible batches
np.random.seed(0)

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
plt.savefig('figs/mnist_logreg.png')
