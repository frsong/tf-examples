"""
Visualize a TensorFlow graph.

"""
import tensorflow as tf

# For feeding in data
x = tf.placeholder(tf.float32, [None], name='x')
y = tf.placeholder(tf.float32, [None], name='y')

# Model parameters
W = tf.Variable(0.0, name='W')
b = tf.Variable(0.0, name='b')

# Define the model
y_pred = tf.add(tf.multiply(W, x), b, name='y_pred')

# Loss function
half = tf.constant(0.5, name='half')
loss = tf.multiply(half, tf.reduce_mean(tf.square(y_pred - y)), name='loss')

# Write graph
writer = tf.summary.FileWriter('logs/visualize', tf.get_default_graph())
writer.close()
