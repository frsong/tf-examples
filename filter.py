"""
Random filter example.

"""
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load original image
image = mpimg.imread('datasets/images/wsp_arch.png')

def weight(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W, name='W')

def apply_convolution(x):
    x = tf.reshape(x, [-1] + x.get_shape().as_list() + [1])
    W = weight([10, 10, 1, 1])
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return tf.squeeze(x, [0, -1])

# Seed random number generator for reproducible initialization
tf.set_random_seed(0)

x = tf.placeholder(tf.float32, image.shape)
y = apply_convolution(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv_image = sess.run(y, {x: image})

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original image")

# After convolution
plt.subplot(1, 2, 2)
plt.imshow(conv_image, cmap='gray')
plt.title("After convolution")

# Save figure
plt.savefig('figs/filter.png')
