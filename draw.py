"""
Implementation of deep recurrent attentive writer (DRAW), based on

    https://github.com/ericjang/draw

Original paper:

    DRAW: A recurrent neural network for image generation.
    https://arxiv.org/abs/1502.04623

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn import LSTMCell

# Load the data
data = input_data.read_data_sets('datasets/mnist', one_hot=True)
IMAGE_WIDTH  = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT

# Hyperparameters
encoder_size  = 256
decoder_size  = 256
T             = 10
batch_size    = 100
learning_rate = 1e-3
num_epochs    = 20
latent_dim    = 10
eps           = 1e-8

# Attention parameters
attention = True
read_n    = 5
write_n   = 5

def linear(x, dim):
    W = tf.get_variable('W', [x.get_shape()[-1].value, dim])
    b = tf.get_variable('b', [dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b

# Seed the TF random number generator for reproducible initialization
tf.set_random_seed(0)

# For feeding in data
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

# VAE
encoder = LSTMCell(encoder_size)
decoder = LSTMCell(decoder_size)

def filter_bank(gx, gy, var, delta, n):
    grid = tf.reshape(tf.cast(tf.range(n), dtype=tf.float32), [1, -1])
    mu_x = tf.reshape(gx + (grid - n/2 - 0.5) * delta, [-1, n, 1]) # Eq. 19
    mu_y = tf.reshape(gy + (grid - n/2 - 0.5) * delta, [-1, n, 1]) # Eq. 20
    a    = tf.reshape(tf.cast(tf.range(IMAGE_WIDTH),  tf.float32), [1, 1, -1])
    b    = tf.reshape(tf.cast(tf.range(IMAGE_HEIGHT), tf.float32), [1, 1, -1])
    var  = tf.reshape(var, [-1, 1, 1])

    # Eq. 25
    Fx    = tf.exp(-tf.square((a - mu_x) / (2*var)))
    norm  = tf.reduce_sum(Fx, 2, keep_dims=True)
    Fx   /= tf.maximum(norm, eps)

    # Eq. 26
    Fy    = tf.exp(-tf.square((b - mu_y) / (2*var)))
    norm  = tf.reduce_sum(Fy, 2, keep_dims=True)
    Fy   /= tf.maximum(norm, eps)

    return Fx, Fy

def attention_window(scope, reuse, decoder_output, n):
    with tf.variable_scope(scope, reuse=reuse):
        params = linear(decoder_output, 5) # Eq. 21
    gx, gy, log_var, log_delta, log_gamma = tf.split(params, 5, axis=1)
    var   = tf.exp(log_var)
    delta = tf.exp(log_delta)
    gamma = tf.exp(log_gamma)

    gx     = (gx + 1) * (IMAGE_WIDTH + 1)/2                       # Eq. 22
    gy     = (gy + 1) * (IMAGE_HEIGHT + 1)/2                      # Eq. 23
    delta  = (max(IMAGE_WIDTH, IMAGE_HEIGHT) - 1) / (n-1) * delta # Eq. 24
    Fx, Fy = filter_bank(gx, gy, var, delta, n)

    return Fx, Fy, gamma

# Eq. 27
def apply_filter(x, Fx, Fy, gamma, n):
    Fx_t = tf.transpose(Fx, perm=[0, 2, 1])
    x = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH])
    x = tf.matmul(Fy, tf.matmul(x, Fx_t))
    return tf.reshape(gamma, [-1, 1]) * tf.reshape(x, [-1, n*n])

# Eq. 29
def apply_filter_rev(x, Fx, Fy, gamma, n):
    Fy_t = tf.transpose(Fy, perm=[0, 2, 1])
    x = tf.reshape(x, [-1, n, n])
    x = tf.matmul(Fy_t, tf.matmul(x, Fx))
    return tf.reshape(1/gamma, [-1, 1]) * tf.reshape(x, [-1, IMAGE_SIZE])

def read(x, x_error, decoder_output, reuse):
    if attention:
        Fx, Fy, gamma = attention_window('read', reuse, decoder_output, read_n)
        x       = apply_filter(x, Fx, Fy, gamma, read_n)
        x_error = apply_filter(x_error, Fx, Fy, gamma, read_n)
    return tf.concat([x, x_error], 1)

def write(decoder_output, reuse):
    if not attention:
        return linear(decoder_output, IMAGE_SIZE)

    with tf.variable_scope('patch', reuse=reuse):
        w = linear(decoder_output, write_n**2)
        w = tf.reshape(w, [batch_size, write_n, write_n])
    Fx, Fy, gamma = attention_window('write', reuse, decoder_output, write_n)
    return apply_filter_rev(w, Fx, Fy, gamma, write_n)

canvas          = tf.zeros_like(x)
reconstruction  = tf.zeros_like(x)
decoder_output  = tf.zeros((batch_size, decoder_size))
encoder_state   = encoder.zero_state(batch_size, tf.float32)
decoder_state   = decoder.zero_state(batch_size, tf.float32)
canvases        = []
reconstructions = []
z_means         = []
z_log_vars      = []
for t in range(T):
    if t == 0:
        reuse = None
    else:
        reuse = True

    # Encoding step
    with tf.variable_scope('encoder', reuse=reuse):
        # The encoder observes previous output
        x_error = x - reconstruction
        r = read(x, x_error, decoder_output, reuse)
        encoder_inputs = tf.concat([r, decoder_output], 1)

        encoder_output, encoder_state = encoder(encoder_inputs, encoder_state)
        with tf.variable_scope('mean'):
            z_mean = linear(encoder_output, latent_dim)
        with tf.variable_scope('var'):
            z_log_var = linear(encoder_output, latent_dim)
        z_means.append(z_mean)
        z_log_vars.append(z_log_var)

    # Latent embedding
    epsilon = tf.random_normal(tf.shape(z_log_var))
    z = z_mean + epsilon * tf.exp(0.5*z_log_var)

    # Decoding step
    with tf.variable_scope('decoder', reuse=reuse):
        decoder_output, decoder_state = decoder(z, decoder_state)
        with tf.variable_scope('reconstruction'):
            # Accumulate the modifications
            canvas = canvas + write(decoder_output, reuse=reuse)
    canvases.append(canvas)

    # Previous reconstruction
    reconstruction = tf.sigmoid(canvas)
    reconstructions.append(reconstruction)

# Reconstruction loss at final time step
CE = tf.nn.sigmoid_cross_entropy_with_logits(logits=canvases[-1], labels=x)
CE = tf.reduce_sum(CE, 1)

# Latent loss for each time step
KLs = []
for z_mean, z_log_var in zip(z_means, z_log_vars):
    KL = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    KL = -0.5 * tf.reduce_sum(KL, 1)
    KLs.append(KL)
KL = tf.add_n(KLs)

# Total loss
loss = tf.reduce_mean(CE + KL)

# Optimizer
optimizer  = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads_vars = optimizer.compute_gradients(loss)
grads_vars = [(tf.clip_by_norm(g, 5), v) for g, v in grads_vars]
train_op   = optimizer.apply_gradients(grads_vars)

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
    current_loss = 0
    for _ in range(num_batches_per_epoch):
        batch_x, _ = data.train.next_batch(batch_size)
        _, loss_val = sess.run([train_op, loss], {x: batch_x})
        current_loss += loss_val

    print("After {} epochs, loss = {}"
          .format(epoch+1, current_loss/num_batches_per_epoch))

def reconstruct(x_):
    return sess.run(reconstructions, {x: x_})

#-------------------------------------------------------------------------------
# Example reconstructions
#-------------------------------------------------------------------------------

nx = ny = 10
images = data.test.images[:nx*ny]
reconstructed_images = reconstruct(images)

for t, reconstructed_image in enumerate(reconstructed_images):
    grid = np.zeros((28*ny, 28*nx))
    for i in range(ny):
        for j in range(nx):
            grid[28*(ny-i-1):28*(ny-i),28*j:28*(j+1)] = (
                reconstructed_image[i*ny+j].reshape((28, 28))
                )

    plt.figure()
    plt.imshow(grid, cmap='gray')
    plt.savefig('figs/draw/reconstruction_t{}.png'.format(t))
    plt.close()
