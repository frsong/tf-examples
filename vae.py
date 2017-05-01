"""
Variational autoencoder for MNIST based on

    https://jmetzen.github.io/2015-11-27/vae.html

Original paper:

    Auto-encoding variational Bayes.
    https://arxiv.org/abs/1312.6114

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

# Latent space dimension
latent_dim = 2

def weight(shape):
    bound = np.sqrt(6.0 / np.sum(shape))
    W = tf.random_uniform(shape, -bound, bound)
    return tf.Variable(W, name='W')

def bias(shape):
    b = tf.constant(0.0, shape=shape)
    return tf.Variable(b, name='b')

def linear(scope, x, dim):
    with tf.name_scope(scope):
        W = weight([x.get_shape()[-1].value, dim])
        b = bias([dim])
    return tf.matmul(x, W) + b

def encoder(x, hidden_dim=500):
    with tf.name_scope('encoder'):
        x = linear('hidden_1', x, hidden_dim)
        x = tf.nn.softplus(x)
        x = linear('hidden_2', x, hidden_dim)
        x = tf.nn.softplus(x)
        z_mean    = linear('mean', x, latent_dim)
        z_log_var = linear('log_var', x, latent_dim)

    return z_mean, z_log_var

def decoder(x, hidden_dim=500):
    with tf.name_scope('decoder'):
        x = linear('hidden_1', x, hidden_dim)
        x = tf.nn.softplus(x)
        x = linear('hidden_2', x, hidden_dim)
        x = tf.nn.softplus(x)
        x = linear('reconstruction', x, 784)

    return x

# Seed the TF random number generator for reproducible initialization
tf.set_random_seed(0)

# For feeding in data
x = tf.placeholder(tf.float32, [None, 784])

# Model
z_mean, z_log_var = encoder(x)
epsilon = tf.random_normal(tf.shape(z_log_var))
z = z_mean + epsilon * tf.exp(0.5*z_log_var) # "Reparametrization trick"
logits = decoder(z)
x_reconstruction = tf.nn.sigmoid(logits)

# Reconstruction loss
CE = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x)
CE = tf.reduce_sum(CE, 1)

# Latent loss
KL = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
KL = -0.5 * tf.reduce_sum(KL, 1)

# Total loss
loss = tf.reduce_mean(CE + KL)

# Hyperparameters
learning_rate = 0.001
num_epochs    = 75
batch_size    = 100

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
print("=> Total number of parameters = {}".format(num_params))

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

    if (epoch+1) % 5 == 0:
        print("After {} epochs, loss = {}"
              .format(epoch+1, current_loss/num_batches_per_epoch))

def encode(x_):
    return sess.run(z_mean, {x: x_})

def generate(z_):
    return sess.run(x_reconstruction, {z: z_})

def reconstruct(x_):
    return sess.run(x_reconstruction, {x: x_})

#-------------------------------------------------------------------------------
# Example reconstructions
#-------------------------------------------------------------------------------

nx = ny = 10
images = data.test.images[:nx*ny]
reconstructed_images = reconstruct(images)

grid = np.zeros((28*ny, 28*nx))
reconstructed_grid = np.zeros((28*ny, 28*nx))
for i in range(ny):
    for j in range(nx):
        grid[28*(ny-i-1):28*(ny-i),28*j:28*(j+1)] = (
            images[i*ny+j].reshape((28, 28))
            )
        reconstructed_grid[28*(ny-i-1):28*(ny-i),28*j:28*(j+1)] = (
            reconstructed_images[i*ny+j].reshape((28, 28))
            )

plt.figure()

# Original images
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(grid, cmap='gray')

# Reconstructed images
plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstructed_grid, cmap='gray')

# Save figure
plt.savefig('figs/vae_reconstructions.png')

#-------------------------------------------------------------------------------
# Plot latent embedding
#-------------------------------------------------------------------------------

if latent_dim == 2:
    images = data.test.images
    labels = data.test.labels
    z_mu   = encode(images)

    plt.figure()
    plt.scatter(z_mu[:,0], z_mu[:,1], c=np.argmax(labels, 1))
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.colorbar()
    plt.savefig('figs/vae_embedding.png')

#-------------------------------------------------------------------------------
# Plot samples at corresponding latent space positions
#-------------------------------------------------------------------------------

if latent_dim == 2:
    nx = ny = 20
    xs = np.linspace(-3, 3, nx)
    ys = np.linspace(-3, 3, ny)

    grid = np.zeros((28*ny, 28*nx))
    for i, y_i in enumerate(ys):
        for j, x_j in enumerate(xs):
            latent = [[x_j, y_i]]
            image  = generate(latent)[0]
            grid[28*(ny-i-1):28*(ny-i),28*j:28*(j+1)] = image.reshape((28, 28))

    plt.figure()
    plt.imshow(grid, cmap='gray', extent=[-3, 3, -3, 3])
    plt.savefig('figs/vae_samples.png')
