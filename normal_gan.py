"""
Based on

    http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
    http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

Original paper:

    Generative adversarial networks.
    https://arxiv.org/abs/1406.2661

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#-------------------------------------------------------------------------------
# "Data"
#-------------------------------------------------------------------------------

class DataDistribution(object):
    def __init__(self, mu=1.0, sigma=0.5):
        self.mu    = mu
        self.sigma = sigma

    def sample(self, size, sort=False):
        x = np.random.normal(self.mu, self.sigma, size)
        if sort:
            x = np.sort(x)

        return x

data = DataDistribution()

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------

class NoiseDistribution(object):
    def __init__(self, bound):
        self.bound = bound

    def sample(self, size, sort=False):
        x = np.random.uniform(-self.bound, self.bound, size)
        if sort:
            x = np.sort(x)

        return x

def weight(shape):
    bound = np.sqrt(6.0 / np.sum(shape))
    init  = tf.random_uniform_initializer(-bound, bound)
    return tf.get_variable('W', shape, initializer=init)

def bias(shape):
    init = tf.constant_initializer(0.0)
    return tf.get_variable('b', shape, initializer=init)

def linear(scope, x, dim):
    with tf.variable_scope(scope):
        W = weight([x.get_shape()[-1].value, dim])
        b = bias([dim])
    return tf.matmul(x, W) + b

def generator(x, hidden_dim=4):
    x = linear('hidden', x, hidden_dim)
    x = tf.nn.softplus(x)
    x = linear('output', x, 1)

    return x

def discriminator(x, hidden_dim=8):
    x = linear('hidden_1', x, hidden_dim)
    x = tf.tanh(x)
    x = linear('hidden_2', x, hidden_dim)
    x = tf.tanh(x)
    x = linear('logits', x, 1)
    x = tf.sigmoid(x)

    return x

def get_train_op(loss, variables, initial_learning_rate,
                 decay=0.96, decay_steps=200):
    # Implement exponential decay of learning rate
    global_step   = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        decay_steps,
        decay,
        staircase=True
        )

    # Note that we restrict to subset of variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op  = optimizer.minimize(loss, global_step, variables)

    return train_op

# Seed the TF random number generator for reproducible initialization
tf.set_random_seed(0)

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

# Generator
with tf.variable_scope('G') as scope:
    z = tf.placeholder(tf.float32, [None, 1])
    G = generator(z)
    G_variables = get_variables(scope)

# Discriminator
with tf.variable_scope('D') as scope:
    x  = tf.placeholder(tf.float32, [None, 1])
    D1 = discriminator(x)
    D_variables = get_variables(scope)

    # Copy of the discriminator that receives generator samples
    scope.reuse_variables()
    D2 = discriminator(G)

#-------------------------------------------------------------------------------
# Train
#-------------------------------------------------------------------------------

# Hyperparameters
learning_rate = 0.005
num_steps     = 10000
batch_size    = 20
check_every   = 1000
noise_bound   = 16.0

# Losses
D_loss = -tf.reduce_mean(tf.log(D1) + tf.log(1 - D2))
G_loss = -tf.reduce_mean(tf.log(D2))

# Train ops
D_train_op = get_train_op(D_loss, D_variables, learning_rate)
G_train_op = get_train_op(G_loss, G_variables, learning_rate)

# Seed random number generator
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

# Noise distribution
noise = NoiseDistribution(noise_bound)

# Train
for step in range(num_steps):
    # Update discriminator
    batch_x   = data.sample(batch_size, sort=True)
    batch_z   = noise.sample(batch_size, sort=True)
    feed_dict = {x: batch_x.reshape((-1, 1)), z: batch_z.reshape((-1, 1))}
    _, current_D_loss = sess.run([D_train_op, D_loss], feed_dict)

    # Update generator
    batch_z   = noise.sample(batch_size)
    feed_dict = {z: batch_z.reshape((-1, 1))}
    _, current_G_loss = sess.run([G_train_op, G_loss], feed_dict)

    # Progress report
    if (step+1) % check_every == 0:
        print("After {} steps, discriminator loss = {}, generator loss = {}"
              .format(step+1, current_D_loss, current_G_loss))

def decision_boundary(x_):
    feed_dict = {x: x_.reshape((-1, 1))}
    return sess.run(D1, feed_dict)

def sample(size):
    sample_z  = noise.sample(size)
    feed_dict = {z: sample_z.reshape((-1, 1))}
    return sess.run(G, feed_dict)[:,0]

#-------------------------------------------------------------------------------
# Compare data and GAN distributions
#-------------------------------------------------------------------------------

# Number of samples
num_samples = 20000

# Specify the bins so the data and samples line up
bins = np.linspace(-noise.bound, noise.bound, 51)
bin_centers = (bins[:-1] + bins[1:])/2

# Data
x_data = data.sample(num_samples)
p_data, _ = np.histogram(x_data, bins=bins, density=True)

# Generated samples
x_gan = sample(num_samples)
p_gan, _ = np.histogram(x_gan, bins=bins, density=True)

# Plot distributions
plt.plot(bin_centers, p_data, 'b', label='Data distribution')
plt.plot(bin_centers, p_gan,  'r', label='GAN distribution')

# Decision boundary
x_dec = np.linspace(-noise.bound, noise.bound, 100)
y_dec = decision_boundary(x_dec)
plt.plot(x_dec, y_dec, color='orange', label='P(x from data)')

# Set limits
plt.xlim(-5, 5)
plt.ylim(0, 1)

# Legend
plt.legend(loc='upper left')

# Axis labels
plt.xlabel('x')
plt.ylabel('Probability density, probability')

# Save figure
plt.savefig('figs/normal_gan.png')
