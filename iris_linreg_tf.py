"""
Linear regression with gradient descent (TensorFlow).

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the data
data = np.loadtxt('datasets/iris/iris.txt', skiprows=1)
x_data = data[:,0]
y_data = data[:,1]

#-------------------------------------------------------------------------------
# Fit
#-------------------------------------------------------------------------------

# For feeding in data
x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])

# Model parameters
W = tf.Variable(0.0)
b = tf.Variable(0.0)

# Define the model
y_pred = W*x + b

# Loss function
loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y))

# Hyperparameters
learning_rate = 0.01
num_epochs    = 10000

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op  = optimizer.minimize(loss)

# TF session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Minimize the loss function
feed_dict = {x: x_data, y: y_data}
for epoch in range(num_epochs):
    sess.run(train_op, feed_dict)

    if (epoch+1) % 1000 == 0:
        current_loss = sess.run(loss, feed_dict)
        print("After {} epochs, loss = {}".format(epoch+1, current_loss))

# Print the result
W_val, b_val = sess.run([W, b])
print("W =", W_val)
print("b =", b_val)

def predict(x_):
    feed_dict = {x: x_}
    return sess.run(y_pred, feed_dict)

#-------------------------------------------------------------------------------
# Figure
#-------------------------------------------------------------------------------

# Plot the data
plt.plot(x_data, y_data, 'o', label='Data')

# Plot the fit
x_fit = np.linspace(x_data.min(), x_data.max())
y_fit = predict(x_fit)
plt.plot(x_fit, y_fit, label='Fit')

# Legend
plt.legend()

# Axis labels
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal legnth (cm)")

# Save figure
plt.savefig('figs/iris_linreg_tf.png')
