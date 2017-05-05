"""
Logistic regression with stochastic gradient descent (NumPy).

"""
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('datasets/iris/iris.txt', skiprows=1)
X_data = data[:,:2]
Y_data = data[:,2]

#-------------------------------------------------------------------------------
# Fit
#-------------------------------------------------------------------------------

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def predict(x, W, b):
    logits = np.matmul(x, W) + b
    return 1 * (logits >= 0)

def compute_accuracy(x, W, b, y):
    labels = predict(x, W, b)
    return np.mean(labels == y)

def compute_gradients(x, W, b, y):
    logits = np.matmul(x, W) + b
    y_pred = sigmoid(logits)
    error  = y_pred - y

    dLdW  = np.mean(error * x.T, axis=1)
    dLdb  = np.mean(error)

    return dLdW, dLdb

# Hyperparameters
learning_rate = 0.01
num_epochs    = 100

# Starting values
W = np.zeros(2)
b = 0.0

# Seed the random number generator for reproducibility
np.random.seed(0)

# Minimize the loss function
for epoch in range(num_epochs):
    # Present each data point once in random order
    idx = np.random.permutation(data.shape[0])
    for i in idx:
        grads = compute_gradients(X_data[i:i+1], W, b, Y_data[i:i+1])
        W -= learning_rate * grads[0]
        b -= learning_rate * grads[1]

    # Progress report
    if (epoch+1) % 10 == 0:
        accuracy = compute_accuracy(X_data, W, b, Y_data)
        print("After {} epochs, accuracy = {}".format(epoch+1, accuracy))

# Print the result
print("W =", W)
print("b =", b)

#-------------------------------------------------------------------------------
# Figure
#-------------------------------------------------------------------------------

# Model predictions
labels = predict(X_data, W, b)

# Find indices for the two species
idx_0, = np.where(labels == 0)
idx_1, = np.where(labels == 1)

# Plot the data
plt.plot(X_data[idx_0,0], X_data[idx_0,1], 'bo', label='I. versicolor')
plt.plot(X_data[idx_1,0], X_data[idx_1,1], 'ro', label='I. virginica')

# Plot the separating hyperplane
x_sep = np.linspace(X_data[:,0].min(), X_data[:,0].max())
y_sep = (-b - W[0]*x_sep)/W[1]
plt.plot(x_sep, y_sep, 'm', label="Decision boundary")

# Legend
plt.legend()

# Axis labels
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal legnth (cm)")

# Save figure
plt.savefig('figs/iris_logreg_np.png')
