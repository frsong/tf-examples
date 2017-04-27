"""
Linear regression with gradient descent (NumPy).

"""
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('datasets/iris/iris.txt', skiprows=1)
x_data = data[:,0]
y_data = data[:,1]

#-------------------------------------------------------------------------------
# Fit
#-------------------------------------------------------------------------------

def predict(x, W, b):
    return W*x + b

def compute_loss(x, W, b, y):
    y_pred = predict(x, W, b)
    error  = y_pred - y

    return 0.5 * np.mean(error**2)

def compute_gradients(x, W, b, y):
    y_pred = predict(x, W, b)
    error  = y_pred - y

    dLdW = np.mean(error*x)
    dLdb = np.mean(error)

    return dLdW, dLdb

# Hyperparameters
learning_rate = 0.01
num_epochs    = 1000

# Starting values
W = 0.0
b = 0.0

# Minimize the loss function
for epoch in range(num_epochs):
    grads = compute_gradients(x_data, W, b, y_data)
    W -= learning_rate * grads[0]
    b -= learning_rate * grads[1]

    if (epoch+1) % 100 == 0:
        loss = compute_loss(x_data, W, b, y_data)
        print("After {} epochs, loss = {}".format(epoch+1, loss))

# Print the result
print("W =", W)
print("b =", b)

#-------------------------------------------------------------------------------
# Figure
#-------------------------------------------------------------------------------

# Plot the data
plt.plot(x_data, y_data, 'o', label='Data')

# Plot the fit
x_fit = np.linspace(x_data.min(), x_data.max())
y_fit = predict(x_fit, W, b)
plt.plot(x_fit, y_fit, label='Fit')

# Legend
plt.legend()

# Axis labels
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal legnth (cm)")

# Save figure
plt.savefig('figs/iris_linreg_np.png')
