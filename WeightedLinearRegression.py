# CMPE 442 Assignment 1 Question 3
# Yasemin Direk

import math
import numpy as np
import matplotlib.pyplot as plt

# Generates synthetic data

m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2 * math.pi * X) + np.random.randn(m, 1)

X_b = np.c_[np.ones((len(X), 1)), X]
y_pred = np.zeros(len(X_b))

def weighted_linear_regression(X, y, iteration_cnt, eta, x, tau):

    theta = np.random.randn(2, 1)
    row, col = np.shape(X)
    w = np.zeros((row, 1))

    for i in range(row):
        diff = x - X[i]
        w[i] = np.exp(diff.dot(diff.T) / (-2 * tau ** 2))

    for j in range(iteration_cnt):
        batch_gradient = (2 / m) * (w * X).T.dot(X.dot(theta) - y)
        theta = theta - eta * batch_gradient

    return theta

iteration_cnt = 100
eta = 0.4

# FOR PART I
tau = 0.1

# # FOR PART II
# tau = 0.001

# # FOR PART II
# tau = 0.01

# # FOR PART II
# tau = 0.3

# # FOR PART II
# tau = 1

# # FOR PART II
# tau = 10

# Calls weighted_linear_regression 100 times
for n in range(m):
    theta = weighted_linear_regression(X_b, y, iteration_cnt, eta, X_b[n], tau)
    y_pred[n] = theta[0] + theta[1] * X[n]

# Prints theta[0] and theta[1] for every eta
print("for tau=%.3f" % tau, "| theta 0 =", theta[0], "theta 1 =", theta[1])
plt.scatter(X,y,s=10)
sorted_zip = sorted(zip(X,y_pred))
X, y_pred = zip(*sorted_zip)
plt.plot(X, y_pred, color='m')
plt.show()

