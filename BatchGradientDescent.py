# CMPE 442 Assignment 1 Question 2
# Yasemin Direk

import numpy as np
import matplotlib.pyplot as plt

# Generates synthetic data

m = 100
X = np.random.rand(m,1)
y = 100 + 3 * X + np.random.randn(m,1)

X_b = np.c_[np.ones((len(X),1)),X]

def linear_regression(X, y, iterNo , eta ):

    theta = np.random.randn(2,1)
    m = y.size
    mse = np.zeros(iterNo)

    for i in range(iterNo):
        batch_gradient = (2 / m * X_b.T.dot(X_b.dot(theta) - y))
        theta = theta - eta * batch_gradient
        predictions = X_b.dot(theta)
        for n in range(m):
            # cost function
            mse[i] = (1 / m) * np.sum(np.square((predictions[n] - y[n])))

    # print('MSE: ', mse)
    return theta, mse

iterNo = 1000
eta = 0.1
eta_list = [0.1, 0.001, 0.01, 0.5]

for eta in eta_list:
    theta, mse = linear_regression(X_b,y,iterNo,eta)
    # Prints theta[0] and theta[1] for every eta
    print("for eta=%.3f" % eta, "| theta 0 =",theta[0], "theta 1 =",theta[1])
    plt.scatter(X,y,s=10)
    plt.plot(X, np.dot(X_b, theta), color='m')
    plt.title("for eta = %.3f" % eta)

    # MSE-iterations graph for every eta
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("for eta = %.3f" % eta)
    ax.set_ylabel('MSE', rotation=0)
    ax.set_xlabel('Iterations')
    theta = np.random.randn(2, 1)
    _ = ax.plot(range(iterNo), mse, 'b.')

    plt.show()



