# CMPE 442 Assignment 1 Question 1
# Yasemin Direk

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generates synthetic data

# FOR PART I
m = 10

# FOR PART II
# m = 100

X = np.random.rand(m,1)*2
y = np.sin(2*math.pi*X)+np.random.randn(m,1)

plt.scatter(X,y,s=10)

# FOR d = 0

polynomial_features= PolynomialFeatures(degree=0)
x_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

sorted_zip = sorted(zip(X,y_poly_pred))
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')

# FOR d = 1

polynomial_features1= PolynomialFeatures(degree=1)
x_poly1 = polynomial_features1.fit_transform(X)
model1 = LinearRegression()
model1.fit(x_poly1, y)
y_poly1_pred = model1.predict(x_poly1)

sorted_zip1 = sorted(zip(X,y_poly1_pred))
X, y_poly1_pred = zip(*sorted_zip1)
plt.plot(X, y_poly1_pred, color='g')

# FOR d = 3

polynomial_features3= PolynomialFeatures(degree=3)
x_poly3 = polynomial_features3.fit_transform(X)
model3 = LinearRegression()
model3.fit(x_poly3, y)
y_poly3_pred = model3.predict(x_poly3)

sorted_zip3 = sorted(zip(X,y_poly3_pred))
X, y_poly3_pred = zip(*sorted_zip3)
plt.plot(X, y_poly3_pred, color='r')

# FOR d = 9

polynomial_features9= PolynomialFeatures(degree=9)
x_poly9 = polynomial_features9.fit_transform(X)
model9 = LinearRegression()
model9.fit(x_poly9, y)
y_poly9_pred = model9.predict(x_poly9)

sorted_zip9 = sorted(zip(X,y_poly9_pred))
X, y_poly9_pred = zip(*sorted_zip9)
plt.plot(X, y_poly9_pred, color='b')

plt.show()

