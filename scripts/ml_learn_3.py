import numpy as np
import matplotlib.pyplot as plt

# Create a dataset for linear regression
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Computer the slope of the linear regression using Normal Equation.
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predict the new y values.
X_new = np.array([[0], [2], [3]])
X_new_b = np.c_[np.ones((3, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

# Plot the fitted linear regression.
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
