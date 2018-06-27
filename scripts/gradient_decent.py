import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]


def normal_equation():
    """normal_equation"""
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta_best)


def batch_gradient_descent():
    """batch gradient descent"""
    eta = 0.1  # Learning rate
    n_iteration = 1000
    m = 100
    theta = np.random.rand(2, 1)
    for iteration in range(n_iteration):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print(theta)


def stochastic_gradient_descent():
    """Stochastic gradient descent"""
    n_epochs = 50
    m = 100
    t0, t1 = 5, 50  # Learning schedule hyperparameters
    theta = np.random.randn(2, 1)  # Random initialization

    def learning_schedule(t):
        return t0 / (t + t1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
    print(theta)


if __name__ == '__main__':
    normal_equation()
    batch_gradient_descent()
    #stochastic_gradient_descent()
