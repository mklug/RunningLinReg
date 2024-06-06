import numpy as np


class LinearRegression(object):

    def __init__(self):
        self.beta0_hat = None
        self.beta1_hat = None
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        x_train = x_train.reshape(-1, 1)
        ones = np.full(shape=x_train.shape, fill_value=1.0)
        X = np.concatenate((ones, x_train), axis=1)
        self.beta0_hat, self.beta1_hat = (
            np.linalg.inv(X.T @ X) @ X.T).dot(y_train)
        return self

    def _gd(x_train, y_train, eta=0.001, epochs=1000, regularization=None, lmbda=0.0):
        N = len(x_train)
        beta0, beta1 = 0.0, 0.0

        def r(x): return 0
        if regularization == "ridge":
            def r(x): return 2*lmbda*x
        elif regularization == "lasso":
            def r(x): return 2*lmbda

        for _ in range(epochs):
            # grad0 = 1/N * -2 * np.sum(y_train - beta1*x_train - beta0)
            # grad1 = 1/N * -2 * x_train.dot(y_train - beta1*x_train - beta0)
            grad0 = -2 * np.sum(y_train - beta1*x_train - beta0) + r(beta0)
            grad1 = -2 * x_train.dot(y_train - beta1 *
                                     x_train - beta0) + r(beta1)
            beta0 = beta0 - eta*grad0
            beta1 = beta1 - eta*grad1
        return beta0, beta1

    def gd_fit(self, x_train, y_train, eta=0.001, epochs=1000):
        self.x_train, self.y_train = x_train, y_train
        self.beta0_hat, self.beta1_hat = LinearRegression._gd(x_train, y_train,
                                                              eta, epochs)

    def ridge_fit(self, x_train, y_train, eta=0.001, epochs=1000, lmbda=1.0):
        self.x_train, self.y_train = x_train, y_train
        self.beta0_hat, self.beta1_hat = LinearRegression._gd(x_train, y_train,
                                                              eta, epochs,
                                                              regularization="ridge",
                                                              lmbda=lmbda)

    def lasso_fit(self, x_train, y_train, eta=0.001, epochs=1000, lmbda=1.0):
        self.x_train, self.y_train = x_train, y_train
        self.beta0_hat, self.beta1_hat = LinearRegression._gd(x_train, y_train,
                                                              eta, epochs,
                                                              regularization="lasso",
                                                              lmbda=lmbda)

    def predict(self, x_test):
        return self.beta1_hat * np.array(x_test) + self.beta0_hat

    def r2(self):
        y_bar = np.mean(self.y_train)
        y_hat = LinearRegression.predict(self, self.x_train)
        ss_res = np.sum((np.array(self.y_train) - y_hat)**2)
        ss_tot = np.sum((np.array(self.y_train) - y_bar)**2)
        return 1 - (ss_res / ss_tot)
