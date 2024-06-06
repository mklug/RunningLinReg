from collections import deque
from math import sqrt
import numpy as np
from running_stats import RunningMean, RunningVariance, RunningCovariance


class RunningLinearRegression(object):

    def __init__(self):
        self.beta0_hat = None
        self.beta1_hat = None
        self.x_train = None
        self.y_train = None

        self._x_mean = None
        self._y_mean = None
        self._x_var = None
        self._y_var = None
        self._xy_cov = None

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = deque(x_train), deque(y_train)
        self._x_mean = RunningMean(x_train)
        self._y_mean = RunningMean(y_train)
        self._x_var = RunningVariance(x_train)
        self._y_var = RunningVariance(y_train)
        self._xy_cov = RunningCovariance(x_train, y_train)

        x_train = x_train.reshape(-1, 1)
        ones = np.full(shape=x_train.shape, fill_value=1.0)
        X = np.concatenate((ones, x_train), axis=1)
        self.beta0_hat, self.beta1_hat = (
            np.linalg.inv(X.T @ X) @ X.T).dot(y_train)
        return self

    def _update_betas(self):
        if len(self.x_train) < 2:
            self.beta0_hat = None
            self.beta1_hat = None
        self.beta1_hat = self._xy_cov.covariance / self._x_var.variance
        self.beta0_hat = self._y_mean.mean - self.beta1_hat * self._x_mean.mean

    def push(self, x, y):
        self.x_train.append(x)
        self.y_train.append(y)
        self._x_mean.push(x)
        self._y_mean.push(y)
        self._x_var.push(x)
        self._y_var.push(y)
        self._xy_cov.push(x, y)
        RunningLinearRegression._update_betas(self)

    def pop(self):
        self.x_train.popleft()
        self.y_train.popleft()
        self._x_mean.pop()
        self._y_mean.pop()
        self._x_var.pop()
        self._y_var.pop()
        self._xy_cov.pop()
        RunningLinearRegression._update_betas(self)

    def pushpop(self, x, y):
        self.x_train.append(x)
        self.y_train.append(y)
        self.x_train.popleft()
        self.y_train.popleft()
        self._x_mean.pushpop(x)
        self._y_mean.pushpop(y)
        self._x_var.pushpop(x)
        self._y_var.pushpop(y)
        self._xy_cov.pushpop(x, y)
        RunningLinearRegression._update_betas(self)

    def predict(self, x_test):
        return self.beta1_hat * np.array(x_test) + self.beta0_hat

    def r2(self):
        return self._xy_cov.covariance / (sqrt(self._x_var.variance * self._y_var.variance))
