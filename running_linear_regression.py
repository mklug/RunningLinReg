from math import sqrt
import numpy as np
from running_simple_stats import RunningSimpleStats


class RunningLinearRegression(object):
    '''
    Follows the sklearn API for linear regression but
    adds support for efficient updating of the learned parameters
    given the addition of new datapoints using the method ``push``,
    removal of the oldest datapoint using the method ``pop``, or 
    combining both operations with the method ``pushpop`` (which
    has improved efficiency over a serial excution of ``push`` 
    and ``pop``).   
    '''

    def __init__(self):
        '''
        Initialized with no arguments.  Prediction can not be
        done until the ``fit`` method has been called.  

        Instance attributes:
        ``beta0_hat`` : intercept parameter.
        ``beta1_hat`` : slope parameter.
        ``_rss``      : running simple statistics using the 
                        ``RunningSimpleStats`` class.
        '''
        self.beta0_hat = None
        self.beta1_hat = None
        self._rss = None

    def fit(self, x_train, y_train):
        '''
        Fits the parameters using the naive closed formula for 
        the least squared coefficients.  Returns the fitted model.  
        '''
        self._rss = RunningSimpleStats(x_train, y_train)
        x_train = np.array(x_train).reshape(-1, 1)
        ones = np.full(shape=x_train.shape, fill_value=1.0)
        X = np.concatenate((ones, x_train), axis=1)
        self.beta0_hat, self.beta1_hat = (
            np.linalg.inv(X.T @ X) @ X.T).dot(y_train)
        return self

    def _update_betas(self):
        '''
        Updates the regression parameters ``beta0_hat`` and 
        ``beta1_hat`` given that the other statistics have already
        been updated.  Raises an exception if there are less than
        2 data points since you can not fit a line then.  
        '''
        if self._rss.N < 2:
            raise Exception("too few points to fit")
        self.beta1_hat = self._rss.xy_cov / self._rss.x_var
        self.beta0_hat = self._rss.ys.mean - self.beta1_hat * self._rss.xs.mean

    def push(self, x, y):
        '''
        Adds the point with coordinates ``x`` and ``y`` to the 
        training data and updates the parameters accordingly.  
        '''
        self._rss.push(x, y)
        self._update_betas()

    def pop(self):
        '''
        Pops the first entry off of the training data and updates 
        the parameters accordingly.  
        '''
        res = self._rss.pop()
        self._update_betas()
        return res

    def pushpop(self, x, y):
        '''
        Adds the point with coordinates ``x`` and ``y`` to the 
        training data, pops the first entry off of the training data
        and updates the parameters accordingly.  
        '''
        res = self._rss.pushpop(x, y)
        self._update_betas()
        return res

    def predict(self, x_test):
        '''
        Returns the array of predictions using the leared parameters
        given an array of test values.  
        '''
        return self.beta1_hat * np.array(x_test) + self.beta0_hat

    def r(self):
        '''
        Computes the sample correlation between the independent 
        and dependent variables.  
        '''
        return self._rss.xy_cov / sqrt((self._rss.x_var * self._rss.y_var))

    def r2(self):
        '''
        Computes the coefficient of determination.  
        '''
        return self.r() ** 2

    def t_score(self):
        '''
        Computes the t-score.  
        '''
        return sqrt(self._rss.N - 2) * self.r() / sqrt(1 - self.r2())
