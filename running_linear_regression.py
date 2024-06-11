from math import sqrt
import numpy as np
import scipy.stats
from running_simple_stats import RunningSimpleStats
from class_is_fit import class_is_fit


class NotFittedError(Exception):
    '''
    Exception that is raised when trying a nonfit method
    before fitting.  
    '''

    def __str__(self):
        return "You need to call the `fit` method first."


class RunningLinearRegression(object):
    '''
    Follows the sklearn API for linear regression but
    adds support for efficient updating of the learned parameters
    given the addition of new datapoints using the method ``push``,
    removal of the oldest datapoint using the method ``pop``, or 
    combining both operations with the method ``pushpop`` (which
    has improved efficiency over a serial excution of ``push`` 
    and ``pop``).  

    Initialized with no arguments.  Prediction can not be
    done until the ``fit`` method has been called.  

    Instance attributes:
    ``beta0_hat_`` : intercept parameter.
    `beta1_hat_``  : slope parameter.
    ``rss_``       : running simple statistics using the 
                    ``RunningSimpleStats`` class.
    '''

    def fit(self, x_train, y_train):
        '''
        Fits the parameters using the naive closed formula for 
        the least squared coefficients.  Returns the fitted model.  
        '''
        self.rss_ = RunningSimpleStats(x_train, y_train)
        x_train = np.array(x_train).reshape(-1, 1)
        ones = np.full(shape=x_train.shape, fill_value=1.0)
        X = np.concatenate((ones, x_train), axis=1)
        self.beta0_hat_, self.beta1_hat_ = (
            np.linalg.inv(X.T @ X) @ X.T).dot(y_train)
        return self

    def _update_betas(self):
        '''
        Updates the regression parameters ``beta0_hat_`` and 
        `beta1_hat_`` given that the other statistics have already
        been updated.  Raises an exception if there are less than
        2 data points since you can not fit a line then.  
        '''
        if self.rss_.N < 2:
            raise Exception("too few points to fit")
        self.beta1_hat_ = self.rss_.xy_cov / self.rss_.x_var
        self.beta0_hat_ = self.rss_.ys.mean - self.beta1_hat_ * self.rss_.xs.mean

    def push(self, x, y):
        '''
        Adds the point with coordinates ``x`` and ``y`` to the 
        training data and updates the parameters accordingly.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        self.rss_.push(x, y)
        self._update_betas()

    def pop(self):
        '''
        Pops the first entry off of the training data and updates 
        the parameters accordingly.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        res = self.rss_.pop()
        self._update_betas()
        return res

    def pushpop(self, x, y):
        '''
        Adds the point with coordinates ``x`` and ``y`` to the 
        training data, pops the first entry off of the training data
        and updates the parameters accordingly.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        res = self.rss_.pushpop(x, y)
        self._update_betas()
        return res

    def predict(self, x_test):
        '''
        Returns the array of predictions using the leared parameters
        given an array of test values.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        return self.beta1_hat_ * np.array(x_test) + self.beta0_hat_

    def r(self):
        '''
        Computes the sample correlation between the independent 
        and dependent variables.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        return self.rss_.xy_cov / sqrt((self.rss_.x_var * self.rss_.y_var))

    def r2(self):
        '''
        Computes the coefficient of determination.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        return self.r() ** 2

    def t_score(self):
        '''
        Computes the t-score.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        return sqrt(self.rss_.N - 2) * self.r() / sqrt(1 - self.r2())

    def _ppf_beta1(N, alpha):
        '''
        Computes the upper percentile point function of the 
        t-distribution with N-2 degrees of freedom for alpha/2.
        Used for obtaining a confidence interval for the slope.  
        '''
        return -scipy.stats.t.ppf(alpha/2, df=N-2)

    def slope_confidence_interval(self, alpha):
        '''
        Returns the 100(1-alpha)% confidence interval for 
        the slope.  Returns the confidence interval as a 
        tuple.  
        '''
        if not class_is_fit(self):
            raise NotFittedError
        radius = (self.beta1_hat_ / self.t_score()) * \
            RunningLinearRegression._ppf_beta1(self.rss_.N, alpha)
        return (self.beta1_hat_ - radius, self.beta1_hat_ + radius)
