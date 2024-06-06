from collections import deque
import numpy as np
# Convention: mean of the empty list is None.


class RunningMean(object):

    def __init__(self, xs=None):
        self.mean = None if xs is None else np.mean(xs)
        self.q = None if xs is None else deque(xs)

    def push(self, x):
        N = len(self.q)
        self.q.append(x)
        self.mean = 1 / (N + 1) * (N * self.mean + x)

    def pop(self):
        N = len(self.q)
        if N == 0:
            print("Popped from an empty list.")
            self.mean = None
            return
        old = self.q.popleft()
        if N > 1:
            self.mean = N / (N - 1) * (self.mean - old / N)
        else:
            self.mean = None
        return old

    def pushpop(self, x):
        N = len(self.q)
        if N == 0:
            print("Pushpopped from an empty list.")
            return
        old = self.q.popleft()
        self.q.append(x)
        self.mean = self.mean + (x - old) / N
        return old

# Using the naive one-pass textbook formula.

# Convention: variance of the empty list is None.


class RunningVariance(object):

    def __init__(self, nums=None):
        self._x_rm = RunningMean(nums)
        self._x2_rm = RunningMean([x**2 for x in nums])
        self.variance = None if nums is None else self._x2_rm.mean - self._x_rm.mean**2

    def push(self, x):
        self._x_rm.push(x)
        self._x2_rm.push(x**2)
        self.variance = self._x2_rm.mean - self._x_rm.mean**2

    def pop(self):
        N = len(self._x_rm.q)
        if N == 0:
            print("Popped from an empty list.")
            self.variance = None
            return
        old = self._x_rm.pop()
        self._x2_rm.pop()
        if N > 1:
            self.variance = self._x2_rm.mean - self._x_rm.mean**2
        else:
            self.variance = None
        return old

    def pushpop(self, x):
        N = len(self._x_rm.q)
        if N == 0:
            print("Pushpopped from an empty list.")
            return
        old = self._x_rm.pushpop(x)
        self._x2_rm.pushpop(x**2)
        self.variance = self._x2_rm.mean - self._x_rm.mean**2
        return old


# Using the naive one-pass textbook formula.

# Convention: covariance of the empty list is None.
class RunningCovariance(object):

    def __init__(self, xs=None, ys=None):
        if xs is not None or ys is not None:
            assert len(xs) == len(ys)
        self._x_rm = RunningMean(xs)
        self._y_rm = RunningMean(ys)
        self._xy_rm = RunningMean([x*y for x, y in zip(xs, ys)])

        self.covariance = None if xs is None else self._xy_rm.mean - \
            self._x_rm.mean * self._y_rm.mean

    def push(self, x, y):
        self._x_rm.push(x)
        self._y_rm.push(y)
        self._xy_rm.push(x*y)
        self.covariance = self._xy_rm.mean - self._x_rm.mean * self._y_rm.mean

    def pop(self):
        N = len(self._x_rm.q)
        if N == 0:
            print("Popped from an empty list.")
            self.variance = None
            return
        old_x = self._x_rm.pop()
        old_y = self._y_rm.pop()
        self._xy_rm.pop()
        if N > 1:
            self.covariance = self._xy_rm.mean - self._x_rm.mean * self._y_rm.mean
        else:
            self.covariance = None
        return old_x, old_y

    def pushpop(self, x, y):
        N = len(self._x_rm.q)
        if N == 0:
            print("Pushpopped from an empty list.")
            return
        old_x = self._x_rm.pushpop(x)
        old_y = self._y_rm.pushpop(y)
        self._xy_rm.pushpop(x*y)
        self.covariance = self._xy_rm.mean - self._x_rm.mean * self._y_rm.mean
        return old_x, old_y
