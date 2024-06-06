from collections import deque
import numpy as np


class RunningMean(object):
    '''
    Stores the mean of a queue of numbers that can be updated with either
    adding a new number on the end (``push``), removing the first number 
    (``pop``), or a combination of both of those (``pushpop`` which is 
    more efficient than a consecutive ``push`` and ``pop``).  The queue
    can be initialized as empty or with a list of entries.  Each
    of the methods runs in O(1) time.  If ``pop`` or ``pushpop`` is 
    called on an empty queue, then an exception is raised.  
    '''

    def __init__(self, xs=[]):
        '''
        Initialization can be either with a queue or with nothing.
        Internally the entries are stored as a queue (namely, ``deque``
        from the collections library).  

        Instance attributes:
        ``q``    : entries as a (double-ended) queue.  
        ``N``    : current number of entries.  
        ``mean`` : current running mean.  
        '''
        self.q = deque(xs)
        self.N = len(xs)
        self.mean = np.mean(xs) if self.N != 0 else 0.0

    def push(self, x):
        '''
        Adds the number ``x`` to the queue. Updates the 
        attributes accordingly.   
        '''
        self.q.append(x)
        self.mean = 1 / (self.N + 1) * (self.N * self.mean + x)
        self.N += 1

    def pop(self):
        '''
        Removes the first entry from the queue. Updates 
        the attributes accordingly.   
        '''
        if self.N == 0:
            raise Exception("popping from an empty queue")
        x_pop = self.q.popleft()
        self.mean = self.N / (self.N - 1) * (self.mean -
                                             x_pop / self.N) if self.N != 1 else 0.0
        self.N -= 1
        return x_pop

    def pushpop(self, x):
        '''
        Adds the number ``x`` to the queue and remove the first 
        entry from the queue. Updates the attributes accordingly.   
        More efficient than running a consecutive ``push`` and 
        ``pop``.
        '''
        if self.N == 0:
            raise Exception("pushpopping from an empty queue")
        x_pop = self.q.popleft()
        self.q.append(x)
        self.mean = self.mean + (x - x_pop) / self.N
        return x_pop


class RunningSimpleStats(object):

    def _cov(xs, ys):
        '''
        Computes the sample covariance between the input lists.
        '''
        # Note that we must set ``bias=True`` to not get the
        # Bessel correction term.
        return np.cov(xs, ys, bias=True)[0][1]

    def __init__(self, xs=[], ys=[]):

        if len(xs) != len(ys):
            raise Exception("features are not of the same length")

        self.xs = RunningMean(xs)
        self.ys = RunningMean(ys)
        self.N = len(xs)

        xs = np.array(xs)
        ys = np.array(ys)
        self._x2s = RunningMean(xs**2)
        self._y2s = RunningMean(ys**2)
        self._xys = RunningMean(xs*ys)

        self.x_var = np.var(xs) if self.N != 0 else 0.0
        self.y_var = np.var(ys) if self.N != 0 else 0.0
        self.xy_cov = RunningSimpleStats._cov(xs, ys) if self.N != 0 else 0.0

    def _update_var_cov(self):
        '''
        Updates the ``x_var``, ``y_var``, and ``xy_cov`` instance
        attributes.  All mean instance attributes must have already been 
        updated.  
        '''
        # Update the variances.
        self.x_var = self._x2s.mean - self.xs.mean**2
        self.y_var = self._y2s.mean - self.ys.mean**2
        # Update Covariance.
        self.xy_cov = self._xys.mean - self.xs.mean*self.ys.mean

    def push(self, x, y):
        '''
        Adds the numbers ``x`` and ``y`` to the queues. Updates the 
        attributes accordingly.   
        '''
        self.xs.push(x)
        self.ys.push(y)
        self._x2s.push(x*x)
        self._y2s.push(y*y)
        self._xys.push(x*y)
        self.N += 1
        self._update_var_cov()

    def pop(self):
        '''
        Removes the first entry from each queue. Updates 
        the attributes accordingly.   
        '''
        if self.N == 0:
            raise Exception("popping from an empty queue")
        x_pop = self.xs.pop()
        y_pop = self.ys.pop()
        self._x2s.pop()
        self._y2s.pop()
        self._xys.pop()
        self.N -= 1
        self._update_var_cov()
        return x_pop, y_pop

    def pushpop(self, x, y):
        '''
        Adds the numbers ``x`` and ``y`` to the queues and remove 
        the first entry entries from the queues. Updates the 
        attributes accordingly. More efficient than running a 
        consecutive ``push`` and ``pop``.
        '''
        if self.N == 0:
            raise Exception("pushpopping from an empty queue")
        x_pop = self.xs.pushpop(x)
        y_pop = self.ys.pushpop(y)
        self._x2s.pushpop(x*x)
        self._y2s.pushpop(y*y)
        self._xys.pushpop(x*y)
        self._update_var_cov()
        return x_pop, y_pop
