This repository contains an implementation of simple linear regression that allows the efficient addition of new data points and removing of old data points.  With a least squares simple linear regression model
$$
y = \beta_1 x + \beta_0
$$
we estimate the parameters 
$$
\hat{\beta_1} = \frac{\sigma_{xy}}{\sigma_{x}^2}
$$
and
$$
\hat{\beta_0} = \bar{y} - \hat{beta_1} \bar{x}
$$
where $\sigma_{xy}$ is the sample covariance of $x$ and $y$, $\sigma_{x}^2$ is the sample variance of $x$, and $\bar{x}$ and $\bar{y}$ are the sample means of $x$ and $y$ respectively (as ratios are being taken, the Bessel correction can either be included or not, as long as this is done consistently, there will be no effect -- we make the convention that we are not using the Bessel correction as this simplifies some formulas).  Rather than recomputing all of the quantities each time a new data point is added (or removed), we can observe that the means can be efficiently updated without looking at all of the data and that the sample variance and covariance can be computed as 
$$
\sigma_x^2 = \bar{x^2} - \bar{x}^2
$$
and
$$
\sigma_{xy}^2 = \bar{xy} - \bar{x} \cdot \bar{y}
$$
Similarly, $r^2$ and the $t$-score of a linear regression can be efficiently updated (recalling that $r^2$ is the square of the sample correlation coeficient between $x$ and $y$, and the $t$ can be computed from $r$).  

The file ``running_simple_stats.py`` contains a supporting class ``RunningSimpleStats`` that, after initialized with equal length two lists of numbers and given new $x$ and $y$, will efficiently update the means, variances and covariance of the data augmented by $x$ and $y$.  The lists are stored as queues (using the [``deque``](https://docs.python.org/3/library/collections.html) container) and as such this operation is called ``push``.  In addition, the class supports the simultaneous efficient removal of the first data points from the lists, together with an update to the relevant statistics -- this operation is denoted ``pop``.  If one wishes to add a new data point and remove the oldest data point from the dataset, this can be done slightly more efficiently than a call to a ``push`` and a ``pop``.  As this sort of updating is rather common, we include it as an operation ``pushpop``.  

The file ``running_linear_regression.py`` contains a class ``RunningLinearRegression`` that roughly follows the [sklearn linear regression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) but allows for efficient support of these push, pop, and pushpop operations (where a complete retraining of the model is not necessary).  

Appropriate generalizations of these classes would allow for support of multiple regression as well as support for updating confidence intervals and other statistics.  

The notebook ``testing.ipynb`` demonstrates that our linear regression class is behaving as it should by comparing the parameters that it computes to the parameters computed by the sklearn ``LinearRegression`` class.  The notebook ``timing`.ipynb`` gives performance comparisons between our linear regression class and the sklearn's ``LinearRegression`` with a (arbitrarily chosen) workload that involves pushing and popping of new and old data points.  We see that our method offers an approximately 20x speedup over the sklearn method (which completely retrains for each change in the dataset).  In addition, we include a comparison of pushing and popping separately and show that by grouping these as a single operation we get a 2x speedup.  

There are several [other methods](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) of computing the sample variance/covariance that have different tradeoffs.  [This](https://www.cs.yale.edu/publications/techreports/tr222.pdf) paper by Chan, Goub and LeVeque is a nice entry-point into these considerations that offers recommendations on when to possibly prefer certain methods over others.  These ideas ould be integrated with ``RunningLinearRegression`` following a suitable modification of the ``RunningSimpleStats`` class.  

The ``old`` directory contains some fragments that I used in development and you should likely not look there.  The ``classis_lin_reg`` directory contains a naive implementation of linear regression with options for lasso and ridge regression and training via gradient descent.  Stochastic gradient descent offers a different avenue towards updating linear regression to new data which I plan to explore and compare with the method outlined here.  
# RunningLinReg
