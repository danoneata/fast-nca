import pdb

import numpy as np

from scipy.optimize import (
    check_grad,
    fmin_cg,

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from sklearn.preprocessing import (
    StandardScaler,
)


def square_dist(x1, x2=None):
    """If x1 is NxD and x2 is MxD (default x1), return NxM square distances."""

    if x2 is None:
        x2 = x1

    return (
        np.sum(x1 * x1, 1)[:, np.newaxis] +
        np.sum(x2 * x2, 1)[np.newaxis, :] -
        np.dot(x1, (2 * x2.T))
    )


def nca_cost(A, xx, yy, reg):

    N, D = xx.shape
    assert(yy.size == N)
    assert(A.shape[1] == D)
    K = A.shape[0]

    # Cost function:
    zz = np.dot(A, xx.T)  # KxN

    # TODO Subsample part of data to compute loss on.
    # kk = np.exp(-square_dist(zz.T, zz.T[idxs]))  # Nxn
    # kk[idxs, np.arange(len(idxs))] = 0

    ss = square_dist(zz.T)
    np.fill_diagonal(ss, np.inf)
    mm = np.min(ss, axis=0)
    kk = np.exp(mm - ss)  # NxN
    # kk = np.exp(-ss)  # NxN
    np.fill_diagonal(kk, 0)
    Z_p = np.sum(kk, 0)  # N,
    p_mn = kk / Z_p[np.newaxis, :]  # P(z_m | z_n), NxN
    mask = yy[:, np.newaxis] == yy[np.newaxis, :]
    p_n = np.sum(p_mn * mask, 0)  # 1xN
    ff = - np.sum(p_n)

    # Back-propagate derivatives:
    kk_bar = - (mask - p_n[np.newaxis, :]) / Z_p[np.newaxis, :]  # NxN
    ee_bar = kk * kk_bar
    zz_bar_part = ee_bar + ee_bar.T
    zz_bar = 2 * (np.dot(zz, zz_bar_part) - (zz * np.sum(zz_bar_part, 0)))  # KxN
    gg = np.dot(zz_bar, xx)  # O(DKN)

    if reg > 0:
        ff = ff + reg * np.dot(A.ravel(), A.ravel())
        gg = gg + 2 * reg * A

    return ff, gg


def nca_cost_batch(self, A, xx, yy, idxs):

    N, D = xx.shape
    n = len(idxs)

    assert(yy.size == N)
    assert(A.shape[1] == D)

    K = A.shape[0]

    # Cost function:
    zz = np.dot(A, xx.T)  # KxN
    Z_p = np.sum(kk, 0)  # N,
    p_mn = kk / Z_p[np.newaxis, :]  # P(z_m | z_n), NxN
    mask = yy[:, np.newaxis] == yy[np.newaxis, :]
    p_n = np.sum(p_mn * mask, 0)  # 1xN
    ff = - np.sum(p_n)

    # Back-propagate derivatives:
    kk_bar = - (mask - p_n[np.newaxis, :]) / Z_p[np.newaxis, :]  # NxN
    zz_bar_part = kk * (kk_bar + kk_bar.T)
    zz_bar = 2 * (np.dot(zz, zz_bar_part) - (zz * sum(zz_bar_part, 0)))  # KxN
    gg = np.dot(zz_bar, xx)  # O(DKN)

    return ff, gg


class NCA(BaseEstimator, TransformerMixin):
    """Neighbourhood Components Analysis: cost function and gradients

        ff, gg = nca_cost(A, xx, yy)

    Evaluate a linear projection from a D-dim space to a K-dim space (K<=D).
    See Goldberger et al. (2004).

    Inputs:
        A KxD Current linear transformation.
        xx NxD Input data
        yy Nx1 Corresponding labels, taken from any discrete set

    Outputs:
        ff 1x1 NCA cost function
        gg KxD partial derivatives of ff wrt elements of A

    Motivation: gradients in existing implementations, and as written in the
    paper, have the wrong scaling with D. This implementation should scale
    correctly for problems with many input dimensions.

    Note: this function should be passed to a MINIMIZER.

    TODO Numerical stability. Currently if the scale of A is too large, so that
    a whole column of kk underflows, Bad Things happen.

    """

    def __init__(self, reg=0, dim=None, optimizer='cg'):
        self.reg = reg
        self.K = dim
        self.standard_scaler = StandardScaler()

        if optimizer in ('cg', 'conjugate_gradients'):
            self._fit = self._fit_conjugate_gradients
        elif optimizer in ('gd', 'gradient_descent'):
            self._fit = self._fit_gradient_descent
        elif optimizer in ('mb', 'mini_batches'):
            self._fit = self._fit_mini_batches
        else:
            raise ValueError("Unknown optimizer {:s}".format(optimizer))

    def fit(self, X, y):

        N, D = X.shape

        if self.K is None:
            self.K = D

        self.A = 0.5 * np.random.randn(self.K, D)

        X = self.standard_scaler.fit_transform(X)
        return self._fit(X, y)

    def _fit_gradient_descent(self, X, y):
        # Gradient descent.
        self.learning_rate = 0.1
        self.error_tol = 0.001
        self.max_iter = 1000

        curr_error = None

        # print(check_grad(costf, costg, 0.1 * np.random.randn(self.K * D)))
        idxs = list(sorted(random.sample(range(len(X)), 100)))

        for it in range(self.max_iter):

            f, g = nca_cost(self.A, X, y, self.reg)
            self.A -= self.learning_rate * g

            prev_error = curr_error
            curr_error = f

            print('{:4d} {:+6.2f}'.format(it, curr_error))

            if prev_error and np.abs(curr_error - prev_error) < self.error_tol:
                break

        return self

    def _fit_conjugate_gradients(self, X, y):
        N, D = X.shape

        def costf(A):
            f, _ = nca_cost(A.reshape([self.K, D]), X, y, self.reg)
            return f 

        def costg(A):
            _, g = nca_cost(A.reshape([self.K, D]), X, y, self.reg)
            return g.ravel()

        # print(check_grad(costf, costg, 0.1 * np.random.randn(self.K * D)))
        self.A = fmin_cg(costf, self.A.ravel(), costg).reshape([self.K, D])
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return np.dot(self.standard_scaler.transform(X), self.A.T)
