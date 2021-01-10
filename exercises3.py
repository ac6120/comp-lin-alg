import numpy as np
from scipy.linalg import solve_triangular

def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape
    R = (1.0+0j)*A
    if kmax is None:
        kmax = n
    kmax = min(m,kmax)
    for k in range(kmax):
        x = 1.0 * R[k:,k]
        v = (1.0+0j) * x
        arg = np.angle(x[0])
        coeff = np.exp(1j * arg)
        v[0] += coeff * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        R[k:,k:] -= 2 * np.outer(v, np.dot(v.conj().transpose(), R[k:,k:]))
    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m, k = b.shape
    Ahat = np.zeros((m,m+1))
    x = np.zeros((m,k))
    for i in range(k):
        Ahat[:,:m] = 1.0*A
        Ahat[:,m] = 1.0*b[:,i]
        Rhat = householder(Ahat, m)
        x[:,i] = solve_triangular(Rhat[:,:m], Rhat[:,m])
    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    I = np.eye(m, dtype = complex)
    Ahat = np.zeros((m, n+m), dtype = complex)
    Ahat[:, :n] = A
    Ahat[:, n:] = I

    Rhat = householder(Ahat)
    R = Rhat[:,:n]
    Q = Rhat[:,n:].transpose().conj()

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, n = A.shape
    Ahat = np.zeros((m, n+1))
    Ahat[:,:n] = 1.0*A
    Ahat[:, n] = 1.0*b

    Rhat = householder(Ahat)
    x = solve_triangular(Rhat[:n,:n], Rhat[:n,n])

    return x
