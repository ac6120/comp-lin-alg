import numpy as np


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
    R = 1.0*A
    if kmax is None:
        kmax = n
    for k in range(n):
        x = R[k:,k]
        v = 1*x
        s = np.sign(x[0])
        if s==0:
            s=1 #fixing sign(0)=1, which is not true in numpy
        v[0] += s * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        R[k:,k:] -= 2 * v * np.dot(R[k:,k:], v.conj().transpose())
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

    raise NotImplementedError

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    raise NotImplementedError

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    raise NotImplementedError

    return x
