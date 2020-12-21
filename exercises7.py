import numpy as np

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    a = 1*p[i]
    p[i] = 1*p[j]
    p[j] = 1*a
    
    return p


def LUP_inplace(A):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
    m = A.shape[0]
    p = np.arange(m)
    for k in range(m-1):
        i = 1*k
        for j in range(k+1,m):
            if abs(A[j,k])>abs(A[i,k]):
                i = 1*j
        p = perm(p, i, k)
        b = 1.0*A[k:,i]
        A[k:,i] = 1.0*A[k:,k]
        A[k:,k] = 1.0*b
        A[k+1:,k] /= A[k,k]
        A[k+1:,k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])
        
    return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
                     
    raise NotImplementedError

def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
                     
    raise NotImplementedError
