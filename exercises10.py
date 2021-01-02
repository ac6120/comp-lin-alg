import numpy as np
import numpy.random as random
from cla_utils import *

def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mxk dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    H = np.zeros((k+1,k), dtype=complex)
    m = A.shape[0]
    Q = np.zeros((m,k+1), dtype=complex)
    Q[:,0] = b / np.linalg.norm(b)
    for n in range(k):
        v = A.dot(Q[:,n])
        for j in range(n+1):
            H[j,n] = np.dot(Q[:,j].conj(), v)
            v -= H[j,n]*Q[:,j]
        H[n+1,n] = np.linalg.norm(v)
        Q[:,n+1] = v / H[n+1,n]
    return Q, H


def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    if x0 is None:
        x0 = b

    conv = False
    
    m = A.shape[0]
    Q = np.zeros((m,maxit+1), dtype=complex)
    Q[:,0] = b / np.linalg.norm(b)

    H = np.zeros((maxit+1,maxit), dtype=complex)

    nits = -1
    
    
    r = []
    rnorms = []
    
    for n in range(maxit):
        v = A.dot(Q[:,n])
        
        for j in range(n+1):
            H[j,n] = np.dot(Q[:,j].conj().transpose(), v)
            v -= H[j,n] * Q[:,j]

        e1b = np.zeros(n+1)
        e1b[0] = np.linalg.norm(b)
        
        H[n+1,n] = np.linalg.norm(v)
        Q[:,n+1] = v / H[n+1,n]
        
        y = np.linalg.lstsq(H[:n+1,:n], e1b, rcond=None)[0]
        x = Q[:,:n] @ y
        r1 = b-A.dot(x)
        r.append(r1)
        rnorms.append(np.linalg.norm(r1))

        nits +=1
        
        if rnorms[n]<tol :
            conv = True
            break

    if not conv:
        nits = -1
    rnorms = np.array(rnorms)
    if return_residual_norms:
        if return_residuals:
            return x, nits, rnorms, r
        else:
            return x, nits, rnorms
    elif return_residuals:
        return x, nits, r
    else:
        return x, nits


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
