import numpy as np

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    x = A[:,0]
    v = 1.0*x
    s = np.sign(x[0])
    if s==0:
        s=1 #fixing sign(0)=1, which is not true in numpy
    v[0] += s * np.linalg.norm(x)
    v = v / np.linalg.norm(v)
    A1 = 1.0*A
    A1 -= 2 * np.dot(np.outer(v, v.conj().transpose()), A1)
    A1 -= 2 * (np.dot(np.outer(v, v.conj().transpose()), A1.T.conj())).T.conj()
    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    m = A.shape[0]
    for k in range(m-2):
        x = A[k+1:,k]
        v = 1.0*x
        s = np.sign(x[0])
        if s==0:
            s = 1 #fixing sign(0)=1, which is not true in numpy
        v[0] += s * np.linalg.norm(x)
        v /= np.linalg.norm(v)
        A[k+1:,k:] -= 2 * np.outer(v, v.conj().dot(A[k+1:,k:]))
        A[:,k+1:] -= 2 * np.outer(A[:,k+1:].dot(v), v.conj())


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m = A.shape[0]
    Q = np.eye(m, dtype=A.dtype)
    for k in range(m-1):
        x = A[k+1:,k]
        v = 1.0*x
        s = np.sign(x[0])
        if s==0:
            s = 1 #fixing sign(0)=1, which is not true in numpy
        v[0] += s * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        A[k+1:,k:] -= 2 * np.dot(np.outer(v, v.conj().transpose()), A[k+1:,k:])
        A[:,k+1:] -= 2 * A[:,k+1:].dot(np.outer(v, v.conj().transpose()))
        Q[k+1:,:] -= 2 * np.outer(v, np.dot(v.conj().transpose(), Q[k+1:,:]))
    return Q.conj().transpose()

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    assert(np.linalg.norm(H[np.tril_indices(m, -1)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvalues and eigenvectors. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of A
    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    raise NotImplementedError
