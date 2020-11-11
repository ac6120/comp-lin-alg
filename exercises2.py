import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r:for an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    (m,n) = Q.shape
    r = v*1.0
    u = np.zeros(n, dtype=complex)
    for i in range(n):
        u[i] = np.vdot(Q[:,i],v)
        r = r - u[i] * Q[:,i]
    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    x=np.dot(Q.conj().transpose(),b)

    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    P=np.dot(Q,Q.conj().transpose())

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mx(m-n)-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """

    Q1,R = np.linalg.qr(V, mode="complete")

    m,n = V.shape

    Q = Q1[:,n:]

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array
    
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    V = A*1.0
    m,n = A.shape
    Q = np.zeros((m, n), dtype=complex)
    R = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for i in range(k-1):
            R[i,k] = np.vdot(Q[:,i], A[:,k])
            V[:,k] -= R[i,k] * Q[:,i]
        R[k,k] = np.linalg.norm(V[:,k])
        Q[:,k] = V[:,k] / R[k,k]
    return Q, R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, producing

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    m,n = A.shape
    Q = A.astype(complex)
    R = np.zeros((n, n), dtype=complex)
    for i in range(n):
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1, n):
            R[i,j] = np.vdot(Q[:,i], Q[:,j])
            Q[:,j] -= R[i,j] * Q[:,i]
    return Q, R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    m, n = A.shape
    Ahat = A.astype(complex)
    R = np.eye(n, dtype=complex)
    R[k,k] = 1 / np.linalg.norm(Ahat[:,k])    
    Ahat[:,k] = Ahat[:,k] / R[k,k]
    for i in range(k,n):
        R[k,i] = -np.linalg.norm(np.vdot(Ahat[:,k], Ahat[:,k])) / R[k,k]
    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
