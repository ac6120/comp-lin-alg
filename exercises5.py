import numpy as np


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = linalg.qr(random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = random.randn(m, m)
    return numpy.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = np.dot(Q1, R1)
        Q2, R2 = np.linalg.qr(A)
        q = np.linalg.norm(Q1 - Q2)
        r = np.linalg.norm(R1 - R2)
        a = np.linalg.norm(A - Q2.dot(R2))
        print("||Q1-Q2|| = ", q)
        print("||R1-R2|| = ", r)
        print("||A-Q2R2|| = ", a)
    return None

def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix 
    and b is an m dimensional vector.

    :param R: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """
    m, _ = R.shape
    x = np.zeros(m)
    x[-1] = b[-1] / R[-1,-1]
    for i in range(m-2, -1, -1):
        x[i] = (b[i] - np.dot(R[i,i+1:], x[i+1:])) / R[i,i]
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = random.randn(m, m)
        R = np.triu(A)

        raise NotImplementedError


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    raise NotImplementedError
