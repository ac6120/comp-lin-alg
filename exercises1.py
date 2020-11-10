import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    
    (m,n)=A.shape
    b=np.zeros(m)
    for i in range(m):
        for j in range(n):
            b[i]=b[i]+A[i,j]*x[j]
    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    (m,n)=A.shape
    b=np.zeros(m)
    for j in range(n):
        b=b+A[:,j].transpose()*x[j]
    return b

def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v1^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    B=np.column_stack([u1,u2])
    C=np.transpose(np.conjugate(np.column_stack([v1,v2])))

    A = np.dot(B,C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    m=u.shape[0]
    alpha=-1/(1+np.dot(np.conjugate(v),u))
    Ainv=np.identity(m)+alpha*np.outer(u,v.conj())
    
    return Ainv


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i<=j and Ahat[i,j] = C[i,j] for i>j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    m=Ahat.shape[0]
    zr=np.zeros(m)
    zi=np.zeros(m)
    """
    zr=Bxr-Cxi
    zi=Bxi+Cxr
    C=-C.T
    """
    for i in range(m):
        zr[i] += np.vdot(Ahat[i,:i+1],xr[:i+1]) + np.vdot(Ahat[i+1:,i],xr[i+1:]) #Bxr
        zr[i] += - np.vdot(Ahat[:i,i],xi[:i]) + np.vdot(Ahat[i,i+1:],xi[i+1:]) #-Cxi
        zi[i] += np.vdot(Ahat[i,:i+1],xi[:i+1]) + np.vdot(Ahat[i+1:,i],xi[i+1:]) #Bxi
        zi[i] += np.vdot(Ahat[:i,i],xr[:i]) - np.vdot(Ahat[i,i+1:],xr[i+1:]) #Cxr
    return zr, zi
