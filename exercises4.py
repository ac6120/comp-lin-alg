import numpy as np
import math
from numpy import random


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    #||Ax||^2=x*A*Ax=x*lambdax=lambda, for lambda max eigenvalue of A*A
    #and x with norm 1, eigenvector of lambda max
    eig, _ = np.linalg.eig(np.dot(A.transpose().conj(), A))
    o2norm = math.sqrt(np.amax(eig))
    return o2norm

def verify_inequality_Ax(m, n):
    "verifies ||Ax|| <= ||A|| ||x||"
    random.seed(2432*m + 7438*n)
    A = random.randn(m, n)
    x = random.randn(n)
    n = np.linalg.norm(np.dot(A, x))
    n0 = cla_utils.operator_2_norm(A) * np.linalg.norm(x)
    assert(n <= n0) #do we need a tolerance?

def verify_inequality_AB(l, m, n):
    "verifies ||AB|| <= ||A|| ||B||"
    random.seed(1878*l + 2432*m + 7438*n)
    A = random.randn(l, m)
    B = random.randn(m, n)
    n = cla_utils.operator_2_norm(np.dot(A, B))
    n0 = cla_utils.operator_2_norm(A) * cla_utils.operator_2_norm(B)
    assert(n <= n0) #do we need a tolerance?

def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    eig, _ = np.linalg.eig(np.dot(A.transpose().conj(), A))
    o2normA = math.sqrt(np.amax(eig))
    o2normAinv = 1 / math.sqrt(np.amin(eig))
    ncond = o2normA * o2normAinv
    return ncond
