import numpy as np

def MatrixSymmetryMetric(A,norm=None):
    """
    Some ad hoc metric for how symmetric a matrix is. Will return a number on [0,1].
    0 if A = Aanti is an antisymmetric matrix.
    1 if A = Asym is a symmetric matrix.
    
    Default norm=None will use ord=None, which is numpy default of Frobenius norm
    or put in whatever other numpy norm you want for numpy.linalg.norm function
    """
    Asym = .5*(A + A.T)
    Aanti = .5*(A - A.T)
    num = np.linalg.norm(Asym,ord=norm) - np.linalg.norm(Aanti,ord=norm)
    denom = np.linalg.norm(Asym,ord=norm) + np.linalg.norm(Aanti,ord=norm)
    return .5*(num/denom + 1.)