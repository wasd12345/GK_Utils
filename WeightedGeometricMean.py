import numpy as np


def WeightedGeometricMean(X,W):
    """
    X - NxM array of values
    W - NxM array of weights

    Orientation: assumes N dimensional vectors, and M vectors to average.
    For each row n of N, returns the weighted geometric mean across row n.
    
    returns:
    WGM - N dimensional vector. Each element is the weighted geometric mean 
    along axis 1.

    """
    
    if X.shape != W.shape:
        Exception('X and W must be same shape: one wight for each measurement')
    
    
    P = np.nanprod(X**W,axis=1)
    exponent = 1./np.nansum(W,axis=1)
    WGM = P**exponent
    
    return WGM