import numpy as np


def WeightedHarmonicMean(X,W):
    """
    X - NxM array of values
    W - NxM array of weights

    Orientation: assumes N dimensional vectors, and M vectors to average.
    For each row n of N, returns the weighted harmonic mean across row n.
    
    returns:
    WHM - N dimensional vector. Each element is the weighted harmonic mean 
    along axis 1.

    """
    
    if X.shape != W.shape:
        Exception('X and W must be same shape: one wight for each measurement')
    
    num = np.nansum(W,axis=1)
    denom = np.nansum(W/X,axis=1)
    WHM = num/denom
    #Anywhere that had X=0 will have denom=inf or -inf, so WHM will be 0 at those elements
    
    return WHM