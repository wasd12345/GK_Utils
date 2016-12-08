import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def KDEplot(xkde,bw_method=None,plot=False):
    """
    Gaussian Kernel Density Estimate of 1D series.
    
    xkde - the 1D series of which to estimate PDF.
        
    If bw_method = scalar, uses that bandwidth. 
    Otherwise uses default heuristic to determine reasonable value for bandwidth.
    """
    kde = gaussian_kde(xkde,bw_method=bw_method)
    
    #Upper bound on number of points for computational reasons (1000),
    #and lower bound for cases where there would otherwise be few points
    #NPoints = int(np.clip(xkde.max()-xkde.min(),100.,1000.))
    NPoints = 1000 #May as well just always sample at 1000 pts.

    xs = np.linspace(xkde.min(), xkde.max(), num=NPoints)
    ykde = kde(xs)
    
    #Normalize KDE to probability
    ykde /= ykde.sum()
    
    if plot==True:
        plt.figure()
        plt.plot(xs,ykde,marker='o')
        
    return xs, ykde