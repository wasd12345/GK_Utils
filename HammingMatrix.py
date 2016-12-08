import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
#from scipy.spatial.distance import hamming



def StringHamming(str1, str2):
        diffs = 0
        for ch1, ch2 in zip(str1, str2):
                if ch1 != ch2:
                        diffs += 1
        return diffs
        
        
        
def HammingMatrix(item_set,plot=False):
    """
    Calculate the Hamming Distnce between all points in the list.
    Make a heatmap of the Hamming Distance between each pair.
    Diagonals elements are 0, and the matrix is symmetric, with all 
    elements nonnegative integers.
    
    item_set - list of strings, all of same length N
    Is a list of N vertex string coordinates between which to calculate the 
    Hamming distance.
    e.g. 
    item_set = ['AATACGC','AATGCTA',...,'TTCGCTT']
    """
    
    #Hamming distance is only defined for comparing 2 strings that have the 
    #same length. If they are not the same length, use another string metric,
    #e.g. Levenshtein distance.
    lens = np.array([len(i) for i in item_set])
    if ~np.all(lens==lens[0]):
        Exception('Not all strings are same length')
    
    
    Npts = len(item_set) #Number of points between which to calculate dist.
    Dim = lens[0] #Dimensionality of space the points are in
    
    
    #Iterate through each pair of strings, updating corresponding element of 
    #Hamming Matrix.
    HM = np.zeros((Npts,Npts))
    for r in xrange(Npts):
        for c in xrange(r,Npts):
            d = StringHamming(item_set[r],item_set[c])
            HM[r,c] = d
            HM[c,r] = d
            
    
    
    if plot==True:
        plt.figure()
        plt.imshow(HM,interpolation='None',cmap='gray')
        plt.title('Hamming Distance Matrix',fontsize=20)
        plt.colorbar()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)

    
    print 'Hamming Distance Matrix'
    print HM
    print 'Unique Values'
    print np.unique(HM)
    
    print 'Npts', Npts
    print 'Dim', Dim
    print
    return HM, Npts, Dim
    
    
    








    
if __name__ == '__main__':
    
    #Test on very simple case    
    item_set = ['AATGC','AATGC','ATGCC','CGTGC','GGTGC','TGTGC']
    HM, Npts, Dim = HammingMatrix(item_set,plot=True)
    
    #Test on strings of tracer fragments
    import pandas as pd
    df = pd.read_csv("uidlist2.csv",header=None)
    item_set = df[0].values
    print item_set
    HM, Npts, Dim = HammingMatrix(item_set,plot=True)
    
    plt.figure()
    plt.hist(HM.flatten(),bins=6)
    