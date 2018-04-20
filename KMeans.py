# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:20:44 2018

@author: GK


very simple custom K-means / K-medians

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances


# =============================================================================
# Define some parameters
# =============================================================================
RANDOM_SEED = 4152018 #None #0
CLUSTERING_TYPE = 'MEAN' #'MEAN' or 'MEDIAN'
N_Clusters = 2 #3 #4 #7 #There are only K=2 different ground truth labeled clusters, but anyway just see 3 seems to visually fit best
N_iterations_max = 1000 #In reality, this data seems to finish clustering in <10 steps most of the time
TRAIN_FILEPATH = r"synth.tr"
#TEST_FILEPATH = r"synth.te" #Actually for this unsupervised clusterign we will not use TEST set
PLOT = True






# =============================================================================
# LOAD the data
# =============================================================================
def load_data(filepath):
    print('Loading data...')
    with open(filepath,'r') as gg:
        xs = []
        ys = []
        yc = []
        for i, line in enumerate(gg.readlines()):
            if i==0:
                #Skip the header line
                continue
            #rest is data
            t = line.split()
            xs+=[float(t[0])]
            ys+=[float(t[1])]
            yc+=[int(t[2])]
        X = np.vstack((np.array(xs),np.array(ys))).T
        y = np.array(yc)
        N = y.size
    return X, y, N






# =============================================================================
#  Basic metrics for analysis
# =============================================================================
def Metrics(y_pred, y_true):
    """
    Accuracy and F1-score as metrics, using predicted labels and ground truth labels
    """
    
    p = (y_pred==y_true)
    N = p.size
    #N_correct = p.sum()
    #accuracy = float(N_correct)/float(N)
    #print('Accuracy: ', accuracy)


    #CONFUSION MATRIX
    #TP - True positive
    #FP- False positive
    #TN - True negative
    #FN - False negative
    
    #Cleaner than below would be to just use set intersection for TP, etc., but this is fine
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.        
    for ex in range(N):
        if y_true[ex] == 1:
            #True Positives:
            if y_pred[ex] == 1:
                TP += 1.
            #False negatives:
            elif y_pred[ex] == 0:
                FN += 1.
        if y_true[ex] == 0:
            #True negatives:
            if y_pred[ex] == 0:
                TN += 1.
            #False Positives:
            elif y_pred[ex] == 1:
                FP += 1.
    print('TP: ',TP)
    print('FP: ',FP)
    print('TN: ',TN)
    print('FN: ',FN)
    print('N examples: ',N)

    #Calculate various classification metrics:
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2.*precision*recall / (precision + recall)
    accuracy = (TP + TN) / N
    print('precision: ',precision)
    print('recall: ',recall)
    print('fscore: ',fscore)
    print('accuracy: ',accuracy)        
    return precision, recall, fscore, accuracy




# =============================================================================
# Initialize algorithm
# =============================================================================
def Clustering(X, y, N, random_seed, clustering_type, plot):


    #Randomly choose (withOUT replacement) the poitns to use as initial clusters
    #For better results, could initialize using KMeans++ algorithm or similar approach instead of uniformly at random
    #But for now just do uniformly at random selection
    #seed for repeatability:
    np.random.seed(random_seed)
    
    center_inds = np.random.choice(N, size=N_Clusters, replace=False)
    centers = X[center_inds]
    centers_previous = centers*np.nan
    
    if clustering_type == 'MEAN':
        clustering_center_func = np.mean 
    if clustering_type == 'MEDIAN':
        clustering_center_func = np.median     
        
        
        
    for i in range(N_iterations_max):
        print('Iteration ',i)
        #Do (hard) assignment step:
        #get distances from all points to cluster center points:
        D = pairwise_distances(X,centers)
        #Do hard assignment: each point assigned fully to exactly 1 cluster label:
        assignments = np.argmin(D,axis=1)
        
        #Now given the assignments, compute new cluster centers:
        #Use MEAN as the metric for centerness:
        for c in range(N_Clusters):
            inds = np.where(assignments==c)
            assigned_points = X[inds]
            #To do K-Means or K-medians or custom function:
            new_center = clustering_center_func(assigned_points,axis=0)
            #Update the center coordinates for this cluster:
            centers[c] = new_center
            
        
        if plot:
            colorlist = ['r','g','b','k','c','y','m']
            plt.figure()
            plt.title('Clustering: Iteration {}'.format(i),fontsize=20)
            for c in range(N_Clusters):
                inds = np.where(assignments==c)
                assigned_points = X[inds]
                xx = assigned_points[:,0]
                yy = assigned_points[:,1]
                #plt.scatter(xx,yy,c=c*np.ones(xx.shape),cmap=plt.cm.hsv,label='Cluster {}'.format(c))
                plt.scatter(xx,yy,color=colorlist[c],label='Cluster {}'.format(c))
            plt.legend(numpoints=1)
            plt.show()
            
        
        
        
        #Check for convergence (no update to assignments):
        if np.alltrue(centers_previous == centers):
            #If converged, return final assignments
            print('Finished clustering after iteration ',i)
            return assignments
        
        #Otherwise keep iterating:
        else:
            centers_previous = centers.copy()
            
            
    #If it gets here, it failed to converge after this many iterations
    print('Failed to converge after iteration ',i)
    print('returning latest assignments')
    return assignments
        
        
    
    
    
    
















if __name__ == "__main__":
    #Load training data
    Xtrain, ytrain, N = load_data(TRAIN_FILEPATH)
    assignments = Clustering(Xtrain, ytrain, N, RANDOM_SEED, CLUSTERING_TYPE, PLOT)
    
    #Load the test data
    #Actually we will not use the TEST data
    #Xtest, ytest, N = load_data(TEST_FILEPATH)
    
    #Analyze predicted assignments:
    precision, recall, fscore, accuracy = Metrics(assignments, ytrain)
    
    
    
    
    
    
    
    
    
    
"""
EXAMPLE OUTPUT:


7 clusters:

    
Loading data...
Iteration  0
Iteration  1
Iteration  2
Iteration  3
Iteration  4
Iteration  5
Iteration  6
Iteration  7
Iteration  8
Iteration  9
Iteration  10
Finished clustering after iteration  10
TP:  20.0
FP:  36.0
TN:  24.0
FN:  10.0
N examples:  250
precision:  0.35714285714285715
recall:  0.6666666666666666
fscore:  0.46511627906976744
accuracy:  0.176






2 clusters:
    
    
Loading data...
Iteration  0
Iteration  1
Iteration  2
Iteration  3
Iteration  4
Iteration  5
Finished clustering after iteration  5
TP:  69.0
FP:  62.0
TN:  63.0
FN:  56.0
N examples:  250
precision:  0.5267175572519084
recall:  0.552
fscore:  0.5390625
accuracy:  0.528
"""