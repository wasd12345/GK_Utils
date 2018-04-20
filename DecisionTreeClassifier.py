# -*- coding: utf-8 -*-
"""
Very basic implementation of a slow multiclass decision tree classifier.
E.g. tested on MNIST digits
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

#And get rid of all the pixels which are ALWAYS 0 for every single image
from sklearn.feature_selection import VarianceThreshold



def Gini(p):
    #Get the proportions of each class (look at last column labels):
    p = np.bincount(data[:,-1])
    #Get the counts for all nonzero coutn classes, normalize to probabilities:
    p = p[p>=1]/p.sum()
    #Calculate the Gini
    return 1. - np.sum(p**2)


def Entropy(data):
    #Get the proportions of each class (look at last column labels):
    p = np.bincount(data[:,-1])
    #Get the counts for all nonzero coutn classes, normalize to probabilities:
    p = p[p>=1]/p.sum()
    #Calculate the (Shannon (base2)) entropy:
    return -(p*np.log2(p)).sum()
    

def GetMajorityLabel(data):
    labels = data[:,-1]
    vals, counts = np.unique(labels,return_counts=True)
    return vals[counts.argmax()]
        
    
def FindBestSplit(data,metric):
    if data.ndim == 2:
        Nfeatures = data.shape[1] - 1 #-1 since the lst column is class label
    else:
        Nfeatures = data.size - 1
        
    #Score/metric of parent node. E.g. entropy
    parent_score = metric(data)

    
    #Below implementation iterates through each feature WITHOUT replacement from features
    #randomly shuffle features
    feature_inds = np.arange(Nfeatures)
    #To take random subsample of features, uncomment end slice:
    feature_inds = np.random.permutation(feature_inds)#[:100]
    
    #Iterate through every feature and evry value of that feature to get best
    #possible split for this dat at this level of the tree. 
    #if wanted to only use random e.g. 100 features: feature_inds[:100], as in
    #Random Forests or Extra Trees, which usually use N' = sqrt(Nfeatures) 
    #randomly sampled features.
    best_score = -123456.
    best_feature = None
    best_threshold = None
    best_below_rows = np.array([])
    best_above_rows = np.array([])

    for feature in feature_inds:

        #For this feature, get all the unique values over this subet of the data
        feature_vals = data[:,feature]
        #(only take unique values so don't repeat in for loop computation below)
        #(Will be a sorted list of subset of elements from [0,1,...,255])
        unique_feature_vals = np.unique(feature_vals)
        unique_feature_vals = np.linspace(feature_vals.min(),feature_vals.max(),10)#SPlit into 10 equal regions
        unique_feature_vals = np.random.permutation(unique_feature_vals)#[:20] #Could take random subset of the 10 thresholds
        
        
        #!!!!Actually, better than below method of checking all thresholds is 
        #to use Extra Trees approach and just randomly choose a threshold.
        #Speeds up computation a lot and reduces variance of the final 
        #bagged classifier. But for now just use traditional decision tree:
        for val in unique_feature_vals:
            below_inds = np.where(feature_vals<=val)[0]
            above_inds = np.where(feature_vals>val)[0]
            N_below = below_inds.size
            N_above = above_inds.size
            #Get the entropy/gini of the labels for potential above/below nodes:
            below_score = metric(data[below_inds])
            above_score = metric(data[above_inds])
            #Weighted Mean of both branch split scores:
            combined_score = (N_below*below_score + N_above*above_score)/(N_below+N_above)
            split_score = parent_score - combined_score
            #print(parent_score,combined_score,split_score)



                
            #If this particular split/threshold looks promising, save it
            if split_score > best_score and len(below_inds)>0 and len(above_inds)>0:
                best_score = split_score
                best_feature = feature
                best_threshold = val
                best_below_rows = data[below_inds]
                best_above_rows = data[above_inds]
                

#        print(best_score,best_feature,best_threshold)

    return {'Feature':best_feature, 'Threshold':best_threshold, 'Below/Above':(best_below_rows,best_above_rows)}



def DoSplit(tree_leaf, max_depth, split_metric, current_depth):

    #Get rid of case where there is no data left (already sorted at higher
    #levels of tree but still got passed through to this step)
    if tree_leaf==None:
        return

    #Otherwise continue splitting the tree:
    below, above = tree_leaf['Below/Above']
    del(tree_leaf['Below/Above'])

        
    
    #If reached max depth of tree, don't split anymore here:
    if current_depth >= max_depth:
#        print('too deep')
        tree_leaf['below'] = GetMajorityLabel(below)
        tree_leaf['above'] = GetMajorityLabel(above)
        return
    
    #If not at max depth, and not empty, then continue splitting:
    m = 5 #This m value has 2 effects. First, from an imlpementation perspective
    #it prevents potential indexing errors if a leaf is split but would have 0
    #data points in either of the splits. More interestingly, it is a limit on 
    #the minimum leaf node size, and increasing it alleviates overfitting since 
    #the tree will not make minor splits to get a few data points more separated.
    if below.shape[0]>=m:
#        print('below m')
        tree_leaf['below'] = FindBestSplit(below,split_metric)
        DoSplit(tree_leaf['below'], max_depth, split_metric, current_depth+1)
    if above.shape[0]>=m:
#        print('above m')
        tree_leaf['above'] = FindBestSplit(above,split_metric)
        DoSplit(tree_leaf['above'], max_depth, split_metric, current_depth+1)

    if below.shape[0]<m or above.shape[0]<m:
        label = GetMajorityLabel(np.vstack((below,above)))
        tree_leaf['below'] = label
        tree_leaf['above'] = label
        return
    
    
	
def TrainTree(data,max_depth,split_metric):
    """
    Build decision tree classifier ontraining data.
    
    max_depth - integer, max # of levels allowed in building tree. 
    
    split metric - function, e.g. Gini or Entropy
    """
    
    tree = FindBestSplit(data,split_metric)
    current_depth = 1
    DoSplit(tree,max_depth,split_metric,current_depth)
    return tree




    
    
def TestTree(data,tree):
    """
    Use the decision tree classifier for classification on the test set
    """
    
#    Nfeatures = data.shape[1] - 1 #-1 since the lst column is class label
    Nexamples = data.shape[0]    
    y_true = data[:,-1]
    
    y_pred = []
    for ex in range(Nexamples):
        tree_test = tree.copy()
        not_converged = True
        while not_converged:
            feature = tree_test['Feature']
            threshold = tree_test['Threshold']
            branch = 'below' if data[ex,feature] <= threshold else 'above'
            tree_test = tree_test[branch]
            if type(tree_test)==np.uint8:
                y_pred.append(tree_test)
                not_converged = False
        
    y_pred = np.array(y_pred)
    
    #Calculate classification accuracy
    accuracy = (y_pred==y_true).sum() / Nexamples

    print(y_pred)
    print(y_true)
    print(accuracy)
    
    return y_pred, accuracy
    
    
    

if __name__ == "__main__":
    
    # =============================================================================
    #     Parameters
    # =============================================================================
    #Seed for repeatability
    np.random.seed(9898)#Used this for 1st figure
    np.random.seed(1234)#Used for 2nd figure, 3rd fig
    Ntraining = 7000 #9500 #Number of examples to use in trainging set 
    #(leaving 10K - Ntraining in test set)
    Ntest = 10000 - Ntraining
    
    
    # =============================================================================
    # MAIN    
    # =============================================================================
    
    #Load MNIST data
    data = scipy.io.loadmat(r'\HW1\hw1data.mat')
    #data['X'] is 10000 x 784 [Nexamples by Nfeatures] of pixel values on [0,255]
    #data['Y'] is 10000 x 1 vector of labels [0,1,2,...,9]
    data = np.hstack((data['X'],data['Y']))
    #Random train test split, taking Ntraining for training and 10K - Ntraining for testing
    np.random.shuffle(data)
    train0 = data[:Ntraining]
    test0 = data[Ntraining:]
    
    
    #FOr this training data, gets rid of about 550 features, leaving best 200.
    selector = VarianceThreshold(10000.)
    train = selector.fit_transform(train0[:,:-1])
    #Append back the labels to use later
    train = np.hstack((train,train0[:,-1].reshape((Ntraining,1))))
    #Same transformation for test set
    test = selector.transform(test0[:,:-1])
    test = np.hstack((test,test0[:,-1].reshape((Ntest,1))))
    
    
    #Explore model complexity / ovrfitting as a function of max tree depth, K:
    K_range = np.arange(1,20) #Just do up to depth K=20 since gets slow for deep trees 
    acc_vector__train = []
    acc_vector__test = []
    tree_list = []
    for i, K in enumerate(K_range):
        
        #Build a tree with max_deph K:
        print('Training Tree with max_depth {} levels'.format(K))
        tree = TrainTree(train,K,split_metric=Entropy)#Is much slower and worse w/ Gini
        tree_list += [tree]
        print(tree)
        
        #Get the TRAINING error
        predictions__train, accuracy__train = TestTree(train,tree)
        acc_vector__train.append(accuracy__train)
        
        #Get the TEST error
        predictions__test, accuracy__test = TestTree(test,tree)
        acc_vector__test.append(accuracy__test)
        
    plt.figure(figsize=(18,12))
    plt.title('Accuracy as a function of tree depth',fontsize=20)
    plt.plot(K_range,acc_vector__train,marker='o',color='g',label='Train')
    plt.plot(K_range,acc_vector__test,marker='s',color='r',label='Test')
    plt.xlabel('Max Tree Depth',fontsize=20)
    plt.ylabel('Classification Accuracy',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(numpoints=1)

    
    
    import pickle
    pickle.dump( tree_list, open( "tree_list.p", "wb" ) )
    pickle.dump( acc_vector__train, open( "acc_vector__train.p", "wb" ) )
    pickle.dump( acc_vector__test, open( "acc_vector__test.p", "wb" ) )
#    acc_vector__test = pickle.load( open( "acc_vector__test.p", "rb" ) )
#    acc_vector__train = pickle.load( open( "acc_vector__train.p", "rb" ) )