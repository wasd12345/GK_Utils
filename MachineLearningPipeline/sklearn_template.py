# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:33:27 2018
Python 3.6.3 64bits
@author: GK
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, classification_report, f1_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import copy

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

#pip install xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance

#pip install keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras import regularizers


#%matplotlib inline




def fit_svd(X, ncomps):
    """
    Find SVD of matrix X.
    Return the summed outer product matrix of the topK components outer products,
    to use later to subtract from each data point in Xtrain and Xtest.
    (**This is different than typical use of SVD. Here we are NOT doing
    dimensionality reduction. We are doing noise subtraction, where the assumption
    is that top 1 or 2 singular components are (mostly) variance due to noise,
    so optionally subtract those out.
    """
    #from sklearn.decomposition import TruncatedSVD
    #svd = TruncatedSVD(n_components=ncomps, n_iter=7)#, algorithm='arrpack', random_state=42)
    #svd.fit(X)
    #Xt = svd.transform(X)
    #print(Xt.shape)
   
    #For each feature vectorn, subtract out topK singular vectors:
    u, s, vh = np.linalg.svd(X, full_matrices=False)#change to False because memory errors...
    print(u.shape, s.shape, vh.shape)
    
    topK = vh[:ncomps]
    print(topK)
    print(topK.shape)
    
    #One-time make the matrix of outer products of the topK components 
    all_VVT = np.zeros((vh[0].size,vh[0].size))
    for v in topK:
        all_VVT += np.outer(v,v)
        
    return all_VVT



def subtract_topK_svd(X, all_VVT):
    """
    Using the outer product matrix of topK components,
    subtract from each row (data point) of X
    """
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i] = X[i] - np.dot(all_VVT, X[i]) 
    print(Xnew.shape)
    return Xnew
    
    

  
    
    
    
    
    
    
def Train_and_test_estimators(train_path, test_path, drop_cols_list, label_col_name, supervised_type, models_list, n_iterations=10, svd_n_components=None, pca_n_components=None, make_plots=True, run_mode='train_validate'):
    """
    ...
    supervised_type = 'cassification' vs. 'regression'
    
    """
    #Use from "make_train_csv"
#     train_path = 'EUweb_vs_USAweb_prediction_experiment.csv'
#     test_path = '' #Don;t use this, just remnant from my boilerplate code
    
    
    
    # =============================================================================
    # PARAMETERS 
    # =============================================================================

    SEED = 3172018 #Random seed for repeatbility
    VALIDATION_PCT = .30 #percent of training data to use as validation data



    # =============================================================================
    # LOADING DATA
    # =============================================================================
    train_all = pd.read_csv(train_path)
    train_all.drop(columns=drop_cols_list,inplace=True)
    
    #Drop rows which have nans: / inf:
    # (#replace inf w nan, then drop nans):
    print('train size before remove nans / infs: ', len(train_all))
    #train_all.replace(['inf', np.inf, -np.inf], np.nan)
    #train_all.dropna(axis=0, how='any', inplace=True)
    with pd.option_context('mode.use_inf_as_null', True):
        train_all.dropna(axis=0, how='any', inplace=True)
    print('train size after remove nans / infs: ', len(train_all))
    
    
    # =============================================================================
    # RANDOM SEED FOR REPEATABILITY
    # =============================================================================
    np.random.seed(SEED)
    
    
    # =============================================================================
    # QUICK LOOK AT THE FULL TRAINING DATA
    # =============================================================================
    print(train_all.describe())


    # =============================================================================
    # TRAIN / VALIDATION SPLIT
    # We'll manually do several iterations of train-test split in a K-fold validation setup
    # =============================================================================
    y = train_all[label_col_name].values
    X = train_all.drop(columns=[label_col_name])
    feature_names = train_all.columns
    print('features: ', feature_names)
    print(X.shape)
    
    
    metrics_names = ['roc_auc_score', 'f1_score'] if supervised_type=='classification' else ['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2_score', 'MAPE', 'SMAPE', 'MAAPE']
    inner_dict = {kk:[] for kk in metrics_names}
    metrics_dict = {modelname:copy.deepcopy(inner_dict) for modelname in models_list}
    
    if supervised_type=='regression':
        metrics_dict['baseline'] = copy.deepcopy(inner_dict)
        
    #Could do uniformly at random guessing for the classification baseline...
        
    
    #At inference time, run single iteration using full training data set.
    #Then do the predictions on the text csv.
    if run_mode=='full_inference':
        n_iterations = 1
        test = pd.read_csv(test_path)
        test_drop_cols_list = set(drop_cols_list).intersection(set(test.columns))
        test.drop(columns=test_drop_cols_list,inplace=True)        
        
    #For each iteration:
    for iteration in range(n_iterations):
        print(f'iteration {iteration}')
    
    
        #If doing classification, do stratified train test split
        strat = y if supervised_type=='classification' else None
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=VALIDATION_PCT, stratify=strat)
        #print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)


        #FOr full inference mode, use 100% of the data, rebuild using best model, and predict 
        if run_mode=='full_inference':
            X_train, X_validation, y_train, y_validation = X, X, y, y


        #For regression, do log transform / Box-Cox on y. Especially for prediction for ------, which has big dynamic range.
        #.....


        #If doing SVD topK subtraction to subtract noise:
        # (in this case, unlike usual way of using SVD
        # instead of keeping top K components, we subtract out top 1 or 2, 
        # which in this application is mostly high variance noise):
        if svd_n_components:
            all_VVT = fit_svd(X_train, svd_n_components)
            X_train = subtract_topK_svd(X_train, all_VVT)
            X_validation = subtract_topK_svd(X_validation, all_VVT)
            print(X_train)  


        #If doing PCA on subset of features (last EMBED_DIM elements of the feature vector)
        if pca_n_components:
            EMBED_DIM = 300 #Do PCA on last 300 dimesions of feature vector, which correspond to --- subsapce
            print(f'Doing PCA on last {EMBED_DIM} subset of dimensions of feature vector...')
            X_train_subset = X_train[:, -EMBED_DIM:]
            X_train = X_train[:, :-EMBED_DIM]
            X_validation_subset = X_validation[:, -EMBED_DIM:]
            X_validation = X_validation[:, :-EMBED_DIM]
            pca = PCA(n_components=PCA_N_COMPONENTS)
            pca.fit(X_train_subset)
            X_train_subset = pca.transform(X_train_subset)
            X_validation_subset = pca.transform(X_validation_subset)
            #Now rebuild X_train and X_validation using the PCA reduced feature components:
            X_train = np.hstack((X_train, X_train_subset))
            X_validation = np.hstack((X_validation, X_validation_subset))




        # =============================================================================
        # # Doing standard scaling:
        # Technically this is only helpful for some of the regressors used below.
        # (e.g. it should not affect Extra Trees)
        # But for those where it is not required, it also doesn't hurt, so just use 
        # standardized features from here on.
        # =============================================================================
        sclr = StandardScaler()
        sclr.fit(X_train)
        X_train = sclr.transform(X_train)
        X_validation = sclr.transform(X_validation)





        # =============================================================================
        # EXTRA TREES REGRESSOR
        # Since Extra Trees is fidning thresholds of features to split tree leaves on,
        # we don't actually need to do any feature scaling here, unlike a few regressors
        # tried later.
        # =============================================================================
        if 'ExtraTrees' in models_list:
            if supervised_type == 'classification':
                estimator = ExtraTreesClassifier
            elif supervised_type == 'regression':
                estimator = ExtraTreesRegressor
            print('Running Extra Trees...')
            etr = estimator(n_estimators=400,
                            #criterion='mse',#'mae',
                            min_samples_split=3,
                            min_samples_leaf=2)
            etr.fit(X_train, y_train)

            #Get predicted values for the training set and validation set:
            y_train_pred = etr.predict(X_train)
            y_validation_pred = etr.predict(X_validation)
            #Look at performance accoridng to a few metrics
            print('Training error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_train, y_train_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_train, y_train_pred, 'Training', make_plots)

            print('Validation error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_validation, y_validation_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_validation, y_validation_pred, 'Validation - (estimator)', make_plots)
            for k,v in m.items():
                metrics_dict['ExtraTrees'][k].extend([v])   
            print()
            print('--------------------------------------------------\n')

            #Also look at the feature importances just out of curiosity and as a simple check:
            if make_plots:
                importances  = etr.feature_importances_
                order = np.argsort(importances)[::-1]
                importances = importances[order]
                x = np.arange(importances.size)
                names = feature_names[order]
                plt.figure(figsize=(18,12))
                plt.title('Feature Importances of Extra Trees',fontsize=20)
                plt.bar(x,importances)
                plt.yticks(fontsize=20)
                plt.ylabel('Feature Importance',fontsize=20)
                plt.xticks(x,names,fontsize=20,rotation='vertical')
                plt.tight_layout()
                plt.show()    






        # =============================================================================
        # SUPPORT VECTOR MACHINE REGRESSOR
        # This just uses a hinge loss instead of regular least squares.
        # =============================================================================
        if 'SVM' in models_list:
            print('Running SVM Regressor...')
            svr = SVR(kernel='poly', #kernel='rbf',
                      degree=4,
                      gamma='auto')
            svr.fit(X_train, y_train)

            #Get predicted values for the training set and validation set:
            y_train_pred = svr.predict(X_train)
            y_validation_pred = svr.predict(X_validation)

            #Look at performance according to a few metrics
            print('Training error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_train, y_train_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_train, y_train_pred, 'Training', make_plots)

            print('Validation error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_validation, y_validation_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_validation, y_validation_pred, 'Validation - (estimator)', make_plots)
            for k,v in m.items():
                metrics_dict['SVM'][k].extend([v])   
            print()
            print('--------------------------------------------------\n')    




        # =============================================================================
        # LINEAR REGRESSION
        # =============================================================================
        if 'LinearRegression' in models_list:
            print('Running LinearRegression...')
            estimator = LinearRegression()
            estimator.fit(X_train, y_train)

            #Get predicted values for the training set and validation set:
            y_train_pred = estimator.predict(X_train)
            y_validation_pred = estimator.predict(X_validation)

            #Look at performance according to a few metrics
            print('Training error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_train, y_train_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_train, y_train_pred, 'Training', make_plots)

            print('Validation error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_validation, y_validation_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_validation, y_validation_pred, 'Validation - (estimator)', make_plots)
            for k,v in m.items():
                metrics_dict['LinearRegression'][k].extend([v])   
            print()
            print('--------------------------------------------------\n')




    #     # =============================================================================
    #     # XGBOOST REGRESSOR
    #     # Because XGBoost uses gradient descent, it is helpful to use the standard scaled features
    #     # =============================================================================
        if 'xgboost' in models_list:
            print('Running XGBoost Regressor...')
            xgbr = XGBRegressor(n_estimators=4000)
            xgbr.fit(X_train, y_train)

            #Get predicted values for the training set and validation set:
            y_train_pred = xgbr.predict(X_train)
            y_validation_pred = xgbr.predict(X_validation)

            #Look at performance according to a few metrics
            print('Training error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_train, y_train_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_train, y_train_pred, 'Training', make_plots)

            print('Validation error:')
            if supervised_type == 'classification':
                m = print_classification_performance_report(y_validation, y_validation_pred)
            elif supervised_type == 'regression':
                m = print_regression_performance_report(y_validation, y_validation_pred, 'Validation - (estimator)', make_plots)
            for k,v in m.items():
                metrics_dict['xgboost'][k].extend([v])   
            print()
            print('--------------------------------------------------\n')    

            #Also look at the feature importances just out of curiosity and as a simple check:
            if make_plots:
                
                plot_importance(xgbr)
                
#                 #importances = xgbr.booster.get_score(importance_type='weight')
#                 importances = xgbr.feature_importances_()#get_booster.get_score(importance_type='weight')
#                 names = list(importances.keys())
#                 importances = list(importances.values())
#                 order = np.argsort(importances)[::-1]
#                 importances = np.array(importances)[order]
#                 x = np.arange(importances.size)
#                 names = np.array(names)[order]
#                 plt.figure(figsize=(18,12))
#                 plt.title('Feature Importances of XGBoost Regressor',fontsize=20)
#                 plt.bar(x,importances)
#                 plt.yticks(fontsize=20)
#                 plt.ylabel('Feature Importance',fontsize=20)
#                 plt.xticks(x,names,fontsize=20,rotation='vertical')
#                 plt.tight_layout()
#                 plt.show()








        # =============================================================================
        # SIMPLE NEURAL NETWORK REGRESSOR
        # May as well give it a try...
        # Again, we'll use the standardized features.
        # Make a very simple neural network with just a few small dense layers.
        # =============================================================================
#        if 'NNRegressor' in models_list:
#            print('Running DNN Regressor...')
#            Nfeats=np.shape(X_train)[1]
#            #Define basic neural network with a few dense layers
#            model = Sequential()
#            model.add(Dense(20, input_dim=Nfeats, kernel_initializer='he_normal',
#                            activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
#            model.add(Dropout(0.25))
#            model.add(Dense(20, input_dim=Nfeats, kernel_initializer='he_normal',
#                            activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
#            model.add(Dropout(0.25))
#        #     model.add(Dense(50, input_dim=Nfeats, kernel_initializer='he_normal',
#        #                     activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
#        #     model.add(Dropout(0.25))
#            model.add(Dense(1, kernel_initializer = 'he_normal'))
#            model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
#            #Fit the model:
#            model.fit(X_train, y_train, epochs=400, batch_size=32)
#            score = model.evaluate(X_validation, y_validation)
#            print('\nValidation error:\nMSE: {}'.format(score[1]))
#            print('MAE: {}'.format(score[2]))
#            print('--------------------------------------------------\n')    

          
        
        
#         else:
#             raise Exception('no model type found in MODELS_LIST')        
        
        
            
        # =============================================================================
        #  Compare all against baseline
        # =============================================================================
        print()
        print('vs. mean over all training data:')           
        y_train_mean = np.ones(y_validation.size)*np.mean(y_validation)
        if supervised_type == 'classification':
            m = print_classification_performance_report(y_validation, y_train_mean)
            metrics_dict['baseline']['roc_auc_score'].extend([roc_auc_score])
            metrics_dict['baseline']['f1_score'].extend([f1_score])
        elif supervised_type == 'regression':
            m = print_regression_performance_report(y_validation, y_train_mean, 'Validation - (estimator)', make_plots)
        for k,v in m.items():
            metrics_dict['baseline'][k].extend([v])   
        print()
        print('--------------------------------------------------\n')  

        
    #Also keep track of which features:
    metrics_dict['features'] = feature_names    



        
    # =============================================================================
    # SAVE PREDICTIONS FROM BEST MODEL / COMBINED MODEL
    # (full_inference MODE ONLY)
    # =============================================================================
    if run_mode=='full_inference':
        
        #Many things we could do here, e.g. bag together various estimators, etc.
        #But here just use best individual model, e.g. ExtraTreesRegressor:
        # ...
        best_model = etr
        
        # Transform the test set using the exact same transformation we did on the train set:
        X_test = test.values
    
        #If did SVD topK subtraction during training:
        if svd_n_components:
            all_VVT = fit_svd(X_test, svd_n_components)
            X_test = subtract_topK_svd(X_test, all_VVT)
    
        #If did PCA durign training:
        if pca_n_components:
            #EMBED_DIM = must necessarily be same as during training, defined already in training loop
            print(f'Doing PCA on last {EMBED_DIM} subset of dimensions of feature vector...')
            X_test_subset = X_test[:, -EMBED_DIM:]
            X_test = X_test[:, :-EMBED_DIM]
            
            pca = PCA(n_components=PCA_N_COMPONENTS)
            pca.fit(X_test_subset)
            X_test_subset = pca.transform(X_test_subset)
            #Now rebuild X_test using the PCA reduced feature components:
            X_test = np.hstack((X_test, X_test_subset))
    
        #Using same standard scaling, sclr, that was fit to all training data earlier:
        X_test = sclr.transform(X_test)
        
        #Do final predictions using whatever the final model is (e.g. best single model, or ensemble):
        y_pred_final = best_model.predict(X_test)
        #For some tasks, rounding (.e.g randomized rounding), or other post-processing:
        #y_pred_final = round_to_ints(y_pred_final)
        #... e.g. clip to all be positive
        
        #Save predictions for test set:
        outname = f'test_data__wPredictions_{run_mode}.csv'
        save_predictions(y_pred_final,test,outname)
                
    
    
    return metrics_dict
    
    
    
    
    
    
    
    
# =============================================================================
# DEFINE SOME HELPERS
# =============================================================================

def MAPE(y_true, y_pred):
    #n = y_true.size
    #return (100./n)*np.sum(np.abs((y_true-y_pred)/y_true))
    #definition 2, ignoring y_true=0 since gives INF:
    inds = np.where(y_true != 0.)[0]
    n = inds.size
    return (100./n)*np.sum(np.abs((y_true[inds]-y_pred[inds])/y_true[inds]))
    
def SMAPE(y_true, y_pred):
    n = y_true.size
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) #* .5 #Without the .5 will put on [0%,100%]
    #Avoid div by 0 error:
    inds = np.where(denom>0.)[0]
    num = num[inds]
    denom = denom[inds]
    return (100./n)*np.sum(num/denom)

def MAAPE(y_true, y_pred):
    """
    Calculate the 
    mean arctangent absolute percentage error (MAAPE)
    
    A new metric of absolute percentage error for intermittent demand forecasts
    Sungil Kim, Heeyoung Kim
    2015
    https://www.sciencedirect.com/science/article/pii/S0169207016000121
    
    MAAPE ranges from [0, pi/2], and I do NOT rescale it e.g. to [0,1].
    """
    n = y_true.size
    inside = (y_true-y_pred)/y_true
    #Numerically it's ok if this has np.inf, which will be handled in arctan,
    #but need to make sure that it does not have nans, which would happen if 
    #y_true = y_pred = 0, since then get 0/0, and the MAAPE should then be 0:
    inside = [0. if np.isnan(i) else i for i in inside]
    AAPE = np.arctan(np.abs(inside))
    return (1./n)*np.sum(AAPE)


def print_classification_performance_report(y_true, y_pred):
    AUC = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('roc_auc_score: {}'.format(AUC))
    print('f1_score: {}'.format(f1))
    print('Confusion Matrix:')
    print('{}'.format(confusion_matrix(y_true, y_pred)))
    print()
    print('Classification Report:')
    print('{}'.format(classification_report(y_true, y_pred)))
    print()
    return {'roc_auc_score':AUC, 'f1_score':f1}
    
def print_regression_performance_report(y_true, y_pred, mode, make_plots):
    meanSE = mean_squared_error(y_true, y_pred)
    meanAE = mean_absolute_error(y_true, y_pred)
    medianAE = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    smape = SMAPE(y_true, y_pred)
    maape = MAAPE(y_true, y_pred)
    
    print('mean_squared_error: {}'.format(meanSE))
    print('mean_absolute_error: {}'.format(meanAE))
    print('median_absolute_error: {}'.format(medianAE))
    print('R^2: {}'.format(r2))
    print('MAPE: {}'.format(mape))
    print('SMAPE: {}'.format(smape))
    print('MAAPE: {}'.format(maape))    
    print()
    
    if make_plots:
        
        FONTSIZE = 40
        MARKSIZE = 15
        
        
        #As (sorted) timeseries plot
        y = y_true.copy()
        order = np.argsort(y)
        y = [y[i] for i in order]
        y_pred_ordered = [y_pred[i] for i in order]
        
        plt.figure(figsize=(18,12))
        plt.title(f'Predicted vs. Actual ({mode})',fontsize=FONTSIZE)
        plt.plot(y_pred_ordered,marker='x',color='r', markersize=MARKSIZE, linestyle='None', label='Pred')
        plt.plot(y,marker='o',color='k', markersize=MARKSIZE, linestyle='None', label='True')
        plt.yticks(fontsize=FONTSIZE)
        plt.ylabel('Value',fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.legend(numpoints=1,fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()

        #Scatter plot correlation
        plt.figure(figsize=(18,12))
        plt.title(f'Predicted vs. Actual ({mode})',fontsize=FONTSIZE)
        plt.plot(y_true, y_pred, marker='o',linestyle='None',color='b')
        val_max = max( max(y_true), max(y_pred) )
        plt.plot([0, val_max], [0, val_max], linestyle='--', color='k')
        plt.ylabel('Predicted',fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.xlabel('True',fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()  


        #Residuals w/ histogram:
        x = np.arange(len(y_pred))
        y = y_pred-y_true
        nullfmt = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left_h, bottom, 0.2, height]
        # start with a rectangular Figure
        plt.figure(1, figsize=(18, 18))
        axScatter = plt.axes(rect_scatter)
        axScatter.set_title(f'Residuals ({mode})',fontsize=FONTSIZE)
        axHisty = plt.axes(rect_histy)
        axHisty.yaxis.set_major_formatter(nullfmt)
        axScatter.scatter(x, y)
        # now determine nice limits by hand:
    #     binwidth = 0.25
    #     xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    #     lim = (int(xymax/binwidth) + 1) * binwidth
    #     bins = np.arange(-lim, lim + binwidth, binwidth)
    #     axHisty.hist(y, bins=bins, orientation='horizontal')
        axHisty.hist(y, orientation='horizontal')
        axHisty.set_ylim(axScatter.get_ylim())
        #plt.savefig('rrrrrrrrrrr.png')
        plt.show()    

    #     plt.figure(figsize=(18,12))
    #     plt.title(f'Residuals ({mode})',fontsize=20)
    #     plt.plot(y_pred-y_true,marker='o')
    #     plt.yticks(fontsize=20)
    #     plt.ylabel('Residuals',fontsize=20)
    #     plt.tight_layout()
    #     plt.show()

    return {'mean_squared_error':meanSE, 'mean_absolute_error':meanAE, 'median_absolute_error':medianAE, 'r2_score':r2, 'MAPE':mape, 'SMAPE':smape, 'MAAPE':maape}

    
    
    
def plot_all_folds_metrics(metrics_dict):
    """
    Look at distributon of metrics over all train-val splits
    """
    
    FONTSIZE = 40
    
    model_names = list(metrics_dict.keys())
    #Since also tracking features:
    if 'features' in model_names:
        model_names.remove('features')
        
    metrics_names = list(metrics_dict[model_names[0]].keys())
    nsplits = len( list(metrics_dict[model_names[0]][metrics_names[0]]) )
    

    MARKERSIZE = 15
    zero_min_bound_metrics = ['roc_auc_score', 'f1_score', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error']
    
    for metric in metrics_names:
        
        plt.figure(figsize=(18,12))
        plt.title(f'{metric} over {nsplits} splits',fontsize=FONTSIZE)
        
        all_means = []
        for mm, model in enumerate(model_names):
            y = metrics_dict[model][metric]
            median = np.median(y)
            mean = np.mean(y)
            all_means.extend([mean])
            ymin = np.min(y)
            ymax = np.max(y)
            sd = np.std(y)
            plt.plot(mm, median, marker='o', color='b', markersize=MARKERSIZE)
            plt.plot(mm, mean, marker='s', color='k', markersize=MARKERSIZE)
            plt.plot(mm, ymin, marker='v', color='g', markersize=MARKERSIZE)
            plt.plot(mm, ymax, marker='^', color='r', markersize=MARKERSIZE)
            plt.errorbar(mm, mean, yerr=sd, ecolor='k', capsize=15) #yerr=[[mean-sd], [mean+sd]]

        #So the legend only has 1 entry per 'median', per 'mean' etc.:    
        plt.plot(mm, median, marker='o', color='b', label='median', markersize=MARKERSIZE)
        plt.plot(mm, mean, marker='s', color='k', label='mean', markersize=MARKERSIZE)
        plt.plot(mm, ymin, marker='v', color='g', label='min', markersize=MARKERSIZE)
        plt.plot(mm, ymax, marker='^', color='r', label='max', markersize=MARKERSIZE)

        #Line connecting means so see which is higher/lower:
        x = [i for i in range(len(model_names))]
        plt.plot(x, all_means, marker='s', color='k', linestyle='-', label=None)
        plt.ylabel(f'{metric}',fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.xlabel(f'{model}',fontsize=FONTSIZE)
        plt.xticks(x, model_names, rotation='vertical', fontsize=FONTSIZE)
        if metric in zero_min_bound_metrics:
            ylims = plt.ylim()
            plt.ylim(min(0.,ylims[0]),None)
        plt.legend(numpoints=1, fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()        
    

    
def round_to_ints(vector):
    #Since the actual count numbers are integers, it may be more meaningful to round
    return vector.astype(int)


def save_predictions(y_pred,df_test,outname):
    #y_pred__rounded = round_to_ints(y_pred)  
    #df_test['PREDICTIONS'] = y_pred__rounded
    df_test['PREDICTIONS'] = y_pred
    df_test.to_csv(outname, index=False)    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def svd_experiment(n_components_list, train_path, test_path, drop_cols_list, label_col_name, supervised_type, models_list, n_iterations=10):
    """
    SVD experiment
    """
    
    MARKERSIZE = 15
    zero_min_bound_metrics = ['roc_auc_score', 'f1_score', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error']
        
    all_comps_dict = {}
    for ncomp in n_components_list:
        metrics_dict = Train_and_test_estimators(train_path, test_path, drop_cols_list, label_col_name, supervised_type, models_list, n_iterations=n_iterations, svd_n_components=ncomp, make_plots=False)
        all_comps_dict[ncomp] = metrics_dict
        
    #Plot for each model, for each metric: the metric as a function of number of components subtracted
    model_names = list(metrics_dict.keys())
    metrics_names = list(metrics_dict[model_names[0]].keys())
    nsplits = len( list(metrics_dict[model_names[0]][metrics_names[0]]) )
    
    for metric in metrics_names:    

        for model in model_names:
            
            plt.figure(figsize=(18,12))
            plt.title(f'{metric} over {nsplits} splits\n{model}',fontsize=20)            
            all_means = []
            for mm, ncomp in enumerate(n_components_list):
                y = all_comps_dict[ncomp][model][metric]
                median = np.median(y)
                mean = np.mean(y)
                all_means.extend([mean])
                ymin = np.min(y)
                ymax = np.max(y)
                sd = np.std(y)
                plt.plot(ncomp, median, marker='o', color='b', markersize=MARKERSIZE)
                plt.plot(ncomp, mean, marker='s', color='k', markersize=MARKERSIZE)
                plt.plot(ncomp, ymin, marker='v', color='g', markersize=MARKERSIZE)
                plt.plot(ncomp, ymax, marker='^', color='r', markersize=MARKERSIZE)
                plt.errorbar(ncomp, mean, yerr=sd, ecolor='k', capsize=15) #yerr=[[mean-sd], [mean+sd]]

        #So the legend only has 1 entry per 'median', per 'mean' etc.:    
        plt.plot(ncomp, median, marker='o', color='b', label='median', markersize=MARKERSIZE)
        plt.plot(ncomp, mean, marker='s', color='k', label='mean', markersize=MARKERSIZE)
        plt.plot(ncomp, ymin, marker='v', color='g', label='min', markersize=MARKERSIZE)
        plt.plot(ncomp, ymax, marker='^', color='r', label='max', markersize=MARKERSIZE)

        #Line connecting means so see which is higher/lower:
        plt.plot(n_components_list, all_means, marker='s', color='k', linestyle='-', label=None)
        plt.ylabel(f'{metric}',fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(f'Number of SVD Components',fontsize=20)
        plt.xticks(n_components_list, fontsize=20)
        if metric in zero_min_bound_metrics:
            ylims = plt.ylim()
            plt.ylim(min(0.,ylims[0]),None)
        plt.legend(numpoints=1, fontsize=20)
        plt.tight_layout()
        plt.show() 
    
    return all_comps_dict

    








if __name__ == "__main__":
    
    TRAIN_PATH = 'training_featurized.csv'
    TEST_PATH = 'test_featurized.csv' #''
    LABEL_COL_NAME = 'y'
    DROP_COLS_LIST = [] #['f4', 'f7', 'f22'] #Names of FEATURES to drop, if any
    MODELS_LIST = ['ExtraTrees', 'LinearRegression']#, 'xgboost', 'SVM', 'NNRegressor']
    SVD_N_COMPONENTS = None #Leave as None to not do SVD noise subtraction
    PCA_N_COMPONENTS = None #Leave as None to not do any PCA on any subsets of dimensions of the feature vector
    MAKE_PLOTS = True#True#False
    N_ITERATIONS = 20 #Number of folds of cross-validation
    RUN_MODE = 'train_validate' #'full-inference'
    SUPERVISED_TYPE = 'regression' #'classification'
    
    
    
    
    #For large scale experiments with a lot of model / feature processing variations:
    #(variaitons tried indepdently not together):
    #E.g. to do different number PCA components and drop some different feature:
    #variation1 = {'PCA_N_COMPONENTS':None, 'DROP_COLS_LIST':['f7']}
    #variation2 = {'PCA_N_COMPONENTS':3, 'DROP_COLS_LIST':['f8']}
    #VARIATIONS = [{'v1__pcaNone_f7': variation1}, {'v2__pca3_f8':variation2}]
    VARIATIONS = [{}] #If want to just use standard setup, i.e. no experiments, just leave as [{}]
    
    #Run the ML pipeline [train/validation mode]:
    print('Starting training/validation mode...')
    for variation in VARIATIONS:
        #Get all relevant variables for this particular variation, e.g. if 
        #experimenting w/ diff number of PCA components, different scaling, 
        #different feature subsets, etc., e.g.:
        #PCA_N_COMPONENTS = variation.values['PCA_N_COMPONENTS']
        #DROP_COLS_LIST = variation.values['DROP_COLS_LIST']

        
        #E.g. to get rid of all ------- embedding features:
        # df = pd.read_csv(TRAIN_PATH)
        # cols = df.columns.tolist()
        # cols = [i for i in cols if i.startswith('g_ave')]
        # DROP_COLS_LIST += cols        

        
        metrics_dict__train_validate = Train_and_test_estimators(TRAIN_PATH, TEST_PATH, DROP_COLS_LIST, LABEL_COL_NAME, SUPERVISED_TYPE, MODELS_LIST, n_iterations=N_ITERATIONS, svd_n_components=SVD_N_COMPONENTS, pca_n_components=PCA_N_COMPONENTS, make_plots=MAKE_PLOTS, run_mode=RUN_MODE)
        with open(f'metrics_dict__{RUN_MODE}.pickle', 'wb') as gg:
            pickle.dump(metrics_dict__train_validate, gg)
        plot_all_folds_metrics(metrics_dict__train_validate)
        #Print out a few of the metrics we really care about for this problem:
        #(The actual metrics that are calculated are hardcoded, so the metrics listed here should be a subset of those calculated emtrics)
        IMPORTANT_METRICS_LIST = ['median_absolute_error', 'MAPE', 'MAAPE', 'SMAPE'] #['roc_auc_score', 'f1_score']
        for estimator_name in MODELS_LIST: 
            for met in IMPORTANT_METRICS_LIST: 
                values = metrics_dict__train_validate[estimator_name][met]
                print(f'mean {met} over {N_ITERATIONS} folds, for {PCA_N_COMPONENTS} PCA comps = ', np.mean(values))
            print()
        print('vs. baseline:')
        for met in IMPORTANT_METRICS_LIST: 
            values = metrics_dict__train_validate['baseline'][met]
            print(f'mean {met} over {N_ITERATIONS} folds, for {PCA_N_COMPONENTS} PCA comps = ', np.mean(values))
        print('\n'*5)             
   



    #Based on the results, decide the best estimator/variation....
    #And now do inference using the best estimator/variation


    #Now that best model and features/scaling variation was chosen via K-fold cross-validation:        
    #Do predictions after training best model on WHOLE 100% of training data, and save out predictions:
    print()
    print('Starting full inference mode...')
    RUN_MODE  = 'full_inference'
    metrics_dict__inference = Train_and_test_estimators(TRAIN_PATH, TEST_PATH, DROP_COLS_LIST, LABEL_COL_NAME, SUPERVISED_TYPE, MODELS_LIST, n_iterations=N_ITERATIONS, svd_n_components=SVD_N_COMPONENTS, pca_n_components=PCA_N_COMPONENTS, make_plots=MAKE_PLOTS, run_mode=RUN_MODE)
    with open(f'metrics_dict__{RUN_MODE}.pickle', 'wb') as gg:
        pickle.dump(metrics_dict__inference, gg)
    plot_all_folds_metrics(metrics_dict__inference)