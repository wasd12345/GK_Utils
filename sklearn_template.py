# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:33:27 2018
Python 3.6.3 64bits
@author: GK
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
#from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import sys



def Train_and_test_regressors(train_path,test_path):
    
    # =============================================================================
    # PARAMETERS 
    # =============================================================================

    SEED = 3172018 #Random seed for repeatbility
    VALIDATION_PCT = .25 #percent of training data to use as validation data
    MAKE_PLOTS = True #False#True


    # =============================================================================
    # LOADING DATA
    # =============================================================================
    train_all = pd.read_csv(TRAIN_PATH)#[3176 rows x 14 columns]
    test = pd.read_csv(TEST_PATH)#[794 rows x 13 columns]
    
    
    # =============================================================================
    # RANDOM SEED FOR REPEATABILITY
    # =============================================================================
    np.random.seed(SEED)
    
    
    # =============================================================================
    # QUICK LOOK AT THE FULL TRAINING DATA
    # =============================================================================
    print(train_all.describe())
    #if MAKE_PLOTS:
	
	

    
    # =============================================================================
    # TRAIN / VALIDATION SPLIT
    # To decide which estimator is best, do an empirical test on an independent validation set
    # Could do K-fold cross validation, but single fold should be fine too.
    # =============================================================================
    y = train_all['groundtruth_walkin'].values
    X = train_all.values[:,:-1]
    feature_names = train_all.columns
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=VALIDATION_PCT)
    #print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)#(2382, 13) (794, 13) (2382,) (794,)
    
    
    
     
    
    
    
    
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
    print('Running Extra Trees Regressor...')
    etr = ExtraTreesRegressor(n_estimators=400,
                              criterion='mse',#'mae',
                              min_samples_split=3,
                              min_samples_leaf=2)
    etr.fit(X_train, y_train)
    
    #Get predicted values for the training set and validation set:
    y_train_pred = etr.predict(X_train)
    y_validation_pred = etr.predict(X_validation)
    #Look at performance accoridng to a few metrics
    print('Training error:')
    print_performance_report(y_train, y_train_pred)
    print('Validation error:')
    print_performance_report(y_validation, y_validation_pred)
    print('--------------------------------------------------\n')
       
    #Also look at the feature importances just out of curiosity and as a simple check:
    if MAKE_PLOTS:
        importances = etr.feature_importances_
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
    print('Running SVM Regressor...')
    svr = SVR(kernel='rbf',
              degree=4,
              gamma='auto')
    svr.fit(X_train, y_train)
    
    #Get predicted values for the training set and validation set:
    y_train_pred = svr.predict(X_train)
    y_validation_pred = svr.predict(X_validation)
    #Look at performance accoridng to a few metrics
    print('Training error:')
    print_performance_report(y_train, y_train_pred)
    print('Validation error:')
    print_performance_report(y_validation, y_validation_pred)
    print('--------------------------------------------------\n')    
    
    
    
    
    
    
    # =============================================================================
    # XGBOOST REGRESSOR
    # Because XGBoost uses gradient descent, it is helpful to use the standard scaled features
    # =============================================================================
    print('Running XGBoost Regressor...')
    xgbr = XGBRegressor(n_estimators=400)
    xgbr.fit(X_train, y_train)
    
    #Get predicted values for the training set and validation set:
    y_train_pred = xgbr.predict(X_train)
    y_validation_pred = xgbr.predict(X_validation)
    #Look at performance according to a few metrics
    print('Training error:')
    print_performance_report(y_train, y_train_pred)
    print('Validation error:')
    print_performance_report(y_validation, y_validation_pred)
    print('--------------------------------------------------\n')
    
    #Also look at the feature importances just out of curiosity and as a simple check:
    if MAKE_PLOTS:
        importances = xgbr.booster().get_score(importance_type='weight')
        names = list(importances.keys())
        importances = list(importances.values())
        order = np.argsort(importances)[::-1]
        importances = np.array(importances)[order]
        x = np.arange(importances.size)
        names = np.array(names)[order]
        plt.figure(figsize=(18,12))
        plt.title('Feature Importances of XGBoost Regressor',fontsize=20)
        plt.bar(x,importances)
        plt.yticks(fontsize=20)
        plt.ylabel('Feature Importance',fontsize=20)
        plt.xticks(x,names,fontsize=20,rotation='vertical')
        plt.tight_layout()
        plt.show()

        
        
    
    

    
    
    # =============================================================================
    # SIMPLE NEURAL NETWORK REGRESSOR
    # May as well give it a try...
    # Again, we'll use the standardized features.
    # Make a very simple neural network with just a few small dense layers.
    # =============================================================================
    print('Running DNN Regressor...')
    Nfeats=np.shape(X_train)[1]
    #Define basic neural network with a few dense layers
    model = Sequential()
    model.add(Dense(50, input_dim=Nfeats, kernel_initializer='he_normal',
                    activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(Dropout(0.25))
    model.add(Dense(100, input_dim=Nfeats, kernel_initializer='he_normal',
                    activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(Dropout(0.25))
    model.add(Dense(50, input_dim=Nfeats, kernel_initializer='he_normal',
                    activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(Dropout(0.25))
    model.add(Dense(1, kernel_initializer = 'he_normal'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    #Fit the model:
    model.fit(X_train, y_train, epochs=1000, batch_size=32)
    score = model.evaluate(X_validation, y_validation)
    print('\nValidation error:\nMSE: {}'.format(score[1]))
    print('MAE: {}'.format(score[2]))
    print('--------------------------------------------------\n')    
    
    
    
    
        
    # =============================================================================
    # SAVE PREDICTIONS FROM BEST MODEL / COMBINED MODEL
    # =============================================================================
    #Many things we could do here, e.g. bag together various estimators, etc.
    #But I'll just use my best individual model, the ExtraTreesRegressor.
        
    #Transform the test set using the exact same transformation we did on the train set:
    X_test = test.values
    X_test = sclr.transform(X_test)
    y_pred_final = etr.predict(X_test)

    #Save predictions for test set:
    outname = 'test_data__wPredictions.csv'
    save_predictions(y_pred_final,test,outname)
    
    
    
	
	
    
    
# =============================================================================
# DEFINE SOME HELPERS
# =============================================================================
def print_performance_report(y_true, y_pred):
    print('mean_squared_error: {}'.format(mean_squared_error(y_true, y_pred)))
    print('mean_absolute_error: {}'.format(mean_absolute_error(y_true, y_pred)))
    print('median_absolute_error: {}'.format(median_absolute_error(y_true, y_pred)))
    print('R^2: {}'.format(r2_score(y_true, y_pred)))
    #print('Mutual Information: {}'.format(mutual_info_regression(y_true, y_pred)))

def round_to_ints(vector):
    #Since the actual count numbers are integers, it may be more meaningful to round
    return vector.astype(int)


def save_predictions(y_pred,df_test,outname):
    y_pred__rounded = round_to_ints(y_pred)  
    df_test['PREDICTIONS'] = y_pred__rounded
    df_test.to_csv(outname, index=False)    
    
	
	

if __name__ == "__main__":
    #Run the main function, which will train several regressors, do some basic
    #analysis, output saved results.
    TRAIN_PATH = sys.argv[1]
    TEST_PATH = sys.argv[2]   
#    TRAIN_PATH = r"train_data.csv"
#    TEST_PATH = r"test_data.csv"     
    Train_and_test_regressors(TRAIN_PATH,TEST_PATH)