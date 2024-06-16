# -*- coding: utf-8 -*-
"""
Script that have the functions to build the train and test cross validation datasets and also to provide all functions that compute the results and plots obtained that were included in both our presentation and report 

"""

# Modules to use
import numpy as np
from statistics import mean
import os
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('Code MLB')
import mlb_charts as mlb

# Adapted from:
# https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
# This function can be imported and used with other classifiers

# function used to perform the cross validation k fold using stratified shuffle splitter
def splitclassify(clf, splt, X, y):
    
    y_test_cnfmtx = np.empty([0], dtype=str)
    y_prd_cnfmtx  = np.empty([0], dtype=str)
    
    accuracy, precision, recall, f1 = {'trn':[],'tst':[]}, {'trn':[],'tst':[]}, {'trn':[],'tst':[]} ,{'trn':[],'tst':[]}

    for train_ndx, test_ndx in splt.split(X, y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        y_test_cnfmtx = np.append(y_test_cnfmtx, test_y)

        clf.fit(train_X, train_y)
        prd_trn = clf.predict(train_X)
        prd_tst = clf.predict(test_X)
        
        y_prd_cnfmtx = np.append(y_prd_cnfmtx, prd_tst)
        
        trn_precision, trn_recall, trn_f1score, trn_support = precision_recall_fscore_support(train_y, prd_trn, average = 'weighted')
        tst_precision, tst_recall, tst_f1score, tst_support = precision_recall_fscore_support(test_y, prd_tst, average = 'weighted')
        trn_accuracy = accuracy_score(train_y, prd_trn)
        tst_accuracy = accuracy_score(test_y, prd_tst)
        
        accuracy['trn']+=[trn_accuracy]
        accuracy['tst']+= [tst_accuracy]
        f1['trn']+= [trn_f1score]
        f1['tst']+= [tst_f1score]
        recall['trn']+= [trn_recall]
        recall['tst']+= [tst_recall]
        precision['trn']+= [trn_precision]
        precision['tst']+= [tst_precision]
        
    evaluation = {'Accuracy': [round(mean(accuracy['trn']),2), round(mean(accuracy['tst']),2)],
                  'Recall': [round(mean(recall['trn']),2), round(mean(recall['tst']),2)],
                  'F1-Score': [round(mean(f1['trn']),2), round(mean(f1['tst']),2)],
                  'Precision': [round(mean(precision['trn']),2), round(mean(precision['tst']),2)]}
    
    return y_test_cnfmtx, y_prd_cnfmtx, evaluation


# function to obtain for each dataset the 4 performance scores and the correspondent confusion matrix
def performance(labels, tst_y, prd_tst, scores, name):
    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
    
    fig, axs = plt.subplots(1, 2, figsize=(2 * mlb.HEIGHT, mlb.HEIGHT))
    mlb.multiple_bar_chart(['Train', 'Test'], scores, ax=axs[0],
                       percentage=True, location='lower center')
    mlb.plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')
    # Optional (if you desire to save the results in a picture) -> uncomment this line and add the desired destination directory
   # plt.savefig('insertdirectory'+ name + '_performance.png', dpi = 300)
    

# Function to obtain a plot of all the performance scores for all data sets for the same classifier
def performance_overall(all_evals: dict):
    
    train_values = {'Accuracy':[],'Recall':[],'F1-Score':[],'Precision':[]}
    test_values = {'Accuracy':[],'Recall':[],'F1-Score':[],'Precision':[]}
    
    for dataset in all_evals:
        evaluation = all_evals[dataset]
        for estimator in train_values: #can be test_values as well
            train_values[estimator].append(evaluation[estimator][0])
            test_values[estimator].append(evaluation[estimator][1])
            
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(14, 5*2), squeeze=False)
    mlb.multiple_bar_chart(list(all_evals.keys()), train_values, ax=axs[0,0], percentage=True, location='lower center', xlabel='Train')
    mlb.multiple_bar_chart(list(all_evals.keys()), test_values, ax=axs[1,0], percentage=True, location='lower center', xlabel='Test')
    # Optional (if you desire to save the results in a picture) -> uncomment this line and add the desired destination directory
    #plt.savefig('insert directory'\\MLP_performance.png', dpi=300)

    
    # missing rank function 




