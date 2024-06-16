# -*- coding: utf-8 -*-
"""
Script that uses the data for classification, using neural networks (Multi Layer Perceptrons)
"""

# Modules to use
import numpy as np
import os
from sklearn import svm
import pandas as pd
import sys
sys.path.append('Code MLB')


# Code to get the performance of each dataset using SVM classifier

##############The best parameters for SVM #############
# kernel = 'rbf'
# C = 100
# gamma = 0.00001

####################################################
# The rest of the parameters are the default ones --> see the sklearn website for MLP classifier

# Adapted from:
# https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
# This function can be imported and used with other classifiers


all_evaluations = {}    
datasets = os.listdir('Data Classification')
target = 'BRAIN REGION'
for dataset in datasets:
    data = pd.read_csv('Data Classification\\'+dataset)
    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    labels.sort()
    sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3)
    SVM = svm.SVC(kernel ='rbf', C=100, gamma=0.00001)

    tst_y, prd_tst, evaluation = splitclassify(SVM, sss, X, y)
    name = dataset[5:-4]
    performance(labels, tst_y, prd_tst, evaluation, name)
    all_evaluations[name] = evaluation
    print('Done dataset : '+ dataset +'\n')

# Generates plot with the performance of MLP over all datasets (in train and test)
performance_overall(all_evaluations)

