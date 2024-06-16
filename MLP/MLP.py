# -*- coding: utf-8 -*-
"""
Script that uses the data for classification, using neural networks (Multi Layer Perceptrons)
"""

# Modules to use
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pandas as pd
import sys
sys.path.append('Code MLB')
from ..performance import *
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
### PARAMETERS OF MLP CLASSIFIER CONSIDERED ###
# hidden_layer_sizes = (10,10)
# activation = 'relu' (based on article)
# solver = 'adam' (based on article)
# learning_rate_init = 0.01 (based on article)
# maximum iterations = 500 (based on article)

# The rest of the parameters are the default ones --> see the sklearn website for MLP classifier

# Adapted from:
# https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
# This function can be imported and used with other classifiers


# Code to get the performance of each dataset using MLP classifier
# Editing the classifier part (lines 91-100), the rest can be used for other classifiers
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
    mlp = MLPClassifier(activation='relu', 
                        solver='adam', 
                        learning_rate_init=0.01, 
                        max_iter=500, 
                        verbose=False)
    #scrs = cross_validate(mlp, X, y, cv = sss, 
                          # scoring=('accuracy','f1_weighted','recall_weighted','precision_weighted'),
                          # return_train_score=True)
    tst_y, prd_tst, evaluation = splitclassify(mlp, sss, X, y)
    name = dataset[5:-4]
    performance(labels, tst_y, prd_tst, evaluation, name)
    all_evaluations[name] = evaluation
    print('Done dataset : '+ dataset +'\n')

# Generates plot with the performance of MLP over all datasets (in train and test)
performance_overall(all_evaluations)

