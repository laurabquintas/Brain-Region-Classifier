# -*- coding: utf-8 -*-
"""
Script that uses the data for classification, using Decision trees
"""

# Modules to use
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import sys
sys.path.append('Code MLB')
from performance import *

### PARAMETERS OF DECISION TREES CLASSIFIER CONSIDERED ###
# criteria = 'gini', 'entropy' (the first article used only gini and the second both)
# depth = '3' (used in the articles)
# minimum impurity decrease = 0.0025, 0.001, 0.0005 (common values used for min_i_d)

# The rest of the parameters are the default ones --> see the sklearn website for MLP classifier


# Code to get the performance of each dataset using DT classifier

##############The best parameters for DT#############

d = 3 #max_depth
f = 'entropy'  #criteria
i_d = 0.0005 #minimum impurity decrease

####################################################
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
    dtClf= DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=i_d,class_weight='balanced')
        

    tst_y, prd_tst, evaluation = splitclassify(dtClf, sss, X, y)
    name = dataset[5:-4]
    performance(labels, tst_y, prd_tst, evaluation, name)
    all_evaluations[name] = evaluation
    print('Done dataset : '+ dataset +'\n')

# Generates plot with the performance of DT over all datasets (in train and test)
performance_overall(all_evaluations)


