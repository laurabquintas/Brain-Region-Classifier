# -*- coding: utf-8 -*-
"""
Script that uses the data for classification, using KNN (k-nearest neighbour)

"""

# Modules to use
from ..performance import *
import numpy as np
import os
import pandas as pd
import sys
import pickle
sys.path.append('Code MLB')


### PARAMETERS OF KNN CLASSIFIER CONSIDERED ###
# k number - we will test several k numbers
# weights - uniform (based on article) 
# p - euclidean distance (based on article)

# The rest of the parameters are the default ones --> see the sklearn website for KNN classifier


# Code to get the performance of each dataset using KNN classifier

all_evaluations = {}    
datasets = os.listdir('Data Classification')
target='BRAIN REGION'

krange = range(1, 31)
store_train_evaluations=[]
store_test_evaluations=[]
for  i in range(4):
    store_train_evaluations.append([])
    store_test_evaluations.append([])

# linhas = k, colunas = type of evaluation

for dataset in datasets:
    for k in krange:
        data = pd.read_csv('Data Classification\\'+dataset)
        y: np.ndarray = data.pop(target).values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(y)
        labels.sort()
        sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3)
        knn = KNeighborsClassifier(n_neighbors = k, 
                            weights = 'uniform', 
                            p = 2)
        
        tst_y, prd_tst, evaluation = splitclassify(knn, sss, X, y)
        
        # plot performance of dataset
        name = dataset[5:-4]
        
        # evaluation = performance(labels, trnY, prd_trn, tstY, prd_tst, name)
        performance(labels, tst_y, prd_tst, evaluation, name)
        all_evaluations[name] = evaluation
        print('Done dataset : '+ dataset +'\n')
        
        store_train_evaluations[0].append(evaluation['Accuracy'][0])
        store_test_evaluations[0].append(evaluation['Accuracy'][1])
        store_train_evaluations[1].append(evaluation['Recall'][0])
        store_test_evaluations[1].append(evaluation['Recall'][1])
        store_train_evaluations[2].append(evaluation['F1-Score'][0])
        store_test_evaluations[2].append(evaluation['F1-Score'][1])
        store_train_evaluations[3].append(evaluation['Precision'][0])
        store_test_evaluations[3].append(evaluation['Precision'][1])
        
#store results
filetrain = open('train_evaluation', 'wb')
pickle.dump(store_train_evaluations, filetrain)
filetrain.close

filetest = open('test_evaluation', 'wb')
pickle.dump(store_test_evaluations, filetest)
filetest.close
    
# Generates plot with the performance of MLP over all datasets (in train and test)
performance_overall(all_evaluations)


