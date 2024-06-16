# -*- coding: utf-8 -*-
"""
This python script performs the ranking and analysis of the hyperparameters 
sets for the SVM classifier

For each set fo parameters the average of the four scores is computed for each 
dataset. Then in each dataset the hyperparameter sets are ranked 
therefore producing the final plot

"""

### Classification of the data using Decison Trees 

#import modules
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('Code MLB')
from mlb_charts import  multiple_line_chart
from sklearn import svm
import matplotlib.pyplot as plt
from statistics import mean
from ..performance import *
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

### PARAMETERS OF DECISION TREES CLASSIFIER CONSIDERED ###
# kernell = 'rgd' (used in the article)
# C values = [0.1,1,10,100] (used in the articles)
# gamma_ values = [0.01,0.001,0.0001,0.00001] (used in the article)
datasets = os.listdir('Data Classification')

# ranking the set of paremeters in order to choose the best set
C_val= [0.1,1,10,100]

gamma_val =[0.01,0.001,0.0001,0.00001]


all_evaluations = {}
rank = {} #for each dataset a list of the ranks of each set of parameters
names =[] #names of the datasets


plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(16, 4), squeeze=False)


for dataset in datasets[0]:
    name = dataset[5:-4] 
    names += [name]
    i=0
    data = pd.read_csv('Data Classification\\'+dataset)

    performances = {}

    #create the splits for each dataset and use the same to compare each set of parameter
    splits =StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    y: np.ndarray = data.pop('BRAIN REGION').values
    X: np.ndarray = data.values
    
    for c in C_val:

        for g in gamma_val:
                      

             #building the classifier (ALTER HERE)
            dtClf= svm.SVC(kernel ='rbf', C=c, gamma=g)
            #cross validate the results (ALTER HERE)
            dt_cv = cross_validate( dtClf, X, y, cv =splits, scoring=('accuracy','f1_weighted','recall_weighted','precision_weighted'), return_train_score=True)

            #stores the results in variables (only test parameters are being used)

            accuracy_test =round(mean(dt_cv['test_accuracy']),2)
            recall_test = round(mean(dt_cv['train_recall_weighted']),2)
            f1_test = round(mean(dt_cv['test_f1_weighted']),2)
            precision_test = round(mean(dt_cv['test_precision_weighted']),2)

            #creates a string for each set os parameters that will be displayed as label in the multipleline plot (ALTER HER)
            parameters = 'C_'+str(c) + 'g_'+ str(g)

            #computes the mean for the four parameters of permformance
            performances[parameters] = np.mean(np.array([accuracy_test,recall_test, f1_test, precision_test]))
    
    #order the dict in descending order of performance and transforms it into a list
    ordered_performance= list(dict(sorted(performances.items(), key=lambda x:x[1],reverse=True)))
    
    #cycle to obtain the rank of each set of parameters (by getting its index in the ordered list)
    for element in ordered_performance:

        if element in rank:

            rank[element].append(ordered_performance.index(element)+1)
        else:

            rank[element] = [ordered_performance.index(element)+1]


print(rank)
#plt.figure()
#fig, axs = plt.subplots(1, 1, figsize=(16, 4), squeeze=False) 
#multiple_line_chart(names, rank, title='SVM', 
                              # xlabel='dataset', ylabel='rank')


#uncomment this line to store the ran for SVM parameters
#plt.savefig('Results SVM v2\\SVM_rank.png', dpi=300)