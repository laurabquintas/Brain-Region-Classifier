# -*- coding: utf-8 -*-
"""
This python script performs the ranking and analysis of the hyperparameters 
sets for the Decision tree classifier

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
import mlb_charts as mlb
sys.path.append('Code MLB')
from mlb_charts import  multiple_line_chart
from sklearn.tree import DecisionTreeClassifier
from statistics import mean
import matplotlib.pyplot as plt
from ..performance import *


### PARAMETERS OF DECISION TREES CLASSIFIER CONSIDERED ###
# criteria = 'gini', 'entropy' (the first article used only gini and the second both)
# depth = '3' (used in the articles)
# minimum impurity decrease = 0.0025, 0.001, 0.0005 (common values used for min_i_d)

datasets = os.listdir('Data Classification')

# ranking the set of paremeters in order to choose the best set
min_impurity_decrease = [ 0.001, 0.0005,0.0025]
max_depths = [3]
criteria = ['gini','entropy']

all_evaluations = {}
rank = {} #for each dataset a list of the ranks of each set of parameters
names =[]


plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(16, 4), squeeze=False)


for dataset in datasets:
    name = dataset[5:-4] 
    names += [name]
    i=0
    data = pd.read_csv('Data Classification'+dataset)

    set_performance = {}

    #create the splits for each dataset and use the same to compare each set of parameter
    splits =StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    y: np.ndarray = data.pop('BRAIN REGION').values
    X: np.ndarray = data.values
    
    for k in range(len(criteria)):
        
        f=criteria[k]

        for d in max_depths:
            
            yvalues =[]

            for i_d in min_impurity_decrease:                

                dtClf= DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=i_d,class_weight='balanced')
                dt_cv = cross_validate( dtClf, X, y, cv =splits, scoring=('accuracy','f1_weighted','recall_weighted','precision_weighted'), return_train_score=True)

                accuracy_train =round(mean(dt_cv['train_accuracy']),2) 
                accuracy_test =round(mean(dt_cv['test_accuracy']),2)
                recall_train =round(mean(dt_cv['train_recall_weighted']),2)
                recall_test = round(mean(dt_cv['train_recall_weighted']),2)
                f1_train = round(mean(dt_cv['train_f1_weighted']),2)
                f1_test = round(mean(dt_cv['test_f1_weighted']),2)
                precision_train = round(mean(dt_cv['train_precision_weighted']),2)
                precision_test = round(mean(dt_cv['test_precision_weighted']),2)


                parameters = f + '_' + str(d) + '_'+ str(i_d)
                set_performance[parameters] = np.mean(np.array([accuracy_test,recall_test, f1_test, precision_test]))

    ordered_performance= list(dict(sorted(set_performance.items(), key=lambda x:x[1],reverse=True)))
    for element in ordered_performance:

        if element in rank:

            rank[element].append(ordered_performance.index(element)+1)
        else:

            rank[element] = [ordered_performance.index(element)+1]
            
print(rank)

#multiple_line_chart(names, rank, title=f'Rank of the set of parameters for Decision Trees in each dataset ',
                              #xlabel='dataset', ylabel='rank')

#uncomment this line to save the final plot (pls change the directory)
#plt.savefig('Results DT\\DT_rank_parameters_performance.png', dpi=300)
