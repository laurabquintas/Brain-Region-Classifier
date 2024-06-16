# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:38:56 2022

@author: 35193
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('Code CD')

#import the files with performance measures for the train and test datasets

filetrain = open('train_evaluation', 'rb')
train_data = pickle.load(filetrain)
filetrain.close()

filetest = open('test_evaluation', 'rb')
test_data = pickle.load(filetest)
filetrain.close()

datasets=['logCPM', 'logCPM_centered', 'logCPM_FS', 'logCPM_centered_FS', 'logCPM_zscore', 'raw_data', 'raw_data_filtered', 'raw_data_filtered_FS']
evs=['Accuracy', 'Recall', 'F1-Score', 'Precision']

i=0
j=30
n=0

while n<8:   #all performance evaluations for the train and test datasets
    
    fig, axs = plt.subplots(4, 2)
    fig.set_size_inches(17,10)
    
    print(datasets[n])
    
    #select the the values from all values of k for a specific dataset and evaluation
    y_train = train_data[0][i:j]
    y_test = test_data[0][i:j]
    

    print('The maximum accuracy of the train dataset is at k=' + str(y_train.index(max(y_train))+1))
    print('The maximum accuracy of the test dataset is at k=' + str(y_test.index(max(y_test))+1))
    
    #bar plots for accuracy in all values of k for test and train datasets
    bp1 = axs[0,0].bar(np.arange(1,31,1) , y_train)
    bp2 = axs[0,1].bar(np.arange(1,31,1) , y_test)
    axs[0,0].bar_label(bp1, fontsize=6.5, padding =2)
    axs[0,1].bar_label(bp2, fontsize=6.5, padding =2)
    axs[0,0].set_title('Train Dataset ' + str(evs[0]), fontsize='medium')
    axs[0,1].set_title('Test Dataset ' + str(evs[0]), fontsize='medium')
    

    print('The maximum recall of the train dataset is at k=' + str(y_train.index(max(y_train))+1))
    print('The maximum recall of the test dataset is at k=' + str(y_test.index(max(y_test))+1))
    
    #bar plots for recall in all values of k for test and train datasets
    bp3 = axs[1,0].bar(np.arange(1,31,1) , y_train)
    bp4 = axs[1,1].bar(np.arange(1,31,1) , y_test)
    axs[1,0].bar_label(bp3, fontsize=6.5, padding =2)
    axs[1,1].bar_label(bp4, fontsize=6.5, padding =2)
    axs[1,0].set_title('Train Dataset ' + str(evs[1]), fontsize='medium')
    axs[1,1].set_title('Test Dataset ' + str(evs[1]), fontsize='medium')
    
    y_train = train_data[2][i:j]
    y_test = test_data[2][i:j]
    

    print('The maximum F1-Score of the train dataset is at k=' + str(y_train.index(max(y_train))+1))
    print('The maximum F1-Score of the test dataset is at k=' + str(y_test.index(max(y_test))+1))
     
    #bar plots for f-score in all values of k for test and train datasets
    bp5 = axs[2,0].bar(np.arange(1,31,1) , y_train)
    bp6 = axs[2,1].bar(np.arange(1,31,1) , y_test)
    axs[2,0].bar_label(bp5, fontsize=6.5, padding =2)
    axs[2,1].bar_label(bp6, fontsize=6.5, padding =2)
    axs[2,0].set_title('Train Dataset ' + str(evs[2]), fontsize='medium')
    axs[2,1].set_title('Test Dataset ' + str(evs[2]), fontsize='medium')
    
    y_train = train_data[3][i:j]
    y_test = test_data[3][i:j]
    

    print('The maximum precision of the train dataset is at k=' + str(y_train.index(max(y_train))+1))
    print('The maximum precision of the test dataset is at k=' + str(y_test.index(max(y_test))+1))
    
    #bar plots for precision in all values of k for test and train datasets
    bp7 = axs[3,0].bar(np.arange(1,31,1) , y_train)
    bp8 = axs[3,1].bar(np.arange(1,31,1) , y_test)
    axs[3,0].bar_label(bp7, fontsize=6.5, padding =2)
    axs[3,1].bar_label(bp8, fontsize=6.5, padding =2)
    axs[3,0].set_title('Train Dataset ' + str(evs[3]), fontsize='medium')
    axs[3,1].set_title('Test Dataset ' + str(evs[3]), fontsize='medium')
    
    for ax in axs.flat:
        ax.set(xlabel='k')
    fig.suptitle(str(datasets[n]), fontsize = 'large')
    fig.tight_layout()
    
    plt.savefig('Results KNN\\Comparing_k_'+ datasets[n] +'.png', dpi=300)
    
    #next dataset
    i=i+30
    j=j+30
    n=n+1
    
