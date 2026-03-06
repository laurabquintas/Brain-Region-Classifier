"""Evaluate K-Nearest Neighbors classifier across all dataset variants."""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from evaluation import splitclassify, performance, performance_overall

DATA_DIR = os.path.join('data', 'processed')
TARGET = 'BRAIN REGION'
K_RANGE = range(1, 31)

all_evaluations = {}
datasets = os.listdir(DATA_DIR)

store_train = [[] for _ in range(4)]
store_test = [[] for _ in range(4)]

for dataset in datasets:
    for k in K_RANGE:
        data = pd.read_csv(os.path.join(DATA_DIR, dataset))
        y = data.pop(TARGET).values
        X = data.values
        labels = np.sort(pd.unique(y))

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)

        tst_y, prd_tst, evaluation = splitclassify(knn, sss, X, y)

        name = dataset[5:-4]
        performance(labels, tst_y, prd_tst, evaluation, name)
        all_evaluations[name] = evaluation
        print(f'Done: {dataset} (k={k})')

        for i, metric in enumerate(['Accuracy', 'Recall', 'F1-Score', 'Precision']):
            store_train[i].append(evaluation[metric][0])
            store_test[i].append(evaluation[metric][1])

with open('train_evaluation.pkl', 'wb') as f:
    pickle.dump(store_train, f)

with open('test_evaluation.pkl', 'wb') as f:
    pickle.dump(store_test, f)

performance_overall(all_evaluations)
