"""Rank SVM hyperparameter sets by cross-validated performance."""

import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from statistics import mean

DATA_DIR = os.path.join('data', 'processed')
TARGET = 'BRAIN REGION'

# Hyperparameter search space
C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = [0.01, 0.001, 0.0001, 0.00001]

datasets = os.listdir(DATA_DIR)
rank = {}
names = []

for dataset in datasets:
    name = dataset[5:-4]
    names.append(name)
    data = pd.read_csv(os.path.join(DATA_DIR, dataset))

    splits = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    y = data.pop(TARGET).values
    X = data.values

    performances = {}

    for c in C_VALUES:
        for g in GAMMA_VALUES:
            clf = svm.SVC(kernel='rbf', C=c, gamma=g)
            cv_results = cross_validate(
                clf, X, y, cv=splits,
                scoring=('accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted'),
                return_train_score=True,
            )

            acc_test = round(mean(cv_results['test_accuracy']), 2)
            rec_test = round(mean(cv_results['test_recall_weighted']), 2)
            f1_test = round(mean(cv_results['test_f1_weighted']), 2)
            prec_test = round(mean(cv_results['test_precision_weighted']), 2)

            param_key = f'C={c}_gamma={g}'
            performances[param_key] = np.mean([acc_test, rec_test, f1_test, prec_test])

    ordered = list(dict(sorted(performances.items(), key=lambda x: x[1], reverse=True)))
    for param_set in ordered:
        rank.setdefault(param_set, []).append(ordered.index(param_set) + 1)

print(rank)
