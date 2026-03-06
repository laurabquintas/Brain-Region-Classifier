"""Rank SVM hyperparameter sets by cross-validated performance."""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from statistics import mean
from utils.evaluation import DATA_DIR, TARGET

C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = [0.01, 0.001, 0.0001, 0.00001]

rank = {}

for filename in sorted(os.listdir(DATA_DIR)):
    data = pd.read_csv(os.path.join(DATA_DIR, filename))
    splits = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    y = data.pop(TARGET).values
    X = data.values

    performances = {}

    for c in C_VALUES:
        for g in GAMMA_VALUES:
            clf = SVC(kernel='rbf', C=c, gamma=g)
            cv = cross_validate(
                clf, X, y, cv=splits,
                scoring=('accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted'),
                return_train_score=True,
            )
            param_key = f'C={c}_gamma={g}'
            performances[param_key] = np.mean([
                round(mean(cv['test_accuracy']), 2),
                round(mean(cv['test_recall_weighted']), 2),
                round(mean(cv['test_f1_weighted']), 2),
                round(mean(cv['test_precision_weighted']), 2),
            ])

    ordered = sorted(performances, key=performances.get, reverse=True)
    for param_set in ordered:
        rank.setdefault(param_set, []).append(ordered.index(param_set) + 1)

print(rank)
