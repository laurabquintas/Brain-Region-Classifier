"""Rank Decision Tree hyperparameter sets by cross-validated performance."""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from statistics import mean
from utils.evaluation import DATA_DIR, TARGET

CRITERIA = ['gini', 'entropy']
MAX_DEPTHS = [3]
MIN_IMPURITY_DECREASE = [0.001, 0.0005, 0.0025]

rank = {}

for filename in sorted(os.listdir(DATA_DIR)):
    data = pd.read_csv(os.path.join(DATA_DIR, filename))
    splits = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    y = data.pop(TARGET).values
    X = data.values

    set_performance = {}

    for criterion in CRITERIA:
        for depth in MAX_DEPTHS:
            for imp_dec in MIN_IMPURITY_DECREASE:
                dt = DecisionTreeClassifier(
                    max_depth=depth, criterion=criterion,
                    min_impurity_decrease=imp_dec, class_weight='balanced',
                )
                cv = cross_validate(
                    dt, X, y, cv=splits,
                    scoring=('accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted'),
                    return_train_score=True,
                )
                param_key = f'{criterion}_d{depth}_id{imp_dec}'
                set_performance[param_key] = np.mean([
                    round(mean(cv['test_accuracy']), 2),
                    round(mean(cv['test_recall_weighted']), 2),
                    round(mean(cv['test_f1_weighted']), 2),
                    round(mean(cv['test_precision_weighted']), 2),
                ])

    ordered = sorted(set_performance, key=set_performance.get, reverse=True)
    for param_set in ordered:
        rank.setdefault(param_set, []).append(ordered.index(param_set) + 1)

print(rank)
