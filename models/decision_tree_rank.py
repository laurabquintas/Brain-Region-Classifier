"""Rank Decision Tree hyperparameter sets by cross-validated performance."""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from statistics import mean
from utils.charts import multiple_line_chart
import matplotlib.pyplot as plt

DATA_DIR = os.path.join('data', 'processed')
TARGET = 'BRAIN REGION'

# Hyperparameter search space
CRITERIA = ['gini', 'entropy']
MAX_DEPTHS = [3]
MIN_IMPURITY_DECREASE = [0.001, 0.0005, 0.0025]

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

    set_performance = {}

    for criterion in CRITERIA:
        for depth in MAX_DEPTHS:
            for imp_dec in MIN_IMPURITY_DECREASE:
                dt = DecisionTreeClassifier(
                    max_depth=depth,
                    criterion=criterion,
                    min_impurity_decrease=imp_dec,
                    class_weight='balanced',
                )
                cv_results = cross_validate(
                    dt, X, y, cv=splits,
                    scoring=('accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted'),
                    return_train_score=True,
                )

                acc_test = round(mean(cv_results['test_accuracy']), 2)
                rec_test = round(mean(cv_results['test_recall_weighted']), 2)
                f1_test = round(mean(cv_results['test_f1_weighted']), 2)
                prec_test = round(mean(cv_results['test_precision_weighted']), 2)

                param_key = f'{criterion}_d{depth}_id{imp_dec}'
                set_performance[param_key] = np.mean([acc_test, rec_test, f1_test, prec_test])

    ordered = list(dict(sorted(set_performance.items(), key=lambda x: x[1], reverse=True)))
    for param_set in ordered:
        rank.setdefault(param_set, []).append(ordered.index(param_set) + 1)

print(rank)
