"""Evaluate Decision Tree classifier across all dataset variants."""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from evaluation import splitclassify, performance, performance_overall

DATA_DIR = os.path.join('data', 'processed')
TARGET = 'BRAIN REGION'

# Best hyperparameters (from ranking analysis)
MAX_DEPTH = 3
CRITERION = 'entropy'
MIN_IMPURITY_DECREASE = 0.0005

all_evaluations = {}
datasets = os.listdir(DATA_DIR)

for dataset in datasets:
    data = pd.read_csv(os.path.join(DATA_DIR, dataset))
    y = data.pop(TARGET).values
    X = data.values
    labels = np.sort(pd.unique(y))

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    dt = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        criterion=CRITERION,
        min_impurity_decrease=MIN_IMPURITY_DECREASE,
        class_weight='balanced',
    )

    tst_y, prd_tst, evaluation = splitclassify(dt, sss, X, y)

    name = dataset[5:-4]
    performance(labels, tst_y, prd_tst, evaluation, name)
    all_evaluations[name] = evaluation
    print(f'Done: {dataset}')

performance_overall(all_evaluations)
