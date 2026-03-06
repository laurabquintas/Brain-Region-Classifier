"""Evaluate Decision Tree classifier across all dataset variants."""

from sklearn.tree import DecisionTreeClassifier
from utils.evaluation import run_all_datasets

# Best hyperparameters (from ranking analysis)
MAX_DEPTH = 3
CRITERION = 'entropy'
MIN_IMPURITY_DECREASE = 0.0005

run_all_datasets(lambda: DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    criterion=CRITERION,
    min_impurity_decrease=MIN_IMPURITY_DECREASE,
    class_weight='balanced',
))
