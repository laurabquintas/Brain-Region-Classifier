"""Evaluate SVM classifier across all dataset variants."""

from sklearn.svm import SVC
from utils.evaluation import run_all_datasets

# Best hyperparameters (from ranking analysis)
KERNEL = 'rbf'
C = 100
GAMMA = 0.00001

run_all_datasets(lambda: SVC(kernel=KERNEL, C=C, gamma=GAMMA))
