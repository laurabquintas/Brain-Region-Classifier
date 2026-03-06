"""Evaluate Multi-Layer Perceptron classifier across all dataset variants."""

from sklearn.neural_network import MLPClassifier
from utils.evaluation import run_all_datasets

run_all_datasets(lambda: MLPClassifier(
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=500,
    verbose=False,
))
