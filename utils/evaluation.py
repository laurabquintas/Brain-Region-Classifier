"""Cross-validation utilities and performance reporting."""

import os
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.plotting import HEIGHT, multiple_bar_chart, plot_confusion_matrix

DATA_DIR = os.path.join('data', 'processed')
TARGET = 'BRAIN REGION'


def splitclassify(clf, splitter, X, y):
    """Perform stratified cross-validation, returning aggregated predictions and metrics."""
    y_test_all = np.empty([0], dtype=str)
    y_pred_all = np.empty([0], dtype=str)

    accuracy = {'trn': [], 'tst': []}
    precision = {'trn': [], 'tst': []}
    recall = {'trn': [], 'tst': []}
    f1 = {'trn': [], 'tst': []}

    for train_idx, test_idx in splitter.split(X, y):
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]

        y_test_all = np.append(y_test_all, test_y)
        clf.fit(train_X, train_y)
        prd_trn = clf.predict(train_X)
        prd_tst = clf.predict(test_X)
        y_pred_all = np.append(y_pred_all, prd_tst)

        trn_prec, trn_rec, trn_f1, _ = precision_recall_fscore_support(train_y, prd_trn, average='weighted')
        tst_prec, tst_rec, tst_f1, _ = precision_recall_fscore_support(test_y, prd_tst, average='weighted')

        accuracy['trn'].append(accuracy_score(train_y, prd_trn))
        accuracy['tst'].append(accuracy_score(test_y, prd_tst))
        precision['trn'].append(trn_prec)
        precision['tst'].append(tst_prec)
        recall['trn'].append(trn_rec)
        recall['tst'].append(tst_rec)
        f1['trn'].append(trn_f1)
        f1['tst'].append(tst_f1)

    return y_test_all, y_pred_all, {
        'Accuracy': [round(mean(accuracy['trn']), 2), round(mean(accuracy['tst']), 2)],
        'Recall': [round(mean(recall['trn']), 2), round(mean(recall['tst']), 2)],
        'F1-Score': [round(mean(f1['trn']), 2), round(mean(f1['tst']), 2)],
        'Precision': [round(mean(precision['trn']), 2), round(mean(precision['tst']), 2)],
    }


def plot_performance(labels, tst_y, prd_tst, scores, name):
    """Plot performance bar chart and confusion matrix for one dataset."""
    cnf_mtx = confusion_matrix(tst_y, prd_tst, labels=labels)
    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], scores, ax=axs[0],
                       percentage=True, location='lower center')
    plot_confusion_matrix(cnf_mtx, labels, ax=axs[1], title='Test')


def plot_performance_overall(all_evals):
    """Plot aggregated train/test performance across all datasets."""
    train_values = {'Accuracy': [], 'Recall': [], 'F1-Score': [], 'Precision': []}
    test_values = {'Accuracy': [], 'Recall': [], 'F1-Score': [], 'Precision': []}

    for evaluation in all_evals.values():
        for metric in train_values:
            train_values[metric].append(evaluation[metric][0])
            test_values[metric].append(evaluation[metric][1])

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), squeeze=False)
    multiple_bar_chart(list(all_evals.keys()), train_values, ax=axs[0, 0],
                       percentage=True, location='lower center', xlabel='Train')
    multiple_bar_chart(list(all_evals.keys()), test_values, ax=axs[1, 0],
                       percentage=True, location='lower center', xlabel='Test')


def run_all_datasets(make_classifier):
    """Evaluate a classifier across all dataset variants.

    Args:
        make_classifier: callable returning a fresh sklearn estimator.

    Returns:
        dict mapping dataset name -> evaluation metrics.
    """
    all_evaluations = {}

    for filename in sorted(os.listdir(DATA_DIR)):
        data = pd.read_csv(os.path.join(DATA_DIR, filename))
        y = data.pop(TARGET).values
        X = data.values
        labels = np.sort(pd.unique(y))

        splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
        clf = make_classifier()

        tst_y, prd_tst, evaluation = splitclassify(clf, splitter, X, y)

        name = filename[5:-4]
        plot_performance(labels, tst_y, prd_tst, evaluation, name)
        all_evaluations[name] = evaluation
        print(f'Done: {filename}')

    plot_performance_overall(all_evaluations)
    return all_evaluations
