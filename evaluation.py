import numpy as np
from statistics import mean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from utils import charts as mlb


def splitclassify(clf, splt, X, y):
    """Perform stratified k-fold cross-validation and collect predictions."""
    y_test_all = np.empty([0], dtype=str)
    y_pred_all = np.empty([0], dtype=str)

    accuracy = {'trn': [], 'tst': []}
    precision = {'trn': [], 'tst': []}
    recall = {'trn': [], 'tst': []}
    f1 = {'trn': [], 'tst': []}

    for train_idx, test_idx in splt.split(X, y):
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

    evaluation = {
        'Accuracy': [round(mean(accuracy['trn']), 2), round(mean(accuracy['tst']), 2)],
        'Recall': [round(mean(recall['trn']), 2), round(mean(recall['tst']), 2)],
        'F1-Score': [round(mean(f1['trn']), 2), round(mean(f1['tst']), 2)],
        'Precision': [round(mean(precision['trn']), 2), round(mean(precision['tst']), 2)],
    }

    return y_test_all, y_pred_all, evaluation


def performance(labels, tst_y, prd_tst, scores, name):
    """Plot performance metrics and confusion matrix for a single dataset."""
    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)

    fig, axs = plt.subplots(1, 2, figsize=(2 * mlb.HEIGHT, mlb.HEIGHT))
    mlb.multiple_bar_chart(['Train', 'Test'], scores, ax=axs[0],
                           percentage=True, location='lower center')
    mlb.plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')


def performance_overall(all_evals: dict):
    """Plot aggregated performance across all datasets for a classifier."""
    train_values = {'Accuracy': [], 'Recall': [], 'F1-Score': [], 'Precision': []}
    test_values = {'Accuracy': [], 'Recall': [], 'F1-Score': [], 'Precision': []}

    for dataset in all_evals:
        evaluation = all_evals[dataset]
        for metric in train_values:
            train_values[metric].append(evaluation[metric][0])
            test_values[metric].append(evaluation[metric][1])

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), squeeze=False)
    mlb.multiple_bar_chart(list(all_evals.keys()), train_values, ax=axs[0, 0],
                           percentage=True, location='lower center', xlabel='Train')
    mlb.multiple_bar_chart(list(all_evals.keys()), test_values, ax=axs[1, 0],
                           percentage=True, location='lower center', xlabel='Test')
