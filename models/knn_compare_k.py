"""Visualize KNN performance across k values for each dataset."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

with open('train_evaluation.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('test_evaluation.pkl', 'rb') as f:
    test_data = pickle.load(f)

DATASETS = [
    'logCPM', 'logCPM_centered', 'logCPM_FS', 'logCPM_centered_FS',
    'logCPM_zscore', 'raw_data', 'raw_data_filtered', 'raw_data_filtered_FS',
]
METRICS = ['Accuracy', 'Recall', 'F1-Score', 'Precision']
K_VALUES = np.arange(1, 31)

for n, ds_name in enumerate(DATASETS):
    fig, axs = plt.subplots(4, 2, figsize=(17, 10))
    fig.suptitle(ds_name, fontsize='large')

    start, end = n * 30, (n + 1) * 30

    for m, metric in enumerate(METRICS):
        y_train = train_data[m][start:end]
        y_test = test_data[m][start:end]

        best_train = y_train.index(max(y_train)) + 1
        best_test = y_test.index(max(y_test)) + 1
        print(f'{ds_name} | {metric} | best k: train={best_train}, test={best_test}')

        for col, (y_vals, label) in enumerate([(y_train, 'Train'), (y_test, 'Test')]):
            bp = axs[m, col].bar(K_VALUES, y_vals)
            axs[m, col].bar_label(bp, fontsize=6.5, padding=2)
            axs[m, col].set_title(f'{label} - {metric}', fontsize='medium')

    for ax in axs.flat:
        ax.set(xlabel='k')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'knn_compare_k_{ds_name}.png'), dpi=300)
