"""Compare KNN performance across k values for each dataset."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('train_evaluation.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('test_evaluation.pkl', 'rb') as f:
    test_data = pickle.load(f)

DATASETS = [
    'logCPM', 'logCPM_centered', 'logCPM_FS', 'logCPM_centered_FS',
    'logCPM_zscore', 'raw_data', 'raw_data_filtered', 'raw_data_filtered_FS',
]
METRICS = ['Accuracy', 'Recall', 'F1-Score', 'Precision']
K_VALUES = np.arange(1, 31, 1)

for n, ds_name in enumerate(DATASETS):
    fig, axs = plt.subplots(4, 2, figsize=(17, 10))
    fig.suptitle(ds_name, fontsize='large')

    start = n * 30
    end = start + 30

    for m, metric in enumerate(METRICS):
        y_train = train_data[m][start:end]
        y_test = test_data[m][start:end]

        best_k_train = y_train.index(max(y_train)) + 1
        best_k_test = y_test.index(max(y_test)) + 1
        print(f'{ds_name} | {metric} | best k: train={best_k_train}, test={best_k_test}')

        bp_trn = axs[m, 0].bar(K_VALUES, y_train)
        bp_tst = axs[m, 1].bar(K_VALUES, y_test)
        axs[m, 0].bar_label(bp_trn, fontsize=6.5, padding=2)
        axs[m, 1].bar_label(bp_tst, fontsize=6.5, padding=2)
        axs[m, 0].set_title(f'Train - {metric}', fontsize='medium')
        axs[m, 1].set_title(f'Test - {metric}', fontsize='medium')

    for ax in axs.flat:
        ax.set(xlabel='k')
    fig.tight_layout()
    plt.savefig(f'results/knn_compare_k_{ds_name}.png', dpi=300)
