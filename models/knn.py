"""Evaluate K-Nearest Neighbors across all dataset variants (k=1..30)."""

import pickle
from sklearn.neighbors import KNeighborsClassifier
from utils.evaluation import run_all_datasets

K_RANGE = range(1, 31)

store_train = [[] for _ in range(4)]
store_test = [[] for _ in range(4)]

for k in K_RANGE:
    results = run_all_datasets(lambda: KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2))

    for i, metric in enumerate(['Accuracy', 'Recall', 'F1-Score', 'Precision']):
        for evaluation in results.values():
            store_train[i].append(evaluation[metric][0])
            store_test[i].append(evaluation[metric][1])

with open('train_evaluation.pkl', 'wb') as f:
    pickle.dump(store_train, f)
with open('test_evaluation.pkl', 'wb') as f:
    pickle.dump(store_test, f)
