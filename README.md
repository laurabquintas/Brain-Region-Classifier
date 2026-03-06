# Brain Region Classifier

Multiclass classification of human brain regions from bulk RNA-seq transcriptomic data. Four machine learning models (KNN, SVM, Decision Tree, MLP) are benchmarked across multiple normalization and feature-selection pipelines to identify the most robust approach for mapping gene expression profiles to one of four macro-regions: **Forebrain**, **Basal Ganglia**, **Midbrain**, and **Hindbrain**.

## Motivation

Accurate identification of brain regional identity from gene expression is critical for validating brain organoid models and studying neurological disease. Traditional approaches rely on a handful of marker genes, which fail to capture the full transcriptomic landscape. This project explores whether standard ML classifiers, trained on genome-wide expression data from 25 sub-regions across 265 adult brain samples, can reliably distinguish macro-regions, a necessary step toward automated organoid characterization.

## Dataset

The data originates from bulk RNA-seq profiling of 25 anatomically distinct human brain sub-regions ([Dong et al., 2022](https://doi.org/10.1101/2022.09.02.506419)). Each of the 265 samples is labeled with one of four macro-regions based on neuroanatomical grouping:

| Macro-Region | Sub-Regions |
|---|---|
| Forebrain | PVC, PAC, EC, PMC, VLPFC, PSC, OFC, PVPC, AMY, DLPFC, PSTC, HIPP, ITC, ACC, INS |
| Basal Ganglia | NAC, CN, GP |
| Midbrain | ARC, MDT, RMTG, VTA, DRN, HAB |
| Hindbrain | CRBLM |

### Preprocessing Pipelines

Eight dataset variants are generated from the raw counts to evaluate the impact of normalization and dimensionality reduction:

| Dataset | Description |
|---|---|
| `data_raw` | Raw read counts |
| `data_raw_filtered` | Filtered raw counts (low-expression genes removed) |
| `data_raw_filtered_FS` | Filtered + variance-based feature selection |
| `data_logCPM` | Log-CPM normalized |
| `data_logCPM_centered` | Log-CPM, mean-centered |
| `data_logCPM_zscore` | Log-CPM, z-score standardized |
| `data_logCPM_FS` | Log-CPM + feature selection |
| `data_logCPM_centered_FS` | Log-CPM centered + feature selection |

Feature selection uses `VarianceThreshold`, retaining features that explain >20% of the maximum observed variance (threshold adapted per normalization scheme due to differing value ranges).

## Methods

All classifiers are evaluated with **10-fold stratified shuffle split** (70/30 train/test) and scored on weighted **accuracy, precision, recall, and F1-score**.

| Model | Key Hyperparameters |
|---|---|
| **K-Nearest Neighbors** | k = 1-30 (grid search), uniform weights, Euclidean distance |
| **Support Vector Machine** | RBF kernel, C in {0.1, 1, 10, 100}, gamma in {0.01, 0.001, 0.0001, 0.00001} |
| **Decision Tree** | criterion in {gini, entropy}, max_depth = 3, min_impurity_decrease in {0.001, 0.0005, 0.0025} |
| **Multi-Layer Perceptron** | ReLU activation, Adam solver, lr = 0.01, 500 max iterations |

Hyperparameter ranking is performed by averaging test-set metrics across all eight datasets and ranking parameter sets per dataset.

## Project Structure

```
.
├── data_preparation.py          # Raw data -> structured CSVs with normalization & feature selection
├── models/
│   ├── knn.py                   # KNN evaluation across k=1..30
│   ├── knn_compare_k.py         # Visualize optimal k per dataset
│   ├── svm.py                   # SVM evaluation (best hyperparams)
│   ├── svm_rank.py              # SVM hyperparameter ranking
│   ├── decision_tree.py         # Decision Tree evaluation (best hyperparams)
│   ├── decision_tree_rank.py    # DT hyperparameter ranking
│   └── mlp.py                   # MLP evaluation
├── utils/
│   ├── evaluation.py            # Cross-validation, metrics, shared dataset runner
│   ├── plotting.py              # Matplotlib theme, bar charts, confusion matrices
│   └── mlblabs.mplstyle         # Custom matplotlib stylesheet
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/laurabquintas/Brain-Region-Classifier.git
cd Brain-Region-Classifier

# Install dependencies
pip install -r requirements.txt

# Install the custom matplotlib style
cp utils/mlblabs.mplstyle "$(python -c 'import matplotlib; print(matplotlib.get_data_path())')/stylelib/"
```

### Data

The processed datasets are available on [Google Drive](https://drive.google.com/drive/folders/10PnGUiTsRo8V7TpiiXkZqfoCdFkmzNSh?usp=sharing). Download and place the CSV files in a `data/` directory at the project root.

To regenerate from the original RNA-seq files ([source](https://drive.google.com/drive/folders/1fRamU0TPYpsXJ0sy3RPsnxNNeROhOMif?usp=sharing)):

```bash
python data_preparation.py
```

## Usage

Run any classifier against all dataset variants:

```bash
python -m models.knn           # K-Nearest Neighbors
python -m models.svm           # Support Vector Machine
python -m models.decision_tree # Decision Tree
python -m models.mlp           # Multi-Layer Perceptron
```

Run hyperparameter analysis:

```bash
python -m models.knn_compare_k      # Compare k values for KNN
python -m models.decision_tree_rank  # Rank DT hyperparameter sets
python -m models.svm_rank            # Rank SVM hyperparameter sets
```

## References

1. Zheng, H., Feng, Y., Tang, J., & Ma, S. (2022). Interfacing brain organoids with precision medicine and machine learning. *Cell Reports Physical Science*, 3(7), 100974. [doi:10.1016/j.xcrp.2022.100974](https://doi.org/10.1016/j.xcrp.2022.100974)

2. Tanaka, Y., Cakir, B., Xiang, Y., Sullivan, G.J., & Park, I.H. (2020). Synthetic Analyses of Single-Cell Transcriptomes from Multiple Brain Organoids and Fetal Brain. *Cell Reports*, 30(6), 1682-1689.e3. [doi:10.1016/j.celrep.2020.01.038](https://doi.org/10.1016/j.celrep.2020.01.038)

3. Dong, P., Bendl, J., Misir, R., et al. (2022). Transcriptome and chromatin accessibility landscapes across 25 distinct human brain regions expand the susceptibility gene set for neuropsychiatric disorders. *bioRxiv*. [doi:10.1101/2022.09.02.506419](https://doi.org/10.1101/2022.09.02.506419)
