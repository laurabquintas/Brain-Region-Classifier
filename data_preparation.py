"""
Preprocesses raw RNA-seq expression data into structured datasets
for brain region classification. Outputs multiple normalization variants
and feature-selected subsets as CSV files.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

DATA_INPUT = os.path.join('data', 'original')
DATA_OUTPUT = os.path.join('data', 'processed')

# Sub-region to macro-region mapping (neuroanatomical grouping)
SUBREGION_MAP = {
    'ARC': 'MidBr', 'PVC': 'ForBr', 'PAC': 'ForBr', 'EC': 'ForBr',
    'MDT': 'MidBr', 'RMTG': 'MidBr', 'PMC': 'ForBr', 'VLPFC': 'ForBr',
    'VTA': 'MidBr', 'CRBLM': 'HinBr', 'PSC': 'ForBr', 'OFC': 'ForBr',
    'PVPC': 'ForBr', 'AMY': 'ForBr', 'NAC': 'BasGan', 'CN': 'BasGan',
    'DRN': 'MidBr', 'DLPFC': 'ForBr', 'PSTC': 'ForBr', 'GP': 'BasGan',
    'HAB': 'MidBr', 'HIPP': 'ForBr', 'ITC': 'ForBr', 'ACC': 'ForBr',
    'INS': 'ForBr',
}

TARGET = 'BRAIN REGION'


def structurize(df):
    """Transpose gene-by-sample matrix, extract brain region labels from sample IDs,
    and map sub-regions to four macro-regions."""
    df = df.T
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(index=df.index[0], axis=0, inplace=True)

    df[TARGET] = [idx[4:len(idx) - 6] for idx in df.index]
    df.index = range(len(df))
    df = df.convert_dtypes()

    df[TARGET] = df[TARGET].map(SUBREGION_MAP)
    df = df.astype({TARGET: 'category'})
    return df


def feature_select(df, thresh: float):
    """Remove features with variance below the given threshold."""
    numeric_cols = list(df.columns[:-1])
    df_features = df[numeric_cols]
    df_target = df[TARGET]

    selector = VarianceThreshold(threshold=thresh).fit(df_features)
    reduced = pd.DataFrame(selector.transform(df_features), index=df.index)
    result = pd.concat([reduced, df_target], axis=1)

    print(f'Features: {len(numeric_cols)} -> {reduced.shape[1]} '
          f'(dropped {len(numeric_cols) - reduced.shape[1]})')
    return result


if __name__ == '__main__':
    os.makedirs(DATA_OUTPUT, exist_ok=True)

    # Load raw expression data
    df_raw = pd.read_table(os.path.join(DATA_INPUT, 'GSE211792_readcount_matrix.txt'))
    df_raw_filtered = pd.read_csv(os.path.join(DATA_INPUT, 'raw_AAB.csv'))
    df_logCPM = pd.read_csv(os.path.join(DATA_INPUT, 'GSE211792_logCPM.csv'))
    df_logCPM_cent = pd.read_csv(os.path.join(DATA_INPUT, 'GSE211792_logCPM_cent.csv'))

    # Structure all dataframes
    df_raw = structurize(df_raw)
    df_raw_filtered = structurize(df_raw_filtered)
    df_logCPM = structurize(df_logCPM)
    df_logCPM_cent = structurize(df_logCPM_cent)

    # Save base datasets
    df_raw.to_csv(os.path.join(DATA_OUTPUT, 'data_raw.csv'), index=False)
    df_raw_filtered.to_csv(os.path.join(DATA_OUTPUT, 'data_raw_filtered.csv'), index=False)
    df_logCPM.to_csv(os.path.join(DATA_OUTPUT, 'data_logCPM.csv'), index=False)
    df_logCPM_cent.to_csv(os.path.join(DATA_OUTPUT, 'data_logCPM_centered.csv'), index=False)

    # Z-score normalization of logCPM
    numeric_vars = list(df_logCPM.columns[:-1])
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_logCPM[numeric_vars])
    df_zscore = pd.DataFrame(scaler.transform(df_logCPM[numeric_vars]),
                             index=df_logCPM.index, columns=numeric_vars)
    df_logCPM_zscore = pd.concat([df_zscore, df_logCPM[TARGET]], axis=1)
    df_logCPM_zscore.to_csv(os.path.join(DATA_OUTPUT, 'data_logCPM_zscore.csv'), index=False)

    # Feature selection (variance threshold)
    # Raw filtered: threshold = 0.005% of max variance (retains ~441 features)
    df_raw_filtered_FS = feature_select(df_raw_filtered, 0.00005 * max(df_raw_filtered.var()))
    df_raw_filtered_FS.to_csv(os.path.join(DATA_OUTPUT, 'data_raw_filtered_FS.csv'), index=False)

    # LogCPM variants: retain features explaining >20% of maximum variance
    df_logCPM_FS = feature_select(df_logCPM, 0.20 * max(df_logCPM.var()))
    df_logCPM_FS.to_csv(os.path.join(DATA_OUTPUT, 'data_logCPM_FS.csv'), index=False)

    df_logCPM_cent_FS = feature_select(df_logCPM_cent, 0.20 * max(df_logCPM_cent.var()))
    df_logCPM_cent_FS.to_csv(os.path.join(DATA_OUTPUT, 'data_logCPM_centered_FS.csv'), index=False)
