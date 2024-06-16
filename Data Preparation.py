# -*- coding: utf-8 -*-
"""
Script that takes as input the data from Sofia Agostinho
And processes it in order to be viable for classification
The datasets are saved in.csv format in the 'Data Classification' folder
"""

# Modules to use
import os
import pandas as pd
import sys
sys.path.append('Code MLB')
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# View the working directory
print(os.getcwd())

# View the files given by Sofia from papers
print(os.listdir('Data Original'))
# ['GSE211792_logCPM.csv', 'GSE211792_logCPM_cent.csv', 'GSE211792_readcount_matrix.txt', 'raw_AAB.csv']

# Load the data 
df_raw = pd.read_table('Data Original\\GSE211792_readcount_matrix.txt')
df_raw_filtered = pd.read_csv('Data Original\\raw_AAB.csv')
df_logCPM = pd.read_csv('Data Original\\GSE211792_logCPM.csv')
df_logCPM_cent = pd.read_csv('Data Original\\GSE211792_logCPM_cent.csv')

# Check the shape of the dataframes to assert the order of their creation
# Can be seen in the variable explorer panel as well
print('%s\n %s\n %s\n %s\n' % (df_raw.shape, df_raw_filtered.shape, df_logCPM.shape, df_logCPM_cent.shape))


def structurize(df):
    # transpose the df
    df = df.T 
    # set columns names as gene IDs
    df.rename(columns=df.iloc[0], inplace=True) 
    # drop gene IDs rows
    df.drop(index=df.index[0], axis=0, inplace=True) 
    # make a new column for brain region
    # note that the person's ID and sample origin are not relevant
    # classification should be based only on gene expression data
    df['BRAIN REGION'] = pd.Series()
    # fill the two new columns using the indexes info 
    for el in list(df.index):
        df['BRAIN REGION'][df.index == el] = el[4:len(el)-6]
    # Rename indexes as numbers
    indxs = [i for i in range(0,265)]
    df.index = indxs
    # The types are all messed up. Let's change them to their original meaning
    df = df.convert_dtypes()
    df = df.astype({"BRAIN REGION":'category'})
    # Map the sub-brain regions into the 4 big ones
    # ForBr (1), BasGan (2), MidBr (3) and HinBr (4)
    df['BRAIN REGION'] = df['BRAIN REGION'].map({
        'ARC': 3,
        'PVC': 1,   
        'PAC': 1,    
        'EC': 1,   
        'MDT': 3,  
        'RMTG': 3,   
        'PMC': 1, 
        'VLPFC': 1,
        'VTA': 3, 
        'CRBLM': 4,   
        'PSC': 1,   
        'OFC': 1,  
        'PVPC': 1,   
        'AMY': 1,   
        'NAC': 2,    
        'CN': 2,
        'DRN': 3, 
        'DLPFC':1 ,  
        'PSTC':1,    
        'GP':2,   
        'HAB':3,  
        'HIPP':1,   
        'ITC':1,   
        'ACC':1,
        'INS':1
        })
    # In fact, sklearn encodes the target variable into numeric values on its own
    # Thus, we can map the sub regions into strings named after the big regions
    # Note: if u are going to use other packages bear in mind that numerical mapping
    # of the target may be necessary
    df['BRAIN REGION'] = df['BRAIN REGION'].map({
        1:'ForBr',
        2:'BasGan',
        3:'MidBr',
        4:'HinBr'
        })
    # Turn brain region variable as category again, since it became a string
    df = df.astype({"BRAIN REGION":'category'})
    # Check resuls
    print(df.dtypes)
    print(df.head())
    print(df.shape)
    return df


# Structurize current dataframes
# May take a while
df_raw = structurize(df_raw)
df_raw_filtered = structurize(df_raw_filtered)
df_logCPM = structurize(df_logCPM)
df_logCPM_cent = structurize(df_logCPM_cent)

# Let's save the datasets we already have
# May take a while
df_raw.to_csv('Data Classification\\data_raw.csv', index = False)
df_raw_filtered.to_csv('Data Classification\\data_raw_filtered.csv', index = False)
df_logCPM.to_csv('Data Classification\\data_logCPM.csv', index = False)
df_logCPM_cent.to_csv('Data Classification\\data_logCPM_centered.csv', index = False)

# Add the zscore of logCPM to the dataset files
# It is the logCPM centered to mean 0 (as already done) and standard deviation 1 
# I.e, normally distributed variables
# MinMax could also be done, but it is not relevant
numeric_vars = list(df_logCPM.columns[:-1])
df_logCPM_nr = df_logCPM[numeric_vars]
df_logCPM_sb = df_logCPM['BRAIN REGION']
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_logCPM_nr)
tmp = pd.DataFrame(transf.transform(df_logCPM_nr), index=df_logCPM.index, columns= numeric_vars)
df_logCPM_zscore  = pd.concat([tmp, df_logCPM_sb], axis=1)
df_logCPM_zscore.to_csv('Data Classification\\data_logCPM_zscore.csv', index = False)
    
### Feature selection ###
# Note that feature selection will be made in the structurized datasets
# The only relevant difference is that the variables of interest are now columns instead of rows
# The numeric values remain the same

# Feature selection will be made regarding three datasets: raw_filtered, logCPM and logCPM_centered
# using variance threshold only (other methods are available: correlation,PCA, etc., but are out of
# the scope of our project)

# This is done since feature selection with and without centering is very different
# since variables may have very different distributions (range of value) and thus variance
# levels differ greatly. Centering sets all variables with mean 0, thus considering all the variables

# See values to define thresholds
print('Min_Var, Max_var\n %s, %s\n %s, %s\n %s, %s\n' % (
    min(df_raw_filtered.var()),
    max(df_raw_filtered.var()),
    min(df_logCPM.var()),
    max(df_logCPM.var()),
    min(df_logCPM_cent.var()),
    max(df_logCPM_cent.var())
    ))
# Min_Var, Max_var
#  0.15208690680388795, 140538900147.0951
#  0.01974006133989726, 30.290584732278997
#  0.019740061339897287, 30.29058473227902

# Deletes variables with variance < threshold
# Regarding thresholds of variance, i tried to find values for our type of data in articles
# But i was unsuccessfull. Did not find any
def FS(df, thresh: float):
    initvars = df.shape[1]-1
    numeric_vars = list(df.columns[:-1])
    df_nr = df[numeric_vars]
    df_sb = df['BRAIN REGION']
    transf = VarianceThreshold(threshold = thresh).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=df.index)
    df = pd.concat([tmp, df_sb], axis=1)
    print('Original number of features: %s' % initvars)
    print('\n')
    print('New number of features: %s' % tmp.shape[1])
    print('\n')
    print('Number of variables dropped: %s' % (initvars-tmp.shape[1]))
    return df


# For the raw_filtered, the threshold needs to be much because of the range of values (very high)
# The maximum variance registered must be only present in one column
# So let's see the top 100 maximum variances
maxvars = sorted(list(df_raw_filtered.var()), reverse=True)[:100] 
maxvars
# High percentage --> eliminates all features but one since the maximum variance registered is off the charts
# Low percentage --> generates a threshold small enough to cover some features
# Some examples:
# 0.05*max(df_raw_filtered.var()) -> 1 feature
# 0.005*max(df_raw_filtered.var()) -> 15 features
# 0.0005*max(df_raw_filtered.var()) -> 63 features
# 0.00005*max(df_raw_filtered.var()) -> 441 features  <-- CHOOSEN FOR FEATURE SELECTION
# 0.000010*max(df_raw_filtered.var()) -> 1425 features
# 0.000005*max(df_raw_filtered.var()) -> 2176 features
df_raw_filtered_FS = FS(df_raw_filtered, 0.00005*max(df_raw_filtered.var()))
df_raw_filtered_FS.to_csv('Data Classification\\data_raw_filtered_FS.csv', index = False)


# Eliminates features that explain less than 20% of maximum variance of the data
# In terms of stddev, for the logCPM and logCPM_centered, it eliminates features
# who have stddev < 0.20*np.sqrt(30.29058473227902) ~ 1.10
df_logCPM_FS = FS(df_logCPM, 0.20*max(df_logCPM.var()))
df_logCPM_FS.to_csv('Data Classification\\data_logCPM_FS.csv', index = False)
df_logCPM_cent_FS = FS(df_logCPM_cent, 0.20*max(df_logCPM_cent.var()))
df_logCPM_cent_FS.to_csv('Data Classification\\data_logCPM_centered_FS.csv', index = False)

### END OF DATA PREPARATION ##










