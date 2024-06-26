# Gene expression data for brain region classification
## Machine Learning in Bioengineering 2022/2023
**Introduction**

Human brain organoids are powerful tools, not only to study normal brain development, but 
also to gain knowledge on neurological diseases. However, a key aspect for these studies is 
the precise determination of cellular maturity and regional identity. Brain region identification 
based solely on a few expression markers, although useful and generally inexpensive, does
not fully capture the complete range of gene expression patterns that may be presented by
the organoids In fact, distinct differentiation trajectories across protocols have been identified [1], 
posing the need to a more efficient and complete characterization of developing organoids.
Machine learning approaches have already been applied to study different aspects of brain 
organoids culture and development [2]. In the future, ML models may be used to rapidly classify 
organoids into brain regions and maturity stages. In this work a primary study on brain region classification will be done using a transcriptomic 
data from adult brain samples.

**References:**

**1 -** Honghui Zheng, Yilin Feng, Jiyuan Tang, Shaohua Ma, Interfacing brain organoids with 
precision medicine and machine learning, Cell Reports Physical Science, Volume 3, Issue 7, 
2022, 100974, ISSN 2666-3864, https://doi.org/10.1016/j.xcrp.2022.100974.

**2 -** Yoshiaki Tanaka, Bilal Cakir, Yangfei Xiang, Gareth J. Sullivan, In-Hyun Park, Synthetic 
Analyses of Single-Cell Transcriptomes from Multiple Brain Organoids and Fetal Brain, Cell 
Reports, Volume 30, Issue 6, 2020, Pages 1682-1689.e3, ISSN 2211-1247, 
https://doi.org/10.1016/j.celrep.2020.01.038.

**3 -** Dong, P., Bendl, J., Misir, R., Shao, Z., Edelstien, J., Davis, D. A., Haroutunian, V., Scott, 
W. K., Acker, S., Lawless, N., Hoffman, G. E., Fullard, J. F., & Roussos, P. (2022). 
Transcriptome and chromatin accessibility landscapes across 25 distinct human brain regions 
expand the susceptibility gene set for neuropsychiatric disorders. BioRxiv, 
2022.09.02.506419. https://doi.org/10.1101/2022.09.02.506419

#################################   Attention    ###################################

before running any of the scripts place the file mlblabs.mpltstyles that is in the MLB Code folder in the following directory:

C:\Users\user\anaconda3\Lib\site-packages\matplotlib\mpl-data\stylelib\mlblabs.mplstyle

##########################################################################

This project uses the following folders:

- Data Sofia which contains the original files of the RNA seq data provided by Sofia Agostinho (https://drive.google.com/drive/folders/1fRamU0TPYpsXJ0sy3RPsnxNNeROhOMif?usp=sharing)
- Data classification which contains the files obtained after executing the Data preparation script on the files contained in Data Sofia (https://drive.google.com/drive/folders/10PnGUiTsRo8V7TpiiXkZqfoCdFkmzNSh?usp=sharing)
- Code MLB which contains scripts for obtaining multiple line charts and confuse matrixes for each classifier (Based on https://web.ist.utl.pt/~claudia.antunes/DSLabs/config/)
- Final Report which includes the report, titled "MLB_Report," that is presented in article format and includes all the algorithms, analyses, and results.

and the following scripts:

- the performance.py script where the functions for the execution of cross validations and the construction of the report graphics are defined
- one for each classifier to obtain the performance scores and confusion matrix for all datasets which are organized in the Folders with the algorithms name
- one for SVM, KNN and DT to analyze and rank each set of parameters in the Folders with the algorithms name

_Final Grade_ : 19/20
