import scanpy as sc
import numpy as np
import pandas as pd

import anndata as ad

sc.settings.verbosity = 3

# read the All_cell_matrix into AnnData matrices
filepath_mtx = "matrix.mtx"
mtx_all = sc.read_mtx(filepath_mtx)
sc.pp.calculate_qc_metrics(mtx_all, inplace=True)

# read annotations (cell information)
filepath_ann = "annotations.tsv"
annotations = pd.read_table(filepath_ann)
print("number of cells", annotations.shape) #166533 cells in the GC dataset

print(mtx_all.X.shape)
print("number of stored values:", mtx_all.X.nnz) #269783730 combinations of cells and expressed genes

# convert cell matrix from CSR to pandas dataframe
c = mtx_all.X.tocoo()
df = pd.DataFrame({'gene': c.row, 'cell': c.col, 'gene expression level': c.data})
print(df) #col 1 is the gene, col 2 is the cell, col 3 is the (normalized?) gene expression level
df.describe() #mean, std, min, max of gene expression values

# appends a column to the matrix dataframe consisting of the respective cell types of each gene/cell combination
celltype = np.empty(df.shape[0], dtype=object)
for i in range(celltype.size):
    celltype[i] = annotations['cluster'][df['cell'][i]] #takes a while to run
df.insert(3, "cell type", celltype)

# splits into different dataframes based on cell type
tcells = df[df['cell type']=="T cells & NK cells"]
myeloid = df[df['cell type']=="Myeloid cells"]
bcells = df[df['cell type']=="B cells"]
erythrocytes = df[df['cell type']=="Erythrocytes"]
mast = df[df['cell type']=="Mast cells"]
epithelial = df[df['cell type']=="Epithelial cells"]
endothelial = df[df['cell type']=="Endothelial cells"]
endocrine = df[df['cell type']=="Endocrine cells"]
muscle = df[df['cell type']=="Smooth muscle cells"]
fibroblasts = df[df['cell type']=="Fibroblasts"]
plasma = df[df['cell type']=="B cells(Plasma cells)"]

# erythrocytes have the least data (only 302679 datapoints), most manageable
# ordering the erythrocyte dataframe by cells rather than genes
erythrocytes = erythrocytes.sort_values(by=['cell', 'gene'])
erythrocytes = erythrocytes.reset_index(drop=True)
erythrocytes_ndatapoints = len(erythrocytes)

# muscle cells have second least data
muscle = muscle.sort_values(by=['cell', 'gene'])
muscle = muscle.reset_index(drop=True)
muscle_ndatapoints = len(muscle)

# finding the number of distinct erythrocyte and muscle cells
n_erythrocytes = len(np.unique(erythrocytes['cell'].to_numpy()))
n_muscle = len(np.unique(muscle['cell'].to_numpy()))
n_genes = 24850

# creating an array for gene expression information for erythrocytes
# each row is an individual erythrocyte cell
# each column represents one of 24850 different genes
# the array element is the standardized gene expression value (mostly zeros for no UMIs found)
erythrocytes_dataset = np.zeros([n_erythrocytes, n_genes])
index = 0
currentcell = erythrocytes['cell'].loc[erythrocytes.index[index]]
for i in range(len(erythrocytes_dataset)):
    while (index < erythrocytes_ndatapoints and erythrocytes['cell'].loc[erythrocytes.index[index]] == currentcell):
        erythrocytes_dataset[i][erythrocytes['gene'].loc[erythrocytes.index[index]]] = erythrocytes['gene expression level'].loc[erythrocytes.index[index]]
        index += 1
    if index < erythrocytes_ndatapoints:
        currentcell = erythrocytes['cell'].loc[erythrocytes.index[index]]

# creates an array for smooth muscle cells
muscle_dataset = np.zeros([n_muscle, n_genes])
index = 0
currentcell = muscle['cell'].loc[muscle.index[index]]
for i in range(len(muscle_dataset)):
    while (index < muscle_ndatapoints and muscle['cell'].loc[muscle.index[index]] == currentcell):
        muscle_dataset[i][muscle['gene'].loc[muscle.index[index]]]= muscle['gene expression level'].loc[muscle.index[index]]
        index += 1
    if index < muscle_ndatapoints:
        currentcell = muscle['cell'].loc[muscle.index[index]]

# creates training and testing sets for erythrocytes and muscle cells data
erythrocytes_train = erythrocytes_dataset[:500] #arbitrarily chose the first 500 erythrocytes for the training data (follow the 8:2 rule)
erythrocytes_test = erythrocytes_dataset[500:]

muscle_train = muscle_dataset[:1000] #arbitrarily chose the first 1000 muscle cells for training data
muscle_test = muscle_dataset[1000:]

x_train = np.concatenate((erythrocytes_train, muscle_train))

y_train = np.empty(1500, dtype=object)
for i in range(500):
    y_train[i] = "Erythrocytes"
for i in range(500, 1500):
    y_train[i] = "Smooth muscle cell"
print(y_train.shape)

# shuffles the training data for higher quality model training
shuffle_index = np.random.permutation(1500)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]
y_train_e = (y_train == "Erythrocytes") #using binary classifier (SGD), so converting the y_train values to True and False

from sklearn.linear_model import SGDClassifier #using SGD right now, but easily expandable to other machine learning classification models

# training the model
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(x_train, y_train_e)

# testing the model, finding the number of true positives, true negatives, false positive, and false negatives
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for i in range(286):
    if sgd_clf.predict([erythrocytes_test[i]])==True:
        truepos += 1
    else:
        falseneg += 1
for i in range(451):
    if sgd_clf.predict([muscle_test[i]])==False:
        trueneg += 1
    else:
        falsepos += 1
print(truepos)
print(falseneg)
print(trueneg)
print(falsepos)
