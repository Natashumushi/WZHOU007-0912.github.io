---
layout: post
title: scRNA-seq Data Analysis
date: 2020-09-21
tags: Unsupervised Learning
---

### 0. Introduction  
This kernel uses the data from [Tabula Muris](https://tabula-muris.ds.czbiohub.org/), follows the course developed by [the Hemberg Lab](https://scrnaseq-course.cog.sanger.ac.uk/website/resources.html) and [the Computational Biology team at the Chan Zuckerberg Initiative](https://chanzuckerberg.github.io/scRNA-python-workshop/intro/about.html) for data preprocessing, builds an Autoencoder (in Keras) + t-SNE for dimensionality reduction, then compares the performance of Kmeans and Agglomerative method in clustering, and finally discusses what makes each cluster different from other cells in the dataset (differential expression).  


### 1. Reading the data  
> The Tabula Muris is a collaborative effort to profile every mouse tissue at a single-cell level.  
The full dataset includes both high throughput but low-coverage 10X data and lower throughput but high-coverage Smartseq2 data.    

For this kernel, the Smartseq2 data from the mouse brain is used.


```python
import pandas as pd

count_dataframe = pd.read_csv('.../data/brain_counts.csv',
                              index_col=0)

print(count_dataframe.head(2)) 
print(count_dataframe.shape)
```

                           0610005C13Rik  0610007C21Rik  0610007L01Rik  \
    A1.B003290.3_38_F.1.1              0            125             16   
    A1.B003728.3_56_F.1.1              0              0              0   
    
                           0610007N19Rik  0610007P08Rik  0610007P14Rik  \
    A1.B003290.3_38_F.1.1              0              0              0   
    A1.B003728.3_56_F.1.1              0              0            324   
    
                           0610007P22Rik  0610008F07Rik  0610009B14Rik  \
    A1.B003290.3_38_F.1.1              0              0              0   
    A1.B003728.3_56_F.1.1              0              0              0   
    
                           0610009B22Rik  ...  Zxdb  Zxdc  Zyg11a  Zyg11b  Zyx  \
    A1.B003290.3_38_F.1.1              0  ...     0     0       0       0    0   
    A1.B003728.3_56_F.1.1              0  ...     0     0       0       0    0   
    
                           Zzef1  Zzz3  a  l7Rn6  zsGreen_transgene  
    A1.B003290.3_38_F.1.1      0     0  0     54                  0  
    A1.B003728.3_56_F.1.1      0     0  0      0                  0  
    
    [2 rows x 23433 columns]
    (3401, 23433)


The column names represent genes.   
The row names represent unique cell identifiers that were assigned by the authors of the dataset.  
We have 3401 cells and 23433 genes.

### 2. Reading the metadata


```python
metadata_dataframe = pd.read_csv('.../data/brain_metadata.csv',
                                index_col=0)

print(metadata_dataframe.head())
print(metadata_dataframe.shape)
```

                            cell_ontology_class    subtissue mouse.sex mouse.id  \
    cell                                                                          
    A1.B003290.3_38_F.1.1             astrocyte     Striatum         F   3_38_F   
    A1.B003728.3_56_F.1.1             astrocyte     Striatum         F   3_56_F   
    A1.MAA000560.3_10_M.1.1     oligodendrocyte       Cortex         M   3_10_M   
    A1.MAA000564.3_10_M.1.1    endothelial cell     Striatum         M   3_10_M   
    A1.MAA000923.3_9_M.1.1            astrocyte  Hippocampus         M    3_9_M   
    
                            plate.barcode  
    cell                                   
    A1.B003290.3_38_F.1.1         B003290  
    A1.B003728.3_56_F.1.1         B003728  
    A1.MAA000560.3_10_M.1.1     MAA000560  
    A1.MAA000564.3_10_M.1.1     MAA000564  
    A1.MAA000923.3_9_M.1.1      MAA000923  
    (3401, 5)


We have 5 columns of information about 3401 cells.  
Now check out the number of times each cell appears in a label (column):


```python
for i in metadata_dataframe.columns.values:
    print(pd.value_counts(metadata_dataframe[i]))
```

    oligodendrocyte                   1574
    endothelial cell                   715
    astrocyte                          432
    neuron                             281
    oligodendrocyte precursor cell     203
    brain pericyte                     156
    Bergmann glial cell                 40
    Name: cell_ontology_class, dtype: int64
    Cortex         1149
    Hippocampus     976
    Striatum        723
    Cerebellum      553
    Name: subtissue, dtype: int64
    M    2694
    F     707
    Name: mouse.sex, dtype: int64
    3_10_M    980
    3_9_M     871
    3_8_M     590
    3_38_F    355
    3_11_M    253
    3_39_F    241
    3_56_F    111
    Name: mouse.id, dtype: int64
    MAA000560    287
    MAA000926    263
    MAA000581    190
    MAA000944    184
    MAA000932    174
    MAA001894    147
    MAA000564    143
    MAA000942    136
    MAA000935    131
    MAA000941    125
    MAA000930    111
    MAA000923    108
    MAA000947    107
    B003290       98
    MAA000561     97
    MAA000615     95
    B003275       93
    MAA000641     67
    B003728       66
    MAA000940     63
    MAA001895     60
    MAA000563     57
    MAA000925     55
    B003277       52
    MAA000638     51
    MAA000902     40
    MAA000553     39
    MAA000424     39
    MAA000578     38
    MAA000928     36
    MAA000550     34
    MAA001845     33
    B001688       32
    B003274       27
    B000621       24
    MAA001854     23
    MAA001853     22
    B000404       21
    MAA000924     14
    MAA000538     10
    MAA001856      9
    Name: plate.barcode, dtype: int64


### 3. Building an AnnData object  

`AnnData` stands for "annotated data," and is the standard format used by the analysis library, `SCANPY`.  
This data structure has four areas where we can store information:  
![flowchart](https://github.com/WZHOU007-0912/images/raw/master/flowchart.png)

`AnnData.X` stores the count matrix  
`AnnData.obs` stores metadata about the observations (cells)  
`AnnData.var` stores metadata about the variables (genes)  
`AnnData.uns` stores any additional, unstructured information we decide to attach later   


```python
import scanpy as sc

adata = sc.AnnData(X = count_dataframe, obs = metadata_dataframe)
print(adata)
```

    AnnData object with n_obs × n_vars = 3401 × 23433
        obs: 'cell_ontology_class', 'subtissue', 'mouse.sex', 'mouse.id', 'plate.barcode'


Labeling spilke-ins:


```python
is_spike_in = {}
number_of_spike_ins = 0

for gene_name in adata.var_names:
    if 'ERCC' in gene_name:
        is_spike_in[gene_name] = True 
        # record that we found a spike-in
        number_of_spike_ins += 1 
        # bump the counter
    else:
        is_spike_in[gene_name] = False 
        # record that this was not a spike-in
        
adata.var['ERCC'] = pd.Series(is_spike_in) 
# because the index of adata.var and the keys of is_spike_in match, 
# anndata will take care of matching them up

print('found this many spike ins: ', number_of_spike_ins)
```

    found this many spike ins:  92


### 4. Quality control

#### 4.1 Cell filtering

>One of the measures of cell quality is the **ratio** between ERCC spike-in RNAs and endogenous RNAs.  
This ratio can be used to estimate the total amount of RNA in the captured cells.  
**Cells with a high level of spike-in RNAs had low starting amounts of RNA, likely due to the cell being dead or stressed which may result in the RNA being degraded.**    


```python
print(adata)
```

    AnnData object with n_obs × n_vars = 3401 × 23433
        obs: 'cell_ontology_class', 'subtissue', 'mouse.sex', 'mouse.id', 'plate.barcode'
        var: 'ERCC'



```python
# The calculate_qc_metrics function returns two dataframes: 
# one containing quality control metrics about cells, 
# and one containing metrics about genes. 
qc = sc.pp.calculate_qc_metrics(adata, qc_vars = ['ERCC'])

cell_qc_dataframe = qc[0]
gene_qc_dataframe = qc[1]
```


```python
print(cell_qc_dataframe.head(2))
```

                           n_genes_by_counts  log1p_n_genes_by_counts  \
    cell                                                                
    A1.B003290.3_38_F.1.1               3359                 8.119696   
    A1.B003728.3_56_F.1.1               1718                 7.449498   
    
                           total_counts  log1p_total_counts  \
    cell                                                      
    A1.B003290.3_38_F.1.1      390075.0           12.874097   
    A1.B003728.3_56_F.1.1      776439.0           13.562474   
    
                           pct_counts_in_top_50_genes  \
    cell                                                
    A1.B003290.3_38_F.1.1                   25.884766   
    A1.B003728.3_56_F.1.1                   43.051933   
    
                           pct_counts_in_top_100_genes  \
    cell                                                 
    A1.B003290.3_38_F.1.1                    32.847017   
    A1.B003728.3_56_F.1.1                    52.912721   
    
                           pct_counts_in_top_200_genes  \
    cell                                                 
    A1.B003290.3_38_F.1.1                    42.219573   
    A1.B003728.3_56_F.1.1                    65.313309   
    
                           pct_counts_in_top_500_genes  total_counts_ERCC  \
    cell                                                                    
    A1.B003290.3_38_F.1.1                    59.472666            10201.0   
    A1.B003728.3_56_F.1.1                    87.315423            67351.0   
    
                           log1p_total_counts_ERCC  pct_counts_ERCC  
    cell                                                             
    A1.B003290.3_38_F.1.1                 9.230339         2.615138  
    A1.B003728.3_56_F.1.1                11.117688         8.674345  



```python
print(gene_qc_dataframe.head(2))
```

                   n_cells_by_counts  mean_counts  log1p_mean_counts  \
    0610005C13Rik                 28     0.118201           0.111721   
    0610007C21Rik               2399   206.211990           5.333742   
    
                   pct_dropout_by_counts  total_counts  log1p_total_counts  
    0610005C13Rik              99.176713         402.0            5.998937  
    0610007C21Rik              29.461923      701327.0           13.460731  



```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(cell_qc_dataframe['pct_counts_ERCC'], bins=1000, color = 'gold')
plt.xlabel('Percent counts ERCC')
plt.ylabel('N cells')
```

![cell filtering](https://github.com/WZHOU007-0912/images/raw/master/QC-cell.png)

The majority of cells have less than 10% ERCC counts,   
but there's a long tail of cells that have very high spike-in counts; **these are likely dead cells and should be removed**:


```python
low_ERCC_mask = (cell_qc_dataframe['pct_counts_ERCC'] < 10)
adata = adata[low_ERCC_mask]
```

In addition to ensuring sufficient sequencing depth for each sample, we also want to make sure that the reads are distributed across the transcriptome.   
Thus, we count the total number of unique genes detected in each sample.


```python
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(cell_qc_dataframe['n_genes_by_counts'], bins=100, color = 'plum')
plt.xlabel('N genes')
plt.ylabel('N cells')
```

![cell distribution](https://github.com/WZHOU007-0912/images/raw/master/output_25_1.png)

> If detection rates were equal across the cells then the distribution should be approximately normal. 

**Thus we remove those cells in the tail of the distribution (fewer than ~1000 detected genes):**


```python
sc.pp.filter_cells(adata, min_genes = 1000)
```

    Trying to set attribute `.obs` of view, copying.


#### 4.2 Gene filtering

> A gene is defined as **detectable** if at least two cells contain more than 5 reads from the gene (however, the threshold strongly depends on the sequencing depth).  
  
It is typically a good idea to remove genes whose expression level is considered **"undetectable"**:


```python
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(gene_qc_dataframe['n_cells_by_counts'], bins=1000, color = 'darkseagreen')
plt.xlabel('N cells expressing > 0')
plt.ylabel('log(N genes)') # for visual clarity
plt.yscale('log') 
```

![genes filtering](https://github.com/WZHOU007-0912/images/raw/master/output_31_0.png)


```python
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(gene_qc_dataframe['total_counts'], bins=1000, color = 'mediumpurple')
plt.xlabel('Total counts')
plt.ylabel('log(N genes)') # for visual clarity
plt.yscale('log') 
```

![genes filtering](https://github.com/WZHOU007-0912/images/raw/master/output_32_0.png)


```python
sc.pp.filter_genes(adata, min_cells = 2)
sc.pp.filter_genes(adata, min_counts = 5)
```

### 5.  Normalization

#### 5.1 Normalizing cell library size - CPM

> Library sizes vary for many reasons, including natural differences in cell size, variation of RNA capture.  
While the volume of a cell is informative of a cell's phenotype, there is much more variation in size due to technical factors, and so cells are commonly normalized to have comparable RNA content, becuase this is known to exclude much more technical than biological variation.   
  
> The simplest way to normalize this data is to convert it to **counts per million** (CPM) by dividing each row by a size factor (the sum of all counts in the row), then multiplying by 1,000,000.  
Note that this method assumes that each cell originally contained the same amount of RNA.  
A potential **drawback of CPM** is if your sample contains genes that are both very highly expressed and differentially expressed across the cells. In this case, the total molecules in the cell may depend of whether such genes are on/off in the cell and normalizing by total molecules may hide the differential expression of those genes and/or falsely create differential expression for the remaining genes.  
One way to mitigate this is to **exclude highly expressed genes** from the size factor estimation.


```python
adata.raw = adata.copy()
```


```python
sc.pp.normalize_total(adata, target_sum = 1e6, 
                      exclude_highly_expressed = True)
```

#### 5.2 Normalizing gene expression - centering and scaling


```python
sc.pp.log1p(adata)
sc.pp.scale(adata)
```

### 6. Dimension reduction - Autoencoders


```python
print(adata)
```

    AnnData object with n_obs × n_vars = 3191 × 18877
        obs: 'cell_ontology_class', 'subtissue', 'mouse.sex', 'mouse.id', 'plate.barcode', 'n_genes'
        var: 'ERCC', 'n_cells', 'n_counts', 'mean', 'std'
        uns: 'log1p'



```python
import numpy as np 
from keras.layers import Dense, Dropout
from keras.optimizers import Adam 
from keras.models import Sequential, Model 
from keras import regularizers

model = Sequential()
model.add(Dense(40,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(40,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(35,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(30,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(20,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(10,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(4,
                activation='linear',
                kernel_initializer='he_uniform', name="bottleneck"))
model.add(Dense(10,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(20,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(30,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(35,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(40,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(40,
                activation='elu',
                kernel_initializer='he_uniform'))
model.add(Dense(adata.shape[1], activation='tanh'))
model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 0.0005))
```


```python
history = model.fit(adata.X, adata.X, batch_size = 100, epochs= 300, 
                   shuffle = False, verbose = 0)

print("\n" + "Training Loss: ", history.history['loss'][-1])
plt.figure(figsize=(10, 10)) 
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
```

![loss](https://github.com/WZHOU007-0912/images/raw/master/output_44_1.png)


```python
from sklearn.manifold import TSNE

encoder = Model(model.input, model.get_layer('bottleneck').output)
bottleneck_representation = encoder.predict(adata.X)

model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, 
                       random_state = 0, perplexity = 50, 
                       n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
```

    [t-SNE] Computing 151 nearest neighbors...
    [t-SNE] Indexed 3191 samples in 0.001s...
    [t-SNE] Computed neighbors for 3191 samples in 0.113s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 3191
    [t-SNE] Computed conditional probabilities for sample 2000 / 3191
    [t-SNE] Computed conditional probabilities for sample 3000 / 3191
    [t-SNE] Computed conditional probabilities for sample 3191 / 3191
    [t-SNE] Mean sigma: 1.310199
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 59.288960
    [t-SNE] KL divergence after 1000 iterations: 0.537335



```python
import matplotlib.colors as colors
import matplotlib.cm as cmx

uniq = list(set(adata.obs['cell_ontology_class']))

z = range(1,len(uniq))
hot = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax = len(uniq))
scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = hot)


plt.figure(figsize = (10, 10))
for i in range(len(uniq)):
    indx = adata.obs['cell_ontology_class'] == uniq[i]
    plt.scatter(tsne_auto[indx, 0], 
                tsne_auto[indx, 1],
                s = 30, color = scalarMap.to_rgba(i), 
                label = uniq[i])
    
    
plt.title('Autoencoder')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(loc='upper left')
plt.show()
```

![t-SNE](https://github.com/WZHOU007-0912/images/raw/master/output_46_0.png)

### 7. Clustering

#### 7.1 Kmeans clustering


```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 4)
km_fit = km.fit(tsne_auto)
km_labels = km_fit.labels_

plt.figure(figsize = (10, 10))
plt.scatter(tsne_auto[:, 0], 
            tsne_auto[:, 1], 
            c = km_labels.astype(np.float))
```

![k-means](https://github.com/WZHOU007-0912/images/raw/master/output_49_1.png)

performance evaluation:


```python
from sklearn import metrics

metrics.adjusted_rand_score(labels_true = adata.obs['cell_ontology_class'], 
                            labels_pred = km_labels)
```




    0.7601204227996344



Since we have the knowledge of class assignments `adata.obs['cell_ontology_class']` and our clustering algorithm assignments of the same samples `labels_pred` (`km_labels`), we can use the `adjusted Rand index` function that measures the similarity of the two assignments, for which perfect labeling is scored 1.0 while bad  is close to 0.0.  
We can see that the `adjusted Rand index` for Kmeans clustering is 0.76.

#### 7.2 Agglomerative clustering


```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters = 4).fit(tsne_auto)
agg_labels = agg.labels_ 

plt.figure(figsize = (10, 10))
plt.scatter(tsne_auto[:, 0], 
            tsne_auto[:, 1], 
            c = agg_labels.astype(np.float))
```

![agglomerate](https://github.com/WZHOU007-0912/images/raw/master/output_54_1.png)


```python
metrics.adjusted_rand_score(labels_true = adata.obs['cell_ontology_class'], 
                            labels_pred = agg_labels)

```




    0.8391778928818092



Here we see that Agglomerative clustering does a better job of cell clustering than Kmeans with the score of 0.84.

### 8. Differential expression   
>For well-defined cell types, we expect marker genes to show large differences in expression between the cell type of interest and the rest of the dataset, allowing us to use simple methods. 


```python
raw = pd.DataFrame(data=adata.raw.X, index=adata.raw.obs_names, columns=adata.raw.var_names)
```


```python
adata.obs['agg'] = agg_labels
adata.obs['agg'] = adata.obs['agg'].astype(str)
```


```python
adata.obs['agg']
```




    cell
    A1.B003290.3_38_F.1.1      2
    A1.B003728.3_56_F.1.1      3
    A1.MAA000560.3_10_M.1.1    0
    A1.MAA000564.3_10_M.1.1    1
    A1.MAA000923.3_9_M.1.1     3
                              ..
    P9.MAA000926.3_9_M.1.1     2
    P9.MAA000930.3_8_M.1.1     3
    P9.MAA000932.3_11_M.1.1    1
    P9.MAA000935.3_8_M.1.1     0
    P9.MAA001894.3_39_F.1.1    3
    Name: agg, Length: 3191, dtype: object




```python
cluster3 = raw[adata.obs['agg'] == '3']
notcluster3 = raw[adata.obs['agg'] != '3']
```

Say we are interested in genes named **Gja1**:


```python
astrocyte_marker = 'Gja1'
cluster3_marker_exp = cluster3[astrocyte_marker].values
notcluster3_marker_exp = notcluster3[astrocyte_marker].values

plt.figure(figsize = (10, 10))
plt.hist(cluster3_marker_exp, bins=100, 
         color='Mediumpurple',alpha=0.4, label='Cluster 3')
plt.hist(notcluster3_marker_exp, bins=100, 
         color='Deeppink', alpha=0.9, label='not Cluster 3')
plt.ylim(0,200)
plt.xlabel('%s expression'%astrocyte_marker) 
plt.ylabel('N cells')
plt.legend()
plt.show()

```

![distribution](https://github.com/WZHOU007-0912/images/raw/master/output_63_0.png)

#### 8.1 T-test  
Since we expect the differences in expression to be relatively large for marker genes, we can use hypothesis testing methods. In our case, $\sigma$ for the population is unknown and t-test will be used in here.   
$h_0$: The cluster 3 and non-cluster 3 populations had no true difference in mean expression  
$h_1$: The difference in expression between cluster 3 population and non-cluster 3 is significant


```python
from scipy.stats import ttest_ind

ttest = ttest_ind(cluster3_marker_exp, 
          notcluster3_marker_exp, 
          equal_var=False, 
          nan_policy='omit') 
print(ttest)
```

    Ttest_indResult(statistic=10.360312783138038, pvalue=9.01980462938619e-23)


The result for p-value tells us that we reject the null hypothesis, meaning that we can not say that the distibution of **Gja1's expression** in cluster 3 and in non-cluster 3 has no true difference.   
