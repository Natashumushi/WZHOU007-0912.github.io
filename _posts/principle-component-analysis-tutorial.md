# Unsupervised Dimentionality Reduction  
**via Principle Component Analysis**

In the context of dimensionality reduction, feature extraction can be understood as an approach to data compression with the goal of maintaining most of the relevant information.  
Feature extraction is not only used to improve storage space or the computational efficiency of the learning algorithm, but can also improve the predictive performance by reducing the curse of dimensionality—especially if we are working with non-regularized models.  


> $x_1$,$x_2$ are original feature axes, $pc_1$,$pc_2$ are the principle components.

**Principle Component Analysis (PCA)** aims to find the *directions* of **maximum variance** in high dimensional data and projects it onto a new subspace with fewer dimensions than original one.  
The orthogonal axes (principle components) of the new subspace can be interpreted as the directions of maximum given the constraints that new features are **orthogonal** to each other.  
(*orthogonal means uncorrelated*)

- What is an orthogonal matrix?  

Say we have:  
  
  $$a_1^2 + a_2^2=1$$   
  $$b_1^2 + b_2^2=1$$    
  $$a_1 b_1 + a_2 b_2=0$$   
  $$b_1 a_1 + b_2 a_2=0$$    
  
that is:  

 $$\begin{bmatrix} a_1 & a_2\\ b_1 & b_2 \end{bmatrix} \begin{bmatrix} a_1 & b_1\\a_2 & b_2 \end{bmatrix} =
  \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix} $$  
   
 for $$ A = \begin{bmatrix} a_1 & a_2\\ b_1 & b_2 \end{bmatrix} $$  
     $$ {A^T}= \begin{bmatrix} a_1 & b_1\\ a_2 & b_2 \end{bmatrix} $$  
  
  $$AA^T = I$$    
    
Then A is an orthogonal matrix.
  


# Extracting the principle components step by step

# 1.Standardizing the data  
- PCA directions are highly sensitive to data scaling. We need to standardize the features **prior** to PCA.  
- We will start by loading the Wine datase from:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 


```python
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)

df_wine.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>



- The first row is the label and the rest are features.  
- Next, we process the Wine data into separate training and test sets—using 70 percent and 30 percent of the data.


```python
from sklearn.model_selection import train_test_split

X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size = 0.3,
                stratify = y, 
                random_state = 0)
```

- Standardize the features


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
```

# 2. Constructing the covariance matrix  
- The covariance between two features x<sub>i</sub> and x<sub>j</sub> on population level can be calculated as:  
  
 $$\sigma_ij = \sum_{i=1}^{n}(x_j^{(i)} - \mu_j)(x_k^{(i)}-\mu_k)$$
  
  
- Since we already have standardized dataset, then covariance matrix can be calculated as:  
  
  $$\Sigma = \frac{1}{n}X X^T$$    
  
  Where $\Sigma$ is the covariance matrix of features, $X$ is the feature matrix.  
    
      
- **Why we need to calculate the covariance matrix for features?**
  
  The covariance matrix is calculating the correlation between the matrix.  
  The goal of PCA can be interpreted as maximize the main diagonal (the covariance of feature and itself) while minimize  the rest of the diagonals (the covariance between different features).  


```python

```