# we are algerian forest dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset=pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv' ,header=1)
dataset.head()
# DATA CLEANING
## missing values
dataset[dataset.isnull().any(axis=1)]
# add new colmn to the region
dataset.loc[:122,"Region"]=0
dataset.loc[122:,"Region"]=1
df=dataset
df[['Region']]=df[['Region']].astype(int)
df.head()
df.isnull().sum()
## Removing the null values
df=df.dropna().reset_index(drop=True)
df.head()
df.isnull().sum()
df.iloc[[122]]
##remove the 122nd row
df=df.drop(122).reset_index(drop=True)
