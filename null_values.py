#pip install seaborn
import seaborn as sns
import numpy as np
import pandas as pd
df = sns.load_dataset('titanic')
df.isnull().sum()
df.shape()
df.dropna().shape
sns.histplot(df['age'])
# Mean value imputation
df['age_mean']= df['age'].fillna(df['age'].mean())
df[['age','age_mean']]
sns.histplot(df['age_mean'],kde=True)

# Median value imputation-if we have outliers in the dataset
df['age_median'] = df['age'].fillna(df['age'].median())
sns.scatterplot(df['age'])
sns.scatterplot(df['age_median'])
df[['age','age_mean','age_median']]
df.isnull().sum()
df['embarked'].unique()

# Mode imputation teq-categorical values
df[df['embarked'].isnull()]
mode_value = df[df['embarked'].notna()]['embarked'].mode()[0]
df['embarked_mode'] = df['embarked'].fillna(mode_value)
df['embarked_mode'] = df['embarked'].fillna(mode_value)
df['embarked_mode'].isnull().sum()
df['embarked'].isnull().sum()
