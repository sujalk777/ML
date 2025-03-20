# wine dataset
import pandas as pd
df = pd.read_csv('winequality-red (1).csv')
df.head()
df.info()
# descriptive summary of dataset
df.describe()
df.shape
df.columns
df['quality'].unique()
## missing values in the dataset
df.isnull().sum()
## Duplicate records
df[df.duplicated()]
# reamove the duplicates
df.drop_duplicates(inplace=True)
df.corr()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
df.quality.value_counts().plot(kind='bar')
plt.xlabel("wine quality")
plt.ylabel("count")
plt.show()
for colum in df.columns:
    sns.histplot(df[colum],kde=True)
sns.histplot(df['alcohol'])
sns.catplot(x='quality',y='alcohol',data=df,kind='box')
sns.scatterplot(x='alcohol',y='pH',hue='quality',data=df)

