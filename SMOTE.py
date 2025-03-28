# SMOTE (Synthetic Minority Over-sampling Technique) is a technique used in machine learning to address imbalanced datasets 
# where the minority class has significantly fewer instances than the majority class.
# SMOTE involves generating synthetic instances of the minority class by interpolating between existing instances.
from sklearn.datasets   import make_classification
x,y = make_classification(n_samples=1000,n_redundant=0,n_features=2,n_clusters_per_class=1,weights=[0.90],random_state=42)
x.shape()
y.shape()
len(y[y==1])
len(y[y==0])
import pandas as pd 
df1=pd.DataFrame(x,columns=['f1','f2'])
df2 = pd.DataFrame(y, columns=['traget'])
final_df = pd.concat([df1,df2],axis=1)
final_df
final_df['traget'].value_counts()
import matplotlib.pyplot as plt
plt.scatter(final_df['f1'],final_df['f2'],c=final_df['traget'])
from imblearn.over_sampling import SMOTE
#transform the dataset
oversample = SMOTE()
x,y = oversample.fit_resample(final_df[['f1','f2']],final_df['traget'])
x.shape
y.shape
len(y[y==0])
len(y[y==1])
df1=pd.DataFrame(x,columns=['f1','f2'])
df2 = pd.DataFrame(y, columns=['traget'])
oversample_df = pd.concat([df1,df2],axis=1)
import matplotlib.pyplot as plt
plt.scatter(final_df['f1'],final_df['f2'],c=final_df['traget'])
