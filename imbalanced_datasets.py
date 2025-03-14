# in this we follow 2 appraoches
# up sampling
#down sampling
import numpy as np
import pandas as pd
# set the random seed for reproducibility
np.random.seed(123)
# Create a dataframe with two classes
n_sample = 1000
class_0_ratio = 0.9
n_class_0 = int(n_sample*class_0_ratio)
n_class_1 = n_sample - n_class_0
n_class_1, n_class_0

#CREATE A DATAFRAME WITH IBALANCEC DATASET
class_0 = pd.DataFrame({
    'feature 1': np.random.normal(loc=0,scale=1,size=n_class_0),
    'feature 2': np.random.normal(loc=0,scale=1,size=n_class_0),
    'Target': [0]*n_class_0

})
class_1 = pd.DataFrame({
    'feature 1': np.random.normal(loc=2,scale=1,size=n_class_1),
    'feature 2': np.random.normal(loc=2,scale=1,size=n_class_1),
    'Target': [1]*n_class_1

})
df = pd.concat([class_0,class_1]).reset_index(drop=True)
df['Target'].value_counts()

# upsampling
df_minority = df[df['Target']==1]
df_majority = df[df['Target']==0]
df_minority
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority,replace=True,#sample with replacment
                                 n_samples=len(df_majority),
                                 random_state=42)
df_minority.shape
df_minority_upsampled.shape
df_minority_upsampled
df_upsampled = pd.concat([df_majority,df_minority_upsampled])
df_upsampled
df_upsampled['Target'].value_counts()
df_majority_downsampled = resample(df_majority,replace=False,
                                   n_samples= len(df_minority),
                                   random_state=42)
df_majority_downsampled.shape
df_downsampled = pd.concat([df_minority,df_majority_downsampled])
df_downsampled['Target'].value_counts()
