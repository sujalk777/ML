import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
path = r"C:\Users\kaifk\lpth\.vscode\DataSciencePrac\002-random-forest-example\mobile_ads.csv"
df = pd.read_csv(path)
df.describe()
df.shape()
df = df.drop_duplicates() #dropped the duplicated values if any 
