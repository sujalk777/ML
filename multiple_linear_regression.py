# y = mx+c / y = b1*x1 +b2*x2+....bn*xn +b0
# california housing dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
california = fetch_california_housing()
california
california.keys()
print(california.DESCR)
california.target_names
california.data
dataset = pd.DataFrame(california.data,columns=california.feature_names)
dataset
dataset['price'] = california.target
dataset
dataset.isnull().sum()
dataset.describe()
dataset.corr()
sns.heatmap(dataset.corr(),annot=True)
dataset.head()
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain ,xtest ,ytrain ,ytest = train_test_split(x,y,test_size=0.3,random_state=42)
xtrain.shape,xtest.shape
from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler()
xtrain = sclaer.fit_transform(xtrain)
xtest = sclaer.transform(xtest)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression
regression.fit(xtrain,ytrain)
##  slope or coff
regression.coef_
# intercept
regression.intercept_
yprd = regression.predict(xtest)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mean_squared_error(ytest,yprd)
mean_absolute_error(ytest,yprd)
r2_score(ytest,yprd)
# for saving the trained model we have to use pickle
import pickle
pickle.dump(regression,open('modelLR.pkl','wb'))
model = pickle.load(open('modelLR.pkl','rb'))
model.predict(xtest)
