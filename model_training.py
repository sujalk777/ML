import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df =pd.read_csv(Algerian_forest_fires_cleaned_dataset.csv)
df.columns
df.drop(['day','month','year'],axis=1,inplace= True)
df.value_counts()
df['classes']=np.where(dp['classes'].str.contains("not fire"),0,1)
df.tail()
#independent and dependent feature
x=df.drop('FWI',axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=43)
xtrain.shape,xtest.shape
## feature selection and correlation
xtrain.corr
##check for multicollinearity
plt.figure(figsize=12,10)
corr=xtrain.corr()
sns.heatmap(corr,annot=True)

def correlation(dataset,threshold):
  col_corr=set()
  corr_matrix=dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        colname=corr_matrix.columns[i]
        col_corr.add(columns)
  return col_corr
#threshold --domain expertise
corr_feature=correlation(xtrain,0.85)
corr_feature
#drop feature where correlatioj is more than 0.85
xtrain.drop(corr_feature,axis=1,inplace=True)
xtest.drop(corr_feature,axis=1,inplace=True)
xtrain.shape,xtest.shape

#Feature scaling or stndardization
from sklearn.preprocessing import StandardScaler
sclaer =Standardscaler()
xtrain_scaled=scaler.fit_transform(xtrain)
#box plot to understand effect of standard scaler
plt.subplot(figsize=(15,5))
pt.subplot(1,2,1)
sns.boxplot(data=xtrain)
plt.title("Xtrain before scaling")
plt.subplot(1,2,2)
sns.boxplot(data=xtrain_scaled)
plt.title("xtrain after scaling")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
linreg=LinearRegression()
linreg=fit(xtrain_scaled,ytrain)
ypred=linreg.predict(xtrain_scaled)
print("mean absolute error:",mean_absolute_error(ytest,ypred))
print("R2 score: " r2_score(ytest,yperd))

from sklearn.linear_model import lasso
lasso=lasso()
lasso.fit(xtrain_scaled,ytrain)
ypred=lasso.predict(xtest_scaled)
print("mean_absolute_error: ",mean_absolute_error(ytest,ypred))
print("R2_score: ",r2_score(ytest,ypred))
from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(xtrain_scaled,ytrain)
ypred=lassocv.predict(xtest_scaled)
print("mean_absolute_error  :",mean_absolute_error(ytest,pred))
print("R2 score: ",r2_score(ytest,pred))

from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(xtrain_scaled,ytrain)
ypred=ridge.predict(xtest_scaled)
print("mean_avsolute_error:"mean_absolute_eror(ytest,ypred))
print("R2 score: ",r2_score(ytest,ypred))

from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV()
ridgecv.fit(xtrain_scaled,ytrain)
ypred=ridgecv.predict(xtest_scaled)
print("mean_absolute_error ",mean_absolute_eror(ytest,ypred))
print("R2 score:"r2_score(ytest,ypred))

from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(xtrain_scaled,ytrain)
ypred=elastic.predict(xtest_scaled)
print("mean absolute error :",mean_absolute_error(ytest,ypred))
print("R2 score: ",r2_score(ytest,ypred))

