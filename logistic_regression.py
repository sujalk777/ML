from sklearn.datasets import load_iris
dataset = load_iris()
print(dataset.DESCR)
dataset.keys()
import numpy as np
import pandas as pd
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df
df['Target'] = dataset.target
df.head()
df['Target'].unique()
df[df['Target']!=2]
df_copy['Target'].unique()
#independent and dependent feature
x = df_copy.iloc[:,:-1]
y=df_copy.iloc[:,-1]
from sklearn.model_selection import train_test_split
xtrain ,xtest ,ytrain ,ytest = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression
classification = LogisticRegression(max_iter=200)
classification.fit(xtrain,ytrain)
ypred = classification.predict(xtest)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_matrix(ytest,ypred)
accuracy_score(ytest,ypred)
print(classification_report(ytest,ypred))


#cross validation
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)
from sklearn.model_selection import cross_val_score
score = cross_val_score(classification,xtrain,ytrain,scoring='accuracy',cv=cv)
score
np.mean(score)
#complex verison
from sklearn.datasets import make_classification
x,y = make_classification(n_samples=1000,n_features=10,n_redundant=5,n_classes=2,random_state=1)
xtrain ,xtest ,ytrain,ytest= train_test_split(x,y ,test_size=0.3,random_state=42)
complex_class_model =  LogisticRegression(max_iter=200)
complex_class_model
complex_class_model.fit(xtrain,ytrain)
ypred = complex_class_model.predict(xtest)
confusion_matrix(ytest,ypred)
accuracy_score(ytest,ypred)
cv = KFold(n_splits=5)
score =cross_val_score(complex_class_model,xtrain,ytrain,cv=cv)
score
np.mean(score)



# Multiple linear regression
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
