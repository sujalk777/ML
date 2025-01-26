import pandas as pd
df =  pd.read_csv("height-weight.csv")
#  divide daatset into independent and dependent feature
x = df[['Height']] # input
y = df['Weight'] # output
import matplotlib.pyplot as plt
plt.scatter(x,y)
#train test split
from sklearn.model_selection import train_test_split
xtrain ,xtest ,ytrain ,ytest = train_test_split(x,y,test_size=0.3,random_state=42)
#standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain.head()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
plt.scatter(xtrain,ytrain)
# model train 
from sklearn.linear_model import LinearRegression
reg =LinearRegression()
reg.fit(xtrain,ytrain)
print("the slop or coff of data is: ", reg.coef_)
print("intercept ",reg.intercept_)
plt.scatter(xtrain,ytrain)
plt.plot(xtrain,reg.predict(xtrain),'r')
ypred = reg.predict(xtest)
ypred
plt.scatter(xtest,ytest)
plt.plot(xtest,reg.predict(xtest),'r')
from sklearn.metrics import r2_score
r2_score(ytest,ypred)
xtrain
# new data is 150
new_data = 200
scaler_height = scaler.transform([[new_data]])
scaler_height
print("the weight prediction for height 151 is: ",reg.predict([scaler_height[0]]))
