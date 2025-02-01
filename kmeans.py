import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=1000,centers=3,n_features=2,random_state=23)
plt.scatter(x[:,0],x[:,1])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.cluster import Kmeans
wcsss=[]
for k in range(1,11):
  kmeans=Kmeans(n_cluster=k,init='k-means++')
  kmeans.fit(xtrain)
  wcss.append(kmeans.inertia_)
wcss
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("No of Cluster")
kmeans=Kmeans(n_cluster=3,init='kmeans++')
y_label=Kmeans.fit_predict(xtrain)
plt.scatter(xtrain[:,0],xtrain[:,1],c=y_label)

from kneed import KneeLocator
k1=KneeLocator(range(1,11),wcss,curvw='convex',direction='decreasing')
k1.elbow

from sklearn.metrics import silhoutte_score
silhoutte_score=[]
