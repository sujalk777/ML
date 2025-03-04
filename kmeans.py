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
for k in range(2,11):
  kMeans=Kmeans(n_clusters=k,init='k-means++')
  kMeans.fit(x.train)
  score=silhoutte_score(xtrain,kMeans.lebels_)
  silhoutte_score.append(score)
silhoutte_score
plt.plot(range(2,11),silhoutte__score)
plt.xticks(range(2,11))
plt.xlabel("No of Cluster")
plt.ylabel("score")
plt.show()


# HIERARICHAL CLUSTERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn import datasets
#import iris dataset 
iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data)
iris_data.columns=iris.features_name
iris_data
#standardization
from sklearn.preprocessign import StandardScaler
scaler=StandardScaler()
x_sclaed=scaler.fit_transform(iris_data)
# Apply PCA
from sklearn.decomposition import PCA
pca_scaled=pca.fit_transform
plt.scatter(pca_scaled[:,0])
# agglomerative clustering
# to construct a dendogram
import scipy.cluster.hierarchy as sc 
# plot diagram
plt.figure(figsize=(20,7))
plt.title("Dendogram")

# create dendogram
sc.dendogram(sc.linkage(pca_scaled,method='ward'))
plt.title("Dendogram")
plt.xlabel("sample Index")
plt.ylabel("Euclidean Distance")

from sklearn.cluster import Agglomerativeclustering
cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
cluter.fit(pca_scaled)
cluster.labels_
plt.scatter(pca_scaled[:0],pca_scaled[:,1])

# silhoutte score
from sklearn.metrics import silhoutte_score
silhoutte_coff=[]
for k in range(2,11):
  agglo=AgglomerativeClustering(n_cluster=2,affinity='euclidean',linkage='ward')
  agglo.fit(x_scaled)
  score=silhoutte_score(x_scaled,agglo.labels_)
  silhoutte_coff.append()

plt.plot(range(2,11),silhoutte_coff)
plt.xticks(range(2,11))
plt.xlabel("No of cluster")
plt.ylabel("Silhoutte_coff")
plt.show()













