from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
%matplotlib inline
X,y=make_moons(n_samples=250,noise=0.05)
plt.scatter(X[:,0],X[:,1])
##feature scaling(Standard Scaling)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled
dbcan=DBSCAN(eps=0.3)
dbcan.fit(X_scaled)
dbcan.labels_
plt.scatter(X[:,0],X[:,1],c=dbcan.labels_)
# <matplotlib.collections.PathCollection at 
plt.scatter(X[:,0],X[:,1],c=y)
