
import numpy as np
l = [1,2,3,4,5]
l
type(l)
arr = np.array(l)
arr
np.asarray(l)
arr1 = np.array([[1,2,3],[2,3,4]])
arr1
arr.ndim
arr1.ndim
l
np.matrix(l)
arr
arr[0] = 50
arr
b = np.copy(arr)
b[0]=34
arr
b
list(i*i for i in range(5))
np.fromstring('23 45 56',sep=' ')
np.fromstring('23,45,56',sep=',')
arr1
arr1.size
arr1.shape
arr1.dtype
list(range(5))
a=list(range(9,20))
list(range(0.4,10.4))
list(np.arange(0.4,10.4))
list(np.arange(0.4,10.4,0.2))
np.linspace(1,5,20)
np.zeros(5)
np.zeros((3,4))
np.ones(5)
arr =np.ones((3,4))
arr
arr + 5
arr*4
arr1 = np.eye(5)
arr1
import pandas as pd
e =pd.DataFrame(arr1)
e
e[1]
np.rnadom.rand(2,3)
np.random.randn(2,3)
arr2 = np.random.randint(1,5,(3,4))
arr2
arr1 = np.random.randint(1,3,(3,3))
arr2 = np.random.randint(1,3,(3,3))
arr1
arr2
arr1+arr2
arr1*arr2
