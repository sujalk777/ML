# data interpolation is the process of estimating unknown value within a dataset based on the knowwn values.in python, there are vaious libraries available tht can be used for interpolation, such as numpy,scipy, and pandas
# Here is an expaple of how to perform data interpolation the numpy library

#LINEAR INTERPOLATION 
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])
plt.scatter(x,y)

x_new=np.linspace(1,5,10)
y_interp=np.interp(x_new,x,y)
plt.scatter(x_new,y_interp)


# CUBIC INTERPOLATION
x = np.array([1,2,3,4,5])
y = np.array([1,8,27,64,125])
plt.scatter(x,y)
from scipy.interpolate import interp1d
## create a cubic interpolation function 

f = interp1d(x,y,kind='cubic')
#interpolate the data
x_new = np.linspace(1,5,10)
y_interp =  f(x_new)
plt.scatter(x_new,y_interp)

# POLYNOMIAL INTERPOLATION
x = np.array([1,2,3,4,5])
y = np.array([1,4,9,16,25])
plt.scatter(x,y)
# interpolate the data using polynomial interpolation
p = np.polyfit(x,y,2)
x_new = np.linspace(1,5,20)#create new x value
y_interp = np.polyval(p,x_new)# interpolate y value
plt.scatter(x_new,y_interp)
