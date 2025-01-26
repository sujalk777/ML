import matplotlib.pyplot as plt
import numpy as np
x = np.random.rand(50)
y = np.random.rand(50)
x
y
plt.scatter(x,y)
plt.figure(figsize=(6,2))
plt.scatter(x,y,c='red',alpha=0.9,marker='x')
plt.xlabel("this is x axis")
plt.ylabel("this is y axis")
plt.title("this is x vs y ")
plt.grid()
x=np.linspace(1,10,100);
y=np.sin(x)
plt.scatter(x,y)
plt.figure(figsize=(8,2))
plt.plot(x,y,'--r')
plt.xlabel("this is x data")
plt.ylabel("this is y data")
plt.title("this is sin wave graph")
plt.grid()
x = ['a','b','c','d','e']
y=np.random.rand(5)
plt.figure(figsize=(5,3))
plt.bar(x,y,color='green')
plt.xlabel("this rep cato data")
plt.ylabel("this is num data")
plt.title("this is bar graph")
plt.grid()
plt.figure(figsize=(5,3))
plt.barh(x,y,color='purple')
plt.xlabel("this rep cato data")
plt.ylabel("this is num data")
plt.title("this is bar graph")
plt.grid()
x = [3,4,5,6,7,8]
y=[4,5,6,7,8,9]
plt.figure(figsize=(6,2))
plt.plot(x,y)
data= [1,2,3,4,5,6,5,6,6,7,7,8,8,8,8,8]
plt.figure(figsize=(6,3))
plt.hist(data,color='red')
plt.xlabel("this rep unique val")
plt.ylabel("represent frequency")
plt.title("hist plot exapmle")
plt.grid()
plt.savefig("test.png")
