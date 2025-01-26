import seaborn as sns
iris = sns.load_dataset('iris')
iris
iris['species'].unique()
sns.scatterplot(x=iris.sepal_length,y=iris.sepal_width)
sns.scatterplot(x=iris.sepal_length,y=iris.petal_length)
sns.displot(iris['sepal_length'])
tips = sns.load_dataset('tips')
tips
sns.scatterplot(x= tips.total_bill,y=tips.tip)
tips.head()
tips['smoker'].value_counts()
tips['size'].value_counts()
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,hue='smoker')
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,hue='size')
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,style='size')
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,style='size',hue='time')
sns.catplot(x='day',y='total_bill',data=tips)
sns.pairplot(iris)
df =sns.load_dataset('titanic')
df
df.isnull().sum()
df.shape()
df.dropna().shape
