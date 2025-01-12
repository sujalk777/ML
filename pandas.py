pip install pandas

import pandas as pd
df=pd.read_csv("services (1).csv")
df.head()
df.tail(3)
type(df)
df.info()
df.columns
df.dtypes
df.head(5)
df['location_id']
df[['appiiacation_process','audience']]
df1 =pd.read_excel('LUSID Excel - Setting up your market data (1).xlsx')
df.head()
df.dtypes
df.dtypes=='int64'
df.dtypes[df.dtypes=='int64'].index
df[df.dtypes[df.dtypes=='int64'].index]
df[df.dtypes[df.dtypes=='object'].index].head()
df[df.dtypes[df.dtypes=='float64'].index]
df.describe()
df[['fees','location_id']][12:20]
df[['fees','location_id']][10:20:2]
df.info()
df['Pandas']=0
df['Pandas']=0
df['Pandas']=0
df['new_col']= df['location_id']+df['program_id']
df.info()
df['new_col']



df =pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.head()
df.tail()
df.columns
df.dtypes
df[['Name','Sex','Age','Ticket']]
df.dtypes=='object'
df.dtypes[df.dtypes=='object'].index
df[df.dtypes[df.dtypes=='object'].index]
df[df.dtypes[df.dtypes=='object'].index].describe()
df['Cabin']
pd.Categorical(df['Cabin'])
df['cabin'].unique()
df[df['age']>18]
len(df)-len(df[df['age']>18])
df.describe()
df[df['Fare']<32.204208]
df[df['Fare']==0]['Name']
len(df[df['Fare']==0])
df[(df['Sex'] == 'female')& (df['Fare']>32)]
df[df['Fare']==max(df['Fare'])]['Name']
len(df)
df[0:100:3]
df.iloc[0:5,[0,1,2,3]] //integer location
