#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


iris=datasets.load_iris()  #load the dataset


a=np.array(iris['data'])   
b=np.array(iris['target'])  #a and b have to be concatenated to be sent as an 'data' argument in p.Dataframe

new_data=np.column_stack((a,b))   #concatenated
col=np.char.add(iris['feature_names'],['target'])    #name of the coloumns
df=pd.DataFrame(new_data,columns=iris['feature_names']+['species'])  #final dataframe




df.species.replace([0,1,2], ['setosa', 'versicolor', 'virginica'], inplace=True)  # replacing target number with target names
df.head(10)



df_setosa=df.loc[df['species']=='setosa']
df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']

#Histogram

g = sns.FacetGrid(df, hue="species",size=5)
g.map(sns.distplot,'petal length (cm)')
plt.legend(['setosa','versicolor','virginica'])
plt.show()



fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

# Box plot


sns.boxplot(x='species',y='petal length (cm)',data=df)
plt.grid()
plt.show()


# QQ plot

import statsmodels.api as sm
fig.set_size_inches(11.7, 8.27)

sm.qqplot(df_setosa['petal length (cm)'], line='s')
plt.xlabel('Setosa', fontsize=12)
plt.show()
sm.qqplot(df_versicolor['petal length (cm)'], line='s')
plt.xlabel('versicolor', fontsize=12)
plt.show()
sm.qqplot(df_virginica['petal length (cm)'], line='s')
plt.xlabel('virginica', fontsize=12)
plt.show()








# Applying shapiro wilk test


from scipy import stats
from scipy.stats import shapiro
stat, p = shapiro(df_setosa['petal length (cm)'])
print('p value for petal length for setosa :',p)
if p>0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')




stat, p = shapiro(df_versicolor['petal length (cm)'])
print('p value for petal length for versicolor :',p)
if p>0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')




stat, p = shapiro(df_virginica['petal length (cm)'])
print('p value for petal length for virginica :',p)
if p>0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')






