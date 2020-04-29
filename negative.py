

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy




bat=pd.read_csv('battery_life.csv')
arr=np.array(bat['battery_percentage'])


# Histogram


g = sns.FacetGrid(bat,size=5)
g.map(sns.distplot,'battery_percentage',bins=10)
plt.show()

# Box Plot


sns.boxplot(y='battery_percentage',data=bat)
plt.show()

# QQ Plot


import statsmodels.api as sm
arr=np.array(np.array(bat['battery_percentage']))
sm.qqplot(arr, line='s')
plt.show()

# Applying shapiro wilk test


from scipy.stats import shapiro

stat, p = shapiro(np.array(bat['battery_percentage']))




print('p value',p)


# In[18]:


if p>0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:




