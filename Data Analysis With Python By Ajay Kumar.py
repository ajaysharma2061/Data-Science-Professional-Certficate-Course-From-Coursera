#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing the Data

# In[2]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[3]:


df.head()


# # Question 1

# In[6]:


print(df.dtypes)


# In[7]:


df.describe()


# # Question 2

# In[8]:



df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
df.describe()


# In[9]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[10]:


mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace= True)


# In[11]:


mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace = True)


# In[12]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # Question 3

# In[13]:


df['floors'].value_counts().to_frame()


# # Question 4

# In[14]:


sns.boxplot(x='waterfront', y='price', data=df)


# # Question 5

# In[15]:


sns.regplot(x='sqft_above', y= 'price',data = df)


# In[16]:


df.corr()['price'].sort_values()


# # Model Development

# In[17]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[18]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# # Question 6

# In[19]:


X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)


# # Question 7 

# In[20]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]


# In[21]:


X = df[features]
Y= df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)


# In[22]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# # Question 8

# In[23]:


pipe=Pipeline(Input)
pipe


# In[24]:


pipe.fit(X,Y)


# In[25]:


pipe.score(X,Y)


# # Evaluation and REFINEMENT

# In[26]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[27]:


eatures =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# # Question 9

# In[28]:


from sklearn.linear_model import Ridge


# In[29]:


RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)


# # Question 10

# In[30]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = Ridge(alpha=0.1)
poly.fit(x_train_pr, y_train)
poly.score(x_test_pr, y_test)


# In[ ]:




