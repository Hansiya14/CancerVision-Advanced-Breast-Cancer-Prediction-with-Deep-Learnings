#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


pwd


# In[6]:


df=pd.read_csv('House Price India.csv')
df.head()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


le=LabelEncoder()


# In[12]:


df['number of bedrooms'] = le.fit_transform(df['number of bedrooms'])


# In[13]:


df.head()


# In[14]:


x=df.iloc[:,:-1].values
y=df.iloc[:,4:5].values


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.4,random_state = 0)


# In[17]:


xtrain.shape,xtest.shape


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


regressor = Sequential()


# In[20]:


regressor.add(Dense(4,activation='relu'))


# In[21]:


regressor.add(Dense(12,activation='relu'))


# In[22]:


regressor.add(Dense(8,activation='relu'))


# In[23]:


regressor.add(Dense(9,activation='relu'))


# In[24]:


regressor.add(Dense(1,activation='linear'))


# In[25]:


regressor.compile(optimizer='adam',loss='mse',metrics=['mse'])


# In[26]:


regressor.fit(xtrain,ytrain,batch_size = 10,epochs=300)


# In[28]:


ypred = regressor.predict(xtest)
ypred


# In[29]:


from sklearn.metrics import r2_score


# In[30]:


r2_score(ytest,ypred)*100


# In[31]:


ypred.flatten()


# In[32]:


pd.DataFrame({'Actual_value':ytest.flatten(),'predicted_value':ypred.flatten()})


# In[ ]:




