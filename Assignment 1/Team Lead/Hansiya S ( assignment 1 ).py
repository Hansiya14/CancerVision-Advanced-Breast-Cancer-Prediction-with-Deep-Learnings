#!/usr/bin/env python
# coding: utf-8

# In[2]:


pwd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('House Price India.csv')
df.head()
sns.displot(df.id)
sns.displot(df.Date)
sns.lineplot(x=df.id, y=df.Date)
sns.pairplot(df)


# In[1]:


pwd


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv("House Price India.csv")
df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().any()


# In[8]:


df.describe()


# In[9]:


import seaborn as sns


# In[10]:


sns.displot(df.Date)


# In[11]:


import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[12]:


plt.pie(df.living_area_renov)
plt.title('LAR')
plt.show


# In[13]:


sns.lineplot(x=df.Price,y=df.Date)


# In[14]:


sns.scatterplot(x=df.Price,y=df.lot_area_renov)


# In[15]:


sns.lineplot(x=df.Price,y=df.lot_area_renov)


# In[16]:


sns.pairplot(df)


# In[17]:


df.corr()


# In[19]:


sns.heatmap(df.corr(),annot=True)


# In[21]:


df.head()


# In[22]:


sns.boxplot(df.living_area_renov)


# In[23]:


df.describe(include=['float'])


# In[ ]:




