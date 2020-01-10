#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from numpy import *


# In[5]:


df=pd.read_csv("uncleandata.csv") #read the file
df


# In[6]:


df.duplicated().sum() # total duplicated value in the table


# In[7]:


len(df.columns)


# In[8]:


len(df.index)


# In[9]:


df.loc[df.duplicated(), :] # I locate all rows that have duplicated value


# In[10]:


df.drop_duplicates() #remove dublicated value


# In[11]:


df[['satisfaction_level']].describe()


# In[12]:


df[['last_evaluation']].describe()


# In[13]:


df[['number_project']].describe()


# In[14]:


df[['average_montly_hours']].describe()


# In[15]:


df[['time_spend_company']].describe()


# In[16]:


df[['work_accident']].describe()


# In[17]:


df[['left']].describe()


# In[18]:


df[['promotion_last_5years']].describe()


# In[19]:


df[['is_smoker']].describe()


# In[20]:


df[['department']].describe()


# In[21]:


df[['salary']].describe()


# In[22]:


df.isnull().sum() #find the how many missing value in each features


# In[41]:


df.drop_duplicates(keep=False) #remove all duplicates in the table


# In[38]:


df.drop('is_smoker',axis=1)


# In[24]:


df.replace(NaN,0) #replace with the value 


# In[25]:


df.dropna(how='any') #I drop any rows in the columns that have at least 1 missing value


# In[29]:


df.dtypes #find the datatype of each column


# In[36]:


df.replace('yes',1).replace('no',0)


# In[37]:


#I combine all code from above to create a file again with different name : longd1.
df.drop_duplicates().drop('is_smoker',axis=1).replace(NaN,0).dropna(how='any').replace('yes',1).replace('no',0).to_csv('longd1.csv')


# In[ ]:




