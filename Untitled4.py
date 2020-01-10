#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori


# In[42]:



df=pd.read_csv("SuperCenterDataNew.csv",header=None)
display(df)


# In[47]:


df.shape


# In[ ]:


#number of colums
len(df.columns)


# In[ ]:


asscociation_rules = apriori(records, min_support=0.0055)

