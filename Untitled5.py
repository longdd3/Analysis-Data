#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("SuperCenterDataNew.csv")
df.head()


# In[ ]:


transactions = []
for i in range(0, 7501):
    transactions.append([str(df.values[i,j]) for j in range(0, 20)])


# In[ ]:


from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# In[ ]:





# In[10]:


def getConfidence(items_given, support_items, countDict):
    items_given.sort()
    support_items = support_items + items_given
    support_items.sort()
    items_given_str = "_".join(items_given)
    item_support_str = "_".join(support_items)
    item_support = item_support_str
    items = list(countDict.keys())
    if (items_given_str not in items) or (item_support not in items):
        return 0
    else:
        return (countDict[item_support]/ countDict[items_given_str])


# In[ ]:




