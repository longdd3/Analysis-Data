#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


# In[3]:


DigitData =load_digits()


# In[4]:


DigitData


# In[5]:


DigitData.data.shape


# In[6]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)


# In[7]:


trs= pca.fit_transform(DigitData.data)


# In[8]:


trs


# In[9]:


plt.scatter(trs[:, 0], trs[:, 1], c=DigitData.target, cmap="Paired")


# In[10]:


#we saw the color of each compoment


# In[12]:


pca.explained_variance_ratio_


# In[ ]:


#here,  the first principal component contains 14.89% of the variance and the second principal component contains 13.61% of the variance
#Together, the two components contain 28,40% of the information. It is not sufficent , we should find more components

