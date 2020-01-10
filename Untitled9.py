#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.datasets import load_digits


# In[6]:


DigitData =load_digits()


# In[9]:


from sklearn import preprocessing
Data_scaled= preprocessing.scale(DigitData.data)


# In[10]:


Data_scaled


# In[15]:


import matplotlib.pyplot as plt


# In[24]:


images_and_labels = list(zip(DigitData.images, DigitData.target))


# In[25]:


for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots 
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))

plt.show()


# In[31]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(DigitData.data, DigitData.target, DigitData.images, test_size=0.20, random_state=42)


# In[37]:


from sklearn import cluster
clusterss = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clusterss.fit(X_train)


# In[40]:


y_pred=clusterss.predict(X_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))


# In[ ]:




