#!/usr/bin/env python
# coding: utf-8

# In[614]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[615]:


df=pd.read_csv("longd1.csv",index_col=0) #read the file,drop unamed column
df


# In[616]:


df['department'].value_counts()


# In[617]:


df['salary'].value_counts()


# In[618]:


df.dtypes #see what is the datatypes in this table


# In[619]:


#I change object in 'salary' column as 'low' to 1, 'medium' to  2, 'high' to 3 
#I change object in 'department' column as 'sales' to 1, 'technical' to  2, 'support' to 3, 'IT' to 4, 'product_mng' to 5, 'RandD' to 6, 'marketing' to 7,'accounting' to 8, 'hr' to 9 and , 'management' to 10      
replace=df.replace('sales',1).replace('technical',2).replace('support',3).replace('IT',4).replace('product_mng',5).replace('RandD',6).replace('marketing',7).replace('accounting',8).replace('hr',9).replace('management',10).replace('low',1).replace('medium',2).replace('high',3).to_csv('longd2.csv')


# In[620]:


df.shape


# In[621]:


df=pd.read_csv("longd2.csv",index_col=0)
df


# In[622]:


df.dtypes


# In[623]:


x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','work_accident','left','promotion_last_5years']]
y=df['department']


# In[635]:


#test case 3) 65,35
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)


# In[636]:


y_train


# In[637]:


print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[638]:


#random training data
import numpy as np
X_train1=np.random.permutation(X_train)
Y_train1=np.random.permutation(y_train)


# In[639]:



from sklearn.linear_model import SGDClassifier
lm =SGDClassifier()
lm.fit(X_train, y_train)


# In[640]:


#Find pridicted value
from sklearn.model_selection import cross_val_predict
y_pred =cross_val_predict(model, X_train1, Y_train1, cv=5)


# In[641]:


y_pred=y_pred.astype(int)


# In[642]:


y_pred


# In[643]:


# I make confusion table of confusin_matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_train, y_pred)


# In[644]:


from sklearn.metrics import precision_score
precision_score(y_train,y_pred,average=None)


# In[645]:


from sklearn.metrics import f1_score
f1_score(y_train,y_pred,average=None)


# In[ ]:





# In[ ]:




