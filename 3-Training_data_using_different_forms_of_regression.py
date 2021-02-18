#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\hp\Downloads\Project_3.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# # dropping useless columns 

# In[5]:


df.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo', 'Special', 'Real Face', 'Release Clause','Joined', 'Contract Valid Until'], inplace=True)
df


# In[6]:


df.drop(columns=['Loaned From'], inplace=True)

##players with no club
df['Club'].fillna(value='No Club', inplace=True)
df


# # filling missing values

# In[7]:


df['Preferred Foot'].isnull().sum()


# In[8]:


df.drop(columns=['Preferred Foot'], inplace=True)


# In[9]:


print(df['Position'].where(df['Position']=='GK'))


# In[10]:


len(df[df['Position'] == 'GK'])


# In[11]:


##filling null values for goal keeper as 0
df.fillna(value=0, inplace=True)


# In[12]:


df['LS'].isnull().sum()


# In[13]:


df['ST'].isnull().sum()         ##means all goal keeper values are filled


# In[14]:


df.isnull().sum().sum()


# In[15]:


df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']]


# In[16]:


drop_cols = df.columns[17:43]
df = df.drop(drop_cols, axis = 1)


# # training the data

# In[17]:


X=df.iloc[:,17:87]   ## independent feature
X


# In[18]:


X=X.astype(int)
X


# In[19]:


y=df.iloc[:,[3]]      ## dependent feature
y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[21]:


from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
lin_regressor.fit(X_train,y_train)
prediction_linear=lin_regressor.predict(X_test)


# In[22]:


from sklearn.linear_model import Lasso
lasso_regressor=Lasso()
lasso_regressor.fit(X_train,y_train)
prediction_lasso=lasso_regressor.predict(X_test)


# In[23]:


from sklearn.linear_model import Ridge
ridge_regressor=Ridge()
ridge_regressor.fit(X_train,y_train)
prediction_ridge=ridge_regressor.predict(X_test)


# In[24]:


from sklearn.linear_model import ElasticNet
model_enet = ElasticNet()
model_enet.fit(X_train, y_train) 
prediction_elasticnet= model_enet.predict(X_test)


# In[25]:


from sklearn import metrics


# In[26]:


print("Linear regression")
mae_linear=metrics.mean_absolute_error(y_test, prediction_linear)
print('Mean Absolute Error:',mae_linear )   
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_ridge)))


# In[27]:


print("Ridge regression")
mae_ridge=metrics.mean_absolute_error(y_test, prediction_ridge)
print('Mean Absolute Error:',mae_ridge )   
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_ridge)))


# In[28]:


print("Lasso regression")
mae_lasso=metrics.mean_absolute_error(y_test,prediction_lasso)
print('Mean Absolute Error:',mae_lasso)    
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_lasso)))


# In[29]:


print("Elastic_net regression")
mae_elasticnet=metrics.mean_absolute_error(y_test, prediction_elasticnet)
print('Mean Absolute Error:',mae_elasticnet )   
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_elasticnet)))


# In[30]:


plt.figure(figsize=(13,9))
plt.plot(["Linear","Ridge","Lasso","ElasticNet"],[mae_linear,mae_ridge,mae_lasso,mae_elasticnet],marker='o')
plt.grid()
plt.xlabel("Types of Regularization")
plt.ylabel("Mean Absolute Error")

