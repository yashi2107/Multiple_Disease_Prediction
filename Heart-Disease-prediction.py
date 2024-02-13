#!/usr/bin/env python
# coding: utf-8

# ###  Importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ###  Data Collection and Preprocessing

# In[2]:


#loading the csv data to a pandas Dataframe
heart_data = pd.read_csv('heart_disease_data.csv')


# In[3]:


#print first five rows of the dataset
heart_data.head()


# In[4]:


#print last five rows of the dataset
heart_data.tail()


# In[5]:


#number of rows and columns in the dataset
heart_data.shape


# In[6]:


#getting some info about the data
heart_data.info()


# In[7]:


#checking for missing values
heart_data.isnull().sum()


# In[8]:


#statistical measures about the data
heart_data.describe()


# In[11]:


#checking the distribution of target variable
heart_data['target'].value_counts()


# In[12]:


#Spliting the features and target
X = heart_data.drop(columns = 'target',axis = 1)
Y = heart_data['target']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[15]:


#Splitting the data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, stratify = Y, random_state =2)


# In[16]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model Training

# In[17]:


#Logistic Regression 
model = LogisticRegression()


# In[18]:


#training he Logistic Regression model with the training data
model.fit(X_train, Y_train)


# ### Model Evaluation

# In[19]:


#Accuracy Score
#accuracy on the training data
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[20]:


print('Accuracy on Training Data : ', train_data_accuracy)


# In[21]:


#Accuracy Score
#accuracy on the testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[22]:


print('Accuracy on Testing Data : ', test_data_accuracy)


# ### Building the Predictive System

# In[41]:


input_data = (66,0,3,150,226,0,1,114,0,2.6,0,0,2)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else: 
    print('The Person has a Heart Disease')


# In[42]:


input_data = (57,1,0,130,131,0,1,115,1,1.2,1,1,3)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else: 
    print('The Person has a Heart Disease')


# ### saving the trained model

# In[47]:


import pickle


# In[48]:


filename = 'heart_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[49]:


#loading the saved model
loaded_model = pickle.load(open('heart_model.sav', 'rb'))


# In[50]:


input_data = (57,1,0,130,131,0,1,115,1,1.2,1,1,3)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else: 
    print('The Person has a Heart Disease')


# In[ ]:




