#!/usr/bin/env python
# coding: utf-8

# ## Importing the dependencies

# In[45]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection And Analysis

# In[46]:


#loading the diabetes dataset to a pandas dataframe
diabetes = pd.read_csv('diabetes.csv')


# In[47]:


#printing the first five rows
diabetes.head()


# In[48]:


#printing the last five rows
diabetes.tail()


# In[49]:


diabetes.shape


# In[50]:


diabetes.info()


# In[51]:


#statistical measures of the data
diabetes.describe()


# In[52]:


#checking the distribution of Outcome variable
diabetes['Outcome'].value_counts()


# In[53]:


diabetes.groupby('Outcome').mean()


# In[54]:


#Splitting the data and labels
X = diabetes.drop(columns = 'Outcome', axis = 1)
Y = diabetes['Outcome']


# In[55]:


print(X)


# In[56]:


print(Y)


# In[57]:


#Standardize the data
#scaler = StandardScaler()


# In[58]:


#scaler.fit(X


# In[59]:


#standardized_data = scaler.transform(X)


# In[60]:


#print(standardized_data)


# In[61]:


#standardized_data.shape


# In[62]:


#X = standardized_data


# In[63]:


Y = diabetes['Outcome']


# In[64]:


print(X)
print(Y)


# Split the data into train and test data

# In[65]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)


# In[66]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model Training

# In[67]:


classifier = svm.SVC(kernel = 'linear')


# In[68]:


#training the svm classifier 
classifier.fit(X_train, Y_train)


# In[69]:


#Model Evaluation 
#Accuracy Score
X_train_prediction = classifier.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[70]:


print('Accuracy score of the training data :', train_data_accuracy)


# In[71]:


#Accuracy Score
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[72]:


print('Accuracy score of the testing data :', test_data_accuracy)


# ### Building the predictive system

# In[79]:


input_data = (4,110,92,0,0,37.6,0.191,30)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#std_data = scaler.transform(input_data_reshaped)
#print(std_data) 

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have Diabetes')
else: 
    print('The Person has Diabetes')


# In[80]:


input_data = (11,143,94,33,146,36.6,0.254,51)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#std_data = scaler.transform(input_data_reshaped)
#print(std_data) 

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have Diabetes')
else: 
    print('The Person has Diabetes')


# ### Saving the trained model

# In[82]:


import pickle


# In[83]:


filename = "diab_model.sav"
pickle.dump(classifier, open(filename, 'wb'))


# In[84]:


#loading the saved model
loaded_model = pickle.load(open('diab_model.sav', 'rb'))


# In[85]:


input_data = (4,110,92,0,0,37.6,0.191,30)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#std_data = scaler.transform(input_data_reshaped)
#print(std_data) 

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have Diabetes')
else: 
    print('The Person has Diabetes')


# In[ ]:




