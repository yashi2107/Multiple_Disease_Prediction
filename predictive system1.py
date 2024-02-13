# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:17:24 2023

@author: YASHIKA
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('D:/project/trained_model2.sav', 'rb'))

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
