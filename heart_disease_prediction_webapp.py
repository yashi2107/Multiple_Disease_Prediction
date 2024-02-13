# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:23:08 2023

@author: YASHIKA
"""

import numpy as np
import pickle 
import streamlit as st

#loading the saved model
loaded_model1 = pickle.load(open('D:/project/trained_model2.sav', 'rb'))


#creating a function for prediction
def heart_disease_predic(input_data):
    
    #change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

    #reshape the numpy array as we are predicting for only one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model1.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Person does not have a Heart Disease'
    else: 
        return 'The Person has a Heart Disease'
    
    
    
def main():
    
    # giving a title for the user interface 
    st.title('Heart Disease Prediction Web App')
    
    #getting the input data from the user
    
    
    age = st.text_input('Age of the person')
    sex = st.text_input('Gender')
    cp = st.text_input('Chest Pain level')
    tresttbps = st.text_input('Blood Pressure value')
    chol = st.text_input('Cholestrol level in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar value > 120 mg/dl')
    restecg  =  st.text_input('ECG value')
    thalach = st.text_input('Maximum Heart Rate acheived')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('St depression value')
    slope = st.text_input('Slope of peak exercise ST segment')
    ca = st.text_input('Major Vessels colored by flouroscopy')
    thal = st.text_input('thal = 0(normal), 1(fixed defect), 2(reversible defect)')
    
    
    #code for prediction    
    diagnosis = ''
     
    #creating a button for prediction
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_predic([age, sex, cp, tresttbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
         
         
    st.success(diagnosis)
     
     
     
     
if __name__ == '__main__':
    main()
     
    