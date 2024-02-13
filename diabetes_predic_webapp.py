# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:33:32 2023

@author: YASHIKA
"""

import numpy as np
import pickle 
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('D:/project/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetes_predic(input_data):
    
    #change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the numpy array as we are predicting for only one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped )
    print(prediction)

    if (prediction[0] == 0):
        return 'The Person does not have Diabetes'
    else: 
        return 'The Person has Diabetes'
    
    
    
def main():
    
    # giving a title for the user interface 
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure level')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction  =  st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')

   #code for prediction    
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_predic([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
