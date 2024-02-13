# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:58:25 2023

@author: YASHIKA
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


#loading the saved models 

diabetes_model = pickle.load(open("C:/Users/YASHIKA/OneDrive/Desktop/Multiple Disease Prediction/trained_model.sav", 'rb'))

heart_disease_model = pickle.load(open("C:/Users/YASHIKA/OneDrive/Desktop/Multiple Disease Prediction/trained_model2.sav", 'rb'))

#sidebars for navigation

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           default_index= 0)
    
# Diabetes pred  page
if (selected == 'Diabetes Prediction'):
    
   st.title('Diabetes Prediction using ML')
    
   Pregnancies = st.text_input('Number of Pregnancies')
   Glucose = st.text_input('Glucose level')
   BloodPressure = st.text_input('Blood Pressure level')
   SkinThickness = st.text_input('Skin Thickness value')
   Insulin = st.text_input('Insulin level')
   BMI = st.text_input('BMI value')
   DiabetesPedigreeFunction  =  st.text_input('Diabetes Pedigree Function value')
   Age = st.text_input('Age of the person')

  #code for prediction    
   diabetes_diagnosis = ''
   
   #creating a button for prediction
   if st.button('Diabetes Test Result'):
       diabetes_prediction = diabetes_model([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
       
       if(diabetes_prediction[0] == 1):
           diabetes_diagnosis = 'The person is diabetic'
           
       else:    
           diabetes_diagnosis = 'The person is not diabetic'
           
             
   st.success(diabetes_diagnosis)
          
          
              
if (selected == 'Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction using ML')
    
    
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
    heart_diagnosis = ''
     
    #creating a button for prediction
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model([[age, sex, cp, tresttbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
         
        if(heart_prediction[0] == 1):
            heart_diagnosis = 'The person have heart disease'
            
        else:    
            heart_diagnosis = 'The person does not have heart disease'
         
    st.success( heart_diagnosis)
        
        
