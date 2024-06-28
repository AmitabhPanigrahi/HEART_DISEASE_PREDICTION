# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 02:07:51 2024

@author: Amitabh
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained model
loaded_model = joblib.load('E:\DEPLOY/trained_model.pkl')

# Function to preprocess input data and make predictions
def heart_disease_predictor(features):
    # Convert string inputs to numeric values if needed
    features = [float(x) if str(x).replace('.', '', 1).isdigit() else x for x in features]
    
    if features[1] == 'Male':
       features[1] = 1
    else:
       features[1] = 0

    input_data = np.array(features).reshape(1, -1)
    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)
    if (prediction[0]==0):
        return "The person does not have Heart Disease"
    else:
        return "The person has Heart Disease"
    return prediction

def main():
    st.title('Heart Disease Predictor')
    
    # Add input widgets for user input
    features = []
    features.append(st.number_input('Age', min_value=0, max_value=150, value=25))
    features.append(st.selectbox('Sex', ['Male', 'Female']))
    features.append(st.number_input('chest pain'))
    features.append(st.number_input('Resting Blood Pressure in mm Hg'))
    features.append(st.text_input("Cholestoral in mg/dl"))
    features.append(st.text_input("Fasting Blood Sugar > 120 mg/dl"))
    features.append(st.text_input("Resting ECG Results"))
    features.append(st.text_input("Maximum Heart Rate"))
    features.append(st.text_input("Exercise Induced Angina"))
    features.append(st.text_input("ST Depression Induced"))
    features.append(st.text_input("Slope of the Peak Exercise ST Segment"))
    features.append(st.text_input("Number of Major Vessels"))
    features.append(st.text_input("Thalassemia"))
    
    # Add other input widgets for remaining features
    
    # Call the predictor function when a button is clicked
    if st.button('Predict'):
        # Call the predictor function with user input
        diagnosis = heart_disease_predictor(features)
        
        # Display the prediction result
        st.write('Diagnosis:', diagnosis)

if __name__ == '__main__':
    main()
