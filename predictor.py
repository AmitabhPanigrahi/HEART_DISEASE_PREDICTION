# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 00:32:34 2024

@author: Amitabh
"""


import numpy as np
import pickle 

# loading the saved model

loaded_model=pickle.load(open("E:\DEPLOY/trained_model.pkl",'rb'))

input_data=(37,0,2,120,215,0,1,170,0,0,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("The person does not have Heart Disease")
else:
  print("The person has Heart Disease")