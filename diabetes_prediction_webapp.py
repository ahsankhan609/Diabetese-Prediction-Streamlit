import numpy as np
import pickle
import streamlit as st

# load the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediciton(input_data):
  #input_data = (5,166,72,19,175,25.8,0.587,51)

  # changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  # standardize the input data
#   std_data = scaler.transform(input_data_reshaped)
#   print(std_data)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person is not diabetic'
  else:
    return 'The person is diabetic'


# Coding the Streamlit app

def main():
  # give title to the WebApp
  st.title('Diabetes WebApp - Streamlit')

  # getting the data from user
  Pregnancies = st.text_input('Number of Pregnencies')
  Glucose = st.text_input('glucose')
  BloodPressure = st.text_input('BP')
  SkinThickness = st.text_input('SKIn thickness')
  Insulin = st.text_input('insulin')
  BMI = st.text_input('bmi')
  DiabetesPedigreeFunction = st.text_input('dibfunction')
  Age = st.text_input('enter age')

  #code for prediction
  diagnosis = ''
  
  # creating a button for prediciton
  if st.button('Diabetes Result'):
    diagnosis = diabetes_prediciton([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

  st.success(diagnosis)


if __name__ == '__main__':
  main()