import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción para saber tu nivel de acondicionamiento fisico ''')
st.image("fit.jpg", caption="Cuidarme a mí mismo también es una forma de querer.")

st.header('Datos del paciente')

def user_input_features():
  # Entrada
  age = st.number_input('Edad:', min_value=1, max_value=100, value = 1, step = 1)
  height_cm = st.number_input('Altura (en cm):', min_value=1, max_value=254, value = 0, step = 1)
  weight_kg = st.number_input('Peso (en kg):', min_value=0, max_value=100, value = 0, step = 1)
  heart_rate = st.number_input('Latidos por minuto:',min_value=0, max_value=200, value = 0, step = 1)
  blood_pressure = st.number_input('presión arterial:', min_value=0, max_value=250, value = 0, step = 1)
  nutrition_quality = st.number_input('nutrición alimenticia:', min_value=0, max_value=150, value = 0, step = 1)
  activity_index = st.number_input('condicion fisica ( 1 a 10):', min_value=0, max_value=10, value = 0, step = 1)
  smokes = st.number_input('fuma? (si con 1 o no con 0):', min_value=0, max_value=1, value = 0, step = 1)

  user_input_data = {'edad': age,
                     'altura en cm': height_cm,
                     'peso en kg': weight_kg,
                     'latidos por minuto': heart_rate,
                     'presión arterial': blood_pressure,
                     'nutrición alimenticia': nutrition_quality,
                     'condición fisica': activity_index,
                     '¿Fuma?': smokes}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

titanic =  pd.read_csv('Fit_prediction.csv', encoding='latin-1')
X = titanic.drop(columns='is_fit')
Y = titanic['is_fit']

classifier = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=25, max_features=7, random_state=0) # agregar los parametros dados
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No es fit')
elif prediction == 1:
  st.write('ES fit')
else:
  st.write('Sin predicción')
