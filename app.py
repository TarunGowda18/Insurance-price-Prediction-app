import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# load the model
model = pickle.load(open('gb_model.pkl','rb'))

# title for app
st.title('Insurance price Prediction app')

# define inputs to take inputs from user
age = st.number_input('Age',min_value=1 , max_value=100,value=25)
gender = st.selectbox('Gender',('male','female'))
bmi = st.number_input('BMI',min_value=10.0 , max_value=80.0,value=30.0)
smoker =  st.selectbox('Smoker',('yes','no'))
children=st.number_input('Children',min_value=0 , max_value=10,value=2)
region = st.selectbox('Region',('southwest','southeast','northwest','northeast'))

# Encoding techniques
# smoker
Smoker = 1 if smoker=='yes' else 0
# Gender
sex_male = 1 if gender=='male' else 0
sex_female = 1 if gender=='female' else 0

# region
region_dict = {'southeast':3,'northeast':2,'northwest':1,'southwest':0}
Region = region_dict[region]

# create dataframe
input_features = pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male],
    'Region':[Region]
})

scaler = StandardScaler()
input_features[['age','bmi']]=scaler.fit_transform(input_features[['age','bmi']])

# predictions
if st.button('Predict'):
  predictions=model.predict(input_features)
  output = round(np.exp(predictions[0]),2)
  st.success(f'Price Predictions: ₹{output}')
