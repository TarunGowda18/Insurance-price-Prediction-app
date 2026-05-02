import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 1. Load the model
# Make sure gb_model.pkl is in the same folder as this script
model = pickle.load(open('gb_model.pkl', 'rb'))

# 2. Setup the App UI
st.title('Insurance Price Prediction App')
st.markdown("Enter details below to estimate insurance costs.")

# Define inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.selectbox('Gender', ('male', 'female'))
bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, value=30.0)
smoker = st.selectbox('Smoker', ('yes', 'no'))
children = st.number_input('Children', min_value=0, max_value=10, value=2)
region = st.selectbox('Region', ('southwest', 'southeast', 'northwest', 'northeast'))

# 3. Encoding techniques
# Smoker
Smoker = 1 if smoker == 'yes' else 0
# Gender (Matching your training feature names)
sex_male = 1 if gender == 'male' else 0
sex_female = 1 if gender == 'female' else 0

# Region
region_dict = {'southeast': 3, 'northeast': 2, 'northwest': 1, 'southwest': 0}
Region = region_dict[region]

# 4. Create input DataFrame
input_features = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'Smoker': [Smoker],
    'sex_female': [sex_female],
    'sex_male': [sex_male],
    'Region': [Region]
})

# 5. Scaling (The critical fix)
# NOTE: Using fit_transform on a single row makes the values 0.
# You should load the scaler used during training:
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    input_features[['age', 'bmi']] = scaler.transform(input_features[['age', 'bmi']])
except FileNotFoundError:
    st.warning("Warning: 'scaler.pkl' not found. Using raw values (results may be inaccurate).")

# 6. Predictions
if st.button('Predict'):
    # Get prediction from model
    prediction = model.predict(input_features)
    
    # If you used log-transformation during training (np.log(y)), use np.exp
    # Otherwise, use the prediction directly
    output = round(np.exp(prediction[0]), 2)
    
    st.success(f'Estimated Insurance Price: ${output}')
