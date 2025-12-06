import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('train_delay_model.pkl')

st.title('Train Delay Prediction App')

st.write("Enter the details below to predict the historical train delay.")

# Define the categorical features and their possible values
categorical_features_info = {
    'Weather Conditions': ['Clear', 'Foggy', 'Rainy'],
    'Day of the Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Time of Day': ['Afternoon', 'Evening', 'Morning', 'Night'],
    'Train Type': ['Express', 'Local', 'Superfast'],
    'Route Congestion': ['High', 'Low', 'Medium']
}

# Create a dictionary to store LabelEncoders for consistent encoding
label_encoders_streamlit = {}
for feature, categories in categorical_features_info.items():
    le = LabelEncoder()
    le.fit(categories) # Fit on all possible categories
    label_encoders_streamlit[feature] = le

# Input features from the user
distance = st.slider('Distance Between Stations (km)', min_value=0, max_value=1000, value=100)
weather_condition = st.selectbox('Weather Conditions', options=categorical_features_info['Weather Conditions'])
day_of_week = st.selectbox('Day of the Week', options=categorical_features_info['Day of the Week'])
time_of_day = st.selectbox('Time of Day', options=categorical_features_info['Time of Day'])
train_type = st.selectbox('Train Type', options=categorical_features_info['Train Type'])
route_congestion = st.selectbox('Route Congestion', options=categorical_features_info['Route Congestion'])

# Encode categorical inputs
encoded_weather_condition = label_encoders_streamlit['Weather Conditions'].transform([weather_condition])[0]
encoded_day_of_week = label_encoders_streamlit['Day of the Week'].transform([day_of_week])[0]
encoded_time_of_day = label_encoders_streamlit['Time of Day'].transform([time_of_day])[0]
encoded_train_type = label_encoders_streamlit['Train Type'].transform([train_type])[0]
encoded_route_congestion = label_encoders_streamlit['Route Congestion'].transform([route_congestion])[0]

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Distance Between Stations (km)': [distance],
    'Weather Conditions': [encoded_weather_condition],
    'Day of the Week': [encoded_day_of_week],
    'Time of Day': [encoded_time_of_day],
    'Train Type': [encoded_train_type],
    'Route Congestion': [encoded_route_congestion]
})

# Predict button
if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    st.success(f'The predicted delay is: {prediction[0]:.2f} minutes')
