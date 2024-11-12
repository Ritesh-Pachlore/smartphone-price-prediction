import streamlit as st
import pickle
import pandas as pd

# Load the saved model, scaler, label encoders, and feature columns
with open("E:/SEM V/ML CA2/models/random_forest_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("E:/SEM V/ML CA2/models/scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open("E:/SEM V/ML CA2/models/label_encoders.pkl", 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

with open("E:/SEM V/ML CA2/models/feature_columns.pkl", 'rb') as feature_columns_file:
    feature_columns = pickle.load(feature_columns_file)

# Load the external CSS styling file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit interface with styled header
st.markdown('<div class="title">Smartphone Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate smartphone prices based on features</div>', unsafe_allow_html=True)

# Function to encode input data using the label encoders
def encode_input(data, column):
    if data in label_encoders[column].classes_:
        return label_encoders[column].transform([data])[0]
    else:
        st.warning(f"Warning: The label '{data}' for column '{column}' is unseen. Using default encoding.")
        return label_encoders[column].transform([label_encoders[column].classes_[0]])[0]

# Input fields section
st.markdown('<div class="section-title">Select Smartphone Specifications:</div>', unsafe_allow_html=True)
processor = st.selectbox('Select Processor', label_encoders['Processor'].classes_.tolist())
camera_details = st.selectbox('Select Camera Details', label_encoders['Camera_details'].classes_.tolist())
storage_details = st.selectbox('Select Storage', label_encoders['Storage_details'].classes_.tolist())
screen_size = st.selectbox('Select Screen Size', label_encoders['Screen_size'].classes_.tolist())
battery_details = st.selectbox('Select Battery', label_encoders['Battery_details'].classes_.tolist())

# Encode the inputs
encoded_processor = encode_input(processor, 'Processor')
encoded_camera_details = encode_input(camera_details, 'Camera_details')
encoded_storage_details = encode_input(storage_details, 'Storage_details')
encoded_screen_size = encode_input(screen_size, 'Screen_size')
encoded_battery_details = encode_input(battery_details, 'Battery_details')

# Prepare the data for prediction and ensure it has the correct feature names and order
input_data = pd.DataFrame({
    'Processor': [encoded_processor],
    'Camera_details': [encoded_camera_details],
    'Storage_details': [encoded_storage_details],
    'Screen_size': [encoded_screen_size],
    'Battery_details': [encoded_battery_details]
})

# Reorder columns based on feature_columns to match training order
input_data = input_data.reindex(columns=feature_columns)

# Button to trigger prediction
if st.button('Predict Price'):
    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict the price using the trained model
    predicted_price = model.predict(scaled_input)[0]

    # Display the predicted price with styling
    st.markdown(f'<div class="prediction-text">Predicted Price: ₹{predicted_price:.2f}</div>', unsafe_allow_html=True)
