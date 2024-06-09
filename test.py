import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import xgboost

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models and scalers
models = {}
scalers = {}

for hotel_number in [1, 2]:  # 1 for City Hotel, 2 for Resort Hotel
    for model_number in [4]:  # Model numbers

        model_filename = os.path.join(current_dir, f'model_{hotel_number}_{model_number}.pkl')
        scaler_filename = os.path.join(current_dir, f'scaler_{hotel_number}_{model_number}.pkl')
        
        with open(model_filename, 'rb') as file:
            models[(hotel_number, model_number)] = pickle.load(file)
        
        with open(scaler_filename, 'rb') as file:
            scalers[(hotel_number, model_number)] = pickle.load(file)

# Fungsi untuk mengonversi kolom non-numerik menjadi numerik
def convert_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df

# Define a function to make predictions
def make_prediction(hotel_type, model_type, features):
    model = models[(hotel_type, model_type)]
    scaler = scalers[(hotel_type, model_type)]
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    print(features_df)
    
    # Preprocess the features
    features_df = convert_to_numeric(features_df)
    print('after converted to numeric ', features_df)
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    print('after scaled', features_df)
    
    # Make a prediction
    prediction = model.predict(features_scaled)
    return prediction

# Streamlit app
st.title("Hotel Booking Cancellation Prediction")

# Choose hotel type
hotel_type = st.selectbox("Select Hotel Type", ["City Hotel", "Resort Hotel"])
hotel_number = 1 if hotel_type == "City Hotel" else 2

# Choose model type
model_type = st.selectbox("Select Model", ["XGBoost"])
model_number = { "XGBoost": 4}[model_type]

# Define the input fields based on the features used in the model
if hotel_number == 1:
    st.header("City Hotel Features")
    arrival_date_year = st.selectbox('Pilih tahun kedatangan', ['2015','2016','2017','2018','2019', '2020', '2021', '2022','2023', '2024'])
    lead_time = st.number_input('lead_time')
    distribution_channel = st.selectbox('distribution_channel',["TA/TO", "Direct", "Undefined", "Corporate", "GDS"])
    required_car_parking_spaces = st.selectbox('required_car_parking_spaces',['0','1','2','3'])
    country = st.selectbox('country',['PRT', 'ITA', 'ESP', 'DEU', 'FRA', 'NLD', 'GBR', 'ROU', 'BRA', 'SWE', 'AUT', 'Others', 'BEL', 'CHE', 'RUS', 'IRL', 'POL', 'CHN', 'USA', 'CN'])
    total_of_special_requests = st.selectbox('total_of_special_requests',['0', '1', '2', '3', '4', '5'])
    adr = st.number_input('adr')
    market_segment = st.selectbox('market_segment', ['Offline', 'TA/TO', 'Online TA', 'Groups', 'Others', 'Direct', 'Corporate'])
    customer_type = st.selectbox('customer_type',['Transient', 'Transient-Party', 'Contract', 'Group'])
    deposit_type = st.selectbox('deposit_type',['No Deposit', 'Non Refund', 'Refundable'])
    is_use_agent = st.selectbox('is_use_agent',['1','0'])
    stays_in_weekend_nights = st.selectbox('stays_in_weekend_nights',[0, 2, 1, 3, 4])
    stays_in_week_nigths = st.selectbox('stays_in_week_nights',[2, 4, 3, 5, 1, 6, 7, 0])
    reserved_room_type = st.selectbox('reserved_room_type',	['A', 'B', 'D', 'F', 'E', 'G', 'C'])
    assigned_room_type = st.selectbox('assigned_room_type',['A', 'B', 'F', 'D', 'G', 'E', 'K', 'C'])
    is_changed_room = 1 if reserved_room_type == assigned_room_type else 0
    total_stay_duration = stays_in_weekend_nights + stays_in_week_nigths
    total_expense = adr * total_stay_duration
    

    features = {
        'arrival_date_year': arrival_date_year,
        'is_changed_room' : is_changed_room,
        'lead_time': lead_time,
        'distribution_channel': distribution_channel,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_expense': total_expense,
        'country': country,
        'total_of_special_requests': total_of_special_requests,
        'adr': adr,
        'market_segment': market_segment,
        'customer_type': customer_type,
        'deposit_type': deposit_type,
        'is_use_agent': is_use_agent,
        
    }
else:
    st.header("Resort Hotel Features")
    arrival_date_year = st.selectbox('Pilih tahun kedatangan', ['2015','2016','2017','2018','2019', '2020', '2021', '2022','2023', '2024'])
    lead_time = st.number_input('lead_time')
    distribution_channel = st.selectbox('distribution_channel',["TA/TO", "Direct", "Undefined", "Corporate", "GDS"])
    required_car_parking_spaces = st.selectbox('required_car_parking_spaces',['0','1','2','3'])
    country = st.selectbox('country',['PRT', 'ITA', 'ESP', 'DEU', 'FRA', 'NLD', 'GBR', 'ROU', 'BRA', 'SWE', 'AUT', 'Others', 'BEL', 'CHE', 'RUS', 'IRL', 'POL', 'CHN', 'USA', 'CN'])
    total_of_special_requests = st.selectbox('total_of_special_requests',['0', '1', '2', '3', '4', '5'])
    adr = st.number_input('adr')
    market_segment = st.selectbox('market_segment', ['Offline', 'TA/TO', 'Online TA', 'Groups', 'Others', 'Direct', 'Corporate'])
    customer_type = st.selectbox('customer_type',['Transient', 'Transient-Party', 'Contract', 'Group'])
    deposit_type = st.selectbox('deposit_type',['No Deposit', 'Non Refund', 'Refundable'])
    is_use_agent = st.selectbox('is_use_agent',['1','0'])
    stays_in_weekend_nights = st.selectbox('stays_in_weekend_nights',[0, 2, 1, 3, 4])
    stays_in_week_nigths = st.selectbox('stays_in_week_nights',[2, 4, 3, 5, 1, 6, 7, 0])
    reserved_room_type = st.selectbox('reserved_room_type',	['A', 'B', 'D', 'F', 'E', 'G', 'C'])
    assigned_room_type = st.selectbox('assigned_room_type',['A', 'B', 'F', 'D', 'G', 'E', 'K', 'C'])
    is_changed_room = 1 if reserved_room_type == assigned_room_type else 0
    total_stay_duration = stays_in_weekend_nights + stays_in_week_nigths
    total_expense = adr * total_stay_duration
    arrival_date_month = st.selectbox("arrival date month",['July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June']
)


    features = {
        'is_changed_room' : is_changed_room,
        'lead_time': lead_time,
        'assigned_room_type': assigned_room_type,
        'distribution_channel': distribution_channel,
        'total_stay_duration' : total_stay_duration,
        'arrival_date_month' : arrival_date_month,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_expense': total_expense,
        'country': country,
        'adr': adr,
        'market_segment': market_segment,
        'is_use_agent': is_use_agent,
        
    }

# Make a prediction when the user clicks the button
if st.button("Predict"):
    features_list = list(features.values())
    prediction = make_prediction(hotel_number, model_number, features_list)
    print(prediction)
    st.write(f"The prediction is: {'Canceled' if prediction[0] else 'Not Canceled'}")
