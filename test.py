import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
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
st.markdown("<h1 style='text-align: center;'>Final Project Data Science (Cancellation Hotel Booking)</h1>", unsafe_allow_html=True)
image = Image.open('hotel.jpg')
new_size = (700, 300)  # Specify the new size (width, height)
image = image.resize(new_size)
# Display the image in Streamlit with a caption
st.image(image, caption='Hotel Demand Booking', use_column_width=False)

st.divider()
with st.container():
    st.subheader("Kelompok 1")
    st.write("1. Aulia Salsabila			 	(23/530951/PPA/06752)\n 2. Muhammad Salam		         	(23/512107/PPA/06497)\n 3. Mochammad Itmamul Wafa 		(23/526555/PPA/06658)")

st.divider()
st.subheader("Content")
with st.container():
    st.write("Pembatalan pesanan adalah salah satu tantangan terbesar yang dihadapi industri perhotelan karena dapat menghilangkan penghasilan dari kamar yang tidak berpenghuni. Berdasarkan permasalahan tersebut, perlu dibangun kerangka kerja teknologi yang dapat memprediksi kemungkinan terjadinya pembatalan setiap pesanan secara akurat, sehingga pihak hotel dapat membuat kebijakan dan strategi yang tepat dalam mengatasi permasalahan tersebut")
    
# Subheader for Dataset
st.subheader("Dataset")
with st.container():
    st.write("""
    Dataset yang digunakan dalam menganalisa pembatalan hotel bersumber dari kaggle yang berjudul 
    [“Hotel Booking Demand”](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) 
    yang dapat diakses melalui tautan ini. Dataset ini menyediakan informasi lengkap mengenai pemesanan 
    di dua jenis hotel, yaitu Resort Hotel dan City Hotel, dengan jumlah dataset mencapai  119390 data dan 32 feature.
    """)
    df = pd.read_csv('hotel_bookings.csv')
    st.write(df)

    st.write("""<span style="color:red">**Note:**</span> Data aslinya berasal dari artikel Hotel Booking Demand Datasets yang ditulis oleh <b>Nuno Antonio, Ana Almeida, dan Luis Nunes untuk Data in Brief, Volume 22, Februari 2019.</b>""", unsafe_allow_html=True)
    
st.subheader("kesimpulan")
with st.container():
    st.write("""Dalam menentukan prediksi pembatalan pemesanan hotel, model yang digunakan adalah XGBoost, yang menunjukkan kinerja dengan nilai akurasi 0.80 untuk city hotel dan 0.85 untuk resort hotel. Penentuan fitur-fitur dalam dataset yang digunakan untuk prediksi dilakukan melalui metode seleksi fitur Anova dan Mutual Information, serta teknik rekayasa fitur (feature engineering) yang membantu dalam mengidentifikasi fitur penting.

Untuk city hotel, fitur-fitur yang dipilih meliputi tahun kedatangan, perubahan kamar, waktu tunggu, saluran distribusi, kebutuhan ruang parkir, total pengeluaran, negara asal tamu, jumlah permintaan khusus, rata-rata tarif harian, segmen pasar, tipe pelanggan, jenis deposit, dan penggunaan agen pemesanan. Sedangkan untuk resort hotel, fitur-fitur yang dipilih meliputi perubahan kamar, waktu tunggu, tipe kamar yang ditugaskan, saluran distribusi, durasi total menginap, bulan kedatangan, kebutuhan ruang parkir, total pengeluaran, negara asal tamu, rata-rata tarif harian, segmen pasar, dan penggunaan agen pemesanan.

Dengan menggunakan model XGBoost dan pemilihan fitur yang tepat, prediksi pembatalan pemesanan hotel dapat dilakukan dengan akurasi yang cukup tinggi, memberikan informasi yang berharga untuk pengelolaan dan pengambilan keputusan operasional hotel.""", unsafe_allow_html=True)

# Subheader for Model
st.divider()
st.subheader("Hotel Booking Cancellation Prediction")
with st.container():
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
    lead_time = st.number_input('Jumlah hari sebelum kedatangan saat pemesanan')
    distribution_channel = st.selectbox('Nama saluran yang digunakan untuk pemesanan',["TA/TO", "Direct", "Undefined", "Corporate", "GDS"])
    required_car_parking_spaces = st.selectbox('Jumlah tempat parkir mobil yang dibutuhkan',['0','1','2','3'])
    country = st.selectbox('Identifikasi ISO negara pemegang pemesanan',['PRT', 'ITA', 'ESP', 'DEU', 'FRA', 'NLD', 'GBR', 'ROU', 'BRA', 'SWE', 'AUT', 'Others', 'BEL', 'CHE', 'RUS', 'IRL', 'POL', 'CHN', 'USA', 'CN'])
    total_of_special_requests = st.selectbox('Jumlah permintaan khusus yang dibuat',['0', '1', '2', '3', '4', '5'])
    adr = st.number_input('Rata-rata Tarif harian')
    market_segment = st.selectbox('Segmentasi pasar tempat pemesanan ditetapkan', ['Offline', 'TA/TO', 'Online TA', 'Groups', 'Others', 'Direct', 'Corporate'])
    customer_type = st.selectbox('customer_type',['Transient', 'Transient-Party', 'Contract', 'Group'])
    deposit_type = st.selectbox('deposit_type',['No Deposit', 'Non Refund', 'Refundable'])
    is_use_agent = st.selectbox('is_use_agent',['1','0'])
    stays_in_weekend_nights = st.selectbox('Total lama menginap, berapa malam di akhir pekan (Sabtu dan Minggu)',[0, 2, 1, 3, 4])
    stays_in_week_nigths = st.selectbox('Total lama menginap, berapa malam di hari kerja (Senin sampai Jumat)',[2, 4, 3, 5, 1, 6, 7, 0])
    reserved_room_type = st.selectbox('Jenis kamar yang diminta oleh tamu',	['A', 'B', 'D', 'F', 'E', 'G', 'C'])
    assigned_room_type = st.selectbox('Jenis kamar ditetapkan untuk pemesanan',['A', 'B', 'F', 'D', 'G', 'E', 'K', 'C'])
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
