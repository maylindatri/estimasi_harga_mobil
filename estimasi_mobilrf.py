import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Fungsi khusus untuk memuat model dengan dtype yang dikoreksi
# Memuat model
def custom_load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Memuat model terlatih dan encoder
model_filename = 'best_random_forest_model.sav'
model_mapping_filename = 'model_mapping.pkl'
transmission_mapping_filename = 'transmission_mapping.pkl'
fuelType_mapping_filename = 'fuelType_mapping.pkl'

model = custom_load_model(model_filename)
with open(model_mapping_filename, 'rb') as file:
    model_mapping = pickle.load(file)
with open(transmission_mapping_filename, 'rb') as file:
    transmission_mapping = pickle.load(file)
with open(fuelType_mapping_filename, 'rb') as file:
    fuelType_mapping = pickle.load(file)

# Membalik mapping untuk lookup yang mudah
reverse_model_mapping = {k: model_mapping.inverse_transform([k])[0] for k in range(len(model_mapping.classes_))}
reverse_transmission_mapping = {k: transmission_mapping.inverse_transform([k])[0] for k in range(len(transmission_mapping.classes_))}
reverse_fuelType_mapping = {k: fuelType_mapping.inverse_transform([k])[0] for k in range(len(fuelType_mapping.classes_))}

st.title('Estimasi Harga Mobil Bekas Toyota')

# Input untuk atribut mobil
model_input = st.selectbox('Model Mobil', list(reverse_model_mapping.values()))
year_input = st.text_input('Tahun Mobil', '')
transmission_input = st.selectbox('Transmisi Mobil', list(reverse_transmission_mapping.values()))
mileage_input = st.text_input('Jarak Tempuh (dalam km)', '')
tax_input = st.text_input('Pajak Mobil (dalam IDR)', '')
fuelType_input = st.selectbox('Jenis Bahan Bakar', list(reverse_fuelType_mapping.values()))
mpg_input = st.text_input('Konsumsi BBM Mobil (dalam mpg)', '')
engineSize_input = st.text_input('Ukuran Mesin (dalam L)', '')

if st.button('Estimasi Harga'):
    if year_input == '' or mileage_input == '' or tax_input == '' or mpg_input == '' or engineSize_input == '':
        st.warning('HARAP LENGKAPI SEMUA INPUT SEBELUM MENEKAN TOMBOL "Estimasi Harga"')
    else:
        try:
            year = int(year_input)
            mileage = float(mileage_input)
            tax = int(tax_input)
            mpg = float(mpg_input)
            engineSize = float(engineSize_input)

            model_encoded = model_mapping.transform([model_input])[0]
            transmission_encoded = transmission_mapping.transform([transmission_input])[0]
            fuelType_encoded = fuelType_mapping.transform([fuelType_input])[0]

            logging.debug(f"Model Encoded: {model_encoded}")
            logging.debug(f"Transmission Encoded: {transmission_encoded}")
            logging.debug(f"FuelType Encoded: {fuelType_encoded}")

            input_data = np.array([[model_encoded, year, mileage, tax, transmission_encoded, fuelType_encoded, mpg, engineSize]])

            logging.debug(f"Input Data: {input_data}")

           # Prediksi harga dalam IDR
            prediction = model.predict(input_data)
            prediction_idr = max(int(round(prediction[0])), 0)

            # Konversi ke GBP
            conversion_rate = 19000  # 1 GBP = 19,000 IDR
            prediction_gbp = round(prediction_idr / conversion_rate)

            # Format hasil prediksi
            prediction_idr_formatted = f"Rp {prediction_idr:,.0f}"
            prediction_gbp_formatted = f"£{prediction_gbp:,}"

            st.write('Estimasi harga mobil bekas dalam IDR:', prediction_idr_formatted)
            st.write('Estimasi harga mobil bekas dalam Poundsterling:', prediction_gbp_formatted)
        except ValueError as e:
            st.warning('HARAP MASUKKAN ANGKA YANG VALID UNTUK SETIAP INPUT')
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            logging.error(f"Exception: {e}")