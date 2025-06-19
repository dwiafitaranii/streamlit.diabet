import pickle
import streamlit as st 

#membaca model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))


#judul web
st.title('PREDIKSI DIABETES MENGGUNAKAN DATA MINING !')

pregnancies = st.text_input ('Input Nilai Pregnancies')
Glucose = st.text_input ('Input Nilai Glucose')
BloodPressure = st.text_input ('Input Nilai Blood Pressure')
SkinThickness = st.text_input ('Input Nilai Skin Thickness')
Insulin = st.text_input ('Input Nilai Insulin')
BMI = st.text_input ('Input Nilai BMI')
DiabetesPedigreeFunction = st.text_input ('Input Nilai Diabetes Pedigree Function')
Age = st.text_input ('Input Nilai Age')


#membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):
    # Membuat array input untuk prediksi
    input_data = [[
    float(pregnancies), float(Glucose), float(BloodPressure),
    float(SkinThickness), float(Insulin), float(BMI),
    float(DiabetesPedigreeFunction), float(Age)
    ]]
    
    # transform input agar sesuai dengan data training
    input_data = scaler.transform(input_data)


    diab_prediction = diabetes_model.predict(input_data)

    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien Tidak Terkena Diabetes'

    st.success(diab_diagnosis)
    st.error("Masukkan semua nilai input dengan benar (angka).")
