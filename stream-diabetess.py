import pickle
import streamlit as st

# CSS Styling
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: #e6f7ff;  /* Biru muda */
            color: #000000;
        }
        
        h1 {
            color: #004466;
            text-align: center;
        }

        .stTextInput > label {
            font-weight: bold;
            color: #00334d;
        }

        .stButton > button {
            background-color: #0099cc;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# Membaca model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Judul web
st.title('🔍 Prediksi Diabetes dengan Data Mining')

st.markdown("Masukkan data pasien untuk mengetahui prediksi diabetes berdasarkan model ML.", unsafe_allow_html=True)

# Input fields
pregnancies = st.text_input('🧬 Nilai Pregnancies')
Glucose = st.text_input('🍬 Nilai Glucose')
BloodPressure = st.text_input('🩸 Nilai Blood Pressure')
SkinThickness = st.text_input('📏 Nilai Skin Thickness')
Insulin = st.text_input('💉 Nilai Insulin')
BMI = st.text_input('⚖️ Nilai BMI')
DiabetesPedigreeFunction = st.text_input('👨‍👩‍👧‍👦 Nilai Diabetes Pedigree Function')
Age = st.text_input('🎂 Nilai Age')

# Tombol prediksi
if st.button('🔎 Test Prediksi Diabetes'):
    try:
        input_data = [[
            float(pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]]

        input_data = scaler.transform(input_data)
        diab_prediction = diabetes_model.predict(input_data)

        if diab_prediction[0] == 1:
            st.error('🚨 Pasien Terkena Diabetes')
        else:
            st.success('✅ Pasien Tidak Terkena Diabetes')

    except:
        st.warning('⚠️ Masukkan semua nilai input dengan benar (angka).')
