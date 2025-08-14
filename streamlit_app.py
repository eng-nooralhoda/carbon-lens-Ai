import streamlit as st
import pandas as pd
st.title('🤖 carbon lens AI')
st.info('carbon lens AI is a cloud- powered machine learning tool that predicts CO2 emissions based on energy consumption data')
with st.expander('Data'):
  st.write('**Data set**')
df= pd.read_csv('https://raw.githubusercontent.com/eng-nooralhoda/carbon-lens-Ai/refs/heads/master/CO2_cleaned_1950_onwards.csv')
st.write (df)

import numpy as np
import joblib

st.set_page_config(page_title="Carbon Lens AI", layout="centered")

st.title("🌍 Carbon Lens AI")
st.write("تنبؤ بانبعاثات ثاني أكسيد الكربون باستخدام نموذج Random Forest")

# تحميل المودل
@st.cache_resource
def load_model():
    return joblib.load("carbon_model.pkl")  # ضعي ملف المودل في نفس المجلد

try:
    model = load_model()
except:
    st.error("⚠️ ملف carbon_model.pkl غير موجود في المجلد الحالي")
    st.stop()

# إدخال البيانات
year = st.number_input("السنة", min_value=1950, max_value=2100, value=2025)
country_code = st.number_input("كود الدولة", min_value=0, value=120)
coal = st.number_input("الفحم (MtCO₂)", min_value=0.0, value=30000.0)
oil = st.number_input("النفط (MtCO₂)", min_value=0.0, value=20000.0)
gas = st.number_input("الغاز (MtCO₂)", min_value=0.0, value=15000.0)
cement = st.number_input("الإسمنت (MtCO₂)", min_value=0.0, value=3000.0)
flaring = st.number_input("الفليرنج (MtCO₂)", min_value=0.0, value=800.0)
other = st.number_input("أخرى (MtCO₂)", min_value=0.0, value=500.0)

if st.button("🔮 تنبؤ"):
    data = np.array([[year, country_code, coal, oil, gas, cement, flaring, other]])
    prediction = model.predict(data)
