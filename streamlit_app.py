import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# عنوان التطبيق
st.title('🌍Carbon Lens AI🤖')
st.info('Carbon Lens AI is a cloud-powered machine learning tool that predicts CO2 emissions based on energy consumption data.')

# تحميل البيانات
df = pd.read_csv('https://raw.githubusercontent.com/eng-nooralhoda/carbon-lens-Ai/refs/heads/master/CO2_cleaned_1950_onwards.csv')

# تحويل Country إلى كود رقمي
df['Country_Code'] = df['Country'].astype('category').cat.codes

# تحضير المتغيرات
X = df[['Year', 'Country_Code', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']]
y = df['Total']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء النموذج وتدريبه
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التوقع
y_pred = model.predict(X_test)

# حساب الدقة والأخطاء
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# أهمية الخصائص
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# عرض البيانات والمتغيرات داخل Expander
with st.expander('Data and Features'):
    st.subheader('Original Data')
    st.write(df)
    st.subheader('Features (X)')
    st.write(X)
    st.subheader('Target (y)')
    st.write(y)

# عرض النتائج
st.subheader('Model Results')
st.write("R² Score:", r2)
st.write("RMSE:", rmse)
st.write("Feature Importances:")
st.write(feature_importance)
import matplotlib.pyplot as plt

st.subheader("💨 احسب نسبة الانبعاثات بنفسك")

# مدخلات المستخدم
year = st.number_input("أدخل السنة", min_value=1900, max_value=2100, step=1)
country = st.text_input("أدخل اسم الدولة")
coal = st.number_input("كمية الفحم (Coal)", min_value=0.0)
oil = st.number_input("كمية النفط (Oil)", min_value=0.0)
gas = st.number_input("كمية الغاز (Gas)", min_value=0.0)
cement = st.number_input("كمية الاسمنت (Cement)", min_value=0.0)

# زر للحساب
if st.button("احسب الانبعاثات"):
    # مجموع الانبعاثات
    total = coal + oil + gas + cement

    if total == 0:
        st.warning("⚠️ الرجاء إدخال قيم أكبر من صفر")
    else:
        # عرض النتيجة
        st.success(f"✅ النتيجة: نسبة الانبعاثات في {country} لسنة {year} هي {total:.2f} طن CO2")

        # تجهيز البيانات للرسم
        sources = ['Coal', 'Oil', 'Gas', 'Cement']
        values = [coal, oil, gas, cement]

        # رسم Pie Chart
        fig, ax = plt.subplots()
        ax.pie(values, labels=sources, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"مصادر الانبعاثات في {country} لسنة {year}")

        # عرض الرسم في Streamlit
        st.pyplot(fig)
        import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# ضبط الخط (ممكن تغيّري لـ "Cairo" أو "Amiri" لو عندك خط عربي ثاني)
plt.rcParams['font.family'] = 'Arial'

# جملة عربية
title_text = "انبعاثات الكربون"
xlabel_text = "السنة"
ylabel_text = "الانبعاثات"

# إصلاح الحروف العربية
title = get_display(arabic_reshaper.reshape(title_text))
xlabel = get_display(arabic_reshaper.reshape(xlabel_text))
ylabel = get_display(arabic_reshaper.reshape(ylabel_text))

# مثال رسم
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Total'])  # غيّري حسب الأعمدة اللي عندك
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

st.pyplot(fig)
