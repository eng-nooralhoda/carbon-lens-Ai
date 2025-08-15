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
