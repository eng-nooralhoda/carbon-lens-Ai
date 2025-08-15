import streamlit as st
import pandas as pd
st.title('🌍carbon lens AI')
st. info('carbon lens AI is a cloud-powered machine learning tool that predicts CO2 emissions based on energy consumption data')
with st.expander('Data'):
  st.write('**data set**')
  df =pd. read_csv('https://raw.githubusercontent.com/eng-nooralhoda/carbon-lens-Ai/refs/heads/master/CO2_cleaned_1950_onwards.csv')
  df
  df = pd.read_csv('https://raw.githubusercontent.com/eng-nooralhoda/carbon-lens-Ai/refs/heads/master/CO2_cleaned_1950_onwards.csv') 
  df['Country_Code'] = df['Country'].astype('category')
  cat.codes
  X = df[['Year', 'Country_Code', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']] 
  y = df['Total'] 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
  model = RandomForestRegressor(n_estimators=100, random_state=42) 
  model.fit
  (X_train, y_train) 
  y_pred = model.predict(X_test) 
  r2 = r2_score(y_test, y_pred) 
  rmse = root_mean_squared_error(y_test, y_pred) 
  feature_importance = pd.Series(model.feature_importances_, 
  index=X.columns).sort_values(ascending=False) 
  print("R² Score:", r2) 
  print("RMSE:", rmse) 
  print("\nFeature Importances:") 
  print(feature_importance) 
