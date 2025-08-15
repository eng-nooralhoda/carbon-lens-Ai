import streamlit as st
import pandas as pd
st.title('🌍carbon lens AI')
st. info('carbon lens AI is a cloud-powered machine learning tool that predicts CO2 emissions based on energy consumption data')
with st.expander('Data'):
  st.write('**data set**')
  df =pd. read_csv('https://raw.githubusercontent.com/eng-nooralhoda/carbon-lens-Ai/refs/heads/master/CO2_cleaned_1950_onwards.csv')
  df
  st.write('**X**')
  X= df.drop('Year', 'Country_Code', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other')
  X
  st.write('**Y**')
  Y = df('Total')
  Y
