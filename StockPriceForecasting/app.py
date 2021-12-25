import streamlit as st 
st.set_page_config(
page_title="Copyright © 2021 Hasnain",
page_icon="🎢",
layout="wide",
initial_sidebar_state="expanded")

import numpy as np 
import pandas as pd 
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from keras.layers import Dense, Dropout, LSTM 
from keras.models import Sequential
from keras.models import load_model
import streamlit as st 

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
 
st.title('Stock Trend Prediction')
col1, col2, col3 = st.columns(3)
with col1 :
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
with col2 :
    start_date = st.text_input('Enter Start Date e.g, 2010-01-01 ', '2010-01-01')
with col3:
    end_date = st.text_input('Enter End Date e.g, 2020-12-31 ', '2020-12-31')
df = data.DataReader(user_input, 'yahoo', start_date, end_date)

#Describing Data
st.subheader(f'Data Summary from {start_date[0:4]} - {end_date[0:4]}')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Daily Stock Trend') 
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r', label = '100 Days MA')
plt.plot(df.Close, 'b',label = 'Daily Stock Trend')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b', label = 'Daily Stock Trend')
plt.plot(ma100, 'r', label = '100 Days MA')
plt.plot(ma200, 'g', label = '200 Days MA')
plt.legend()
st.pyplot(fig)



data_training = pd.DataFrame(df.Close[0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df.Close[int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Load_Model
model = load_model('keras_model.h5')

#Testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

y_predicted = model.predict(X_test)

scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Actual Vs Predicted Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Actual Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)