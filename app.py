import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import keras

st.title('Exchange Rate Prediction')

user_input = st.text_input('Enter Currency Ticker', 'EURUSD=X')

start_interval = "12/01/2009"
end_interval = date.today().strftime("%m/%d/%y")

df = get_data(user_input, start_date = start_interval , end_date = end_interval, index_as_date = True, interval="1wk")  #poznámka - interval


st.subheader('RAW DATA')
st.write(df.tail())

#describing data


st.subheader('Closing Price vs Moving Average')

st.write('Closing Price vs 50 Day Moving Average')
ma50 = df.close.rolling(50).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma50, 'r')
plt.plot(df.close)
plt.legend(['values', 'moving average: 50'], loc='upper right')
st.pyplot(fig)

st.write('Closing Price vs 200 Day Moving Average')
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma200, 'g')
plt.plot(df.close)
plt.legend(['values', 'moving average: 200'], loc='upper right')
st.pyplot(fig)

st.write('Closing Price vs 50 Day Moving Average and 200 Day Moving Average')
ma50 = df.close.rolling(50).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma50, 'r')
plt.plot(ma200, 'g')
plt.plot(df.close)
plt.legend(['values', 'moving average: 50', 'moving average: 200'], loc='upper right')
st.pyplot(fig)


