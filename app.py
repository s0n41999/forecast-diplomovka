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

#splitting data
data = df.filter(['close'])

dataset = data.values

training_data_len = int(np.ceil( len(dataset) * .80 ))

#scaling 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#test train
train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(26, len(train_data)):   #sequence length in time series forecasting, lookback parameter in time series models, time series prediction window size 
    x_train.append(train_data[i-26:i, 0])  
    y_train.append(train_data[i, 0])
    if i<= 26:
        print(x_train)
        print(y_train)
        print()
        

x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#loading model
model = keras.models.load_model('keras_model_2.keras')

#prediction
test_data = scaled_data[training_data_len - 26: , :]  #52 - rok nebol dobry vysledok Root Mean Squared Error (RMSE): 0.02373457503629023 LSTM(50, batch_size=32, epochs=10)

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(26, len(test_data)):
    x_test.append(test_data[i-26:i, 0])
    

x_test = np.array(x_test)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


train = data[:training_data_len].copy()
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions


st.subheader('Training and Testing the model')
fig = plt.figure(figsize=(12,6))
plt.title('LSTM Model')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')  #NÁJDI LEPŠIE POMENOVANIE
st.pyplot(fig)
