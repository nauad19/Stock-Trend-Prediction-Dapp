import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


# ------------------------ APP TITLE ------------------------
st.title("📈 Stock Trend Prediction App")


# ------------------------ USER INPUT ------------------------
user_input = st.text_input("Enter Stock Ticker", "AAPL")

start = '2010-01-01'
end = '2019-12-31'

df = yf.download(user_input, start=start, end=end)

df = df.reset_index()
# Remove Date column safely
if 'Date' in df.columns:
    df = df.drop('Date', axis=1)

# Remove Adj Close only if it exists
if 'Adj Close' in df.columns:
    df = df.drop('Adj Close', axis=1)



# ------------------------ DATA DESCRIPTION ------------------------
st.subheader("Data from 2010 - 2019")
st.write(df.describe())


# ------------------------ VISUALIZATION ------------------------
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.legend()
st.pyplot(fig)


st.subheader("Closing Price with 100MA")
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, label='100 Moving Avg')
plt.legend()
st.pyplot(fig)


st.subheader("Closing Price with 100MA and 200MA")
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, label='100 MA')
plt.plot(ma200, label='200 MA')
plt.legend()
st.pyplot(fig)


# ------------------------ DATA SPLIT ------------------------
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])


# ------------------------ SCALING TRAINING DATA ------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)


# ------------------------ LOAD LSTM MODEL ------------------------
model = load_model("keras_model.h5")


# ------------------------ PREPARE TESTING DATA ------------------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, len(input_data)):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)


# ------------------------ MAKE PREDICTIONS ------------------------
y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# ------------------------ FINAL GRAPH ------------------------
st.subheader("Prediction vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
