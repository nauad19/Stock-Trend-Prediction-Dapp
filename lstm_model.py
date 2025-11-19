import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# ---------------- DATA DOWNLOAD ----------------
start = '2010-01-01'
end = '2019-12-31'

df = yf.download('AAPL', start=start, end=end)

df = df.reset_index()

# Some yfinance versions do not give "Adj Close", so we remove only if exists
if 'Adj Close' in df.columns:
    df = df.drop(['Adj Close'], axis=1)

# Drop Date column
df = df.drop(['Date'], axis=1)

# ---------------- MOVING AVERAGES ----------------
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# ---------------- TRAIN / TEST SPLIT ----------------
training_data_len = int(len(df) * 0.7)

data_training = pd.DataFrame(df['Close'][:training_data_len])
data_testing = pd.DataFrame(df['Close'][training_data_len:])

# ---------------- SCALING ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# ---------------- TRAIN DATA PREP ----------------
x_train = []
y_train = []

for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape for LSTM: (samples, timesteps, features)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)


# ---------------- LSTM MODEL ----------------
model = Sequential()

model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(100, 1)))
model.add(Dropout(0.2))

model.add(LSTM(60, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))

model.add(LSTM(80, return_sequences=True, activation='relu'))
model.add(Dropout(0.4))

model.add(LSTM(120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("🚀 Training LSTM Model...")
model.fit(x_train, y_train, epochs=50)

model.save("keras_model.h5")

print("\n✅ Model saved as keras_model.h5")
