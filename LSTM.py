import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import tensorflow as tf
print("Process Start!")
dataset = pd.read_csv("stockprice.csv",index_col='Date', parse_dates=['Date'])
train_set = dataset[:'2016'].iloc[:, 1:2].values # High value from 2006 to 2016
test_set = dataset['2017':].iloc[:,1:2].values # High value in 2017

def pic(test_result, predict_restult, title):
    plt.plot(test_result, color='red', label='Ground Truth')
    plt.plot(predict_restult, color='blue', label='Prediction')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

sc = MinMaxScaler(feature_range=[0, 1])
train_set_scaled = sc.fit_transform(train_set)

# 60 timestamp a sample, one output
X_train = []
y_train = []
for i in range(60, 2769):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Input of LSTM：(samples, sequence_length, features)
# reshape: training set (2709,60)  ---> (2709, 60, 1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# ***************LSTM Model ***************
lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1))) # input shape is 60
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(128, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(128)) # the default return_sequence is false, therefore, it will return 1 deminsion
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='rmsprop', loss='mse')

time_start = time.time()  # Record start time
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)
time_end = time.time()  # Record finish time
time_sum = time_end - time_start  # Unit is second
print("Time for training LSTM model is: ",time_sum," second!")
dataset_total = pd.concat((dataset['High'][:"2016"], dataset['High']["2017":]), axis=0)
inputs = dataset_total[len(train_set):].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.fit_transform(inputs)
dataset_total = pd.concat((dataset['High'][:"2016"], dataset['High']["2017":]), axis=0)

inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 311):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict_test = lstm_model.predict(X_test) # Predict
predict_stock_price = sc.inverse_transform(predict_test)  # Inverse transform, otherwise, the result is in range 0-1

pic(test_set, predict_stock_price,"LSTM_Model")
# Calculate MSE of LSTM model's result
LSTM_MSE = mean_squared_error(test_set,predict_stock_price)
print("MSE of LSTM is: ",LSTM_MSE)