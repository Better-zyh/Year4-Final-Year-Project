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
# ***************Bi-LSTM Model ***************
bilstm_model = Sequential()
bilstm_model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], 1)))
bilstm_model.add(Dropout(0.2))
bilstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
bilstm_model.add(Dropout(0.2))
bilstm_model.add(Bidirectional(LSTM(64)))
bilstm_model.add(Dropout(0.2))
bilstm_model.add(Dense(1))
# Compile
bilstm_model.compile(optimizer='rmsprop', loss='mse')
# Train
time_start = time.time()  # Record start time
bilstm_model.fit(X_train, y_train, epochs=20, batch_size=32)
time_end = time.time()  # Record finish time
time_sum = time_end - time_start
print(time_sum)
dataset_total = pd.concat((dataset['High'][:"2016"], dataset['High']["2017":]), axis=0)
inputs = dataset_total[len(train_set):].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.fit_transform(inputs)
dataset_total = pd.concat((dataset['High'][:"2016"], dataset['High']["2017":]), axis=0)

inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Prepare x_test
X_test = []
for i in range(60, 311):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
bilstm_predicted = bilstm_model.predict(X_test)
print(bilstm_predicted)
print(bilstm_predicted.shape)
bilstm_predicted_stock_price = sc.inverse_transform(bilstm_predicted)
# Visualization
pic(test_set, bilstm_predicted_stock_price, "Bi-LSTM Model")
bilstm_MSE = mean_squared_error(test_set,bilstm_predicted_stock_price)
print("MSE of Bi-LSTM is: ",bilstm_MSE)