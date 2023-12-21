import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('stocks.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# data.set_index('Date', inplace=True)
# # plt.title('Stock Prices')
# # plt.xlabel('Dates')
# # plt.ylabel('Close Prices')
# # plt.plot(data['Adj Close'])
# # plt.show()
# train_data, test_data = data[0:int(len(data)*0.5)], data[int(len(data)*0.54):]
# plt.title('Stock Prices')
# plt.xlabel('Dates')
# plt.ylabel('Close Prices')
# plt.plot(data['Adj Close'], 'blue', label='Training Data')
# plt.plot(test_data['Adj Close'], 'green', label='Testing Data')
# plt.legend()
# plt.show()

# f = plt.figure(figsize=(12, 6))
# plt.matshow(df.corr(), fignum=f.number)
# plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
# plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.show()

dec_2020 = '2020-12-31'
mask = (df['Date'] <= dec_2020)
data_20 = df.loc[mask]
mask = (df['Date'] > dec_2020)
data_23 = df.loc[mask]
training_set = data_20.iloc[:,4:5].values
scaler = MinMaxScaler(feature_range = (0,1)) 
scaled_training_set = scaler.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60, 1795):
    X_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis
model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")
hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)
plt.figure(figsize = (16, 8))
plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
testing_set = data_18.iloc[:,1:2]
y_test = testing_set.iloc[60:,0:].values
testing_set = testing_set.iloc[:,0:].values 
scaled_testing_set = scaler.transform(testing_set)