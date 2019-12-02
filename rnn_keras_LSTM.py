import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv('StockMarket/SP500_train.csv')
dataset_test = pd.read_csv('StockMarket/SP500_test.csv')

trainingset = dataset_train.iloc[:,5:6].values
testset = dataset_test.iloc[:,5:6].values

minMax = MinMaxScaler()
scaled_trainingset = minMax.fit_transform(trainingset)

X_train = []
y_train = []

for i in range(40,len(dataset_train)):
    X_train.append(scaled_trainingset[i-40:i,0])
    y_train.append(scaled_trainingset[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# shape - (num_of_samples,num_of_features,1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)


df = pd.concat((dataset_train['adj_close'], dataset_test['adj_close']),axis=0)
inputs = df[len(df) - len(dataset_test) - 40:].values
inputs = inputs.reshape(-1,1)

inputs = minMax.transform(inputs)

X_test = []
for i in range(40,len(dataset_test) + 40):
    X_test.append(inputs[i-40:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predictions = model.predict(X_test)
pred = minMax.inverse_transform(predictions)

plt.plot(testset,color='blue',label='Actual Prices')
plt.plot(pred,color='green',label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


