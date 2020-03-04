#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

data = pd.read_excel('database_backup.xlsx', head=1).drop('name', axis=1)
data = data[data.molweight != 'f'].sample(frac=1).to_numpy()
print('shape of data:', np.shape(data))
data_x, data_y = data[:,:3], data[:,3:]
print('shape of data_x, data_y:', np.shape(data_x), np.shape(data_y))

#x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
#scaled_x = x_scaler.fit_transform( data_x )
#scaled_y = y_scaler.fit_transform( data_y )

train_x, test_x = np.asarray(data_x[:5400], dtype=np.float), \
    np.asarray(data_x[5400:], dtype=np.float)
train_y, test_y = np.asarray(data_y[:5400], dtype=np.float), \
    np.asarray(data_y[5400:], dtype=np.float)

print('shape of train_x, test_x:', np.shape(train_x), np.shape(test_x))
print('shape of train_y, test_y:', np.shape(train_y), np.shape(test_y))


# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from time import time

neuron_count = 30

model = Sequential()
model.add(Dense(neuron_count, input_dim=3, activation='relu', kernel_initializer='normal'))
#TODO: add dropout
model.add(Dense(neuron_count, activation='relu'))
model.add(Dense(1, activation='linear', kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# In[9]:


history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                   epochs = 100, verbose=1, callbacks=[tensorboard])
results = model.evaluate(test_x, test_y, batch_size=64, verbose=1)
print(results)


# In[13]:
# Guessing time!
while True:
    answer = input('give me 3 comma-delimited vals: ')
    if answer == '!q':
        print('ending')
        break
    params = list( map( float, answer.split(',') ) )
    custom_test_x = np.array([params[0],params[1],params[2]]).reshape(1, -1)
    #scaled_custom_test_x = x_scaler.fit_transform(custom_test_x)
    prediction = model.predict( custom_test_x )
    #descaled_prediction = y_scaler.inverse_transform(prediction)
    print('prediction: ', prediction)
