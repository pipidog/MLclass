'''
This code shows how to use RNN, LSTM and GRU in Keras to recongize 
MNIST dataset. The basic idea is to consider the MNIST image as 
a puzzle. For each time step, we input a single row to RNN.  
'''

import numpy as np
import sys

np.random.seed(1)  # make results reproducible
from keras.datasets import mnist
from keras import utils 
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, GRU, LSTM
from keras.optimizers import Adam

# parameters ========================
time_steps = 28     # height of images
input_size = 28     # width of images
batch_size = 50
cell_size = 100
learning_rate = 0.001
model_type='LSTM'   # 'RNN', 'LSTM', 'GRU'

# preprocessing data =================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# renormalize images to 0-1
x_train = x_train.reshape(-1, 28, 28) / 255.      # normalize
x_test = x_test.reshape(-1, 28, 28) / 255.        # normalize

# use one-hot representation for categories. 
y_train = utils.to_categorical(y_train, num_classes=10)
y_test = utils.to_categorical(y_test, num_classes=10)

# build model =========================
# RNN layer
model = Sequential()
if model_type=='RNN':
    model.add(SimpleRNN(
        batch_input_shape=(None, time_steps,input_size),
        units=cell_size))
elif model_type=='LSTM':
    model.add(LSTM(
        batch_input_shape=(None, time_steps,input_size),
        units=cell_size))  
elif model_type=='GRU':        
    model.add(GRU(
        batch_input_shape=(None, time_steps,input_size),
        units=cell_size))  

# output layer, 0-9, 10 categories        
model.add(Dense(10)) 
model.add(Activation('softmax'))

# optimizer
model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# output model summary
print(model.summary())
utils.plot_model(model,to_file='model.png',show_shapes=True)

# run model =========================
model.fit(
    x_train, 
    y_train,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=False
)

# save trained model
model.save('model.h5')