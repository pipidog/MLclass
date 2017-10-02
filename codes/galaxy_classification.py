'''
This code shows how to perform binary classification of galaxy data provided
by Jen-Wei Hsueh. 

Note: to use it, please also download "galaxy_table.txt". 

Author: pipidog@gmail.com 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import keras.utils as utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Parameters ======================================
layer_nodes=[100,100]
act_func='relu'
dropout_rate=0.5
epochs=10

# preprocessing data ===============================
with open('galaxy_table.txt','r') as file:
    flines=file.readlines()

raw_data=np.array([line.split() for line in flines[2:]]).astype(np.float)    
print(raw_data.shape)
x=raw_data[:,0:2]
y=raw_data[:,-1]

# build model =======================================
model=Sequential()
# construct hidden layer
for n,nodes in enumerate(layer_nodes):
    if n==0:
        model.add(Dense(units=nodes,input_dim=2,kernel_initializer='normal',activation=act_func))
    else:
        model.add(Dense(units=nodes,kernel_initializer='normal',activation=act_func))
    model.add(Dropout(dropout_rate))
# output layer
model.add(Dense(units=1,kernel_initializer='normal',activation='sigmoid'))
print('* Model Summary -------------')
print(model.summary())

# Train model ======================================
model.compile(loss='binary_crossentropy',
  optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=x,y=y,validation_split=0.2, 
                        epochs=epochs, batch_size=200,verbose=2)

# plot train results ===============================
def show_train_history(train_history,train_item,valid_item):
    plt.plot(train_history.history[train_item])
    plt.plot(train_history.history[valid_item])
    plt.title('Train History')
    plt.ylabel(train_item)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

# prediction & confusion matrix =====================
prediction=model.predict_classes(x)
print(' confusion matrix ----------- ')
datatab=pd.crosstab(y,np.squeeze(prediction),
            rownames=['label'],colnames=['predict'])
print(datatab)

# plot data =========================================
plt.subplot(1,2,1)
plt.title('true galaxy data')
plt.scatter(x[:, 0], x[:, 1], c=y, s=10, lw=0, cmap='tab10')
plt.subplot(1,2,2)
plt.title('predicted galaxy data')
plt.scatter(x[:, 0], x[:, 1], c=prediction, s=10, lw=0, cmap='tab10')
plt.show()
