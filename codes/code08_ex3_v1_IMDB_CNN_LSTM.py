import sys
import numpy as np
np.random.seed(1)
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.datasets import imdb

# parameters =============================
# model type
model_type='CNN+LSTM'   # 'CNN' / 'CNN+LSTM

# Dense layer if CNN only
hidden_nodes=250

# Embedding
num_words = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

# Main ====================================

# load data -------------------------------
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# build model -----------------------------
print('Build model...')
model = Sequential()
model.add(Embedding(num_words, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

if model_type is 'CNN':
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dropout(0.2))
elif model_type is 'CNN+LSTM':
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
sys.exit()
# train model ------------------------------
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)