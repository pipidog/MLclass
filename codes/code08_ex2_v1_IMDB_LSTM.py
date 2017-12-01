import numpy as np

np.random.seed(1)
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# parameters =======================
num_words = 20000    # size of dictionary
maxlen = 80          # size of article
batch_size = 32      # train batch size

# main =============================
# load IMDB data (already in sequence) -----
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# pad sequences ----------------------------
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# build LSTM model -------------------------
print('Build model...')
model = Sequential()
model.add(Embedding(num_words, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# optimizer --------------------------------
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model ------------------------------
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          verbose=2,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)