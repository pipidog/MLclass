import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# parameters =============================
task='predict'  # 'train' / 'predict'
layer_nodes=[128,64,32]  
encoding_dim = 5 # size of encoding
epoch=30
batch_size=300
lr=0.0001
pred_size=5000
# main ===================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

if task is 'train':
	# this is our input placeholder
	input_img = Input(shape=(784,))

	# encoder layers
	encoded = Dense(layer_nodes[0], activation='relu')(input_img)
	encoded = Dense(layer_nodes[1], activation='relu')(encoded)
	encoded = Dense(layer_nodes[2], activation='relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	# decoder layers
	decoded = Dense(layer_nodes[2], activation='relu')(encoder_output)
	decoded = Dense(layer_nodes[1], activation='relu')(decoded)
	decoded = Dense(layer_nodes[0], activation='relu')(decoded)
	decoded = Dense(784, activation='tanh')(decoded)

	# construct the autoencoder model
	autoencoder = Model(input=input_img, output=decoded)

	# construct the encoder model for plotting
	encoder = Model(input=input_img, output=encoder_output)

	# compile autoencoder
	autoencoder.compile(optimizer=Adam(lr=lr), loss='mse')

	# training
	autoencoder.fit(x_train, x_train,
	                epochs=epoch,
	                verbose=2,
	                batch_size=batch_size,
	                shuffle=True)

	# save model
	encoder.save('encoder.h5')
	autoencoder.save('autoencoder.h5')

elif task is 'predict':
	encoder=load_model('encoder.h5')
	# plotting
	encoded_imgs = encoder.predict(x_test[:pred_size,:])
	print(encoded_imgs.shape)
	if encoding_dim > 2:
		#plot_only = 500
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); 
		low_dim_embs = tsne.fit_transform(encoded_imgs)
		plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=y_test[:pred_size],cmap='tab10')
	else:
		plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test[:pred_size],cmap='tab10')

	plt.colorbar()
	plt.show()