import numpy as np
import pickle
np.random.seed(1)
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
# hyperparameters ===================
task='denosing'   # 'train' / 'denosing'
ephcos=30
batch_size=128

# preprocesing data =================
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255. # renormalize data
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  

# add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# define plot image function =========
def plot_img(x_data,name_flag,is_show=False):
	assert x_data.shape[0] is 25
	for i in range(25):
	    ax = plt.subplot(5, 5, i+1)
	    plt.imshow(x_data[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	    plt.savefig('AE_denosing_'+str(name_flag)+'.png')
	if is_show:
		plt.show()

if task is 'train':
	# build CNN model ====================
	input_img = Input(shape=(28, 28, 1))  

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	# at this point the representation is (7, 7, 32)

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	# train mode by batch ==================
	autoencoder.fit(x_train_noisy, x_train,
	                epochs=ephcos,
	                batch_size=128,
	                shuffle=True,
	                verbose=2,
	                validation_data=(x_test_noisy, x_test))
	autoencoder.save('AE_denosing.h5')
elif task is 'denosing':
	autoencoder=load_model('AE_denosing.h5')
	denosing_img=autoencoder.predict(x_test_noisy[:25])
	plot_img(x_test_noisy[:25],0,True)
	plot_img(denosing_img,1,True)









#plot_img(x_train_noisy[:25],)
# batch_step=int(x_train.shape[0]/batch_size)
# for n in range(ephcos):
# 	print('ephcos',n)
# 	start=time.time()
# 	for m in range(batch_step):
# 		shuffle_int=np.random.randint(0, x_train.shape[0], size=batch_size)
# 		autoencoder.train_on_batch(x_train_noisy[shuffle_int],x_train[shuffle_int])

# 		if m % int(batch_step/10) == 0:
# 			print(str(int(100*m/batch_step))+'% .. ',end='')
# 			sys.stdout.flush()
# 	print('=> time elapsed = {0:.1f}s'.format(time.time()-start))

# 	# perform denosing
# 	denoising_img=autoencoder.predict(x_train_noisy[:25])
# 	plot_img(denoising_img,n)