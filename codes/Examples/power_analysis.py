# this code is to build a model to analysze the household power consupmtion
import sys
from datetime import datetime
import pickle 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout
from keras.optimizers import Adam

# Hyper Parameters =================
task = 'train_model'         # clean_data -> view_data -> train_model -> prediction
# data cleaning --------------------
raw_data_name = 'household_power_consumption.txt'
clean_data_name = 'clean_data.csv'
n_resample_freq = 20         # int, in unit of minutes

# train model ------------------------
n_reference_step = 3         # int, in unit of n_resample_freq
n_predict_step = 1           # int, in unit of n_resample_freq

predict_feature = ['Global_active_power']
model_type='LSTM'            # 'RNN' / 'LSTM' / 'GRU'
epochs = 10                  # num of epoch
learning_rate = 0.001        # learning rate
batch_size = 72              # training bacth size
train_split = 0.8            # precentage of data used as train set (0.1~0.9)
# 
# ==================================
# check hyperparameters 
if task is 'clean_data':
	assert type(n_resample_freq) is int
	# define data parser
	def date_parser(x,y):
		return datetime.strptime(x+' '+y, '%d/%m/%Y %H:%M:%S')

	# read data
	df	= pd.read_csv(raw_data_name,sep=';', 
		parse_dates = [[0,1]], date_parser=date_parser, index_col=0, 
		header=0, low_memory=False)

	# interpolate missing time series data
	df = df.replace('?',np.nan).astype(float).interpolate(method='linear')
	print('size of raw data: ', df.shape)

	# resample data to lower frequency
	if n_resample_freq is not 1:
		print('resampling data with frequency: ', n_resample_freq)
		sys.stdout.flush()
		df = df.resample(str(n_resample_freq)+'min').mean()
		print('size of resampled data: ', df.shape)
	df.to_csv(clean_data_name)

elif task is 'view_data':	
	assert type(n_reference_step) is int
	assert type(n_predict_step ) is int 

	# reload data
	df = pd.read_csv(clean_data_name, index_col=0, header=0)
	values = df.values
	for n in range(7):
		plt.subplot(7,1,n+1)
		plt.plot(values[:,n])
		plt.title(df.columns[n], y=0.5, loc='left')
		if n!=6:
			plt.xticks([])

	plt.show()

elif task is 'train_model':
	# define function to reshape time series data to RNN compatible format
	def time_series_to_RNN(data_values, n_reference_step, n_predict_step, predict_feature ):
		tot_step, tot_feature=data_values.shape
		tot_RNN_sample=tot_step-n_predict_step-n_reference_step+1
		RNN_x=np.zeros((tot_RNN_sample,n_reference_step, tot_feature))
		RNN_y=np.zeros((tot_RNN_sample,n_predict_step*len(predict_feature)))

		for n in range(tot_RNN_sample):
			RNN_x[n,:,:]=data_values[n:n+n_reference_step,:]
			RNN_y[n,:]=data_values[n+n_reference_step:n+n_reference_step+n_predict_step,predict_feature].reshape(1,-1)

		return RNN_x, RNN_y

	# get index from header labels
	def header_index(df_header,feature_names):
		feature_index=[]
		for name in feature_names: 	
			feature_index.append(np.where(df_header==name)[0][0])

		return feature_index

	# check hyperparameters 
	assert (train_split >= 0.1) & (train_split <=0.9)

	# reload data
	df = pd.read_csv(clean_data_name, index_col=0, header=0)
	values = df.values	
	tot_step, tot_feature = values.shape

	# scale data for training
	data_scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_values = data_scaler.fit_transform(values)

	# split data to train set and test set
	x_train, y_train = time_series_to_RNN(scaled_values[:int(train_split*tot_step)],
	 n_reference_step, n_predict_step, header_index(df.columns,predict_feature))

	x_test, y_test = time_series_to_RNN(scaled_values[int(train_split*tot_step)+1:], 
		n_reference_step, n_predict_step, header_index(df.columns,predict_feature))

	# build RNN model
	model = Sequential()
	if model_type is 'RNN':
		model.add( SimpleRNN(
			batch_input_shape=(None , n_reference_step, tot_feature),
			units=100,
			return_sequences=False,
			stateful=False))
	elif model_type is 'LSTM':
		model.add( LSTM(
			batch_input_shape=(None , n_reference_step, tot_feature),
			units=100,
			return_sequences=False,
			stateful=False))
	elif model_type is 'GRU':
		model.add( GRU(
			batch_input_shape=(None , n_reference_step, tot_feature),
			units=100,
			return_sequences=False,
			stateful=False))

	model.add(Dense(20,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(predict_feature)*n_predict_step))
	print(model.summary())

	model.compile(optimizer=Adam(learning_rate),loss='mse')
	history=model.fit(
	    x_train, 
	    y_train,
	    batch_size=batch_size,
	    epochs=epochs,
	    verbose=2,
	    validation_data=(x_test, y_test),
	    shuffle=False
		)

	# plot training results
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Error')
	plt.legend()
	plt.savefig('train_history.png')
	plt.show()	
	
	# save all data
	model.save('model.h5')
	model_data = {'x_test': x_test, 'y_test': y_test, 
		'x_train': x_train, 'y_train': y_train,
		'scale_factor': np.array(data_scaler.scale_)[header_index(df.columns,predict_feature)]}
	pickle.dump(model_data, open('model_data.pkl','wb'))

elif 'prediction':
	model = load_model('model.h5')
	model_data = pickle.load(open('model_data.pkl','rb'))
	x_test = model_data['x_test']
	y_test = model_data['y_test']
	scale_factor = model_data['scale_factor']


	y_pred=model.predict(model_data['x_test'])
	print(y_pred.shape)
	print(model_data['y_test'].shape)
	#model_data['data_scaler']=
	plt.plot(scale_factor*y_test[500:800],'b',label = 'True Data')
	plt.plot(scale_factor*y_pred[500:800],'r', label = 'Predicted')
	plt.xlabel('time ( 20 min)')
	plt.ylabel('Global_active_power')
	plt.legend()
	plt.savefig('prediction.png')
	plt.show()

	







