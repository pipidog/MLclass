'''
This code shows how to use RNN to learn time series data. 
Assume you have an input singal sin(wt) and the output singal is sin(wt_phi).
If so, can we use RNN to predict the output using the input_singal?
'''
import numpy as np
import sys

np.random.seed(1)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Dense
from keras.optimizers import Adam

# parameters ========================
# fake output singal
phi_shift=np.pi/2

# RNN parameters 
model_type='GRU'  #'RNN','LSTM','GRU'
time_steps = 20
batch_size = 50
cell_size = 50

# training parameters
learning_rate = 0.006
tot_steps=300

# generate data ======================
def probe_data(time_start,time_steps,batch_size,phi_shift=np.pi/2):
    # set time unit by setting omega=1
    omega=1
    # assume probing limit is Pi/200 seconds per probe
    probe_interval=(np.pi)/200
    time_interval=time_steps*probe_interval*batch_size
    
    tot_probe=time_steps*batch_size
    time_end=time_start+time_interval
    
    # generate time series
    t=np.linspace(time_start,time_end,tot_probe).reshape(batch_size,time_steps)
    
    # x: probed input singal, y: probed output singal    
    
    input_singal=np.sin(omega*t).reshape(batch_size,time_steps,1)
    output_singal=np.sin(omega*t+phi_shift).reshape(batch_size,time_steps,1)
    
    return t, input_singal, output_singal
    
    
# build model ========================
# RNN layer
model = Sequential()
if model_type=='RNN':
    model.add(SimpleRNN(
        batch_input_shape=(batch_size, time_steps, 1),       
        units=cell_size,
        return_sequences=True,      # true: means output state for each cell
        stateful=True,              # true: last state as initial for next batch 
    ))
elif model_type=='LSTM':
    model.add(LSTM(
        batch_input_shape=(batch_size, time_steps, 1),       
        units=cell_size,
        return_sequences=True,      # true: means output state for each cell
        stateful=True,              # true: last state as initial for next batch 
    ))
elif model_type=='GRU':
    model.add(GRU(
        batch_input_shape=(batch_size, time_steps, 1),       
        units=cell_size,
        return_sequences=True,      # true: means output state for each cell
        stateful=True,              # true: last state as initial for next batch 
    ))

# output layer
model.add(TimeDistributed(Dense(1))) # all cell output share the same connections
model.compile(optimizer=Adam(learning_rate),loss='mse',)

print(model.summary())

# train model ======================            
t_batch=np.zeros((batch_size,time_steps))
for step in range(tot_steps+1):
    # note: only shift a "time_step" in data generation for each step. 
    t_batch, x_batch, y_batch=\
    probe_data(t_batch[0,-1],time_steps,batch_size,phi_shift=phi_shift)
    
    loss=model.train_on_batch(x_batch,y_batch)
    prediction=model.predict(x_batch,batch_size)

    # plot time series results
    plt.clf()         
    plot_range=int(batch_size)
    plt.plot(t_batch[0:plot_range].flatten(),y_batch[0:plot_range].flatten(),'r',
             t_batch[0:plot_range].flatten(),prediction[0:plot_range].flatten(),'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('step = {0}, loss = {1}'.format(step,loss))

