'''
<summary>:
use tensorflow to perform:
- single hidder layer with arbitary nodes 
- use arbitary activation_function
- supervised Regression learning

<procedure>: 
- generate raw data: y=x**p-x0
- define tensorflow computation graph
- training 

<difference>:
- using latest 2017 tensorflow API
- layer structure can be generated tf.layers.dense (like Keras)
- sess.run can evaulate several nodes simultaneously
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import timeit


# Parameters ========================
# input data (y=x**x_power+x_shift+noise)
dataset_size=1000
x_power=4          
x_shift=1.0        
noise_std=0.05     # noise standard deviation
# layer 
layer_node=10      # nodes of the hidden layer
act_func=tf.nn.relu  # activation_function
# train
batch_size=10      # batch size
steps=1000         # training steps
step_show=100      # number of steps to show results 
learning_rate=0.5
# generate data ======================
x_data = np.linspace(-1, 1, dataset_size)[:, np.newaxis] # dataset_size x 1
noise = np.random.normal(0, noise_std, x_data.shape)
y_data = x_data**x_power - x_shift + noise

# computation graph ==================
# define placeholder for inputs to network
x_tf = tf.placeholder(tf.float32, [None, 1])
y_tf = tf.placeholder(tf.float32, [None, 1])

# neural network layers
l1 = tf.layers.dense(x_tf, 10, act_func)          # hidden layer
prediction = tf.layers.dense(l1, 1)                     # output layer

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_tf-prediction), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# tensorflow session ==================
# important step
sess = tf.Session()

# initialize variables
sess.run(tf.global_variables_initializer())

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

start = timeit.default_timer()
for i in range(steps+1):
    # random select batch set
    batch_select=np.random.randint(0,len(x_data)-1,batch_size)
    
    _, loss_value, prediction_value=\
    sess.run([train_op,loss,prediction], {x_tf: x_data[batch_select], y_tf: y_data[batch_select]})

    if i % step_show == 0:
        # evaulate values
        prediction_value = sess.run(prediction, feed_dict={x_tf: x_data})
        loss_value=sess.run(loss,feed_dict={x_tf: x_data, y_tf: y_data})
        # plot the prediction
        plt.cla()
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        ax.scatter(x_data, y_data)
        plt.title('step={0}, Loss={1:.4f}'.format(i,loss_value), fontdict={'size': 12, 'color': 'green'})
        plt.pause(0.5)
stop = timeit.default_timer()
print('time elapse={0}'.format(stop-start))        
plt.ioff()
plt.show()