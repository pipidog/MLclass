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
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters ========================
# input data (y=x**x_power+x_shift+noise)
dataset_size=500
x_power=5          
x_shift=0.5        
noise_std=0.05     # noise standard deviation
# layer 
layer_node=10      # nodes of the hidden layer
act_func=tf.nn.relu  # activation_function
# train
steps=1000         # training steps
step_show=100      # number of steps to show results 
learning_rate=0.1

# generate data ======================
x_data = np.linspace(-1, 1, dataset_size)[:, np.newaxis] # dataset_size x 1
noise = np.random.normal(0, noise_std, x_data.shape)
y_data = x_data**x_power - x_shift + noise

# computation graph ==================

# typical layer structure
def add_layer(inputs, dim_in, dim_out, activation_function=None):
    # set weight, initial = random numbers
    Weights = tf.Variable(tf.random_normal([dim_in, dim_out]))
    # set biases, initial = 0.1
    biases = tf.Variable(tf.zeros([1, dim_out]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # set activation_function
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# define placeholder for inputs to network
x_tf = tf.placeholder(tf.float32, [None, 1])
y_tf = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(x_tf, 1, layer_node, activation_function=act_func)

# add output layer
prediction = add_layer(l1, layer_node, 1, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_tf-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# tensorflow session ==================
# important step
sess = tf.Session()

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for i in range(steps+1):
    # training
    sess.run(train_step, feed_dict={x_tf: x_data, y_tf: y_data})
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
        
plt.ioff()
plt.show()