'''
<summary>:
This code show how to generate tensorborad computation
graphe. 

<procedure>: 
This code is essential identical to code02_x but add
tensorborad summary

<difference>:
this version includes histogram and scalar output.
One can visualize the difference of the train 
behaviors using batch dataset or full dataset.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import timeit


# Parameters ========================
# input data (y=x**x_power+x_shift+noise)
dataset_size=1000
x_power=5          
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
train_scheme='all'  # 'batch' / 'all'

# generate data ======================
x_data = np.linspace(-1, 1, dataset_size)[:, np.newaxis] # dataset_size x 1
noise = np.random.normal(0, noise_std, x_data.shape)
y_data = x_data**x_power - x_shift + noise

# computation graph ==================
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    x_tf = tf.placeholder(tf.float32, [None, 1])
    y_tf = tf.placeholder(tf.float32, [None, 1])

# neural network layers

l1 = tf.layers.dense(x_tf, 10, act_func,name='hidden_0')          # hidden layer
tf.summary.histogram('hidden_o',l1)

prediction = tf.layers.dense(l1, 1, name='output')                     # output layer
tf.summary.histogram('prediction',prediction)

# the error between output and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_tf-prediction), reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
summary_merge=tf.summary.merge_all()

summary_write=tf.summary.FileWriter('logs/')  # write summary
# tensorflow session =================

#create session
sess = tf.Session()
summary_write.add_graph(sess.graph)          # add graph to summary

# initialize variables
sess.run(tf.global_variables_initializer())

for i in range(steps+1):
    # random select batch set
    if train_scheme=='batch':
        batch_select=np.random.randint(0,len(x_data)-1,batch_size)
        _, summary_data=\
        sess.run([train_op,summary_merge], {x_tf: x_data[batch_select], y_tf: y_data[batch_select]})
    elif train_scheme=='all':
        _, summary_data=\
        sess.run([train_op,summary_merge], {x_tf: x_data, y_tf: y_data})
    summary_write.add_summary(summary_data, i)
    