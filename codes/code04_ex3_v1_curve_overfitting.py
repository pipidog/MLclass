'''
<summary>:
this code performs a regression fitting of a linear
data w/ noise. One can compare the results w/ and w/o
dropout. Apprently, dropout has less overfitting issues. 

<procedure>: 
- generate a linear curve w/ noise
- construct NN w/ and w/o dropout
- train and plot results

<Note>:

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# parameters =====================
data_num=50
act_func=tf.nn.relu
hidden_nodes=[300,300]
learning_rate=0.01
dropout_rate=0.5
steps=1000
record_steps=50

# generate data ==================
# use seed to make results repeatable
tf.set_random_seed(1)
np.random.seed(1)

# training data
x_train = np.linspace(-1, 1, data_num)[:, np.newaxis]
y_train = x_train + 0.3*np.random.randn(data_num)[:, np.newaxis]

# test data
x_test = x_train.copy()
y_test = x_test + 0.3*np.random.randn(data_num)[:, np.newaxis]

# computation graph =============
# tf placeholders
x_tf = tf.placeholder(tf.float32, [None, 1])
y_tf = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# overfitting and dropout nets
inp_layer_o=x_tf
inp_layer_d=x_tf
h_o=[]
h_d=[]

for n_nodes in hidden_nodes: 
    # overfitting layers
    h_o.append(tf.layers.dense(inp_layer_o,n_nodes,activation=act_func))   
    inp_layer_o=h_o[-1]
    
    # dropout layers
    h_d.append(tf.layers.dense(inp_layer_d,n_nodes,activation=act_func))   
    h_d[-1]=tf.layers.dropout(h_d[-1],rate=dropout_rate,training=tf_is_training)
    inp_layer_d=h_d[-1]
    
prediction_o=tf.layers.dense(inp_layer_o,1)
prediction_d=tf.layers.dense(inp_layer_d,1)

loss_o=tf.losses.mean_squared_error(y_tf,prediction_o)
loss_d=tf.losses.mean_squared_error(y_tf,prediction_d)

train_o=tf.train.AdamOptimizer(learning_rate).minimize(loss_o)
train_d=tf.train.AdamOptimizer(learning_rate).minimize(loss_d)

# session ========================
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   
for n in range(steps+1):
    # training
    sess.run([train_o, train_d],{x_tf: x_train, y_tf: y_train, tf_is_training: True}) 
    
    # show results
    if n % record_steps == 0:
        # evaulate losses and predictions
        loss_o_, loss_d_, pred_o_, pred_d_ = sess.run(
            [loss_o, loss_d, prediction_o, prediction_d],
            {x_tf: x_test, y_tf: y_test, tf_is_training: False} # test, set is_training=False
        )
        # plot data
        plt.cla()        
        plt.title('step={0}'.format(n))
        plt.scatter(x_train, y_train, c='magenta', s=50, alpha=0.3, label='train'); 
        plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(x_test, pred_o_, 'r-', lw=3, label='overfitting'); 
        plt.plot(x_test, pred_d_, 'b--', lw=3, label='dropout={0}'.format(dropout_rate))
        plt.text(0, -1.2, 'overfitting loss={0:.4f}'.format(loss_o_), fontdict={'size': 14, 'color':  'red'}); 
        plt.text(0, -1.5, 'dropout loss={0:.4f}'.format(loss_d_), fontdict={'size': 14, 'color': 'blue'})
        plt.legend(loc='upper left'); 
        plt.pause(0.1)

plt.ioff()
plt.show()