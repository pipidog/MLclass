'''
<summary>:
this code shows how to train MNIST data w/ and w/o
dropout layers. When dropout comes in, the overfitting
isssue is improved. 

<procedure>: 
- load MINST data set
- constrct NN w/ dropout
- training

<Note>:
- compare accuracy when dropout_rate=0.0 and 0.6, which one
  has worse overfitting issue?
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# Parameters ======================================
# hidden layer
hidden_nodes=[250]
act_func=tf.nn.relu

# train
steps=2000
batch_size=200
record_steps=50
learning_rate=0.5
dropout_rate=0.6     # 0~1 

# generate data ===================================
# import mnist data set, class: 0~9, size: 28x28 pixels 
# one_hot: set label to vector form
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 


# computation graph ===============================
tf.set_random_seed(1)
# placeholder for inputs
x_tf = tf.placeholder(tf.float32, [None, 784])    # 28x28
y_tf = tf.placeholder(tf.float32, [None, 10])     # 0~9
tf_is_training = tf.placeholder(tf.bool, None)  # control whether tf is in training

# add hidden layer
inp_layer=x_tf
h=[]
if len(hidden_nodes)!=0:
    for n, n_nodes in enumerate(hidden_nodes): 
        h.append(tf.layers.dense(inp_layer,n_nodes,activation=act_func))   
        h[-1]=tf.layers.dropout(h[-1],rate=dropout_rate,training=tf_is_training)
        # set as next input
        inp_layer=h[-1]

# output layer
prediction = tf.layers.dense(inp_layer,10, activation=tf.nn.softmax)
# define cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_tf * tf.log(prediction),reduction_indices=[1]))                                                   
# define train op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# tensorflow session ==============================
def compute_accuracy(v_x_tf, v_y_tf):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_tf: v_x_tf, tf_is_training: False})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_tf: v_x_tf, y_tf: v_y_tf, tf_is_training: False})
    return result

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

acc_train=np.array([])
acc_test=np.array([])
for n in range(steps+1):
    batch_x_tf, batch_y_tf = mnist.train.next_batch(batch_size)
    sess.run(train_op, feed_dict={x_tf: batch_x_tf, y_tf: batch_y_tf, tf_is_training: True})
    if n % record_steps == 0:
        acc_train=np.hstack((acc_train,compute_accuracy(mnist.train.images, mnist.train.labels)))
        acc_test=np.hstack((acc_test,compute_accuracy(mnist.test.images, mnist.test.labels)))
        print(' ------- ')
        print('step={0:4d}, train accuracy={1:5.3f}'.format(n, acc_train[-1]))
        print('step={0:4d}, test accuracy={1:5.3f}'.format(n, acc_test[-1]))
plt.plot(np.linspace(0,steps,acc_train.shape[0]),acc_train,'b',\
np.linspace(0,steps,acc_test.shape[0]),acc_test,'r')
plt.legend(['train','test'])
#plt.text(int(steps/2),0.55,'Dropout rate={0}'.format(dropout_rate),fontdict={'size': 14, 'color': 'green'})
plt.text(int(steps/2),0.5,'train acc={0:.3f}'.format(acc_train[-1]),fontdict={'size': 14, 'color': 'green'})
plt.text(int(steps/2),0.45,'test acc={0:.3f}'.format(acc_test[-1]),fontdict={'size': 14, 'color': 'green'})
plt.show()