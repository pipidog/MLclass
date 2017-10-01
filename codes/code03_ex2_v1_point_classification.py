'''
<summary>:
this example shows how to classifiy point dataset with labels 
using fully connected NN.

<procedure>: 
- create datasets. the dataset is constructed by genetating 
  random distribution of points around a point center.
  If there are 4 points, then the number of data around a 
  center is contronlled by train_num and test_num. 
- constructed fully connected NN
- Train NN using batch dataset. 

<Note>:
1. compare how accuracy changes if noise_std=0.7 and 1.0
2. If noise_std=1.0, will add layers or nodes help?
3. If noise_std=1.0, will increase batch_size help?
4. If noise_std=1.0, will reducing learning_rate help?
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

# parameters =============================
# generate data
data_center=[[2,2],[2,-2],[-2,2],[-2,-2]] # data distribtion centers
train_num=1000                     # number of data around a data center
test_num=100
noise_std=1.0                    # noise of data around a center
# hidden layer
layer_nodes=[10]
act_func=tf.nn.relu
# train
batch_size=50
step=500
step_show=10
learning_rate=0.5
# generate data =============================
np.random.seed(1)
tot_class=len(data_center)

# create empty numpy array, so we can use vstack and hstack
x_train=np.array([]).reshape(0,2)
x_test=np.array([]).reshape(0,2)
y_train=np.array([])
y_test=np.array([])

# create training data and validation dataset around each data_center
for n, dc in enumerate(data_center):
        x_train=np.vstack((x_train,np.random.normal(np.tile(dc,(train_num,1)),noise_std)))
        y_train=np.hstack((y_train,np.repeat(n,train_num)))
        x_test=np.vstack((x_test,np.random.normal(np.tile(dc,(test_num,1)),noise_std)))
        y_test=np.hstack((y_test,np.repeat(n,test_num)))

# computation graph =========================
tf.set_random_seed(1)

x_tf = tf.placeholder(tf.float32, [None,2])      # input x
y_tf = tf.placeholder(tf.int32, None)            # input y

# hidden layers
h=[]
inp_dat=x_tf
for nodes in layer_nodes:
    h.append(tf.layers.dense(inp_dat,nodes,act_func))
    inp_dat=h[-1]

# output
output = tf.layers.dense(h[-1],tot_class)                         
# loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_tf, logits=output)   
# predictions
prediction = tf.argmax(output, axis=1)
# return (acc, update_op), and create 2 local variables
accuracy = tf.metrics.accuracy(labels=tf.squeeze(y_tf), predictions=prediction)[1]
# train operator
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# TF session ================================
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op) 

plt.ion() 
acc=np.array([]).reshape(0,2)
plt.figure(0,figsize=(18, 6))
plt.subplot(1,3,1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=100, lw=0, cmap='tab10')
plt.title('true test dataset',fontdict={'size': 14, 'color': 'green'})
for n in range(step+1):
    batch_select=np.random.randint(0,len(x_train)-1,batch_size)
    
    # train and net output
    _, acc_train = sess.run([train_op, accuracy], {x_tf: x_train[batch_select], y_tf: y_train[batch_select]})
    
    if n % step_show == 0:
        # plot and show learning process
        _, acc_test, pred = sess.run([train_op, accuracy, output], {x_tf: x_test, y_tf: y_test})
        acc=np.vstack((acc,np.array([acc_train,acc_test])))

        plt.subplot(1,3,2)
        plt.cla()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='tab10')
        plt.title('step={0:4d}, accuracy={1:.3f}'.format(n, acc_test),fontdict={'size': 14, 'color': 'green'})
        plt.pause(0.1)

# # plot accuracy curves
plt.subplot(1,3,3)
plt.plot(np.linspace(0,step,acc.shape[0]),acc[:,0],'b',\
np.linspace(0,step,acc.shape[0]),acc[:,1],'r')
plt.title('accuracy',fontdict={'size': 14, 'color': 'green'})
plt.legend(['train','test'])

plt.ioff()
plt.show()