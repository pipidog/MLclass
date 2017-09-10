'''
This is a simple example of linear neuron networks to perform
binary classification. 
1. We generate a random dataset
2. setup a simple linear neuron network
3. train the network by mininizing the cross entropy
'''
from numpy.random import RandomState
import tensorflow as tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters =============================================
dataset_size=128  # total dataset size
batch_size=8      # batch dataset size

# computation graph ======================================
# wieghts
W1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
W2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# input placeholder
x=tf.placeholder(tf.float32,shape=(None,2),name='input-data')

# output placeholder
y_=tf.placeholder(tf.float32,shape=(None,1),name='output-data')

# define nodes
a=tf.matmul(x,W1)
y=tf.matmul(a,W2)

# define cross entropy 
cross_entropy=-tf.reduce_mean( y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

# define training scheme
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# generate random datasets ===============================
rdm=RandomState(1)
X=rdm.rand(dataset_size,2)

# for simplicity, we define x1+x1 <1 as 1, else 0
Y=[[int(x1+x2<1)] for (x1,x2) in X]
print('initlal x -----')
print(X)
print('initial y -----')
print(Y)

# tensorflow session =====================================
print('session begins --------')
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)

	print('--- W initial ---')
	print(sess.run(W1))
	print(sess.run(W2))

	print('--- Training ---')
	STEPS=5000
	for i in range(STEPS):
		start=(i*batch_size) % dataset_size
		end=min(start+batch_size,dataset_size)
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

		if i % 1000==0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
			print('step = {0:4d}, cross entropy = {1}'.format(i, total_cross_entropy))

	print('--- W final ---')
	print(sess.run(W1))
	print(sess.run(W2))