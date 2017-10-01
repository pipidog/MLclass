'''
<summary>:
this example shows how to learn MINST dataset

<procedure>: 
- load MINST data set
- constrct NN
    number of layers and nodes are controned by hidden_nodes
    activation function is controned by act_func
    use softmax to make output as probability 
- training

<difference>:

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameters ======================================
# hidden layer
hidden_nodes = [50]
act_func = tf.nn.relu

# train
steps = 3000
batch_size = 300
step_show = 50
learning_rate = 0.5
cross_entropy_method = 'old'

# generate data ===================================
# import mnist data set, class: 0~9, size: 28x28 pixels
# one_hot: set label to vector form
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# computation graph ===============================

# placeholder for inputs
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])  # 0~9

# add hidden layer
inp_data = xs
h = []
if len(hidden_nodes) != 0:
    for n, n_nodes in enumerate(hidden_nodes):
        h.append(tf.layers.dense(inp_data, n_nodes, activation=act_func))
        inp_data = h[n]

# there are two ways to have cross_entropy and softmax
if cross_entropy_method == 'old':
    # output layer
    prediction = tf.layers.dense(inp_data, 10, activation=tf.nn.softmax)
    # define cross_entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys *
		tf.log(prediction), reduction_indices=[1]))

elif cross_entropy_method == 'new':
    prediction = tf.layers.dense(inp_data, 10)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))

# define train op
train_op = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)


# tensorflow session ==============================
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for n in range(steps + 1):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys})
    if n % step_show == 0:
        print('step={0:4d}, accuracy={1:5.3f}'.format(
            n, compute_accuracy(mnist.test.images, mnist.test.labels)))
