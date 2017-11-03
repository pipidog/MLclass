import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

# parameters =============================
# CNN
cnn_filter1={'filters':16,'kernel_size':5,'strides':1,'padding':'same','activation':tf.nn.relu}
max_pooling1={'pool_size':2,'strides':2}
cnn_filter2={'filters':32,'kernel_size':5,'strides':1,'padding':'same','activation':tf.nn.relu}
max_pooling2={'pool_size':2,'strides':2}
# MLP
layer_nodes=[10,10]    # only two layers allowed in this code 

# Train
learning_rate = 0.001              # learning rate
batch_size = 50
step_per_epoch=50   # must be integer
epoch=10                           
tsne_plot=True

# lodaing dataset ========================
# set random seed to make result reproducible
tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
x_train = mnist.train.images
y_train = mnist.train.labels 
x_test = mnist.test.images
y_test = mnist.test.labels


# tensorflow computation graph ===========
x_tf = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(x_tf, [-1, 28, 28, 1])              # CNN must use: (tot_data, 2D size, channel)
y_tf = tf.placeholder(tf.int32, [None, 10])            # input y


# CNN 
conv1=tf.layers.conv2d(inputs=image,**cnn_filter1) #(28,28,1) -> (28,28,filters1)
pool1 = tf.layers.max_pooling2d(conv1,**max_pooling1) # (28,28,filters1) -> (14,14,filters1/pool_size1)
conv2 = tf.layers.conv2d(pool1, **cnn_filter2)    # -> (14, 14, filters2)
pool2 = tf.layers.max_pooling2d(conv2, **max_pooling2)    # -> (7, 7, filters2/pool_size2)
flat=tf.contrib.layers.flatten(pool2)  # or use, e.g. tf.reshape(pool2,[-1,7*7*32])

# fully connected
layer1 = tf.layers.dense(flat, layer_nodes[0])              # output layer
output = tf.layers.dense(layer1,layer_nodes[1])

# training objects
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_tf, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# measure performance, will return (acc, update_op), and create 2 local variables
accuracy = tf.metrics.accuracy(          
    labels=tf.argmax(y_tf, axis=1), predictions=tf.argmax(output, axis=1),)[1]


# function for tsne plot =============
def plot_with_labels(lowDWeights, labels):
    plt.cla(); 
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)) 
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()) 
    plt.ylim(Y.min(), Y.max()) 
    plt.title('Visualize last layer') 
    plt.show() 
    plt.pause(0.01)

# tensorflow session =================
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # accuracy_op has local var
sess.run(init_op)     # initialize var in graph

plt.ion()
for epoch_n in range(epoch):
    for step in range(step_per_epoch):
        x_train_batch, y_train_batch = mnist.train.next_batch(batch_size)
        _, loss_ = sess.run([train_op, loss], {x_tf: x_train_batch, y_tf: y_train_batch})


    accuracy_, flat_output = sess.run([accuracy, flat], {x_tf: x_test, y_tf: y_test})
    print('epoch:', epoch_n, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

    if tsne_plot:
        # visualization of trained flatten layer (T-SNE)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); 
        plot_only = 500
        low_dim_embs = tsne.fit_transform(flat_output[:plot_only, :])
        labels = np.argmax(y_test, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {x_tf: x_test[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(y_test[:10], 1), 'real number')