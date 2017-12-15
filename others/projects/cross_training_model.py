'''
author: pipidog@gmail.com

Subject:
peer cross training model

Question: 
consider we have 60000 MNIST data but only 20000 of them are labeled
all the others are not. Can we use the unlabeled 40000 data to improve
our accuracy?

Idea:
randomly pick some of the labeled data to train two or more different 
machines. Since they read different data, their understanding of the 
data shoul also be different.

For the unlabeled data, we ask each machine to pedict their categories
(i.e. the softmax probability) and pick the one with higest probability
as the right anwser to train all machines again.  
'''

import numpy as np
import keras.utils as utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Parameters ======================================
layer_nodes=[100,100]       # node of hidder layers
act_func='relu'             # activation function
label_dataset=20000         # less than 60000
peer_dataset=[3000,10000]   # initial feeding of each peer
cross_train_times=4         # batch=(# of unlabel)/times       
dropout_rate=0.5            # dropout rate
# preprocessing data ===============================
# fix seed to make this code repeatable
np.random.seed(10)

# load mnist data
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()

# reshape to rows and normalize to 0~1
x_train=x_train_image.reshape(-1,784).astype('float32')/255
x_test=x_test_image.reshape(-1,784).astype('float32')/255

# change label to binary representation
y_train=utils.to_categorical(y_train_label)
y_test=utils.to_categorical(y_test_label)

x_train_label=x_train[:label_dataset]
x_train_unlabel=x_train[label_dataset:]
y_train_label=y_train[:label_dataset]
y_train_unlabel=y_train[label_dataset:]

test_result=[] # to receive test results of each retrain 
# build model =======================================
tot_peer=len(peer_dataset)
model=[]
# construct hidden layer
for peer in range(tot_peer):
    model.append(Sequential())
    for n,nodes in enumerate(layer_nodes):
      if n==0:
        model[-1].add(Dense(units=nodes,input_dim=784,kernel_initializer='normal',activation=act_func))
      else:
        model[-1].add(Dense(units=nodes,kernel_initializer='normal',activation=act_func))
        model[-1].add(Dropout(dropout_rate))
    # output layer
    model[-1].add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
    model[-1].compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    print('* Model Summary -------------')
    print(model[-1].summary())

# independent training =============================
train_history=[]
for peer in range(tot_peer):
        print('independent training of peer-{0} ...'.format(peer))
        data_select=np.random.permutation(label_dataset)[:peer_dataset[peer]]
        train_history.append(model[peer].fit(x=x_train_label[data_select],
          y=y_train_label[data_select],validation_split=0.0, 
            epochs=20, batch_size=200,verbose=2))

# test accuracy ====================================
test_result_tmp=[]
for peer in range(tot_peer):
  test_result_tmp.extend(model[peer].test_on_batch(x_test,y_test))
test_result.append(test_result_tmp)

# define peer training labels ======================
# pred is a list, each element is (None, 10) np array 
def peer_training_label(pred):
    tot_peer=len(pred)
    pred_new=np.array([]).reshape(len(pred[0][:,0]),0)
    for peer in range(tot_peer):
        pred_new=np.hstack((pred_new,pred[peer]))
    pred_new=np.remainder(np.argmax(pred_new,1),10)
    pred_new=np.squeeze(pred_new)
    pred_new=utils.to_categorical(pred_new)
    return pred_new

# cross training ===================================
ind_ini=0
for n in range(cross_train_times):
    print('cross training {0} ==========='.format(n))
    if n is not cross_train_times-1:
        ind_fin=ind_ini+int(len(x_train_unlabel)/cross_train_times)
    else:
        ind_fin=len(x_train_unlabel)

    # ask all peers to predict, then create new labels
    pred=[]
    for peer in range(tot_peer):
      pred.append(model[peer].predict(x_train_unlabel[ind_ini:ind_fin]))
    
    # retraining based on new predicted labels
    pred_new=peer_training_label(pred)
    train_history=[]
    for peer in range(tot_peer):
            print('resume training of peer-{0} ...'.format(peer))
            train_history.append(model[peer].fit(x=x_train_unlabel[ind_ini:ind_fin],
                y=pred_new,validation_split=0.0, 
                epochs=20, batch_size=200,verbose=2))
    # accuracy on test set
    test_result_tmp=[]
    for peer in range(tot_peer):
        test_result_tmp.extend(model[peer].test_on_batch(x_test,y_test))
    test_result.append(test_result_tmp)

    ind_ini=ind_fin
print('----- result of cross training -----')
[print('loss-{0} acc-{0} / '.format(n),end='') for n in range(tot_peer)]
print()
print(np.array(test_result))

