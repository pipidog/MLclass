import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(10)  
from keras import utils
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import plot_model

# hyper parameters =========================
task='train'   # 'train' / 'prediction'
# convolution part
tot_filter=(16,32)           # (16,36)
filter_size=(5,5)            # suggest: (5,5) or (3,3)
padding='same'               # suggest: 'same'
cnn_activation='relu'        # suggest: 'relu'
pool_size=(2,2)              # suggest: (2,2)

# connect to DNN
flatten_dropout=0.25         # flatten dropout rate 

# Dense part
dnn_nodes=[20]              # 128              
dnn_dropout=0.5
dnn_activation='relu'

# train model
epochs=5
batch_size=100

# show prediction (for postprocess only)
show_test_data=range(0,10)  # more than 10 will be dropped
check_confusion=[4,9]       # show label=5, but predict=3
# ==========================================
if task is 'train':
    # preprocessing data =======================

    # in CNN, all input must be 4D, (n_sample, n_pixel_x, n_pixel_y, channel)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
    x_test=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

    # one-hot encoding
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    # build model =================================
    model = Sequential()

    # construct CNN part 
    for n_filter in tot_filter:
        # convolution
        model.add(Conv2D(
        filters=n_filter,
        kernel_size=filter_size,
        padding=padding, 
        input_shape=x_train.shape[1:],
        activation=cnn_activation))
        
        # maxpooling
        model.add(MaxPooling2D(pool_size=pool_size))

    # connect to DNN
    model.add(Flatten())
    model.add(Dropout(flatten_dropout))

    for n_nodes in dnn_nodes:
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(dnn_dropout))

    # output
    model.add(Dense(10,activation='softmax'))

    print(model.summary())
    plot_model(model,to_file='model.png')

    # train model =====================================
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy']) 

    train_history=model.fit(x=x_train, 
                            y=y_train,
                            validation_split=0.2, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            verbose=2)

    # show train history ==============================
    def show_train_history(train_acc,test_acc):
        plt.plot(train_history.history[train_acc])
        plt.plot(train_history.history[test_acc])
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    show_train_history('acc','val_acc')
    show_train_history('loss','val_loss')

    # save training results ==========================
    model.save('model.h5')
    np.savez('test_data.npz',x_test=x_test,y_test=y_test)
    
elif task is 'prediction':
    # load trained data
    model=load_model('model.h5')
    data=np.load('test_data.npz')

    x_test=data['x_test']
    y_test=data['y_test']

    # show accuracy ====================================
    scores=model.evaluate(x_test , y_test)
    print()
    print('performance on test datasets: loss = %.4f, accuracy = %.4f' % (scores[0],scores[1]))
    
    # show prediction ===================================
    prediction=model.predict_classes(x_test)
    print()
    def plot_test_images(image_id):
        fig = plt.gcf()
        fig.set_size_inches(10, 4)
        if len(image_id) >10: image_id=image_id[0:10] 
        for n, n_id in enumerate(image_id):
            ax=plt.subplot(2,5, 1+n)
            ax.imshow(np.squeeze(x_test)[n_id],cmap='binary')
            ax.set_title('label={0}, predict={1}'.format(np.argmax(y_test[n_id]),prediction[n_id]),fontsize=10)       
            ax.set_xticks([]);ax.set_yticks([]) 
        plt.show() 
    plot_test_images(show_test_data)

    # confusion matrix ==================================
    print('confusion matrix =============')
    ct=pd.crosstab(np.argmax(y_test,1),prediction,
                rownames=['label'],colnames=['predict'])
    print(ct)

    print('confused data ===========')
    df = pd.DataFrame({'label':np.argmax(y_test,1), 'predict':prediction})
    confuse_data=df[(df.label==check_confusion[0])&(df.predict==check_confusion[1])]
    print(confuse_data)
    plot_test_images(confuse_data.index.tolist())