'''
This code shows how to apply GAN to MNIST dataset. 
It is to train the generator to forge human handwriting numbers.  
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(1) # make result reproducible
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils import plot_model


# define generator and discriminator ==============
class GANetwork:
    def __init__(self,rand_dim,gen_nodes,dis_nodes,dropout_rate):
        # define three models: generator, discriminator, and their combine 
        # check size of gen_nodes and dis_nodes
        if (len(gen_nodes)!=3) or (len(dis_nodes)!=3):
            print('Error: length of gen_nodes and dis_nodes must equal 3!')
            sys.exit()
        
        # define size of input random vector
        self.rand_dim=rand_dim
        
        # generator
        self.generator=Sequential()
        self.generator.add(Dense(gen_nodes[0],input_dim=rand_dim))
        self.generator.add(LeakyReLU(0.2))
        self.generator.add(Dense(gen_nodes[1]))
        self.generator.add(LeakyReLU(0.2))
        self.generator.add(Dense(gen_nodes[2]))
        self.generator.add(LeakyReLU(0.2))
        self.generator.add(Dense(784, activation='tanh'))
        self.generator.compile(loss='binary_crossentropy', optimizer=adam)
        
        # discriminator
        self.discriminator=Sequential()
        self.discriminator.add(Dense(dis_nodes[0],input_dim=784))
        self.discriminator.add(LeakyReLU(0.2))
        self.discriminator.add(Dropout(dropout_rate))
        self.discriminator.add(Dense(dis_nodes[1]))
        self.discriminator.add(LeakyReLU(0.2))
        self.discriminator.add(Dropout(dropout_rate))
        self.discriminator.add(Dense(dis_nodes[2]))
        self.discriminator.add(LeakyReLU(0.2))
        self.discriminator.add(Dropout(dropout_rate))
        self.discriminator.add(Dense(1,activation='sigmoid'))
        self.discriminator.compile(loss='binary_crossentropy', optimizer=adam)
        
        # combination
        self.discriminator.trainable=False # must have !
        comb_input=Input(shape=(rand_dim,))
        comb_output = self.discriminator(self.generator(comb_input))
        self.combination = Model(inputs=comb_input, outputs=comb_output)
        self.combination.compile(loss='binary_crossentropy', optimizer=adam)
    
    def info(self):
        print('\n## model summary: generator ##')
        print(self.generator.summary())
        plot_model(self.generator,to_file='model_generator.png')
        print('\n## model summary: discriminator ##')
        print(self.discriminator.summary())
        plot_model(self.discriminator,to_file='model_discriminator.png')
        print('\n## model summary: combination ##')
        print(self.combination.summary())
        plot_model(self.combination,to_file='model_combination.png')
        print('check model_xxx.png for model flow chart')
    
    def save(self,file_suffix='test'):
        self.generator.save('model_generator_'+file_suffix+'.h5')
        self.discriminator.save('model_discriminator_'+file_suffix+'.h5')
        self.combination.save('model_combination_'+file_suffix+'.h5')
        print('\nall models are saved')
        
    def generator_forge_image(self,batch_size,noise=None):
        if noise is None:
            noise=np.random.normal(0, 1, size=[batch_size, self.rand_dim])
            
        fake_img=self.generator.predict(noise)
        
        return fake_img
        
    def discriminator_judge_image(self,batch_img):
        return self.discriminator.predict(batch_img)
        
    def discriminator_train(self,batch_ture_img,batch_fake_img):
        self.discriminator.trainable=True
        x=np.vstack((batch_real_img,batch_fake_img))      
        y=np.zeros((len(x)))
        y[:len(batch_real_img)]=0.9
        loss=self.discriminator.train_on_batch(x,y)
        
        return loss
        
    def generator_train(self,batch_size):
        self.discriminator.trainable=False
        noise = np.random.normal(0, 1, size=[batch_size, self.rand_dim])
        y=np.ones(batch_size)
        loss=self.combination.train_on_batch(noise,y)
        
        return loss
        
class plotter:
    def loss(self,gen_loss, dis_loss, file_suffix='test',show_img=True):
        plt.clf()
        plt.plot(gen_loss,label='generator loss')
        plt.plot(dis_loss,label='discriminator loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('GAN_training_loss_'+file_suffix+'.png')
        if show_img:
            plt.show()
        
    def image(self,batch_img,file_suffix='test',show_img=True):
        # will only plot first 100 images. 
        plt.clf()
        tot_img=len(batch_img)
        batch_img=batch_img.reshape(tot_img,28,28)
        plt.figure(figsize=[10,10])
        for n in range(tot_img):
            plt.subplot(10,10,n+1)
            plt.imshow(batch_img[n], cmap='binary')
            plt.axis('off')
        plt.savefig('GAN_images_'+file_suffix+'.png')
        if show_img:
            plt.show()
    
if __name__=='__main__':
    # parameters =====================================
    rand_dim=10 # size of random vector to generate image
    gen_nodes=[200,500,1000]  # nodes of three hidden layers in generator
    dis_nodes=[1000,500,200]  # nodes of three hidden layers in discriminator
    dropout_rate=0.3          # dropout rate in discriminator    
    tot_epoch=50               # total traning epoch  
    batch_size=200            # batch size of each training  
    show_img=False            # whether to show images of each epoch    
    fix_test_noise=True       # whether to fix test noise
    
    # Preprocess data ================================
    # Optimizer
    adam = Adam(lr=0.0002, beta_1=0.5)
    
    # Load MNIST data
    (x_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train = x_train.reshape(60000, 784)
    batch_per_epoch=int(x_train.shape[0] / batch_size)
    
    # generate a fix noise
    fix_noise=np.random.normal(0, 1, size=[100,rand_dim])
    
    # train gan =======================================
    gan=GANetwork(rand_dim,gen_nodes,dis_nodes,dropout_rate)
    gan.info()
    view=plotter()
    
    fake_img=gan.generator_forge_image(100)
    view.image(fake_img,str(0),show_img)

    gen_loss=[]
    dis_loss=[]
    print()
    for epoch in range(1,tot_epoch+1):
        print('\n----- epoch = {0: 3d} -----'.format(epoch))        

        # ----- gan training ------ 
        t0=time.time()
        for n in range(batch_per_epoch):
            # show progress
            if (n % int(0.1*batch_per_epoch)==0):         
                print(' {0}% ..'.format(int(100*(n+1)/batch_per_epoch)),end='')
            elif n==batch_per_epoch-1:
                print(' done => ',end='')
            sys.stdout.flush()
            
            # prepare true and fake data
            batch_real_img = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            batch_fake_img=gan.generator_forge_image(batch_size)
            
            # train discriminator
            dis_loss_batch=gan.discriminator_train(batch_real_img,batch_fake_img)
            
            # train generator
            gen_loss_batch=gan.generator_train(batch_size)
        t1=time.time()
        print(' time elpase = {0:.1f}s'.format(t1-t0))
        # -------------------------     
          
        dis_loss.append(dis_loss_batch)
        gen_loss.append(gen_loss_batch)
        
        # check performance of forge images
        if fix_test_noise:
            fake_img=gan.generator_forge_image(100,fix_noise)
        else:
            fake_img=gan.generator_forge_image(100)
            
        view.image(fake_img,str(epoch),show_img)
        cheat_rate=np.sum(gan.discriminator_judge_image(fake_img)>0.7)/len(fake_img)
        
        print('\nloss of discriminator = {0}'.format(dis_loss[-1]))
        print('loss of generator = {0}'.format(gen_loss[-1]))
        print('cheating rate = {0}'.format(cheat_rate))
        
    # show loss curves
    view.loss(gen_loss, dis_loss, str(epoch),show_img)
    
    # save trained models
    gan.save(str(tot_epoch))
        
    