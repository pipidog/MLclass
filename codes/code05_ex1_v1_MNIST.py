'''
<summary>:
this example shows how to visualize MNIST hand-writing dataset

<procedure>: 
- load assigned dataset
- reshape image data from 784 to 28x28
- plot the image

<difference>:

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# parameters ============================
dataset='train'   #'test','train', validation'
image_id=range(0,30)

# Main ==================================
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 
tot_fig=-1
ax=[]
for n, id_n in enumerate(image_id):
    if dataset=='test':
        tot_image=len(mnist.test.images.view())
        image_pixel=mnist.test.images.view()[id_n,:]
        image_label=mnist.test.labels.view()[id_n,:]
    elif dataset=='train':
        tot_image=len(mnist.train.images.view())
        image_pixel=mnist.train.images.view()[id_n,:]
        image_label=mnist.train.labels.view()[id_n,:]    
    elif dataset=='validation':
        tot_image=len(mnist.validation.images.view())
        image_pixel=mnist.validation.images.view()[id_n,:]
        image_label=mnist.validation.labels.view()[id_n,:]
    if n==0:
        print('------------------------------')
        print('total image of dataset-{0} = {1}'.format(dataset,tot_image))
    image_pixel.shape=(28,28)
    txt='image id={0}, label={1}'.format(id_n,image_label.tolist().index(1))
    if n % 10 == 0:
        tot_fig+=1
        fig=plt.figure(tot_fig)
    fig.add_subplot(2,5,(n % 10)+1)
    plt.imshow(image_pixel, cmap='binary')
    plt.title(txt,fontdict={'size': 12})
plt.show()
    