<p align="center">
  <img src="./others/keras-tensorflow-logo.jpg">
</p>

# Welcome to Deep Learning Zoo
This repo is the course website of my deep learning class: **Deep Learning Zoo**
at Physics Department, UC Davis. Each week I will introduce several deep learning
topics, from beginning to advanced. Participants are mostly PhD students or 
researchers who want to use deep learning techniques in their research fields.

Each week I will also prepare a few codes to demonstrate the concepts. Most of
our code are based on **Tensorflow** and **Keras**. If you have further 
questions, please email to spi@ucdavis.edu.

Note that deep learning can hugely speed up by using GPU. Therefore, if you have
nVidia GPUs, it is strongly recommend to use GPUs as backend. 

# Course Schedule:          
[week 01: Introduction to deep learning](https://github.com/pipidog/MLclass/blob/master/slide/slide01%20(introduction%20to%20ML).pdf)             
[week 02: Basis of Deep Learning: Regression](https://github.com/pipidog/MLclass/blob/master/slide/slide02%20(basis%20of%20deep%20learning%20and%20regression).pdf)            
[week 03: Basis of Deep Learning: Tensorboard & Classification](https://github.com/pipidog/MLclass/blob/master/slide/slide03%20(basis%20of%20deep%20learning%20and%20classification).pdf)    
[week 04: Basis of Deep Learning: Look back Deep Learning Again](https://github.com/pipidog/MLclass/blob/master/slide/slide04%20(basis%20of%20deep%20learning%20look%20back).pdf)            
[week 05: Basis of Deep Learning: MNIST & Keras](https://github.com/pipidog/MLclass/blob/master/slide/slide05%20(Keras%20and%20MNIST).pdf)           
[week 06: Conventional Neural Network](https://github.com/pipidog/MLclass/blob/master/slide/slide06%20(CNN).pdf)          
[week 07: Recurrent Neural Network](https://github.com/pipidog/MLclass/blob/master/slide/slide07%20(RNN%2BLSTM%2BGRU%2BIRNN).pdf)   
[week 08: Natural Language Processing](https://github.com/pipidog/MLclass/blob/master/slide/slide08%20(Natural%20Language%20Processing).pdf)    
[week 09: Autoencoder](https://github.com/pipidog/MLclass/blob/master/slide/slide09%20(autoencoder%2Bone-shot%20learning).pdf)    
[week 10: Advanced Neural Networks] 
                
# Example Codes:    
【week 1】:        
[Package Install.txt](https://github.com/pipidog/MLclass/blob/master/codes/01_Package_install/Package%20Install.txt)          
=> This file shows how to install all necessary packages and set up your environments.          
【week 2】:               
[MLP_curve_fitting_1.py](https://github.com/pipidog/MLclass/blob/master/codes/02_MLP_regression/MLP_curve_fitting_1.py)             
=> This code shows how to define a neural layer to fit a curve
[MLP_curve_fitting_2.py](https://github.com/pipidog/MLclass/blob/master/codes/02_MLP_regression/MLP_curve_fitting_2.py)            
=> similar to MLP_curve_fitting_1 but with batch training
[MLP_curve_fitting_3.py](https://github.com/pipidog/MLclass/blob/master/codes/02_MLP_regression/MLP_curve_fitting_3.py)      
=> similar to MLP_curve_fitting_2 but use built-in API to construct layers      
【week 3】:           
[Tensorboard_1.py](https://github.com/pipidog/MLclass/blob/master/codes/03_MLP_Classification/Tensorboard_1.py)     
=> This code shows how to use tensorboard to see computation graph      
[Tensorboard_2.py](https://github.com/pipidog/MLclass/blob/master/codes/03_MLP_Classification/Tensorboard_2.py)     
=> similar to Tensorboard_1.py, but shows how to visualize training results     
[Point_classification.py](https://github.com/pipidog/MLclass/blob/master/codes/03_MLP_Classification/Point_classification.py)       
=> This code shows how to use simple MLP for point classification       
【week 4】:       
[save_tf_model.py](https://github.com/pipidog/MLclass/blob/master/codes/04_Overfitting/curve_overtiffing.py)           
=> this code shows how to save trained model in tensorflow          
[curve_overtiffing.py](https://github.com/pipidog/MLclass/blob/master/codes/04_Overfitting/curve_overtiffing.py)        
=> this code shows how to use dropout to improve overfitting in a curve fitting problem     
【week 5】:       
[MLP_mnist_1.py](https://github.com/pipidog/MLclass/blob/master/codes/05_MLP_MNIST/MLP_mnist_1.py)     
=> this code shows how to load and visualize MNIST handwriting digits in tensorflow     
[MLP_mnist_2.py](https://github.com/pipidog/MLclass/blob/master/codes/05_MLP_MNIST/MLP_mnist_2.py)     
=> similar to MLP_mnist_1.py but now shows how to train a MLP for MNIST recognization       
[MLP_mnist_overfitting.py](https://github.com/pipidog/MLclass/blob/master/codes/05_MLP_MNIST/MLP_mnist_overfitting.py)      
=> similar to MLP_mnist_2.py but include dropout layers to improve overfitting      
[MLP_Keras_mnist.py](https://github.com/pipidog/MLclass/blob/master/codes/05_MLP_MNIST/MLP_Keras_mnist.py)     
=> similar to MLP_mnist_overfitting.py but use Keras        
【week 6】:       
[CNN_mnist_1.py](https://github.com/pipidog/MLclass/blob/master/codes/06_CNN/CNN_mnist_1.py)          
=> this code shows how to construct a CNN to recognize MNIST        
[CNN_mnist_2.py](https://github.com/pipidog/MLclass/blob/master/codes/06_CNN/CNN_mnist_2.py)          
=> similar to CNN_mnist_1.py but use t-SNE reduce dimension from CNN output for visualization       
[CNN_mnist_keras.py](https://github.com/pipidog/MLclass/blob/master/codes/06_CNN/CNN_mnist_keras.py)       
=> similar to CNN_mnist_1.py but use Keras      
【week 7】:       
[RNN_mnist.py](https://github.com/pipidog/MLclass/blob/master/codes/07_RNN/RNN_mnist.py)               
=> this code shows how to use RNN/LSTM to do MNIST      
[RNN_curve_fitting.py](https://github.com/pipidog/MLclass/blob/master/codes/07_RNN/RNN_curve_fitting.py)       
=> this code shows how to use RNN/LSTM to fit time series data     
[IRNN_mnist.py](https://github.com/pipidog/MLclass/blob/master/codes/07_RNN/IRNN_mnist.py)         
=> this code shows how to use IRNN to do MNIST      
【week 8】:
[NLP_preprocessing.py](https://github.com/pipidog/MLclass/blob/master/codes/08_NLP/NLP_preprocessing.py)          
=> this code shows how to download & tokenize text (IMDB) data for later training       
[NLP_MLP_Retuers.py](https://github.com/pipidog/MLclass/blob/master/codes/08_NLP/NLP_MLP_Retuers.py)     
=> this code shows how to use MLP to categorize news from Reuters 46 News dataset
[NLP_IMDB_CNN_RNN.py](https://github.com/pipidog/MLclass/blob/master/codes/08_NLP/NLP_IMDB_CNN_RNN.py)     
=> this code shows how to use CNN+RNN for sentimental prediction on IMDB dataset        
[NLP_IMDB_LSTM.py](https://github.com/pipidog/MLclass/blob/master/codes/08_NLP/NLP_IMDB_LSTM.py)       
=> this code shows how to use LSTM for sentimental prediction on IMDB dataset       
【week 9】:
[Deep_Autoencoder.py](https://github.com/pipidog/MLclass/blob/master/codes/09_Autoencoder/Deep_Autoencoder.py)  
=> this code shows how to use autoencoder for dimension reduction on MNIST
[AE_denosing.py](https://github.com/pipidog/MLclass/blob/master/codes/09_Autoencoder/AE_denosing.py)       
=> this code shows how to use autoencoder for denoising


    
