#  -*- coding: utf-8 -*-


import urllib.request
import os
import tarfile
import sys
import numpy as np
import re
import h5py 
import pickle
from tqdm import tqdm

np.random.seed(1) # make resule reproducible
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#parameter =============
show_comment=[0,24000] # 0~24999
num_words=2000
# main =================
# download data ----------------------
# from Stanford AI lab (only first time run)
url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

if not os.path.exists('IMDB'):
    os.makedirs('IMDB')

fname='IMDB/aclImdb_v1.tar.gz'
if not os.path.isfile(fname):
    print('downloading IMDB dataset')
    result=urllib.request.urlretrieve(url,fname)
    print('download completed:',os.getcwd()+fname)
    
if not os.path.exists('IMDB/aclImdb'):
    print('unzip downloaded file')
    tfile = tarfile.open(fname, 'r:gz')
    result=tfile.extractall('IMDB/')
    print('unzip completed:',os.getcwd()+fname)

# reading data ------------------------
# function to remove html symbols
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

# function to read text in files
def read_files(filetype):
    path = "IMDB/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
    
    all_texts  = []
    for fi in file_list:
        with open(fi, 'r', encoding='utf-8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_texts, all_labels

# read files 
if not os.path.exists('IMDB/y_test.pkl'):   
    print('reading IMDB data ...')
    x_train, y_train,=read_files("train")
    x_test, y_test,=read_files("test")
    pickle.dump(x_train,open('IMDB/x_train.pkl','wb'))
    pickle.dump(y_train,open('IMDB/y_train.pkl','wb'))
    pickle.dump(x_test,open('IMDB/x_test.pkl','wb'))
    pickle.dump(y_test,open('IMDB/y_test.pkl','wb'))
    
# preprocessing for embedding ------------------
print('loading data ...')
x_train=pickle.load(open('IMDB/x_train.pkl','rb'))
y_train=pickle.load(open('IMDB/y_train.pkl','rb'))
x_test=pickle.load(open('IMDB/x_test.pkl','rb'))
y_test=pickle.load(open('IMDB/y_test.pkl','rb'))
for n in show_comment:
    print('** comment x_train[{0}]'.format(n))
    print(x_train[n])
    print('** label y_train[{0}]'.format(n))
    print(y_train[n])


# build dictionary
token = Tokenizer(num_words=num_words)
token.fit_on_texts(x_train)
print('** show token.document_count:')
print(token.document_count)
print('** show token.word_index:')
print(token.word_index)

# convert article to numbers
x_train_seq = token.texts_to_sequences(x_train)
x_test_seq  = token.texts_to_sequences(x_test)
for n in show_comment:
    print(x_train_seq[n])