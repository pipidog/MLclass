import numpy as np
from numba import jit 
import tensorflow as tf
import time
import sys

from threading import Thread
from queue import Queue

# parameters  ===================
tot_mat = 500        
mat_size = (2000,2000)
tot_thread=6

# generator  ====================
np.random.seed(1)

# numpy  =========================
print('Using Numpy')
time_start = time.time()
for n in range(tot_mat+1):
    A_mat = np.random.rand(mat_size[0],mat_size[1])
    B_mat = np.random.rand(mat_size[0],mat_size[1])
    np.dot(A_mat,B_mat)
    if n % 50 == 0:
        print(n)
time_end = time.time()
time_elapse = time_end-time_start
print(time_elapse)

# # tensorflow  ===================
print('Using TensorFlow')
A = tf.placeholder(tf.float32, mat_size)
B = tf.placeholder(tf.float32, mat_size)
matmul = tf.matmul(A,B)
    
time_start = time.time()
sess = tf.Session()
for n in range(tot_mat+1): 
    A_mat = np.random.rand(mat_size[0],mat_size[1])
    B_mat = np.random.rand(mat_size[0],mat_size[1])
    sess.run(matmul,feed_dict = {A:A_mat,B:B_mat}) 
    if n % 50 == 0:
        print(n)
time_end = time.time()
time_elapse = time_end-time_start
print(time_elapse)

# tensorflow + threading =============
print('Using TensorFlow + Multithreads')

# reshape steps to thread_batch set based on tot_thread
div=divmod(tot_mat,tot_thread)
thread_batch=np.linspace(1,tot_mat-div[1],tot_mat-div[1])
thread_batch=thread_batch.reshape(-1,tot_thread).astype(np.int).tolist()
if div[1] != 0:
    thread_batch.append(list(range(tot_mat-div[1]+1,tot_mat+1)))

# define job for multithreading
# Note, computation graph must be redefined for each thread!
def matmul():
    A = tf.placeholder(tf.float32, mat_size)
    B = tf.placeholder(tf.float32, mat_size)
    matmul = tf.matmul(A,B)
    A_mat = np.random.rand(mat_size[0],mat_size[1])
    B_mat = np.random.rand(mat_size[0],mat_size[1])
    sess = tf.Session()
    sess.run(matmul,feed_dict = {A:A_mat,B:B_mat}) 

# begin multithreading
time_start = time.time()
for batch in thread_batch:
    print(batch)
    main_thread=[]
    for n in range(len(batch)):
        t = Thread(target=matmul)
        t.start()
        main_thread.append(t)
    for thread in main_thread:
        thread.join()
    #print(batch)
time_end = time.time()
time_elapse = time_end-time_start
print(time_elapse)

