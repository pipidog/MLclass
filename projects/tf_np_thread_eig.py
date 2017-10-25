import numpy as np
from numba import jit 
import tensorflow as tf
import time
import sys
import numpy.linalg as la

from threading import Thread
from queue import Queue

# parameters  ===================
tot_mat = 100        
mat_size = (1000,1000)
tot_thread=10
method='tf+thread'  # 'np', 'tf', 'tf+thread'

# generator  ====================
np.random.seed(1)

# generate random matrix
C=np.zeros((mat_size[0],mat_size[1],tot_mat+1))+0j
for n in range(tot_mat+1):
    A = np.random.rand(mat_size[0],mat_size[1])
    B = np.random.rand(mat_size[0],mat_size[1])
    C[:,:,n]=A+B*1j
    C[:,:,n]=C[:,:,n]+C[:,:,n].conj().T

# set multithreading batchs
div=divmod(tot_mat,tot_thread)
thread_batch=np.linspace(1,tot_mat-div[1],tot_mat-div[1])
thread_batch=thread_batch.reshape(-1,tot_thread).astype(np.int).tolist()
if div[1] != 0:
    thread_batch.append(list(range(tot_mat-div[1]+1,tot_mat+1)))

# define tf flowchart
def tf_eig(inp_feed):
    inp_mat = tf.placeholder(tf.float32, mat_size)
    eig = tf.self_adjoint_eigvals(inp_mat)
    sess = tf.Session()
    D=sess.run(eig,feed_dict = {inp_mat:inp_feed})

# run main jobs ========================    
time_start = time.time()    
if method=='np':
    for n in range(tot_mat+1):
        print(n)
        D=la.eig(C[:,:,n])
elif method=='tf':
    for n in range(tot_mat+1):
        D=tf_eig(C[:,:,n])
elif method=='tf+thread':
    for batch in thread_batch:
        main_thread=[]
        for n in batch:
            t = Thread(target=tf_eig,args=([C[:,:,n]]))
            t.start()
            main_thread.append(t)
        for thread in main_thread:
            thread.join()
time_end = time.time()
time_elapse = time_end-time_start
print(time_elapse)   


