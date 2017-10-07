'''
<summary>:
this code show how to save and restore the trained results.

<procedure>: 
1. generte tf variables
2. save it
3. reconstruct tf computation graph
4. load it. 

<Note>:

'''

import tensorflow as tf
import numpy as np

# parameter =========================
task='save'  # 'save' or 'restore'

# Main ==============================
if task is 'save':
    # Save data ---------------------
    # must define dtype to avoid type issues
    W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
       sess.run(init)
       save_path = saver.save(sess, "save/tf_save.ckpt")
       print("Save to path: ", save_path)
       
elif task is 'restore':
    # Restore data ------------------
    # must predefine the same stucture to hold the restored data
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # don't initialize variables, their values will be loaded.
        saver.restore(sess, "save/tf_save.ckpt")
        print("weights:", sess.run(W))
        print("biases:", sess.run(b))