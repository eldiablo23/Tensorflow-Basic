print ('Setting up environment...')

import math
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
import scipy.io as sio
import csv
import winsound

print ('Done!\n...\n...')


#######################
#######################


t = sio.loadmat('CIFAR 10/test_batch')
a = t['data']
a = np.reshape(a,[a.shape[0],3,32,32])
a = np.transpose(a, (0,2,3,1))
test_x = np.array(a)
a = t['labels']
t = np.zeros((a.shape[0],10))
t[np.arange(a.shape[0]),a.T] = 1
test_y = np.array(t)

W = sio.loadmat('W')
b = sio.loadmat('b')

print (test_x.shape, test_y.shape)


#######################
#######################


recep_size = 3
conv_1_depth = 12
fc_hidden = 500
input_size = 1024
input_depth = 3
output_size = 10
inp_pad = 1
stride = 1
dropout = 0.75

x = tf.placeholder(tf.float32, shape=(None, 32,32,input_depth))
y = tf.placeholder(tf.float32, shape=(None, output_size))
    
# x_pad = tf.pad(x,tf.constant([[0,0],[inp_pad,inp_pad],[inp_pad,inp_pad],[0,0]]))
conv_1 = tf.nn.conv2d(x, W['Wc'], strides=[1,stride,stride,1], padding='SAME')
conv_1 = tf.nn.bias_add(conv_1, b['bc'][0,:])

relu = tf.nn.relu(conv_1)

pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

fc = tf.contrib.layers.flatten(pool)

hidden_layer = tf.add(tf.matmul(fc, W['Wfc']), b['bfc'])
hidden_layer = tf.nn.relu(hidden_layer)

hidden_layer = tf.nn.dropout(hidden_layer, dropout)

scores = tf.add(tf.matmul(hidden_layer, W['Wout']), b['bout'])

prediction = tf.nn.softmax(scores)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_pred, tf.float32)),100)

init = tf.global_variables_initializer()


#######################
#######################


with tf.Session() as sess:
    sess.run(init)
    acc = sess.run(accuracy, feed_dict = {x: test_x, y: test_y})
    print ("Accuracy =", "{:.3f}".format(acc)+"%")
    
winsound.Beep(2600, 997)