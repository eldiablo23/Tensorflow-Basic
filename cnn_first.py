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


train_x = []
train_y = []
    
for file in ('CIFAR 10/data_batch_1', 'CIFAR 10/data_batch_2', 'CIFAR 10/data_batch_3', 'CIFAR 10/data_batch_4', 'CIFAR 10/data_batch_5'):
    t = sio.loadmat('CIFAR 10/data_batch_1')
    a = t['data']
    a = np.reshape(a,[a.shape[0],3,32,32])
    a = np.transpose(a, (0,2,3,1))
    train_x.append(a)
    a = t['labels']
    t = np.zeros((a.shape[0],10))
    t[np.arange(a.shape[0]),a.T] = 1
    train_y.append(t)

train_x = np.array(train_x) / 255
train_y = np.array(train_y)

t = sio.loadmat('CIFAR 10/test_batch')
a = t['data']
a = np.reshape(a,[a.shape[0],3,32,32])
a = np.transpose(a, (0,2,3,1))
test_x = np.array(a)
a = t['labels']
t = np.zeros((a.shape[0],10))
t[np.arange(a.shape[0]),a.T] = 1
test_y = np.array(t)

print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)


#######################
#######################


epochs = 10
batch_size = 10000
mini_batch_size = 128
num_batches = 5
step_size = 0.001
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

W = {'Wc' : tf.Variable(0.01 * np.random.randn(recep_size, recep_size, input_depth, conv_1_depth).astype(np.float32)),
     'Wfc' : tf.Variable(0.01 * np.random.randn(16*16*12, fc_hidden).astype(np.float32)),
     'Wout' : tf.Variable(0.01 * np.random.randn(fc_hidden, output_size).astype(np.float32))
    }
b = {'bc' : tf.Variable(np.zeros((conv_1_depth)).astype(np.float32)),
     'bfc' : tf.Variable(np.zeros((1,fc_hidden)).astype(np.float32)),
     'bout' : tf.Variable(np.zeros((1,output_size)).astype(np.float32))
    }
    
# x_pad = tf.pad(x,tf.constant([[0,0],[inp_pad,inp_pad],[inp_pad,inp_pad],[0,0]]))
conv_1 = tf.nn.conv2d(x, W['Wc'], strides=[1,stride,stride,1], padding='SAME')
conv_1 = tf.nn.bias_add(conv_1, b['bc'])

relu = tf.nn.relu(conv_1)

pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

fc = tf.contrib.layers.flatten(pool)

hidden_layer = tf.add(tf.matmul(fc, W['Wfc']), b['bfc'])
hidden_layer = tf.nn.relu(hidden_layer)

hidden_layer = tf.nn.dropout(hidden_layer, dropout)

scores = tf.add(tf.matmul(hidden_layer, W['Wout']), b['bout'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y))

optimizer = tf.train.AdamOptimizer().minimize(cost)

prediction = tf.nn.softmax(scores)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_pred, tf.float32)),100)

init = tf.global_variables_initializer()


#######################
#######################


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for batch in range(num_batches):
            avg_cost = 0
            mask = np.arange(train_x.shape[1])
            np.random.shuffle(mask)
            num_mini_bs = int(math.ceil(train_x.shape[1]/float(mini_batch_size)))
            for i in range(num_mini_bs):
                if (i==num_mini_bs-1):
                    mini_batch_x = train_x[batch,mask[i*mini_batch_size:],:,:,:]
                    mini_batch_y = train_y[batch,mask[i*mini_batch_size:],:]
                else:
                    mini_batch_x = train_x[batch,mask[i*mini_batch_size:(i+1)*mini_batch_size],:,:,:]
                    mini_batch_y = train_y[batch,mask[i*mini_batch_size:(i+1)*mini_batch_size],:]
                _, c = sess.run([optimizer, cost], feed_dict = {x: mini_batch_x, y: mini_batch_y})
                avg_cost += c/num_mini_bs
            print ("Epoch:", (epoch+1), "Batch:", (batch+1), "cost =", "{:.5f}".format(avg_cost))
    print ("\nTraining complete!")
    
    w = sess.run(W)
    b = sess.run(b)
    sio.savemat('w.mat',w)
    sio.savemat('b.mat',b)


#######################
#######################


with tf.Session() as sess:
    sess.run(init)
    acc = sess.run(accuracy, feed_dict = {x: test_x, y: test_y})
    print ("Accuracy =", "{:.3f}".format(acc)+"%")


#######################
#######################


winsound.Beep(2600, 997)