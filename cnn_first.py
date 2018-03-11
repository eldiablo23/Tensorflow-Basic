print ('Setting up environment...')

import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
import scipy.io as sio
import csv
import winsound

print ('Done!\n...\n...')

train_x = []
train_y = []
    
for file in ('CIFAR 10/data_batch_1', 'CIFAR 10/data_batch_2', 'CIFAR 10/data_batch_3', 'CIFAR 10/data_batch_4', 'CIFAR 10/data_batch_5'):
    t = sio.loadmat(file)
    train_x.append(t['data'])
    train_y.append(t['labels'])

train_x = np.array(train_x) / 255
train_y = np.array(train_y)
train_x = np.reshape(train_x,(-1,3072))
train_y = np.reshape(train_y,(-1,1))

t = sio.loadmat('CIFAR 10/test_batch')
test_x = t['data']
test_y = t['labels']

print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)

epochs = 20
batch_size = 128
num_batches = 200
step_size = 0.0002
recep_size = 3
conv_1_depth = 12
fc_hidden = 200
input_size = 1024
input_depth = 3
output_size = 10
inp_pad = 1
stride = 1
dropout = 0.75

x = tf.placeholder(tf.float32, [None, 32,32,input_depth])
y = tf.placeholder(tf.float32, [None, output_size])

W = {'Wc' : tf.Variable(0.01 * np.random.randn(recep_size, recep_size, input_depth, conv_1_depth).astype(np.float32)),
     'Wfc' : tf.Variable(0.01 * np.random.randn(16*16*12, fc_hidden).astype(np.float32)),
     'Wout' : tf.Variable(0.01 * np.random.randn(fc_hidden, output_size).astype(np.float32))
    }
b = {'bc' : tf.Variable(np.zeros((conv_1_depth)).astype(np.float32)),
     'bfc' : tf.Variable(np.zeros((1,fc_hidden)).astype(np.float32)),
     'bout' : tf.Variable(np.zeros((1,output_size)).astype(np.float32))
    }
    
x_pad = tf.pad(x,tf.constant([[0,0],[inp_pad,inp_pad],[inp_pad,inp_pad],[0,0]]))
conv_1 = tf.nn.conv2d(x_pad,W['Wc'],strides=[1,stride,stride,1],padding='VALID')
conv_1 = tf.nn.bias_add(conv_1, b['bc'])

relu = tf.nn.relu(conv_1)

pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

fc = tf.reshape(pool, [-1, 16*16*12])

hidden_layer = tf.add(tf.matmul(fc, W['Wfc']), b['bfc'])
hidden_layer = tf.nn.relu(hidden_layer)

scores = tf.add(tf.matmul(hidden_layer, W['Wout']), b['bout'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=step_size).minimize(cost)

init = tf.global_variables_initializer()


#######################
#######################


with tf.Session() as sess:
    sess.run(init)
    # n = int(train_x.shape[0]/batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(1):
            mask = np.random.choice(train_x.shape[0], batch_size, replace=False)
            batch_x = train_x[mask,:]
            batch_x = np.reshape(batch_x,[batch_size,3,32,32])
            batch_x = np.transpose(batch_x, (0,2,3,1))
            batch_y = train_y[mask]
            t = np.zeros((batch_y.shape[0],output_size))
            t[range(batch_y.shape[0]),batch_y] = 1
            batch_y = t
            
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c / num_batches
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    print ("\nTraining complete!")
    
    w = sess.run(W)
    b = sess.run(b)
    sio.savemat('w.mat',w)
    sio.savemat('b.mat',b)


#######################
#######################
    
# with tf.Session() as sess:
    # sess.run(init)
    # n = int(train_x.shape[0]/batch_size)
    # test_x = np.reshape(test_x,[test_x.shape[0],32,32,3])
    # t = np.zeros((test_y.shape[0],output_size))
    # t[range(test_y.shape[0]),test_y] = 1
    # test_y = t
    
    # cost = sess.run(cost, feed_dict = {x: test_x, y: test_y})
            # avg_cost += c / n
        # print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))


winsound.Beep(2600, 997)