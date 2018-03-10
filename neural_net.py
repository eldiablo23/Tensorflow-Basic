print ('Setting up environment...')

import numpy as np
import tensorflow as tf
from scipy.misc import imread
import scipy.io as sio
import csv
import winsound

print ('Done!\n...\n...')

train_x = []
train_y = []
test_x = []
test_y = []

with open('Mnist Data/mnist_train.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for entry in reader:
        train_x.append(list(map(int,entry[1:])))
        train_y.append(int(entry[0]))

with open('Mnist Data/mnist_test.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for entry in reader:
        test_x.append(list(map(int,entry[1:])))
        test_y.append(int(entry[0]))
        
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)

input_layer_size = 28*28
hidden_layer_size = 100
output_layer_size = 10

epochs = 20
batch_size = 128
step_size = 0.00009

x = tf.placeholder(tf.float32, [batch_size, input_layer_size])
y = tf.placeholder(tf.float32, [batch_size, output_layer_size])
W = {'W1' : tf.Variable(0.01 * np.random.randn(input_layer_size, hidden_layer_size).astype(np.float32)),
     'W2' : tf.Variable(0.01 * np.random.randn(hidden_layer_size, output_layer_size).astype(np.float32))
    }
b = {'b1' : tf.Variable(np.zeros((1,hidden_layer_size)).astype(np.float32)),
     'b2' : tf.Variable(np.zeros((1,output_layer_size)).astype(np.float32))
    }
    
hidden_layer = tf.add(tf.matmul(x, W['W1']), b['b1'])
hidden_layer = tf.nn.relu(hidden_layer)
scores = tf.add(tf.matmul(hidden_layer, W['W2']), b['b2'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=step_size).minimize(cost)

init = tf.global_variables_initializer()

#######################
#######################

with tf.Session() as sess:
    sess.run(init)
    n = int(train_x.shape[0]/batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(n):
            mask = np.random.choice(train_x.shape[0], batch_size, replace=False)
            batch_x = train_x[mask,:]
            batch_y = train_y[mask]
            
            t = np.zeros((batch_y.shape[0],output_layer_size))
            t[range(batch_y.shape[0]),batch_y] = 1
            batch_y = t
            batch_x = batch_x.reshape(-1,input_layer_size)
            # batch_x = tf.subtract(batch_x,tf.reduce_mean(batch,axis=1))
            # batch_x = batch_x / batch_x.max()
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c / n
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    print ("\nTraining complete!")
    
    w = sess.run(W)
    b = sess.run(b)
    sio.savemat('w.mat',w)
    sio.savemat('b.mat',b)

#######################
#######################
    
s1 = np.dot(test_x,w['W1']) + b['b1']
relu = np.maximum(0, s1)
s2 = np.dot(relu,w['W2']) + b['b2']
prediction = np.argmax(s2, axis=1)
prediction = prediction.astype(np.uint8)

equals = np.equal(prediction, test_y)
correct = np.sum(equals)
total = test_x.shape[0]
acc = (float(correct)/total)*100
print ('Accurancy achieved using Two-Layer Neural Network is '+str(acc))

    
winsound.Beep(2600, 997)