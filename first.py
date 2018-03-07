from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3, dtype=tf.float32)
b = tf.constant(4, dtype=a.dtype)
c = a + b
print (a)
print (b)
print (c)

# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run(c))