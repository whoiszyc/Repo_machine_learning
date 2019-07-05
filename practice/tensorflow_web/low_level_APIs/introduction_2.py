from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf




# A placeholder is a promise to provide a value later, like a function argument.
# The placeholder is used to define the object of input and output.

# Normally, the first entry represents sample numbers, while the second represents number of neurons.
x = tf.placeholder(tf.float32, shape=[None, 3])

# Without specifying the dimension, the tensor is flexible
y = tf.placeholder(tf.float32, shape=[None])

# Build graph between tensors
z = x + y

# using the feed_dict argument
X_intput = np.array([[1, 2, 3],[3, 4, 5],[3, 4, 5]])
sess = tf.Session()
print(sess.run(z, feed_dict={x: X_intput, y: [3, 4, 5]}))
sess.close()
