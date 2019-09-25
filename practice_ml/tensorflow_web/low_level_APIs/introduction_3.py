from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf




# Define input
x = tf.placeholder(tf.float32, shape=[None, 3])

# Define a dense layer like x * W + b
linear_model = tf.layers.Dense(units=5)
y = linear_model(x)

# Run the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
print()
sess.close()



# Layer Function shortcuts
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print()
sess.close()


# Using linear agebra
x = tf.placeholder(tf.float32, shape=[None, 3])

# Define a dense layer like x * W + b
linear_model = tf.layers.Dense(units=5)
y = linear_model(x)

# Run the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
print()
sess.close()

