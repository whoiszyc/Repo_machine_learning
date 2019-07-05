from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Here we train a model: y = A * x + b

# Define input and output data
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# Define a linear model
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)


# The model hasn't yet been trained, so the four "predicted" values aren't very good.
# Here's what we got; your own output will almost certainly differ:
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('Evaluate the initial output')
print(sess.run(y_pred))
print()

# Loss
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print('Evaluate the initial error')
print(sess.run(loss))
print()

# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Perform training for 100 steps
for i in range(2000):
  _, loss_value = sess.run((train, loss))
  print('Traning step {} with loss {}'.format(i, loss_value))

# Evaluate final output
print()
print('Evaluate the final output')
print(sess.run(y_pred))


# get trained network
a = tf.trainable_variables()