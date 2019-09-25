
import tensorflow as tf

input = tf.placeholder(tf.float32, shape=[None, 10])
# label = tf.placeholder(tf.float32, shape=[None, 12])

# Define hidden layer
W1 = tf.Variable(tf.zeros([10, 18]), tf.float32)
B1 = tf.Variable(tf.zeros([18]), tf.float32)

## ! note that for tensor operation, x*y is not the multiplication
h_1 = tf.nn.relu(tf.tensordot(input, W1, axes=1))

# Define output layer
# W2 = tf.Variable(tf.truncated_normal([32, 12], stddev=0.1), tf.float32)
# B2 = tf.Variable(tf.truncated_normal([1, 12], stddev=0.1), tf.float32)
# y_pred = h_1 * W2

# Run the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
h_1_res, W1_res = sess.run([h_1, W1], feed_dict={input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]})
print(h_1_res)
print(W1_res)
sess.close()