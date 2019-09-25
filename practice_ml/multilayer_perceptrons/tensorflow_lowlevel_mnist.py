# Model Zoo -- Multilayer Perceptron with Backpropagation from Scratch
# This notebook contains three different approaches for training a simple 1-hidden layer multilayer perceptron using TensorFlow:
# (1) Gradient descent via the "high-level" tf.train.GradientDescentOptimizer
# (2) A lower-level implementation to perform backpropagation via tf.gradients
# (3) An implementation of backpropagation and gradient descent learning based on basic linear algebra operations

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


#################################### # 1. Gradient Descent Using tf.train.GradientDescentOptimizer ################################
np.random.seed(123) # set seed for mnist shuffling
mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_input = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Forward Propagation
    h1_z = tf.add(tf.matmul(tf_x, weights['h1']), biases['b1'])
    h1_act = tf.nn.sigmoid(h1_z)
    out_z = tf.matmul(h1_act, weights['out']) + biases['out']
    out_act = tf.nn.softmax(out_z, name='predicted_probabilities')
    out_labels = tf.argmax(out_z, axis=1, name='predicted_labels')

    ######################
    # Forward Propagation
    ######################

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_z, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')

    ##################
    # Backpropagation
    ##################

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    ##############
    # Prediction
    ##############

    correct_prediction = tf.equal(tf.argmax(tf_y, 1), out_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={'features:0': mnist.test.images,
                                             'targets:0': mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)





#################################### 2. Gradient Descent Using tf.gradients (low level) ################################
np.random.seed(123) # set seed for mnist shuffling
mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_input = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    ######################
    # Forward Propagation
    ######################

    h1_z = tf.add(tf.matmul(tf_x, weights['h1']), biases['b1'])
    h1_act = tf.nn.sigmoid(h1_z)
    out_z = tf.matmul(h1_act, weights['out']) + biases['out']
    out_act = tf.nn.softmax(out_z, name='predicted_probabilities')
    out_labels = tf.argmax(out_z, axis=1, name='predicted_labels')

    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_z, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')

    ##################
    # Backpropagation
    ##################

    # Get Gradients
    dc_dw_out, dc_db_out = tf.gradients(cost, [weights['out'], biases['out']])
    dc_dw_1, dc_db_1 = tf.gradients(cost, [weights['h1'], biases['b1']])

    # Update Weights
    upd_w_1 = tf.assign(weights['h1'], weights['h1'] - learning_rate * dc_dw_1)
    upd_b_1 = tf.assign(biases['b1'], biases['b1'] - learning_rate * dc_db_1)
    upd_w_out = tf.assign(weights['out'], weights['out'] - learning_rate * dc_dw_out)
    upd_b_out = tf.assign(biases['out'], biases['out'] - learning_rate * dc_db_out)

    train = tf.group(upd_w_1, upd_b_1, upd_w_out, upd_b_out, name='train')

    ##############
    # Prediction
    ##############

    correct_prediction = tf.equal(tf.argmax(tf_y, 1), out_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={'features:0': mnist.test.images,
                                             'targets:0': mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)



#################################### 3. Gradient Descent from scratch (very low level) ################################

# Dataset
np.random.seed(123) # set seed for mnist shuffling
mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_input = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    ######################
    # Forward Propagation
    ######################

    h1_z = tf.add(tf.matmul(tf_x, weights['h1']), biases['b1'])
    h1_act = tf.nn.sigmoid(h1_z)
    out_z = tf.matmul(h1_act, weights['out']) + biases['out']
    out_act = tf.nn.softmax(out_z, name='predicted_probabilities')
    out_labels = tf.argmax(out_z, axis=1, name='predicted_labels')

    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_z, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')

    ##################
    # Backpropagation
    ##################

    # Get Gradients

    # input/output dim: [n_samples, n_classlabels]
    sigma_out = (out_act - tf_y) / batch_size

    # input/output dim: [n_samples, n_hidden_1]
    softmax_derivative_h1 = h1_act * (1. - h1_act)

    # input dim: [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
    # output dim: [n_samples, n_hidden]
    sigma_h = (tf.matmul(sigma_out, tf.transpose(weights['out'])) *
               softmax_derivative_h1)

    # input dim: [n_features, n_samples] dot [n_samples, n_hidden]
    # output dim: [n_features, n_hidden]
    grad_w_h1 = tf.matmul(tf.transpose(tf_x), sigma_h)
    grad_b_h1 = tf.reduce_sum(sigma_h, axis=0)

    # input dim: [n_hidden, n_samples] dot [n_samples, n_classlabels]
    # output dim: [n_hidden, n_classlabels]
    grad_w_out = tf.matmul(tf.transpose(h1_act), sigma_out)
    grad_b_out = tf.reduce_sum(sigma_out, axis=0)

    # Update weights
    upd_w_1 = tf.assign(weights['h1'], weights['h1'] - learning_rate * grad_w_h1)
    upd_b_1 = tf.assign(biases['b1'], biases['b1'] - learning_rate * grad_b_h1)
    upd_w_out = tf.assign(weights['out'], weights['out'] - learning_rate * grad_w_out)
    upd_b_out = tf.assign(biases['out'], biases['out'] - learning_rate * grad_b_out)

    train = tf.group(upd_w_1, upd_b_1, upd_w_out, upd_b_out, name='train')

    ##############
    # Prediction
    ##############

    correct_prediction = tf.equal(tf.argmax(tf_y, 1), out_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={'features:0': mnist.test.images,
                                             'targets:0': mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)





