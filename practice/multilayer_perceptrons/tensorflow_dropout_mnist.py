# Typically, dropout is applied after the non-linear activation function (a). However, when using rectified linear units (ReLUs), it might make sense to apply dropout before the non-linear activation (b) for reasons of computational efficiency depending on the particular code implementation.
#
# (a): Fully connected, linear activation -> ReLU -> Dropout -> ...
# (b): Fully connected, linear activation -> Dropout -> ReLU -> ...
#
# Why do (a) and (b) produce the same results in case of ReLU?. Let's answer this question with a simple example starting with the following logits (outputs of the linear activation of the fully connected layer):
#
# [-1, -2, -3, 4, 5, 6]
#
# Let's walk through scenario (a), applying the ReLU activation first. The output of the non-linear ReLU functions are as follows:
#
# [0, 0, 0, 4, 5, 6]
#
# Remember, the ReLU activation function is defined as  f(x)=max(0,x)f(x)=max(0,x) ; thus, all non-zero values will be changed to zeros. Now, applying dropout with a probability 0f 50%, let's assume that the units being deactivated are units 2, 4, and 6:
#
# [0*2, 0, 0*2, 0, 0*2, 0] = [0, 0, 0, 0, 10, 0]
#
# Note that in dropout, units are deactivated randomly by default. In the preceding example, we assumed that the 2nd, 4th, and 6th unit were deactivated during the training iteration. Also, because we applied dropout with 50% dropout probability, we scaled the remaining units by a factor of 2.
#
# Now, let's take a look at scenario (b). Again, we assume a 50% dropout rate and that units 2, 4, and 6 are deactivated:
#
# [-1, -2, -3, 4, 5, 6] ->  [-1*2, 0, -3*2, 0, 5*2, 0]
#
# Now, if we pass this array to the ReLU function, the resulting array will look exactly like the one in scenario (a):
#
# [-2, 0, -6, 0, 10, 0] -> [0, 0, 0, 0, 10, 0]




#################################### Low-level Implementation ################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 64
dropout_keep_proba = 0.5

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_input = 784
n_classes = 10

# Other
random_seed = 123

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)

    # Dropout settings
    keep_proba = tf.placeholder(tf.float32, None, name='keep_proba')

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Multilayer perceptron
    layer_1 = tf.add(tf.matmul(tf_x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_proba)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_proba)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='logits')

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

from numpy.random import seed

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    seed(random_seed)  # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y,
                                                            'keep_proba:0': dropout_keep_proba})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels,
                                                      'keep_proba:0': 1.0})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels,
                                                      'keep_proba:0': 1.0})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={'features:0': mnist.test.images,
                                             'targets:0': mnist.test.labels,
                                             'keep_proba:0': 1.0})
    print('Test ACC: %.3f' % test_acc)





#################################### tensorflow.layers Abstraction ################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 64
dropout_rate = 0.5
# note that we define the dropout rate, not
# the "keep probability" when using
# dropout from tf.layers

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_input = 784
training_epochs = 15

# Other
random_seed = 123

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)

    # Dropout settings
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Multilayer perceptron
    layer_1 = tf.layers.dense(tf_x, n_hidden_1, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer_1 = tf.layers.dropout(layer_1, rate=dropout_rate, training=is_training)

    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer_2 = tf.layers.dropout(layer_1, rate=dropout_rate, training=is_training)

    out_layer = tf.layers.dense(layer_2, n_classes, activation=None, name='logits')

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

from numpy.random import seed

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    seed(random_seed)  # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y,
                                                            'is_training:0': True})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels,
                                                      'is_training:0': False})

        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels,
                                                      'is_training:0': False})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.test.images,
                                                 'targets:0': mnist.test.labels,
                                                 'is_training:0': False})
    print('Test ACC: %.3f' % test_acc)