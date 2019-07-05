
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
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_input = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
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
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
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
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_input = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Multilayer perceptron
    layer_1 = tf.layers.dense(tf_x, n_hidden_1, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    out_layer = tf.layers.dense(layer_2, n_classes, activation=None)

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
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

    test_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.test.images,
                                                 'targets:0': mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)