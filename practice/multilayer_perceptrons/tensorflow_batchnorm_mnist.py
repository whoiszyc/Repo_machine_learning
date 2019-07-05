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

# Other
random_seed = 123

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)

    # Batchnorm settings
    training_phase = tf.placeholder(tf.bool, None, name='training_phase')

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Multilayer perceptron
    layer_1 = tf.layers.dense(tf_x, n_hidden_1,
                              activation=None,  # Batchnorm comes before nonlinear activation
                              use_bias=False,  # Note that no bias unit is used in batchnorm
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    layer_1 = tf.layers.batch_normalization(layer_1, training=training_phase)
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.layers.dense(layer_1, n_hidden_2,
                              activation=None,
                              use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer_2 = tf.layers.batch_normalization(layer_2, training=training_phase)
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.layers.dense(layer_2, n_classes, activation=None, name='logits')

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')

    # control dependency to ensure that batchnorm parameters are also updated
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

import numpy as np

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)  # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y,
                                                            'training_phase:0': True})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels,
                                                      'training_phase:0': False})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels,
                                                      'training_phase:0': False})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.test.images,
                                                 'targets:0': mnist.test.labels,
                                                 'training_phase:0': False})
    print('Test ACC: %.3f' % test_acc)
