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
learning_rate = 0.5
training_epochs = 30
batch_size = 256

# Architecture
n_features = 784
n_classes = 10

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_features])
    tf_y = tf.placeholder(tf.float32, [None, n_classes])

    # Model parameters
    params = {
        'weights': tf.Variable(tf.zeros(shape=[n_features, n_classes],
                                        dtype=tf.float32), name='weights'),
        'bias': tf.Variable([[n_classes]], dtype=tf.float32, name='bias')}

    # Softmax regression
    linear = tf.matmul(tf_x, params['weights']) + params['bias']
    pred_proba = tf.nn.softmax(linear, name='predict_probas')

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=linear, labels=tf_y), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Class prediction
    pred_labels = tf.argmax(pred_proba, 1, name='predict_labels')
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), pred_labels)
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
            _, c = sess.run(['train', 'cost:0'], feed_dict={tf_x: batch_x,
                                                            tf_y: batch_y})
            avg_cost += c

        train_acc = sess.run('accuracy:0', feed_dict={tf_x: mnist.train.images,
                                                      tf_y: mnist.train.labels})
        valid_acc = sess.run('accuracy:0', feed_dict={tf_x: mnist.validation.images,
                                                      tf_y: mnist.validation.labels})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={tf_x: mnist.test.images,
                                             tf_y: mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)


