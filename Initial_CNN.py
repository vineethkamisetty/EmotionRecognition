import numpy
import pandas
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

print(datetime.now())

images_train = pandas.read_csv('./images_train.csv').values
images_validate = pandas.read_csv('./images_validate.csv').values
images_test = pandas.read_csv('./images_test.csv').values
labels_train = pandas.read_csv('./labels_train.csv').values
labels_validate = pandas.read_csv('./labels_validate.csv').values
labels_test = pandas.read_csv('./labels_test.csv').values

images_train = images_train.astype(numpy.float)
images_validate = images_validate.astype(numpy.float)
images_test = images_test.astype(numpy.float)

images_train = numpy.multiply(images_train, 1.0 / 255.0)
images_validate = numpy.multiply(images_validate, 1.0 / 255.0)
images_test = numpy.multiply(images_test, 1.0 / 255.0)

labels_train = labels_train.ravel()
labels_validate = labels_validate.ravel()
labels_test = labels_test.ravel()

sess = tf.InteractiveSession()


def one_hot(label, labels_unique):
    index_offset = numpy.arange(label.shape[0]) * labels_unique
    one_hot = numpy.zeros((label.shape[0], labels_unique))
    one_hot.flat[index_offset + label.ravel()] = 1
    labelset = one_hot.astype(numpy.uint8)
    return labelset


labels_unique = 6
labels_trainset = one_hot(labels_train, labels_unique)
labels_validateset = one_hot(labels_validate, labels_unique)
labels_testset = one_hot(labels_test, labels_unique)

x = tf.placeholder(tf.float32, shape=[None, 2304])
y_actual = tf.placeholder(tf.float32, shape=[None, 6])

x_image = tf.reshape(x, [-1, 48, 48, 1])


# Initialize the weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution type - 2D
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Max pooling type - 2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  # [batch, pool size (2x2), channels]
                          strides=[1, 2, 2, 1], padding='SAME')


# Building conv,relu,max_pool layers
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop-out
dropout_prob = tf.placeholder(tf.float32)
O_fc_1_dropout = tf.nn.dropout(h_fc1, dropout_prob)

W_fc2 = weight_variable([1024, 6])
b_fc2 = bias_variable([6])

y_predicted = tf.nn.softmax(tf.matmul(O_fc_1_dropout, W_fc2) + b_fc2)

cost = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predicted), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
final_prediction = tf.argmax(y_predicted, 1)

epochs_completed = 0
index_in_epoch = 0
num_examples = images_train.shape[0]


def next_batch(batch_size):
    global images_train
    global labels_trainset
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = numpy.arange(num_examples)
        numpy.random.shuffle(perm)
        images_train = images_train[perm]
        labels_trainset = labels_trainset[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return images_train[start:end], labels_trainset[start:end]


sess.run(tf.global_variables_initializer())
train_accuracies = []
validation_accuracies = []
x_range = []

VALIDATION_SIZE = len(labels_validateset)
for i in range(2000):
    batch_imageset, batch_labelset = next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_imageset, y_actual: batch_labelset, dropout_prob: 1.0})
        if VALIDATION_SIZE:
            validation_accuracy = accuracy.eval(feed_dict={x: images_validate[0:50],
                                                           y_actual: labels_validateset[0:50],
                                                           dropout_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)
        else:
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
    train_step.run(feed_dict={x: batch_imageset, y_actual: batch_labelset, dropout_prob: 0.5})

if VALIDATION_SIZE > 0:
    for i in range(0, images_validate.shape[0] // 100):
        validation_accuracy = accuracy.eval(feed_dict={x: images_validate[i * 100:(i + 1) * 100],
                                                       y_actual: labels_validateset[i * 100:(i + 1) * 100],
                                                       dropout_prob: 1.0}, session=sess)

    plt.plot(x_range, train_accuracies, '-b', label='Training')
    # plt.plot(x_range, validation_accuracies, '-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.0, ymin=0.0)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()

    print('validation_accuracy => %.4f' % validation_accuracy)
    print('training  => %.4f' % train_accuracy)

for i in range(0, images_test.shape[0] // 100):
    test_accuracy = accuracy.eval(feed_dict={x: images_test[i * 100:(i + 1) * 100],
                                             y_actual: labels_testset[i * 100:(i + 1) * 100],
                                             dropout_prob: 1.0}, session=sess)

print("test_accuracy: ", test_accuracy)
print(datetime.now())

sess.close()
