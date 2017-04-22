import sys

import matplotlib
import numpy
import tensorflow
import tflearn
from tflearn import *
from util import getData
import matplotlib.pyplot as plt
import math

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

train = False
test = False
saved = True
layers = []

x = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 48, 48, 1], name="x-in")
network = input_data(placeholder=x, shape=[None, 48, 48, 1])

conv_1 = conv_2d(network, 32, 3, activation='relu')
conv_2 = conv_2d(conv_1, 32, 5, activation='relu')
network = max_pool_2d(conv_2, 2, strides=2)
network = dropout(network, 0.3)
network = local_response_normalization(network)
conv_3 = conv_2d(network, 64, 3, activation='relu')
conv_4 = conv_2d(conv_3, 64, 5, activation='relu')
network = max_pool_2d(conv_4, 2, strides=2)
network = dropout(network, 0.3)
network = local_response_normalization(network)
conv_5 = conv_2d(network, 128, 3, activation='relu')
conv_6 = conv_2d(conv_5, 128, 5, activation='relu')
network = max_pool_2d(conv_6, 2, strides=2)
network = dropout(network, 0.3)
network = local_response_normalization(network)
fc_1 = fully_connected(network, 1024, activation='relu')
network = dropout(fc_1, 0.7)  # need to change if needed 0.7 -> 0.5
fc_2 = fully_connected(network, 1024, activation='relu')
network = dropout(fc_2, 0.7)  # need to change if needed 0.7 -> 0.5
network = fully_connected(network, len(EMOTIONS), activation='softmax')

network = regression(network,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)  # need to check with different learning rate

model = tflearn.DNN(network, tensorboard_verbose=3)

sess = tensorflow.Session()
init = tensorflow.global_variables_initializer()
sess.run(init)

if train:
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()  # need to train on different data sets
    X_train = X_train.transpose((0, 2, 3, 1))
    X_valid = X_valid.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()  # to reset the model graph. problem with loading the weights
    model.fit(
        X_train, Y_train,
        validation_set=(X_valid, Y_valid),
        n_epoch=50,
        batch_size=100,
        shuffle=True,
        show_metric=True,
        snapshot_step=200,
        snapshot_epoch=True,
        run_id='emotion_recognition_A'
    )
    model.save('./SavedModels/model_A.tfl')

if saved:
    savepath = './SavedModels/model_A.tfl'
    model.load(savepath)

if test:
    _, _, _, _, X_test, Y_test = getData()  # need to test on different datasets
    X_test = X_test.transpose((0, 2, 3, 1))
    tensorflow.reset_default_graph()
    score = model.evaluate(X_test, Y_test, batch_size=50)
    print('Test accuarcy: %0.4f%%' % (score[0] * 100))


def getActivations(layer, stimuli):
    units = sess.run(layer, feed_dict={x: stimuli})
    plotNNFilter(units)


def plotNNFilter(units):
    colormaps = True
    n_filters = units.shape[3]
    fig = plt.figure(figsize=(12, 6))
    for i in range(n_filters):
        ax = fig.add_subplot(n_filters / 16, 16, i + 1)
        if colormaps:
            ax.imshow(units[0, :, :, i], cmap='Blues')  # seq_colors[i]
        else:
            ax.imshow(units[0, :, :, i], cmap=matplotlib.cm.gray)  # matplotlib.cm.gray
        plt.xticks(numpy.array([]))
        plt.yticks(numpy.array([]))
        plt.tight_layout()
    plt.show()


X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()  # need to train on different data sets

imageToUse = X_valid[0:1, :, :, :].transpose((0, 2, 3, 1))
print("imagetouse : ", imageToUse.shape)
getActivations(conv_6, imageToUse)
