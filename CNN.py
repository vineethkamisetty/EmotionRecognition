import tflearn
import os
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

from util import getImageData

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

if (os.path.exists('train_data.npy') is False) and (os.path.exists('train_label.npy') is False) and (os.path.exists(
        'valid_data.npy') is False) and (os.path.exists('valid_label.npy') is False) and (os.path.exists(
        'test_data.npy') is False) and (os.path.exists('test_label.npy') is False):
    print("File not found!!")
    getImageData()

X_train = np.load('train_data.npy')
Y_train = np.load('train_label.npy')
X_valid = np.load('valid_data.npy')
Y_valid = np.load('valid_label.npy')
X_test = np.load('test_data.npy')
Y_test = np.load('test_label.npy')

# reshape X for tf: N x w x h x c
X_train = X_train.transpose((0, 2, 3, 1))
X_valid = X_valid.transpose((0, 2, 3, 1))
X_test = X_test.transpose((0, 2, 3, 1))

print("X_train : ", X_train.shape)
print("Y_train : ", Y_train.shape)

network = input_data(shape=[None, 48, 48, 1])
network = conv_2d(network, 64, 5, activation='relu')
network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 128, 4, activation='relu')
network = dropout(network, 0.3)
network = fully_connected(network, 3072, activation='relu')
network = fully_connected(network, len(EMOTIONS), activation='softmax')

network = regression(network,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=3)

model.fit(
    X_train, Y_train,
    validation_set=(X_valid, Y_valid),
    n_epoch=5,
    batch_size=50,
    shuffle=True,
    show_metric=True,
    snapshot_step=200,
    snapshot_epoch=True,
    run_id='emotion_recognition'
)
