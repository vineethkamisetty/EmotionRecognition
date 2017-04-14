
import tflearn
tflearn.reset_default_graph()
import os
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


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

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


# Building Residual Network
net = tflearn.input_data(shape=[None, 48, 48, 1])
net = tflearn.conv_2d(net, 64, 5, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 3072, activation='relu')
net = tflearn.fully_connected(net, len(EMOTIONS), activation='softmax')

net = tflearn.regression(net,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


model = tflearn.DNN(net, tensorboard_verbose=3)

print(X_train.shape)
print(Y_train.shape)

model.fit(
    X_train, Y_train,
    validation_set=(X_valid, Y_valid),
    n_epoch=1,
    batch_size=50,
    shuffle=True,
    show_metric=True,
    snapshot_step=200,
    snapshot_epoch=True,
    run_id='emotion_recognition'
)
print("end")

