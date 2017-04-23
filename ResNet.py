import tflearn
import tensorflow
import numpy as np

tensorflow.reset_default_graph()
X_train = np.load('./FercData/train_data.npy')
Y_train = np.load('./FercData/train_label.npy')
X_valid = np.load('./FercData/valid_data.npy')
Y_valid = np.load('./FercData/valid_label.npy')
X_test = np.load('./FercData/test_data.npy')
Y_test = np.load('./FercData/test_label.npy')

# reshape X for tf: N x w x h x c
X_train = X_train.transpose((0, 2, 3, 1))
X_valid = X_valid.transpose((0, 2, 3, 1))
X_test = X_test.transpose((0, 2, 3, 1))

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

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

model.load('./SavedModels/model_resnet.tfl')
model.fit(
    X_train, Y_train,
    validation_set=(X_valid, Y_valid),
    n_epoch=20,
    batch_size=50,
    shuffle=True,
    show_metric=True,
    snapshot_step=200,
    snapshot_epoch=True,
    run_id='emotion_recognition_resnet'
)

model.save('./SavedModels/model_resnet.tfl')

score = model.evaluate(X_test, Y_test, batch_size=50)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
print("end")
