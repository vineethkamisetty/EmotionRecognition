import sys
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from util import getData

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def network():
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
    return model


def getrawnetwork():
    return network()


def getsavednetwork(savepath='./SavedModels/model.tfl'):
    model = getrawnetwork()
    model.load(savepath)
    return model


def train():
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()
    X_train = X_train.transpose((0, 2, 3, 1))
    X_valid = X_valid.transpose((0, 2, 3, 1))

    model = getrawnetwork()
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
    model.save('./SavedModels/model.tfl')


def test():
    _, _, _, _, X_test, Y_test = getData()
    X_test = X_test.transpose((0, 2, 3, 1))

    model = getsavednetwork()
    score = model.evaluate(X_test, Y_test, batch_size=50)
    print('Test accuarcy: %0.4f%%' % (score[0] * 100))


def predit(X):
    model = getsavednetwork()
    X = X.transpose((0, 2, 3, 1))
    pred = model.predict(X)
    print
    "Prediction : ", pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train()
        test()
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
