import sys
import tensorflow
import tflearn
from tflearn import *
from util import getData

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

layers = []


def network():
    network = input_data(shape=[None, 48, 48, 1])

    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.3)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.3)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.3)
    network = local_response_normalization(network)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.7)  # need to change if needed 0.7 -> 0.5
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.7)  # need to change if needed 0.7 -> 0.5
    network = fully_connected(network, len(EMOTIONS), activation='softmax')

    network = regression(network,
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)  # need to check with different learning rate

    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def getrawnetwork():
    return network()


def getsavednetwork(savepath='./SavedModels/model_A.tfl'):
    model = getrawnetwork()
    model.load(savepath)
    return model


def train(cont=False):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()  # need to train on different data sets
    X_train = X_train.transpose((0, 2, 3, 1))
    X_valid = X_valid.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()  # to reset the model graph. problem with loading the weights
    if cont:
        model = getsavednetwork()  # need to change back to getrawnetwork()
    else:
        model = getrawnetwork()
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


def test():
    _, _, _, _, X_test, Y_test = getData()  # need to test on different datasets
    X_test = X_test.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()

    model = getsavednetwork()
    score = model.evaluate(X_test, Y_test, batch_size=50)
    print('Test accuarcy: %0.4f%%' % (score[0] * 100))


def predict(X, val=True):
    tensorflow.reset_default_graph()

    model = getsavednetwork()
    if val:
        X = X.transpose((0, 2, 3, 1))
    pred = model.predict(X)
    print("Prediction : ", pred)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train()
        test()
    elif sys.argv[1] == 'train':
        if len(sys.argv) == 3 and sys.argv[2] == 'continue':
            train(cont=True)
        else:
            train()
    elif sys.argv[1] == 'test':
        test()
