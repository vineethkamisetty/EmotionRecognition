import sys
import tensorflow
import tflearn
from tflearn import *
from util import getData

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

unit_values = []


def network_model():
    """
    Base Network Model
    :return: returns the network
    """
    x = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 48, 48, 1])
    network = input_data(placeholder=x, shape=[None, 48, 48, 1])

    '''
    copy/import models from final_Models.txt or write your model
    '''

    network = regression(network,
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)  # need to check with different learning rate

    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def getrawnetwork():
    """
    :return: returns the un-trained network
    """
    return network_model()


def getsavednetwork(savepath='./SavedModels/model_D.tfl'):
    """
    Loads specified network weights in the path
    :param savepath: path of the saved model. Needed to load the model weights for predicting and to continue the 
    training of model
    :return: returns the model with weights
    """
    model = getrawnetwork()
    model.load(savepath)
    return model


def train(cont=False):
    """
    Training the model
    :param cont: if True, it will load specified network model, so that training can be resumed. if False, trains 
    network from starting
    :return: None
    """
    x_train, y_train, x_valid, y_valid, y_test, y_test = getData()
    x_train = x_train.transpose((0, 2, 3, 1))
    x_valid = x_valid.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()  # to reset the model graph. problem with loading the weights
    if cont:
        model = getsavednetwork()  # need to change back to getrawnetwork()
    else:
        model = getrawnetwork()
    model.fit(
        x_train, y_train,
        validation_set=(x_valid, y_valid),
        n_epoch=50,
        batch_size=100,
        shuffle=True,
        show_metric=True,
        snapshot_step=200,
        snapshot_epoch=True,
        run_id='emotion_recognition_D'
    )
    model.save('./SavedModels/model_D.tfl')


def test():
    """
    Testing the Model
    Function to call for testing test_data of ferc data-set
    :return: None
    """
    _, _, _, _, x_test, y_test = getData()  # need to test on different datasets
    x_test = x_test.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()

    model = getsavednetwork()
    score = model.evaluate(x_test, y_test, batch_size=50)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))


def predict(x):
    """
    Used for predicting the emotion
    :param x: image in numpy array format . Needs to be in [1,1,48,48] dimension
    :return: returns an array of probabilities of each emotion
    """
    tensorflow.reset_default_graph()

    model = getsavednetwork()
    x = x.transpose((0, 2, 3, 1))
    prediction = model.predict(x)
    print("Prediction : ", prediction)
    return prediction


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
