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
    conv_1 = conv_2d(network, 64, 5, activation='relu', bias=False)
    # Residual blocks
    res_1 = residual_bottleneck(conv_1, 3, 16, 64)
    res_2 = residual_bottleneck(res_1, 1, 32, 128, downsample=True)
    res_3 = residual_bottleneck(res_2, 2, 32, 128)
    res_4 = residual_bottleneck(res_3, 1, 64, 256, downsample=True)
    res_5 = residual_bottleneck(res_4, 2, 64, 256)
    network = batch_normalization(res_5)
    network = activation(network, 'relu')
    network = global_avg_pool(network)
    # Regression
    fc_1 = fully_connected(network, 3072, activation='relu')
    network = fully_connected(fc_1, len(EMOTIONS), activation='softmax')

    network = regression(network,
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)  # need to check with different learning rate

    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def get_raw_network_model():
    """
    :return: returns the un-trained network
    """
    return network_model()


def get_saved_network_model(savepath='./SavedModels/model_resnet/model_resnet.tfl'):
    """
    Loads specified network weights in the path
    :param savepath: path of the saved model. Needed to load the model weights for predicting and to continue the 
    training of model
    :return: returns the model with weights
    """
    model = get_raw_network_model()
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
        model = get_saved_network_model()  # need to change back to getrawnetwork()
    else:
        model = get_raw_network_model()
    model.fit(
        x_train, y_train,
        validation_set=(x_valid, y_valid),
        n_epoch=10,
        batch_size=50,
        shuffle=True,
        show_metric=True,
        snapshot_step=200,
        snapshot_epoch=True,
        run_id='emotion_recognition_resnet'
    )
    model.save('./SavedModels/model_resnet/model_resnet.tfl')


def test():
    """
    Testing the Model
    Function to call for testing test_data of ferc data-set
    :return: None
    """
    _, _, _, _, x_test, y_test = getData()  # need to test on different datasets
    x_test = x_test.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()

    model = get_saved_network_model()
    score = model.evaluate(x_test, y_test, batch_size=50)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))


def predict(x):
    """
    Used for predicting the emotion
    :param x: image in numpy array format . Needs to be in [1,1,48,48] dimension
    :return: returns an array of probabilities of each emotion
    """
    tensorflow.reset_default_graph()

    model = get_saved_network_model()
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
