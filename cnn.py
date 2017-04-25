import sys
import tensorflow
import tflearn
from tflearn import *
from util import *

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

save_path = './SavedModels/model_C/model_C.tfl'
model_id = 'emotion_recognition_C'


def network_model():
    """
    Base Network Model
    :return: returns the network
    """
    x = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 48, 48, 1])
    network = input_data(placeholder=x, shape=[None, 48, 48, 1])

    '''
    copy/import models from final_Models.txt or write your own model
    '''
    conv_1 = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(conv_1, 2, strides=2)
    network = dropout(network, 0.5)
    network = local_response_normalization(network)
    conv_2 = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(conv_2, 2, strides=2)
    network = dropout(network, 0.5)
    network = local_response_normalization(network)
    conv_3 = conv_2d(network, 128, 5, activation='relu')
    network = dropout(conv_3, 0.5)
    fc_1 = fully_connected(network, 1024, activation='relu')
    network = dropout(fc_1, 0.5)
    fc_2 = fully_connected(network, 1024, activation='relu')
    network = dropout(fc_2, 0.5)
    network = fully_connected(network, len(EMOTIONS), activation='softmax')

    network = regression(network,
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def get_raw_network_model():
    """
    :return: returns the un-trained network
    """
    return network_model()


def get_saved_network_model(saved_path=save_path):
    """
    Loads specified network weights in the path
    :param saved_path: path of the saved model. Needed to load the model weights for predicting and to continue the 
    training of model
    :return: returns the model with weights
    """
    model = get_raw_network_model()
    model.load(saved_path)
    return model


def train(cont=False):
    """
    Training the model
    :param cont: if True, it will load specified network model, so that training can be resumed. if False, trains 
    network from starting
    :return: None
    """
    x_train, y_train, x_valid, y_valid, y_test, y_test = get_data()
    x_train = x_train.transpose((0, 2, 3, 1))  # changing array from [?,1,48,48] to [?,48,48,1]
    x_valid = x_valid.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights
    if cont:
        model = get_saved_network_model()
    else:
        model = get_raw_network_model()
    model.fit(
        x_train, y_train,
        validation_set=(x_valid, y_valid),
        n_epoch=50,
        batch_size=100,
        shuffle=True,
        show_metric=True,
        snapshot_step=200,
        snapshot_epoch=True,
        run_id=model_id
    )
    model.save(save_path)


def test():
    """
    Testing the Model
    Function to call for testing test_data of ferc data-set
    :return: None
    """
    _, _, _, _, x_test, y_test = get_data()
    x_test = x_test.transpose((0, 2, 3, 1))

    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights

    model = get_saved_network_model()
    score = model.evaluate(x_test, y_test, batch_size=50)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))


def predict(x):
    """
    Used for predicting the emotion
    :param x: image in numpy array format . Needs to be in [1,1,48,48] dimension
    :return: returns an array of probabilities of each emotion
    """
    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights

    model = get_saved_network_model()
    x = x.transpose((0, 2, 3, 1))
    prediction = model.predict(x)
    print("Prediction : ", prediction)
    return prediction


def rafd_test():
    """
    Testing RafD data-set and print accuracy
    :return: None
    """
    x_test = np.load('./RafD/RAFD_images.npy')
    y_test = np.load('./RafD/RAFD_labels.npy')

    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights
    x_test = x_test[1000:]
    y_test = y_test[1000:]

    model = get_saved_network_model(saved_path='./SavedModels/model_C_rafd/model_C_rafd.tfl')

    score = model.evaluate(x_test.transpose((0, 2, 3, 1)), y_test, batch_size=50)
    print(score)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))


def rafd_train():
    """
    Testing RafD data-set and print accuracy
    :return: None
    """
    x_test = np.load('./RafD/RAFD_images.npy')
    y_test = np.load('./RafD/RAFD_labels.npy')
    x_test = x_test[0:1000]
    y_test = y_test[0:1000]
    print(x_test.shape)
    print(y_test.shape)

    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights

    x_train = x_test.transpose((0, 2, 3, 1))  # changing array from [?,1,48,48] to [?,48,48,1]

    tensorflow.reset_default_graph()  # reset the model graph. problem with loading the weights
    model = get_saved_network_model()

    model.fit(
        x_train, y_test,
        n_epoch=20,
        batch_size=100,
        shuffle=True,
        show_metric=True,
        snapshot_step=200,
        snapshot_epoch=True,
        run_id=model_id + '_rafd'
    )
    model.save('./SavedModels/model_C_rafd/model_C_rafd.tfl')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        train()
        test()
    elif sys.argv[1] == 'train':
        if len(sys.argv) == 3 and sys.argv[2] == 'continue':
            train(cont=True)
        else:
            train()
        test()
    elif sys.argv[1] == 'rafd_test':
        rafd_test()
    elif sys.argv[1] == 'rafd_train':
        rafd_train()
        rafd_test()
