import numpy
import tensorflow
import tflearn
from tflearn import *
from util import *
import matplotlib.pyplot as plt

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = get_data()

x = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 48, 48, 1], name="x-in")
network = input_data(placeholder=x, shape=[None, 48, 48, 1])

'''
    copy/import models from final_Models.txt or write your model
'''

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
network = dropout(fc_1, 0.7)
fc_2 = fully_connected(network, 1024, activation='relu')
network = dropout(fc_2, 0.7)
network = fully_connected(network, len(EMOTIONS), activation='softmax')

network = regression(network,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)  # need to check with different learning rate

model = tflearn.DNN(network, tensorboard_verbose=3)

sess = tensorflow.Session()
init = tensorflow.global_variables_initializer()
sess.run(init)

save_path = './SavedModels/model_A/model_A.tfl'
model.load(save_path)


def plot(layer, img):
    """
    :param layer: name of the layer
    :param img: image
    :return: None
    """
    units = sess.run(layer, feed_dict={x: img})
    filters = units.shape[3]
    fig = plt.figure(figsize=(20, 20))
    for i in range(filters):
        ax = fig.add_subplot(filters / 16, 16, i + 1)
        ax.imshow(units[0, :, :, i], cmap='Blues')
        plt.xticks(numpy.array([]))
        plt.yticks(numpy.array([]))
        plt.tight_layout()
    plt.show()


image = X_valid[0:1, :, :, :].transpose((0, 2, 3, 1))  # converting into [1,48,48,1] dimensions
plot(conv_2, image)
