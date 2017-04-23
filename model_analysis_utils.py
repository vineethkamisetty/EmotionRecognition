from cnn import *
from util import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_prediction_matrix(y_true, y_pred, cmap=plt.cm.Blues):
    """
    :param y_true: True labels 
    :param y_pred: Predicted labels
    :param cmap: default matrix color to be plotted
    :return: Genrates matrix image plot
    """
    labels = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(0, 6):
        for j in range(0, 6):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    ax.set_title('Prediction Matrix')
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def prediction_matrix(model, X, Y):
    """
    :param model: loaded model
    :param X: images data nd array
    :param Y: labels data nd array
    :return: matrix plot for predicted labels
    """
    X = X.transpose((0, 2, 3, 1))
    print("start prediction")
    y_prob = []

    # runs for 28,000 images ( 280 * 100 batch size)
    batch_size = 100
    iterations = 35
    for i in range(0, iterations):  # change iterations : each batch takes 100 images
        batch = model.predict(X[i * batch_size:(i + 1) * batch_size])
        for batch_element in batch:
            y_prob.append(batch_element)
        print("images predicted ", (i + 1) * batch_size)

    print("end prediction")

    # get max value from softmax output
    y_predicted = [np.argmax(prob) for prob in y_prob]
    y_true = [np.argmax(true) for true in Y]

    # make sure the size of y_true and y_predicted to be same
    plot_prediction_matrix(y_true[0:batch_size * (iterations)], y_predicted, cmap=plt.cm.Oranges)


def softmax_graph_for_sample_images(model, X_test, Y_test):
    """
    :param model: loaded model
    :param X_test: images data nd array
    :param Y_test: labels data nd array
    :return: created softmax histogram plot
    """

    # RaFD image data set
    # images_train = np.load("./RafD/RaFD_images_train.npy")
    # pixels = images_train[105][0] / 255

    index = 15
    pixels = X_test[index]
    y_true = np.argmax(Y_test[index])
    image48 = pixels.reshape(1, 48, 48, 1)

    result = model.predict(image48)
    EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(6, 6, 1)
    # predicted emotion
    # plt.xlabel(EMOTIONS[result[0].index(max(result[0]))], color='blue', fontsize=14)
    plt.xlabel("True label:" + EMOTIONS[y_true], color='blue', fontsize=14)

    plt.imshow(image48.reshape(48, 48), cmap=cm.gray)

    # fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(6, 6, 2)
    y_prob = np.array(result[0])
    ax.bar(np.arange(0, 6), y_prob, color=['red', 'blue', 'green', 'black', 'yellow', 'orange'], alpha=0.5)
    ax.set_xticks(np.arange(0.0, 6.5, 1))
    ax.set_xticklabels(EMOTIONS, rotation=90, fontsize=10)
    ax.set_yticks(np.arange(0.0, 1.1, 0.5))
    plt.tight_layout()
    plt.show()


def top2_accuracy(model, X_test, Y_test):
    """
    :param model: loaded model
    :param X_test: image data nd array
    :param Y_test: label data nd array
    :return: prints Top-2 accuracy for FERC test data set
    """
    X_test = X_test.transpose((0, 2, 3, 1))
    print("start testing")

    y_prob = []
    batch_size = 100
    for i in range(0, 35):  # change iterations : each batch takes 100 images
        batch = model.predict(X_test[i * batch_size:(i + 1) * batch_size])
        for s in batch:
            y_prob.append(s)
        print("images processed ", (i + 1) * batch_size)
    print("end testing")

    count = 0
    for i in range(0, len(y_prob)):
        max_index = np.argmax(y_prob[i])
        y_prob[i][max_index] = 0
        second_max = np.argmax(y_prob[i])
        if max_index == np.argmax(Y_test[i]) or second_max == np.argmax(Y_test[i]):
            count = count + 1
    print("Top 2 Test Accuracy : ", (count / 3500) * 100)


if __name__ == "__main__":
    model = get_saved_network_model()
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = get_data()

    # final prediction confusion matrix across 6 emotions
    prediction_matrix(model, X_test, Y_test)

    # Evaluating top:2 accuracy
    top2_accuracy(model, X_test, Y_test)

    # plot histogram for softmax output
    softmax_graph_for_sample_images(model, X_test, Y_test)
