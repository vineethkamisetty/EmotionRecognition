import os
import numpy as np

X_train_path = './FercData/train_data.npy'
Y_train_path = './FercData/train_label.npy'
X_valid_path = './FercData/valid_data.npy'
Y_valid_path = './FercData/valid_label.npy'
X_test_path = './FercData/test_data.npy'
Y_test_path = './FercData/test_label.npy'
Ferc_csv = './FercData/fer2013.csv'


def one_hot(labels, num_classes=6):
    """
    :param labels: array of labels to be converted
    :param num_classes: number of classes
    :return: return one-hot label vector
    """
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def check_if_present():
    if (os.path.isfile(X_train_path) and os.path.isfile(Y_train_path) and os.path.isfile(
            X_valid_path) and os.path.isfile(Y_valid_path) and os.path.isfile(X_test_path) and os.path.isfile(
          Y_test_path)):
        return True
    else:
        return False


def get_data():
    """
    :return: returns train, validation and test data-set in .npy format
    """
    if check_if_present() is False:
        print("Creating .npy files")
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_csv_data()

        n, d = x_train.shape
        d = int(np.sqrt(d))  # 2304 -> 48
        x_train = x_train.reshape(n, 1, d, d)
        np.save(X_train_path, x_train)
        np.save(Y_train_path, y_train)

        n, d = x_valid.shape
        d = int(np.sqrt(d))
        x_valid = x_valid.reshape(n, 1, d, d)
        np.save(X_valid_path, x_valid)
        np.save(Y_valid_path, y_valid)

        n, d = x_test.shape
        d = int(np.sqrt(d))
        x_test = x_test.reshape(n, 1, d, d)
        np.save(X_test_path, x_test)
        np.save(Y_test_path, y_test)

    return np.load(X_train_path), np.load(Y_train_path), np.load(X_valid_path), np.load(Y_valid_path), np.load(
        X_test_path), np.load(Y_test_path)


def get_csv_data():
    y_train = []
    x_train = []
    y_valid = []
    x_valid = []
    y_test = []
    x_test = []
    first = True
    for line in open(Ferc_csv):
        if first:
            first = False
        else:
            row = line.split(',')
            usage = row[2].rstrip()  # remove '\n' at the end
            if row[0] != '0':
                row[0] = int(row[0]) - 1  # merge disgust with anger and then shifting each emotion indexes
            if usage == 'Training':
                y_train.append(int(row[0]))
                x_train.append([int(p) for p in row[1].split()])
            if usage == 'PublicTest':
                y_valid.append(int(row[0]))
                x_valid.append([int(p) for p in row[1].split()])
            if usage == 'PrivateTest':
                y_test.append(int(row[0]))
                x_test.append([int(p) for p in row[1].split()])

    x_train, y_train = np.array(x_train) / 255.0, np.array(y_train)
    x_valid, y_valid = np.array(x_valid) / 255.0, np.array(y_valid)
    x_test, y_test = np.array(x_test) / 255.0, np.array(y_test)

    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)
    y_test = one_hot(y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
