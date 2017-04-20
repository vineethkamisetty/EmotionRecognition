import os
import numpy as np

X_train_path = './FercData/train_data.npy'
Y_train_path = './FercData/train_label.npy'
X_valid_path = './FercData/valid_data.npy'
Y_valid_path = './FercData/valid_label.npy'
X_test_path = './FercData/test_data.npy'
Y_test_path = './FercData/test_label.npy'
Ferc_csv = './FercData/fer2013.csv'


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def checkispresent():
    if (os.path.isfile(X_train_path) and os.path.isfile(Y_train_path) and os.path.isfile(
            X_valid_path) and os.path.isfile(Y_valid_path) and os.path.isfile(X_test_path) and os.path.isfile(
        Y_test_path)):
        return True
    else:
        return False


def getData():
    if checkispresent() is False:
        print("Creating .npy files")
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getcsvdata()

        N, D = X_train.shape
        d = int(np.sqrt(D))
        X_train = X_train.reshape(N, 1, d, d)
        np.save(X_train_path, X_train)
        np.save(Y_train_path, Y_train)

        N, D = X_valid.shape
        d = int(np.sqrt(D))
        X_valid = X_valid.reshape(N, 1, d, d)
        np.save(X_valid_path, X_valid)
        np.save(Y_valid_path, Y_valid)

        N, D = X_test.shape
        d = int(np.sqrt(D))
        X_test = X_test.reshape(N, 1, d, d)
        np.save(X_test_path, X_test)
        np.save(Y_test_path, Y_test)

    return np.load(X_train_path), np.load(Y_train_path), np.load(X_valid_path), np.load(Y_valid_path), np.load(
        X_test_path), np.load(Y_test_path)


def getcsvdata():
    Y_train = []
    X_train = []
    Y_valid = []
    X_valid = []
    Y_test = []
    X_test = []
    first = True
    for line in open(Ferc_csv):
        if first:
            first = False
        else:
            row = line.split(',')
            usage = row[2].rstrip()  # to remove '\n' at the end

            if row[0] != '0':
                row[0] = int(row[0]) - 1
            if usage == 'Training':
                Y_train.append(int(row[0]))
                X_train.append([int(p) for p in row[1].split()])
            if usage == 'PublicTest':
                Y_valid.append(int(row[0]))
                X_valid.append([int(p) for p in row[1].split()])
            if usage == 'PrivateTest':
                Y_test.append(int(row[0]))
                X_test.append([int(p) for p in row[1].split()])

    X_train, Y_train = np.array(X_train) / 255.0, np.array(Y_train)
    X_valid, Y_valid = np.array(X_valid) / 255.0, np.array(Y_valid)
    X_test, Y_test = np.array(X_test) / 255.0, np.array(Y_test)

    Y_train = dense_to_one_hot(Y_train, 6)
    Y_valid = dense_to_one_hot(Y_valid, 6)
    Y_test = dense_to_one_hot(Y_test, 6)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
