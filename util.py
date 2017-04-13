import numpy as np


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def getImageData():
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()

    N, D = X_train.shape
    d = int(np.sqrt(D))
    X_train = X_train.reshape(N, 1, d, d)
    np.save('train_data.npy', X_train)
    np.save('train_label.npy', Y_train)

    N, D = X_valid.shape
    d = int(np.sqrt(D))
    X_valid = X_valid.reshape(N, 1, d, d)
    np.save('valid_data.npy', X_valid)
    np.save('valid_label.npy', Y_valid)

    N, D = X_test.shape
    d = int(np.sqrt(D))
    X_test = X_test.reshape(N, 1, d, d)
    np.save('test_data.npy', X_test)
    np.save('test_label.npy', Y_test)

    # return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def getData(balance_ones=False):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y_train = []
    X_train = []
    Y_valid = []
    X_valid = []
    Y_test = []
    X_test = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            usage = row[2].rstrip()  # to remove '\n' at the end
            if usage == 'Training':
                Y_train.append(int(row[0]))
                X_train.append([int(p) for p in row[1].split()])
            if usage == 'PublicTest':
                # print("in valid")
                Y_valid.append(int(row[0]))
                X_valid.append([int(p) for p in row[1].split()])
            if usage == 'PrivateTest':
                Y_test.append(int(row[0]))
                X_test.append([int(p) for p in row[1].split()])

    # print("Test s getData: ", X_test, Y_test)
    # print("Valid s getData: ", X_valid.shape, Y_valid.shape)
    # print("Trains  getData: ", X_train.shape, Y_train.shape)

    X_train, Y_train = np.array(X_train) / 255.0, np.array(Y_train)
    X_valid, Y_valid = np.array(X_valid) / 255.0, np.array(Y_valid)
    X_test, Y_test = np.array(X_test) / 255.0, np.array(Y_test)

    Y_train = dense_to_one_hot(Y_train)
    Y_valid = dense_to_one_hot(Y_valid)
    Y_test = dense_to_one_hot(Y_test)

    '''
    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1] * len(X1)))
    '''
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
