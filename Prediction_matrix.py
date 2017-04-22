import numpy as np
from CNN import *
from util import getData

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_prediction_matrix(y_true, y_pred, cmap=plt.cm.Blues):
    labels = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # fig.colorbar(matrix)
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

model = getsavednetwork();
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData()
X_train = X_train.transpose((0, 2, 3, 1))
print(Y_train.shape)
print("start prediction")
y_prob = []
for i in range(0,10):
    small = model.predict(X_train[i*100:(i+1)*100])
    for s in small:
        y_prob.append(s)
    print("images processed ",(i+1)*100)
print("end prediction")

y_pred = [np.argmax(prob) for prob in y_prob]
y_true = [np.argmax(true) for true in Y_train]

plot_prediction_matrix(y_true[0:100*(i+1)], y_pred, cmap=plt.cm.Oranges)
