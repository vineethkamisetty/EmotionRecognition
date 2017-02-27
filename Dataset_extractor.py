import pandas as pd
import numpy as np
import csv
import cv2

FILE_PATH = 'fer2013.csv'
data = pd.read_csv(FILE_PATH)

labels_train = []
images_train = []
labels_validate = []
images_validate = []
labels_test = []
images_test = []
usage = []
index = 1
total = data.shape[0]
for index, row in data.iterrows():
    emotion = row['emotion']
    image = np.fromstring(str(row['pixels']), dtype=np.uint8, sep=' ')
    usage = row['Usage']
    if usage == 'Training':
        if image is not None:
            labels_train.append(emotion)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
            images_train.append(image)
        else:
            print("Error")
    elif usage == 'PrivateTest':
        if image is not None:
            labels_validate.append(emotion)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
            images_validate.append(image)
        else:
            print("Error")
    elif usage == 'PublicTest':
        if image is not None:
            labels_test.append(emotion)
            image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
            images_test.append(image)
        else:
            print("Error")
    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

fl = open('images_train1.csv', 'w', newline='')
writer = csv.writer(fl)
for values in images_train:
    if len(values):
        writer.writerow(values)
fl.close()

fl = open('images_validate1.csv', 'w', newline='')
writer = csv.writer(fl)
for values in images_validate:
    if len(values):
        writer.writerow(values)
fl.close()

fl = open('images_test1.csv', 'w', newline='')
writer = csv.writer(fl)
for values in images_test:
    if len(values):
        writer.writerow(values)
fl.close()

np.savetxt("labels_train1.csv", labels_train,fmt='%i', delimiter=",")
np.savetxt("labels_validate1.csv", labels_validate,fmt='%i', delimiter=",")
np.savetxt("labels_test1.csv", labels_test,fmt='%i', delimiter=",")

