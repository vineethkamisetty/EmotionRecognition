import cv2
import time
import sys
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import six

from CNN import *

EMOTIONS = ['angry','fearful', 'happy', 'sad', 'surprised', 'neutral']
SIZE_FACE = 48
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )

  # Check the number of faces found in the image
  if len(faces) <= 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resizing the image to network input size i.e 48x48
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("Error during resize")
    return None
  return image


if __name__ == "__main__":
  video_capture = cv2.VideoCapture(0)
  #font = cv2.FONT_HERSHEY_SIMPLEX

  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

  model = getsavednetwork()

  while True:
    ret, frame = video_capture.read()
    image48 = format_image(frame)
    result = None
    if image48 is not None:
      image48 = image48.reshape(1, 48, 48, 1)
      print(image48)
      # Predict result with network
      result = model.predict(image48)

    # Appending the emotion image to Frame
    if result is not None:
      face_image = feelings_faces[result[0].index(max(result[0]))]
      for c in range(0, 3):
        frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video_capture.release()
  cv2.destroyAllWindows()