import cv2
import numpy
from cnn import *

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
FACE_PIXELS = 48
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def format_image(image):
    """
    Takes image from camera app and extracts face, finally converst it into 48x48 gray scale image
    :param image: image frame from camera
    :return: converted image
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    # Check the number of faces found in the image
    if len(faces) <= 0:
        print("Face is not detected")
        return None

    # taking max face area in case of detecting multiple faces
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resizing the image to network input size i.e 48x48
    image = cv2.resize(image, (FACE_PIXELS, FACE_PIXELS), interpolation=cv2.INTER_CUBIC) / 255.
    return image


if __name__ == "__main__":

    video_capture = cv2.VideoCapture(0)

    emojis = []
    for index, emotion in enumerate(EMOTIONS):
        emojis.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    # load the saved network
    model = get_saved_network_model()

    while True:
        ret, frame = video_capture.read()
        image48 = format_image(frame)
        result_softmax = None
        if image48 is not None:
            image48 = image48.reshape(1, 48, 48, 1)
            # Predict result with network
            result_softmax = model.predict(image48)

        # Appending the emotion image to Frame
        if result_softmax is not None:
            emoji = emojis[numpy.argmax(result_softmax[0])]  # get max index of softmax values
            for c in range(0, 3):
                frame[200:320, 10:130, c] = emoji[:, :, c] * (emoji[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (
                    1.0 - emoji[:, :, 3] / 255.0)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("End of Session")
