from statistics import mode
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

USE_WEBCAM = True  # If false, loads video file source
COORDINATE = 0, 0, 0, 0
EMOTION_MODE = 'neutral'
COLOR = 0
HAS_FACE = False

# parameters for loading data and images
emotion_model_path = 'models\\emotion_model.hdf5'

emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('models\\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []


def detemotions(frame):
    global COORDINATE, EMOTION_MODE, COLOR, HAS_FACE

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    HAS_FACE = False
    for COORDINATE in faces:

        x, y, width, height = COORDINATE
        x_off, y_off = emotion_offsets
        x1, x2, y1, y2 = x - x_off, x + width + x_off, y - y_off, y + height + y_off

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = gray_face.astype('float32') / 255.0
        gray_face = gray_face - 0.5
        gray_face = gray_face * 2.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            EMOTION_MODE = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            COLOR = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            COLOR = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            COLOR = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            COLOR = emotion_probability * np.asarray((0, 255, 255))
        else:
            COLOR = emotion_probability * np.asarray((0, 255, 0))

        COLOR = COLOR.astype(int)
        COLOR = COLOR.tolist()
        HAS_FACE = True

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return COORDINATE, EMOTION_MODE, COLOR, HAS_FACE
