from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

import numpy as np
import cv2

## Face detection models

#facedetect = cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_profileface.xml")
facedetect = cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
# facedetect2= cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
# facedetect3= cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# facedetect4= cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
# facedetect5= cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
# facedetect6= cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")


## Emotion recognition models

# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_alt2.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

## Face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Test1/venv\Include/recognizer/trainingData.yml")
id = 0
names = ("Siva", "Vidu")

####################

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

########################

cam = cv2.VideoCapture(0)

x_prev = 0
y_prev = 0
w_prev = 0
h_prev = 0

faces_prev = [0, 0, 0, 0]
id_prev = []
cordinates_prev = []

while True:

    ret, rgb_image = cam.read()
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray_image, 1.08, 5)
    if len(faces) == 0:
        for face_coordinates_prev in faces_prev:
            x, y, w, h = face_coordinates_prev
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(rgb_image, "TEST", (x, y + h + 60), cv2.QT_FONT_NORMAL, 1, (255, 255, 0), 2)
    else:
        id_prev = []
        for face_coordinates in faces:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
            x, y, w, h = face_coordinates
            # if abs(x_prev-x)<1and abs(y_prev-y)<1:
            #     x,y,w,h = cordinates_prev

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)

            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            Id, conf = recognizer.predict(gray_image[y:y+h,x:x+w])
            print(Id)
            print(conf)
            Id = names[Id - 1]

            if conf < 80:
                cv2.putText(rgb_image, str(Id) + "" + str(conf), (x, y + h + 60), cv2.QT_FONT_NORMAL, 1, (255, 255, 0),2)
            else:
                cv2.putText(rgb_image, "Unknown," + str(Id) + "" + str(conf), (x, y + h + 60), cv2.QT_FONT_NORMAL, 1, (255, 255, 0), 2)
            Id = None

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(rgb_image, emotion_mode, (x, y + h + 25), cv2.QT_FONT_NORMAL, 1, (255, 255, 0))

            # cordinates_prev.append(face_coordinates)

            id_prev.append(str(Id))

        faces_prev = faces

    cv2.imshow("Face", cv2.resize(rgb_image, (1200, 900)))
    if (cv2.waitKey(10) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
