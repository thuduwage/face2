import numpy as np
import cv2


facedetect = cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

cam = cv2.VideoCapture(0)

while (True):
    ret, img = cam.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray_img,1.1,5)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)

    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()