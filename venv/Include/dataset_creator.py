import numpy as np
import cv2

facedetect = cv2.CascadeClassifier("C:/Test1/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)

id = input('enter id: ')
sam_num=0

while (True):
    ret, img = cam.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray_img,1.08,5)

    for x,y,w,h in faces:
        sam_num = sam_num + 1
        cv2.imwrite("data_set/user."+str(id)+"."+str(sam_num)+".jpg",gray_img[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
        cv2.waitKey(100)

    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sam_num>=100):
        break

cam.release()
cv2.destroyAllWindows()