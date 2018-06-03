import numpy as np
import cv2
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'data_set/'

def getImagewithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    faces=[]
    IDs=[]

    for imagePath in imagePaths:
        print(imagePath)
        faceImage = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImage,'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        print(Id)
        IDs.append(Id)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces

Ids,faces = getImagewithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()