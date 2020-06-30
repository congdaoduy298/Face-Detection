# Face Detection 

import cv2 
import os 
import numpy as np 
from face_recognition import face_locations, load_image_file

"""
    Face haar cascade detection 
"""
def cascadeFaceDetection(img_path):
    # Load haar cascade pretrain model 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (1080, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite('cascade_image.jpg', img)
    return faces, img 


def loadModel(model_path):
    # Model was trained on PyTorch 
    # So we must load model to openCV 
    model = cv2.dnn.readNetFromTorch(model_path)
    return model


# Convert original image to blob image to reduce noise for photo due to lighting 
def blobImage(img, scale_factor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)):
    # Use bloFromImages for batches, load and process multiple images
    imageBlob = cv2.dnn.blobFromImage(img, scalefactor=scale_factor, 
                                    size=size, mean=mean, swapRB=False)
    return imageBlob


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def faceRecognition(img_path, single=True, show=True):
    img = load_image_file(img_path)
    faces = face_locations(img)
    n = len(faces)
    if single:
        faces = [faces[0]]
    if not show:
        if n == 0:
            return None
        else:
            return faces
    # Draw rectangle of faces 
    for (top, right, bottom, left) in faces:
        img = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Save output image
    cv2.imwrite('detected_image.jpg', img)
    return img



if __name__ == "__main__":
    # Show output image from 2 ways 
    show = True

    _, out1 = cascadeFaceDetection('./test2.jpg')
    out2 = faceRecognition('./test2.jpg', single=False, show=show)


    if show == True:
        cv2.imshow('Haar Cascade Detected Image', out1)
        cv2.imshow('Face Recognition Detected Image', out2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    