import cv2 as cv
import os
import numpy as np
# people=['elon musk','Narendra modi','Salman Khan','Shahrukh Khan','harsh','sagar','samrat']

features=np.load('features.npy', allow_pickle=True)
labels=np.load('labels.npy')
people=np.load('people.npy',allow_pickle=True)

#load the harcascade
cascade_image=cv.CascadeClassifier(r'C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\haarcascade_frontalface_default.xml')
#instan of face
face_verify=cv.face.LBPHFaceRecognizer_create()
#lbphfacerecognizer read from face.yml model
face_verify.read('face.yml')
#read the image as input to detect the face
# image=cv.imread(r"C:\Users\kkamb\OneDrive\Desktop\opencv_ocr_modal\python\opencv\photos\Elon_Musk_Royal_Society_crop.jpg")
cap=cv.VideoCapture(0)

while True:
    success, frame=cap.read()
#convert this image into gray
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #detect the face 
    face=cascade_image.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
   #crop the face for detect the person
    for (x,y,w,h) in face:
        face_roi=gray[x:x+h,y:y+h]

        label,confidence=face_verify.predict(face_roi)
        print(f'label={label} with the confidence {confidence}');
        person_name=people[label]
    #create a text on the image using people list label
        cv.putText(frame,person_name,(30,30),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
        #creat a rectangle on coordinates 
        rectangle=cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),thickness=3)

    cv.imshow('Face Recognition',frame)

    key= cv.waitKey(1) & 0xff
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
