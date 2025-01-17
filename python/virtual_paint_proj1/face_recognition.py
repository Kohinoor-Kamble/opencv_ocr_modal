import cv2 as cv
import os
#used for the matrix m
import numpy as np


DIR=r"C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\faces"

cascade_image=cv.CascadeClassifier(r'C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\haarcascade_frontalface_default.xml')


def capture_image(person_name):
    path_person=os.path.join(DIR,person_name)
    os.mkdir(path_person)

    cap = cv.VideoCapture(0)
    # count=0

    while True:
        success, frame=cap.read()
        cv.imshow('Image',frame)

        if not success:
            break
        
        #convert the grey
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces=cascade_image.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

       #crop the faces
        for (x,y,w,h) in faces:
            face_ro=gray[y:y+h,x:x+w]


            #save the path in peson directroy
            face_path=os.path.join(path_person,f"{person_name}.jpg")
            cv.imwrite(face_path, face_ro)
            # count = count + 1

            #draw a rectange  around the face
            rectangle=cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),thickness=3)

        cv.imshow('Detected face capture',frame)
        
        key=cv.waitKey(1) & 0xff
        if key == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()          

    print(f"Captured image of {person_name}. stored in{face_path}")   

# print(person)
# person_name="balu"

# people=['Shahrukh Khan','elon musk','Salman Khan','Narendra modi','harsh','sagar']
people=[]

def create_model():
 
    features=[]
    labels=[]
   
#finding path directories of path
    for person in os.listdir(DIR):
        path=os.path.join(DIR,person)
       
        if not os.path.isdir(path):
                continue
             
   
        label=len(people) - 1

        print(people)
        print(labels)
      #read the path directories of person directory
        for image in os.listdir(path):
            image_path=os.path.join(path, image)
            print(image_path)

             #read the image path
            image_array=cv.imread(image_path)
            gray=cv.cvtColor(image_array,cv.COLOR_BGR2GRAY)
            cv.imshow('grey',gray)
          
            features.append(gray)
            labels.append(label)

            print(people)

# print(features)
#call the function




    print(f'Length of the feature is {len(features)}')
    print(f'length of the labels is {len(labels)}')
    # print(people)


    features=np.array(features, dtype='object')
    labels=np.array(labels)
    #make the instance of the face
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    #train the recognizer to features list and label list
    face_recognizer.train(features,labels)

    face_recognizer.save('face.yml')
    np.save('people.npy',people)
    np.save('features.npy',features)
    np.save('labels.npy',labels)
    
person_name=input(f"Enter the person name: ")
print(person_name)
capture_image(person_name)
people.append(person_name)
print(people)
create_model()






# p=[]

# for i in os.listdir(r"C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\faces"):
#     p.append(i)

# print(p)
         
